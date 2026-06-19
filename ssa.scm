;;; array-morphisms-ssa.scm
;;;
;;; SSA IR for fused forward+backward computation (Approach B).
;;;
;;; Compiles a morph-variable loss node into a flat SSA program once,
;;; then replays the joint forward+backward computation every training step.
;;; Eliminates the lazy/eager boundary, fresh-copy overhead, and missed
;;; fusion opportunities that arise when forward and backward pass each
;;; build separate lazy trees.
;;;
;;; Phases:
;;;   B-1  morphism-to-ssa  -- DFS over morphism-expr tree → SSA program
;;;   B-2  ssa-vjp          -- symbolic VJP over SSA bindings → extended program
;;;   B-3  ssa-realize      -- sequential executor (no context pooling)
;;;   B-4  ssa-realize/ctx  -- context-pooled executor
;;;
;;; Key invariants:
;;;   - Constants (concrete-array leaves) are stored in a hash-table keyed by
;;;     gensym cid-symbols.  Parameters are concrete-arrays whose underlying
;;;     SRFI-4 vectors are updated in-place by the optimizer; the SSA constants
;;;     table sees the updated values automatically each step.
;;;   - Binding IDs are gensym symbols (bid-NNN for forward, adj-NNN for
;;;     backward adjoints) serving as hash-table keys in the value table.
;;;   - Reduction ops use (list 'reduce rop) as their op field to distinguish
;;;     them from plain element-wise ops.

(module array-morphisms-ssa

  (;; Data-type predicates/accessors
   ssa-value?
   ssa-binding-ref? ssa-const-ref?
   ssa-value-id

   ;; SSA records
   ssa-binding?
   ssa-binding-name ssa-binding-op ssa-binding-inputs
   ssa-binding-shape ssa-binding-dtype ssa-binding-meta
   ssa-program?
   ssa-program-constants ssa-program-morph-to-val
   ssa-program-bindings ssa-program-outputs ssa-program-n-params

   ;; Compilation
   morphism-to-ssa
   ssa-constant-id
   ssa-loss-binding-val

   ;; VJP
   ssa-vjp

   ;; Execution
   ssa-realize
   ssa-realize/ctx)

  (import scheme (chicken base))
  (import (only srfi-1 iota fold filter map for-each append-map filter-map))
  (import (only srfi-4
                f64vector f64vector-set! f64vector-length
                f32vector f32vector-set! f32vector-length))
  (import (only srfi-69
                make-hash-table
                hash-table-ref
                hash-table-ref/default
                hash-table-set!
                hash-table-walk
                eq?-hash))
  (import datatype matchable)
  (import array-morphisms-core)
  (import array-morphisms-index-fn)
  (import array-morphisms-basic-ops)
  (import array-morphisms-structural-ops)
  (import array-morphisms-blas-exec)
  (import array-morphisms-realization)
  (import array-morphisms-context)
  (import (prefix array-morphisms-grad am:))


;;; ============================================================
;;; SSA Value ADT
;;;
;;; An ssa-value identifies either a binding result (binding-ref) or
;;; a constant array (const-ref).  IDs are gensym symbols.
;;; ============================================================

(define-datatype ssa-value ssa-value?
  (binding-ref (id symbol?))
  (const-ref   (cid symbol?)))

(define (ssa-binding-ref? v)
  (cases ssa-value v (binding-ref (_) #t) (else #f)))

(define (ssa-const-ref? v)
  (cases ssa-value v (const-ref (_) #t) (else #f)))

(define (ssa-value-id v)
  "Return the gensym symbol identifying this ssa-value."
  (cases ssa-value v
    (binding-ref (id)  id)
    (const-ref   (cid) cid)))


;;; ============================================================
;;; SSA Records
;;; ============================================================

;; One binding in the SSA program.
;;   name   : gensym symbol (bid-NNN or adj-NNN); key in value table
;;   op     : symbol (e.g. 'add) or list (e.g. '(reduce mean))
;;   inputs : list of ssa-value
;;   shape  : vector (result shape)
;;   dtype  : symbol
;;   meta   : alist of extra info (index-fn, axes, perm, fn, …)
(define-record ssa-binding name op inputs shape dtype meta)

;; The compiled SSA program.
;;   constants   : hash-table cid-symbol → concrete-array
;;   morph-to-val: hash-table morphism-object (eq?) → ssa-value  (for ssa-constant-id)
;;   bindings    : list of ssa-binding in topological order
;;   outputs     : list of ssa-value  (loss first, then param grads)
;;   n-params    : count of trainable parameter constants
(define-record ssa-program constants morph-to-val bindings outputs n-params)


;;; ============================================================
;;; Phase B-1: morphism-to-ssa
;;;
;;; Post-order DFS over the morphism-expr tree rooted at loss-mv.
;;; concrete-array leaves become constants; morphism-expr / reduction-morphism
;;; nodes become SSA bindings.
;;; ============================================================

(define (morphism-to-ssa loss-mv)
  "Compile the morph-variable graph rooted at loss-mv into an SSA program.
   Returns an ssa-program with outputs = (list loss-binding-val) and n-params = 0."
  (let* ((constants   (make-hash-table))          ; cid-symbol → concrete-array
         (visited     (make-hash-table eq? eq?-hash)) ; morphism → ssa-value
         (bindings    '()))                         ; accumulated in topo order

    (define (emit-binding! op inputs shape dtype meta)
      (let* ((bid (gensym 'bid-))
             (b   (make-ssa-binding bid op inputs shape dtype meta)))
        (set! bindings (append bindings (list b)))
        (binding-ref bid)))

    (define (visit m)
      (or (hash-table-ref/default visited m #f)
          (cases array-morphism m

            (concrete-array (data shape strides offset dtype alloc-id batch-axis)
              (let* ((cid (gensym 'cid-))
                     (v   (const-ref cid)))
                (hash-table-set! constants cid m)
                (hash-table-set! visited m v)
                v))

            (morphism-expr (op operands index-fn shape dtype metadata batch-axis)
              (let* ((input-vals (map visit operands))
                     ;; Normalize: morph-transpose stores key 'permutation; we use 'perm
                     (norm-meta  (if (eq? op 'transpose)
                                     (let ((pe (assq 'permutation metadata)))
                                       (if pe
                                           (cons (cons 'perm (cdr pe))
                                                 (filter (lambda (x)
                                                           (not (eq? (car x) 'permutation)))
                                                         metadata))
                                           metadata))
                                     metadata))
                     (meta       (cons (cons 'index-fn index-fn) norm-meta))
                     (v          (emit-binding! op input-vals shape dtype meta)))
                (hash-table-set! visited m v)
                v))

            (reduction-morphism (rop operand reduce-axes index-fn shape dtype batch-axis)
              (let* ((src-val (visit operand))
                     (meta    (list (cons 'axes reduce-axes)
                                    (cons 'index-fn index-fn)
                                    (cons 'src-shape (morph-shape operand))
                                    (cons 'keepdims? (reduction-index-fn-keepdims? index-fn))))
                     (v       (emit-binding! (list 'reduce rop) (list src-val) shape dtype meta)))
                (hash-table-set! visited m v)
                v)))))

    (let* ((loss-m   (am:var-value loss-mv))
           (loss-val (visit loss-m)))
      (make-ssa-program constants visited bindings (list loss-val) 0))))


;;; ============================================================
;;; Accessors into a compiled program
;;; ============================================================

(define (ssa-constant-id prog m)
  "Return (const-ref cid) for morphism m if it is a constant in prog, else #f."
  (let ((v (hash-table-ref/default (ssa-program-morph-to-val prog) m #f)))
    (and v (ssa-const-ref? v) v)))

(define (ssa-loss-binding-val prog)
  "Return the ssa-value for the loss node (first output)."
  (car (ssa-program-outputs prog)))


;;; ============================================================
;;; Phase B-2: ssa-vjp
;;;
;;; Symbolic reverse-mode AD over SSA bindings.
;;; Appends backward bindings to fwd-prog's bindings and returns an
;;; extended ssa-program whose outputs are:
;;;   (loss-binding-val . param-grad-binding-vals)
;;; ============================================================

(define (ssa-vjp fwd-prog param-const-vals loss-binding-val)
  "Compute symbolic VJP of fwd-prog with respect to params.
   param-const-vals: list of (const-ref cid) ssa-values for trainable params.
   loss-binding-val: (binding-ref bid) ssa-value for the loss node.
   Returns extended ssa-program."

  (let* (;; Build shape/dtype lookup from forward bindings
         (fwd-bindings  (ssa-program-bindings fwd-prog))
         (binding-shape (make-hash-table))   ; bid-symbol → vector
         (binding-dtype (make-hash-table))   ; bid-symbol → symbol
         ;; Accumulate backward bindings here
         (bwd-bindings  '())
         ;; Adjoint table: bid-symbol → ssa-value (the accumulated dL/d(bid))
         (adjoint-val   (make-hash-table))
         ;; Param gradient table: cid-symbol → ssa-value
         (param-grad-val (make-hash-table))
         ;; Set of trainable param cid-symbols for fast membership test
         (param-cid-set  (make-hash-table))
         ;; Loss shape/dtype
         (loss-bid-sym  (ssa-value-id loss-binding-val)))

    ;; Index forward binding shapes and dtypes
    (for-each (lambda (b)
                (hash-table-set! binding-shape (ssa-binding-name b) (ssa-binding-shape b))
                (hash-table-set! binding-dtype (ssa-binding-name b) (ssa-binding-dtype b)))
              fwd-bindings)

    ;; Also index constant shapes/dtypes
    (hash-table-walk (ssa-program-constants fwd-prog)
      (lambda (cid m)
        (hash-table-set! binding-shape cid (morph-shape m))
        (hash-table-set! binding-dtype cid (morph-dtype m))))

    ;; Register trainable param cid-symbols
    (for-each (lambda (v)
                (hash-table-set! param-cid-set (ssa-value-id v) #t))
              param-const-vals)

    ;; Lookup helpers
    (define (val-shape v)
      (hash-table-ref/default binding-shape (ssa-value-id v) #f))
    (define (val-dtype v)
      (hash-table-ref/default binding-dtype (ssa-value-id v) #f))

    ;; emit! -- create a new backward SSA binding
    (define (emit! op inputs shape dtype meta)
      (let* ((adj-id (gensym 'adj-))
             (b      (make-ssa-binding adj-id op inputs shape dtype meta)))
        (set! bwd-bindings (append bwd-bindings (list b)))
        ;; Index into binding-shape/dtype tables immediately
        (hash-table-set! binding-shape adj-id shape)
        (hash-table-set! binding-dtype adj-id dtype)
        (binding-ref adj-id)))

    ;; Accumulate adjoint for a binding-ref input
    (define (accumulate-adjoint! bid-sym new-val shape dtype)
      (let ((existing (hash-table-ref/default adjoint-val bid-sym #f)))
        (if existing
            (let ((sum (emit! 'add (list existing new-val) shape dtype '())))
              (hash-table-set! adjoint-val bid-sym sum))
            (hash-table-set! adjoint-val bid-sym new-val))))

    ;; Accumulate gradient for a const-ref param input
    (define (accumulate-param-grad! cid-sym new-val shape dtype)
      (let ((existing (hash-table-ref/default param-grad-val cid-sym #f)))
        (if existing
            (let ((sum (emit! 'add (list existing new-val) shape dtype '())))
              (hash-table-set! param-grad-val cid-sym sum))
            (hash-table-set! param-grad-val cid-sym new-val))))

    ;; Dispatch adjoint accumulation based on ssa-value type
    (define (accumulate-input-adjoint! input-val new-val)
      (let ((shape (val-shape input-val))
            (dtype (val-dtype input-val)))
        (cases ssa-value input-val
          (binding-ref (bid)
            (accumulate-adjoint! bid new-val shape dtype))
          (const-ref (cid)
            (when (hash-table-ref/default param-cid-set cid #f)
              (accumulate-param-grad! cid new-val shape dtype))))))

    ;; emit-reduce-sum-to! -- reduce g-val to target-shape
    (define (emit-reduce-sum-to! g-val g-shape target-shape dtype)
      (let* ((g-rank (vector-length g-shape))
             (t-vec  (if (vector? target-shape)
                         target-shape
                         (list->vector target-shape)))
             (t-rank (vector-length t-vec)))
        ;; Step 1: sum leading extra dims
        (let* ((extra (- g-rank t-rank))
               (cur   (if (> extra 0)
                          (emit! (list 'reduce 'sum) (list g-val)
                                 (vector-drop-left g-shape extra)
                                 dtype
                                 (list (cons 'axes (iota extra))
                                       (cons 'keepdims? #f)
                                       (cons 'src-shape g-shape)))
                          g-val))
               (cur-shape (val-shape cur)))
          ;; Step 2: sum broadcast (size-1) dims
          (let loop ((k 0) (cur cur) (cur-shape cur-shape))
            (if (>= k t-rank)
                cur
                (let ((t-dim (vector-ref t-vec k))
                      (g-dim (vector-ref cur-shape k)))
                  (if (and (= t-dim 1) (> g-dim 1))
                      (let* ((new-shape (let* ((len (vector-length cur-shape))
                                              (nv  (make-vector len)))
                                         (do ((i 0 (+ i 1))) ((= i len) nv)
                                           (vector-set! nv i (vector-ref cur-shape i)))))
                             (_ (vector-set! new-shape k 1))
                             (r (emit! (list 'reduce 'sum) (list cur)
                                       new-shape dtype
                                       (list (cons 'axes (list k))
                                             (cons 'keepdims? #t)
                                             (cons 'src-shape cur-shape)))))
                        (loop (+ k 1) r new-shape))
                      (loop (+ k 1) cur cur-shape))))))))

    ;; vector-drop-left -- remove first n elements of a vector
    (define (vector-drop-left v n)
      (let* ((len (vector-length v))
             (new-len (- len n))
             (result (make-vector new-len)))
        (do ((i 0 (+ i 1)))
            ((= i new-len) result)
          (vector-set! result i (vector-ref v (+ i n))))))

    ;; emit-broadcast-grad! -- broadcast reduced gradient back to src-shape
    ;; src-shape: original input shape; reduced-axes: axes that were reduced
    ;; keepdims?: whether the reduction kept dimensions
    (define (emit-broadcast-grad! g-val g-shape src-shape reduced-axes keepdims? dtype)
      (let* ((in-rank (vector-length src-shape))
             ;; Compute keepdims shape (insert 1s at reduced positions)
             (kd-shape (list->vector
                        (map (lambda (i)
                               (if (member i reduced-axes) 1
                                   (vector-ref src-shape i)))
                             (iota in-rank))))
             ;; Reshape g to keepdims shape if needed
             (g-kd (if keepdims?
                       g-val
                       (emit! 'reshape (list g-val) kd-shape dtype '())))
             ;; Emit ones-like constant for src-shape
             (ones-data (allocate-typed-vector dtype (shape-size src-shape)))
             (_ (let loop ((i 0) (n (shape-size src-shape)))
                  (when (< i n)
                    (typed-vector-set! ones-data dtype i 1.0)
                    (loop (+ i 1) n))))
             (ones-const (make-morphism ones-data (vector->list src-shape) dtype))
             (ones-cid   (gensym 'cid-))
             (_ (hash-table-set! (ssa-program-constants fwd-prog) ones-cid ones-const))
             (_ (hash-table-set! binding-shape ones-cid src-shape))
             (_ (hash-table-set! binding-dtype ones-cid dtype))
             (ones-val (const-ref ones-cid)))
        ;; morph* broadcasts: g-kd * ones → src-shape
        (emit! 'mul (list g-kd ones-val) src-shape dtype '())))

    ;; emit-scalar-const! -- emit a scalar constant of given value
    (define (emit-scalar-const! value dtype)
      (let* ((data (allocate-typed-vector dtype 1))
             (_ (typed-vector-set! data dtype 0 value))
             (m   (make-morphism data '(1) dtype))
             (cid (gensym 'cid-)))
        (hash-table-set! (ssa-program-constants fwd-prog) cid m)
        (hash-table-set! binding-shape cid (vector 1))
        (hash-table-set! binding-dtype cid dtype)
        (const-ref cid)))

    ;; --- Seed: dL/dL = ones-like(loss) stored as a constant ---
    (let* ((loss-bid    (find-fwd-binding fwd-bindings loss-bid-sym))
           (loss-shape  (ssa-binding-shape loss-bid))
           (loss-dtype  (ssa-binding-dtype loss-bid))
           (seed-data   (allocate-typed-vector loss-dtype (shape-size loss-shape)))
           (_ (let loop ((i 0) (n (shape-size loss-shape)))
                (when (< i n)
                  (typed-vector-set! seed-data loss-dtype i 1.0)
                  (loop (+ i 1) n))))
           (seed-const  (make-morphism seed-data (vector->list loss-shape) loss-dtype))
           (seed-cid    (gensym 'cid-))
           (_ (hash-table-set! (ssa-program-constants fwd-prog) seed-cid seed-const))
           (_ (hash-table-set! binding-shape seed-cid loss-shape))
           (_ (hash-table-set! binding-dtype seed-cid loss-dtype))
           (seed-val    (const-ref seed-cid)))
      (hash-table-set! adjoint-val loss-bid-sym seed-val))

    ;; --- Backward sweep: iterate bindings in reverse order ---
    (for-each
     (lambda (b)
       (let* ((bid-sym  (ssa-binding-name b))
              (g-val    (hash-table-ref/default adjoint-val bid-sym #f)))
         (when g-val
           (let* ((op      (ssa-binding-op b))
                  (inputs  (ssa-binding-inputs b))
                  (shape   (ssa-binding-shape b))
                  (dtype   (ssa-binding-dtype b))
                  (meta    (ssa-binding-meta b))
                  (g-shape (val-shape g-val)))

             (cond

               ;; add(x, y) -- dx = reduce-sum-to(g, shape-x); dy = reduce-sum-to(g, shape-y)
               ((eq? op 'add)
                (let* ((x-val (list-ref inputs 0))
                       (y-val (list-ref inputs 1))
                       (x-shape (val-shape x-val))
                       (y-shape (val-shape y-val)))
                  (let ((dx (emit-reduce-sum-to! g-val g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))
                  (let ((dy (emit-reduce-sum-to! g-val g-shape y-shape dtype)))
                    (accumulate-input-adjoint! y-val dy))))

               ;; sub(x, y) -- dx = reduce-sum-to(g, shape-x); dy = reduce-sum-to(-g, shape-y)
               ((eq? op 'sub)
                (let* ((x-val (list-ref inputs 0))
                       (y-val (list-ref inputs 1))
                       (x-shape (val-shape x-val))
                       (y-shape (val-shape y-val)))
                  (let ((dx (emit-reduce-sum-to! g-val g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))
                  (let* ((neg-g (emit! 'negate (list g-val) g-shape dtype '()))
                         (dy    (emit-reduce-sum-to! neg-g g-shape y-shape dtype)))
                    (accumulate-input-adjoint! y-val dy))))

               ;; mul(x, y) -- dx = reduce-sum-to(g*y, shape-x); dy = reduce-sum-to(g*x, shape-y)
               ((eq? op 'mul)
                (let* ((x-val (list-ref inputs 0))
                       (y-val (list-ref inputs 1))
                       (x-shape (val-shape x-val))
                       (y-shape (val-shape y-val)))
                  (let* ((gy    (emit! 'mul (list g-val y-val) g-shape dtype '()))
                         (dx    (emit-reduce-sum-to! gy g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))
                  (let* ((gx    (emit! 'mul (list g-val x-val) g-shape dtype '()))
                         (dy    (emit-reduce-sum-to! gx g-shape y-shape dtype)))
                    (accumulate-input-adjoint! y-val dy))))

               ;; div(x, y) -- dx = g/y; dy = -g*x/y^2
               ((eq? op 'div)
                (let* ((x-val (list-ref inputs 0))
                       (y-val (list-ref inputs 1))
                       (x-shape (val-shape x-val))
                       (y-shape (val-shape y-val)))
                  (let* ((g-over-y (emit! 'div (list g-val y-val) g-shape dtype '()))
                         (dx       (emit-reduce-sum-to! g-over-y g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))
                  (let* ((y2      (emit! 'mul (list y-val y-val) y-shape dtype '()))
                         (x-over-y2 (emit! 'div (list x-val y2) g-shape dtype '()))
                         (neg-dy  (emit! 'mul (list g-val x-over-y2) g-shape dtype '()))
                         (neg-dy2 (emit! 'negate (list neg-dy) g-shape dtype '()))
                         (dy      (emit-reduce-sum-to! neg-dy2 g-shape y-shape dtype)))
                    (accumulate-input-adjoint! y-val dy))))

               ;; pow(x, n) -- dx = g * n * x^(n-1); dn skipped (n usually not a param)
               ((eq? op 'pow)
                (let* ((x-val (list-ref inputs 0))
                       (n-val (list-ref inputs 1))
                       (x-shape (val-shape x-val)))
                  ;; n_minus_1 = n - 1  (using scalar 1 constant)
                  (let* ((one-val (emit-scalar-const! 1.0 dtype))
                         (nm1     (emit! 'sub (list n-val one-val)
                                         (val-shape n-val) dtype '()))
                         (xpow    (emit! 'pow (list x-val nm1) g-shape dtype '()))
                         (n-xpow  (emit! 'mul (list n-val xpow) g-shape dtype '()))
                         (dx-raw  (emit! 'mul (list g-val n-xpow) g-shape dtype '()))
                         (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))))

               ;; negate(x) -- dx = -g
               ((eq? op 'negate)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (neg-g   (emit! 'negate (list g-val) g-shape dtype '()))
                       (dx      (emit-reduce-sum-to! neg-g g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; abs(x) -- dx = g * sign(x)
               ((eq? op 'abs)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (sign-fn (lambda (xv)
                                  (cond ((> xv 0.0)  1.0)
                                        ((< xv 0.0) -1.0)
                                        (else         0.0))))
                       (sign-x  (emit! 'map (list x-val) x-shape dtype
                                       (list (cons 'fn sign-fn))))
                       (dx-raw  (emit! 'mul (list g-val sign-x) g-shape dtype '()))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; sqrt(x) -- dx = g / (2 * sqrt(x))  i.e. g / (2 * fwd-output)
               ;; fwd-output is (binding-ref bid-sym) at this point in execution
               ((eq? op 'sqrt)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (two-val (emit-scalar-const! 2.0 dtype))
                       ;; fwd output = this binding
                       (fwd-out (binding-ref bid-sym))
                       (two-out (emit! 'mul (list two-val fwd-out) g-shape dtype '()))
                       (dx-raw  (emit! 'div (list g-val two-out) g-shape dtype '()))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; exp(x) -- dx = g * exp(x) = g * fwd-output
               ((eq? op 'exp)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (fwd-out (binding-ref bid-sym))
                       (dx-raw  (emit! 'mul (list g-val fwd-out) g-shape dtype '()))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; log(x) -- dx = g / x
               ((eq? op 'log)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (dx-raw  (emit! 'div (list g-val x-val) g-shape dtype '()))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; sin(x) -- dx = g * cos(x)
               ((eq? op 'sin)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (cos-x   (emit! 'cos (list x-val) x-shape dtype '()))
                       (dx-raw  (emit! 'mul (list g-val cos-x) g-shape dtype '()))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; cos(x) -- dx = -g * sin(x)
               ((eq? op 'cos)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (sin-x   (emit! 'sin (list x-val) x-shape dtype '()))
                       (neg-sin (emit! 'negate (list sin-x) x-shape dtype '()))
                       (dx-raw  (emit! 'mul (list g-val neg-sin) g-shape dtype '()))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; matmul(A, B) -- dA = g @ B^T; dB = A^T @ g
               ((eq? op 'matmul)
                (let* ((a-val    (list-ref inputs 0))
                       (b-val    (list-ref inputs 1))
                       (a-shape  (val-shape a-val))
                       (b-shape  (val-shape b-val))
                       ;; Determine 2D shapes; shape = [M K] [K N] → [M N]
                       (rank     (vector-length a-shape))
                       (perm-t   (transpose-perm rank))
                       (b-t-shape (permute-shape b-shape perm-t))
                       (a-t-shape (permute-shape a-shape perm-t))
                       (bt-val  (emit! 'transpose (list b-val) b-t-shape dtype
                                       (list (cons 'perm perm-t))))
                       (at-val  (emit! 'transpose (list a-val) a-t-shape dtype
                                       (list (cons 'perm perm-t))))
                       (da-val  (emit! 'matmul (list g-val bt-val) a-shape dtype '()))
                       (db-val  (emit! 'matmul (list at-val g-val) b-shape dtype '())))
                  (accumulate-input-adjoint! a-val da-val)
                  (accumulate-input-adjoint! b-val db-val)))

               ;; transpose(x, perm) -- dx = transpose(g, inv-perm)
               ((eq? op 'transpose)
                (let* ((x-val    (list-ref inputs 0))
                       (x-shape  (val-shape x-val))
                       (perm     (cdr (assq 'perm meta)))
                       (inv-perm (invert-permutation perm))
                       (dx       (emit! 'transpose (list g-val) x-shape dtype
                                        (list (cons 'perm inv-perm)))))
                  (accumulate-input-adjoint! x-val dx)))

               ;; reshape(x) -- dx = reshape(g, original-shape)
               ((eq? op 'reshape)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (dx      (emit! 'reshape (list g-val) x-shape dtype '())))
                  (accumulate-input-adjoint! x-val dx)))

               ;; (reduce mean) -- dx = broadcast(g / n, src-shape)
               ((equal? op '(reduce mean))
                (let* ((src-val    (list-ref inputs 0))
                       (src-shape  (cdr (assq 'src-shape meta)))
                       (axes       (cdr (assq 'axes meta)))
                       (keepdims?  (cdr (assq 'keepdims? meta)))
                       (n          (apply * (map (lambda (a) (vector-ref src-shape a)) axes)))
                       (n-val      (emit-scalar-const! (exact->inexact n) dtype))
                       (g-over-n   (emit! 'div (list g-val n-val) g-shape dtype '()))
                       (dx         (emit-broadcast-grad! g-over-n g-shape src-shape
                                                         axes keepdims? dtype)))
                  (accumulate-input-adjoint! src-val dx)))

               ;; (reduce sum) -- dx = broadcast(g, src-shape)
               ((equal? op '(reduce sum))
                (let* ((src-val   (list-ref inputs 0))
                       (src-shape (cdr (assq 'src-shape meta)))
                       (axes      (cdr (assq 'axes meta)))
                       (keepdims? (cdr (assq 'keepdims? meta)))
                       (dx        (emit-broadcast-grad! g-val g-shape src-shape
                                                        axes keepdims? dtype)))
                  (accumulate-input-adjoint! src-val dx)))

               ;; (reduce max) -- not needed for standard MLP; skip
               ((equal? op '(reduce max))
                (void))

               ;; map(x, fn) -- no grad (non-differentiable; sign fn used in abs backward)
               ((eq? op 'map)
                (void))

               (else
                (void)))))))
     (reverse fwd-bindings))

    ;; Collect param grad outputs in order of param-const-vals
    (let* ((grad-vals (filter-map
                       (lambda (pcv)
                         (let ((cid-sym (ssa-value-id pcv)))
                           (hash-table-ref/default param-grad-val cid-sym #f)))
                       param-const-vals))
           (all-outputs (cons loss-binding-val grad-vals))
           (all-bindings (append fwd-bindings bwd-bindings)))
      (make-ssa-program
       (ssa-program-constants fwd-prog)
       (ssa-program-morph-to-val fwd-prog)
       all-bindings
       all-outputs
       (length param-const-vals)))))


;;; Helper: find a forward binding by its bid-symbol
(define (find-fwd-binding bindings bid-sym)
  (let loop ((bs bindings))
    (cond
      ((null? bs) (error "ssa-vjp: loss binding not found" bid-sym))
      ((eq? (ssa-binding-name (car bs)) bid-sym) (car bs))
      (else (loop (cdr bs))))))


;;; Helper: build the transpose permutation for a rank-r tensor
;;; For rank 2: (1 0).  For rank 3: (0 2 1) (transpose last two axes).
(define (transpose-perm rank)
  (if (= rank 2)
      '(1 0)
      (let ((base (iota (- rank 2))))
        (append base (list (- rank 1) (- rank 2))))))


;;; Helper: permute shape vector according to perm list
(define (permute-shape shape perm)
  (list->vector (map (lambda (i) (vector-ref shape i)) perm)))


;;; ============================================================
;;; Phase B-3: rebuild-morphism
;;;
;;; Given an ssa-binding and a list of already-realized concrete-array
;;; inputs, construct a lazy morphism to pass to realize.
;;; ============================================================

(define (rebuild-morphism b inputs)
  "Reconstruct a lazy morphism from an SSA binding and concrete inputs."
  (let ((op   (ssa-binding-op b))
        (meta (ssa-binding-meta b))
        (shp  (ssa-binding-shape b)))
    (cond
      ((equal? op '(reduce mean))
       (let ((axes     (cdr (assq 'axes meta)))
             (keepdims? (cdr (assq 'keepdims? meta))))
         (morph-reduce 'mean (car inputs) axes keepdims?)))
      ((equal? op '(reduce sum))
       (let ((axes     (cdr (assq 'axes meta)))
             (keepdims? (cdr (assq 'keepdims? meta))))
         (morph-reduce 'sum (car inputs) axes keepdims?)))
      ((equal? op '(reduce max))
       (let ((axes     (cdr (assq 'axes meta)))
             (keepdims? (cdr (assq 'keepdims? meta))))
         (morph-reduce 'max (car inputs) axes keepdims?)))
      ((eq? op 'matmul)
       (morph-matmul (car inputs) (cadr inputs)))
      ((eq? op 'transpose)
       (let ((perm (cdr (assq 'perm meta))))
         (morph-transpose (car inputs) perm)))
      ((eq? op 'reshape)
       (morph-reshape (car inputs) shp))
      ((eq? op 'add)    (morph+ (car inputs) (cadr inputs)))
      ((eq? op 'sub)    (morph- (car inputs) (cadr inputs)))
      ((eq? op 'mul)    (morph* (car inputs) (cadr inputs)))
      ((eq? op 'div)    (morph/ (car inputs) (cadr inputs)))
      ((eq? op 'pow)    (morph-pow (car inputs) (cadr inputs)))
      ((eq? op 'negate) (morph-negate (car inputs)))
      ((eq? op 'abs)    (morph-abs (car inputs)))
      ((eq? op 'sqrt)   (morph-sqrt (car inputs)))
      ((eq? op 'exp)    (morph-exp (car inputs)))
      ((eq? op 'log)    (morph-log (car inputs)))
      ((eq? op 'sin)    (morph-sin (car inputs)))
      ((eq? op 'cos)    (morph-cos (car inputs)))
      ((eq? op 'map)
       (let ((fn (cdr (assq 'fn meta))))
         (morph-map fn (car inputs))))
      (else
       (error "rebuild-morphism: unknown op" op)))))


;;; ============================================================
;;; Phase B-3: ssa-realize
;;;
;;; Sequential executor.  Runs every binding once, returns outputs
;;; as a list of concrete-arrays.
;;; ============================================================

(define (ssa-realize prog)
  "Execute the SSA program sequentially.
   Returns a list of concrete-arrays for each output ssa-value."
  (let ((values (make-hash-table)))
    ;; Pre-populate constants
    (hash-table-walk (ssa-program-constants prog)
      (lambda (k v) (hash-table-set! values k v)))
    ;; Execute bindings in topological order
    (for-each
     (lambda (b)
       (let* ((inputs (map (lambda (v)
                             (hash-table-ref values (ssa-value-id v)))
                           (ssa-binding-inputs b)))
              (result (realize (rebuild-morphism b inputs))))
         (hash-table-set! values (ssa-binding-name b) result)))
     (ssa-program-bindings prog))
    ;; Collect outputs
    (map (lambda (ov)
           (hash-table-ref values (ssa-value-id ov)))
         (ssa-program-outputs prog))))


;;; ============================================================
;;; Phase B-4: ssa-realize/ctx
;;;
;;; Like ssa-realize but routes each allocation through a
;;; morphism context for buffer pooling.
;;; ============================================================

(define (ssa-realize/ctx ctx prog)
  "Like ssa-realize but uses realize/ctx for buffer pooling."
  (let ((values (make-hash-table)))
    (hash-table-walk (ssa-program-constants prog)
      (lambda (k v) (hash-table-set! values k v)))
    (for-each
     (lambda (b)
       (let* ((inputs (map (lambda (v)
                             (hash-table-ref values (ssa-value-id v)))
                           (ssa-binding-inputs b)))
              (result (realize/ctx ctx (rebuild-morphism b inputs))))
         (hash-table-set! values (ssa-binding-name b) result)))
     (ssa-program-bindings prog))
    (map (lambda (ov)
           (hash-table-ref values (ssa-value-id ov)))
         (ssa-program-outputs prog))))

) ; end module array-morphisms-ssa
