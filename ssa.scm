;;; array-morphisms-ssa.scm
;;;
;;; SSA IR for fused forward+backward computation.
;;;
;;; Compiles a morph-variable loss node into a flat SSA program once,
;;; then replays the joint forward+backward computation every training step.
;;; Eliminates the lazy/eager boundary, fresh-copy overhead, and missed
;;; fusion opportunities that arise when forward and backward pass each
;;; build separate lazy trees.
;;;
;;; Phases:
;;;   1.  morphism-to-ssa  -- DFS over morphism-expr tree -> SSA program
;;;   2.  ssa-vjp          -- symbolic VJP over SSA bindings -> extended program
;;;   3.  ssa-realize      -- sequential executor (no context pooling)
;;;   4.  ssa-realize/ctx  -- context-pooled executor
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
   ssa-program-replay-plan

   ;; Compilation
   morphism-to-ssa
   ssa-constant-id
   ssa-loss-binding-val

   ;; VJP
   ssa-vjp

   ;; Execution
   ssa-realize
   ssa-realize/ctx

   ;; Replay plan ADTs
   replay-ref?
   rr-val rr-const
   replay-instruction replay-instruction?
   ri-gemm ri-gemm-strided ri-index ri-reduce ri-view
   ri-flat-unary ri-flat-binary ri-flat-bias-broadcast
   ri-gemm-epilogue ri-alias

   ;; Fusion pass
   ssa-compute-use-counts
   ssa-fusion-eligible?
   ssa-element-wise-fusion-pass

   ;; Replay plan compilation and execution
   compile-replay-plan
   execute-replay-plan)

  (import scheme (chicken base))
  (import (only srfi-1 iota fold filter map for-each append-map filter-map every))
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
  (import array-morphisms-blas-compat)
  (import array-morphisms-realization)
  (import array-morphisms-context)
  (import array-morphisms-morph-env)
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
;;   meta   : alist of extra info (index-fn, axes, perm, fn, ...)
(define-record ssa-binding name op inputs shape dtype meta)

;; The compiled SSA program.
;;   constants   : hash-table cid-symbol -> concrete-array
;;   morph-to-val: morph-env stable-id (symbol) -> ssa-value  (for ssa-constant-id)
;;   bindings    : list of ssa-binding in topological order
;;   outputs     : list of ssa-value  (loss first, then param grads)
;;   n-params    : count of trainable parameter constants
;;   replay-plan   : #f until compiled; vector of replay-instruction (one per binding)
;;   trace-info    : #f until trace; hash-table bid-sym -> (concrete-array . is-pool?)
;;   output-specs  : #f until compiled; list of (integer | concrete-array) per output
;;                   integer = index into vals vector; concrete-array = const-ref value
(define-record ssa-program constants morph-to-val bindings outputs n-params
               replay-plan trace-info output-specs)


;;; ============================================================
;;; Replay Plan ADTs
;;;
;;; Pre-compiled dispatch structures produced by compile-replay-plan.
;;; Used by execute-replay-plan to avoid per-step morphism rebuilding,
;;; BLAS eligibility re-evaluation, and context-vector allocation.
;;; ============================================================

;; Input reference: previous binding result (by 0-based position in vals vector)
;; or a constant (by cid-symbol from constants hash-table).
(define-datatype replay-ref replay-ref?
  (rr-val   (val-idx integer?))
  (rr-const (cid symbol?)))

;; One pre-compiled instruction per SSA binding.
(define-datatype replay-instruction replay-instruction?

  ;; Row-major matmul: strides pre-computed at compile time.
  (ri-gemm
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (in-A         replay-ref?)
    (in-B         replay-ref?))

  ;; Strided matmul (at least one input is a transposed zero-copy view).
  (ri-gemm-strided
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (in-A         replay-ref?)
    (in-B         replay-ref?))

  ;; Element-wise / general index-fn op: strides pre-computed at compile time.
  (ri-index
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (index-fn     index-fn?)
    (in-refs      list?))

  ;; Reduction (sum, mean, max, ...): strides pre-computed at compile time.
  (ri-reduce
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (src-dtype    symbol?)
    (rop          symbol?)
    (reduce-axes  list?)
    (reducer      procedure?)
    (keepdims?    boolean?)
    (in-ref       replay-ref?))

  ;; Zero-copy view (transpose, reshape, slice): shares pool buffer with source.
  (ri-view
    (view-fn  procedure?)
    (in-ref   replay-ref?))

  ;; Element-wise unary: output[i] = combiner(A[i])
  ;; Emitted when: 1 operand, operand and output both row-major with same shape.
  (ri-flat-unary
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (combiner     procedure?)
    (in-A         replay-ref?))

  ;; Element-wise binary same-shape: output[i] = combiner(A[i], B[i])
  ;; Emitted when: 2 operands, all row-major with identical shape.
  (ri-flat-binary
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (combiner     procedure?)
    (in-A         replay-ref?)
    (in-B         replay-ref?))

  ;; Bias broadcast: output[i] = combiner(A[i], B[i mod N])
  ;; Emitted when: A row-major = output shape; B row-major shape [N] = last dim of output.
  ;; N baked in at compile time.
  (ri-flat-bias-broadcast
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (combiner     procedure?)
    (bias-N       integer?)
    (in-A         replay-ref?)
    (in-B         replay-ref?))

  ;; GEMM with in-place element-wise epilogue: one buffer, one combined kernel.
  ;; BLAS GEMM writes out-pool-idx buffer C[M,N] = A[M,K]*B[K,N],
  ;; then epilogue applied in-place to C.
  ;; epilogue-kind: 'unary | 'bias-broadcast
  ;; epilogue-N:    bias length for 'bias-broadcast; 0 for 'unary
  ;; bias-ref:      unused sentinel for 'unary; bias replay-ref for 'bias-broadcast
  (ri-gemm-epilogue
    (out-pool-idx  integer?)
    (out-shape     vector?)
    (out-strides   vector?)
    (out-dtype     symbol?)
    (in-A          replay-ref?)
    (in-B          replay-ref?)
    (epilogue-kind symbol?)
    (epilogue-comb procedure?)
    (epilogue-N    integer?)
    (bias-ref      replay-ref?))

  ;; Alias for the epilogue binding position after ri-gemm-epilogue.
  ;; Advances alloc-ctr to stay in sync with trace-time pool allocation.
  ;; Returns the already-computed epilogue result from in-ref (the GEMM step).
  (ri-alias
    (pool-idx  integer?)
    (shape     vector?)
    (strides   vector?)
    (dtype     symbol?)
    (in-ref    replay-ref?)))


;;; ============================================================
;;; SSA Fusion Helpers (element-wise fusion pass)
;;;
;;; Implements MoA's psi-composition theorem at the SSA IR level:
;;;   Psi(f, Psi(g, A)) = Psi(f o g, A)
;;; Two adjacent element-wise bindings with use-count=1 are collapsed
;;; into a single binding whose combiner is the composition of theirs.
;;; ============================================================

(define (ssa-binding-combiner b)
  "Extract element-wise combiner from an SSA binding's meta.
   Forward bindings carry (index-fn . compute-index-fn); backward bindings
   carry explicit (combiner . proc) embedded during ssa-vjp."
  (let* ((meta (ssa-binding-meta b))
         (cp   (assq 'combiner meta)))
    (if cp
        (cdr cp)
        (let ((ip (assq 'index-fn meta)))
          (if (and ip (compute-index-fn? (cdr ip)))
              (compute-index-fn-combiner (cdr ip))
              (error "ssa-binding-combiner: no combiner in meta"
                     (ssa-binding-name b) (ssa-binding-op b)))))))

(define (ssa-binding-elementwise? b)
  "True when b is an element-wise compute op (not matmul, reduce, or structural)."
  (let ((op (ssa-binding-op b)))
    (and (not (eq? op 'matmul))
         (not (and (pair? op) (eq? (car op) 'reduce)))
         (not (memq op '(reshape transpose broadcast-expand slice))))))

(define (ssa-compute-use-counts bindings)
  "Return hash-table: binding-name-sym -> integer (number of consuming bindings)."
  (let ((counts (make-hash-table)))
    (for-each (lambda (b)
                (hash-table-set! counts (ssa-binding-name b) 0))
              bindings)
    (for-each
     (lambda (b)
       (for-each
        (lambda (v)
          (cases ssa-value v
            (binding-ref (bid)
              (when (hash-table-ref/default counts bid #f)
                (hash-table-set! counts bid
                  (+ 1 (hash-table-ref counts bid)))))
            (const-ref (_) #f)))
        (ssa-binding-inputs b)))
     bindings)
    counts))

(define (ssa-binding-has-combiner? b)
  "True when b carries an extractable element-wise combiner in meta."
  (let ((meta (ssa-binding-meta b)))
    (or (and (assq 'combiner meta) #t)
        (let ((ip (assq 'index-fn meta)))
          (and ip (compute-index-fn? (cdr ip)))))))

(define (ssa-fusion-eligible? producer consumer use-counts)
  "True when producer and consumer can be fused by MoA psi-composition.
   Requires: use-count(producer)=1, consumer's first input is producer's output,
   same shape, both element-wise, both have extractable combiners."
  (let* ((p-name  (ssa-binding-name producer))
         (p-count (hash-table-ref/default use-counts p-name 0))
         (c-in0   (and (pair? (ssa-binding-inputs consumer))
                       (car (ssa-binding-inputs consumer)))))
    (and (= p-count 1)
         (cases ssa-value c-in0
           (binding-ref (bid) (eq? bid p-name))
           (else #f))
         (equal? (ssa-binding-shape producer) (ssa-binding-shape consumer))
         (ssa-binding-elementwise? producer)
         (ssa-binding-elementwise? consumer)
         (ssa-binding-has-combiner? producer)
         (ssa-binding-has-combiner? consumer))))

(define (fuse-two-ssa-bindings producer consumer)
  "Fuse producer into consumer via combiner composition.
   The fused binding inherits consumer's name (preserving downstream refs).
   Producer's intermediate buffer is eliminated."
  (let* ((p-inputs (ssa-binding-inputs producer))
         (c-inputs (ssa-binding-inputs consumer))
         (c-other  (cdr c-inputs))
         (fused-inputs (append p-inputs c-other))
         (p-comb  (ssa-binding-combiner producer))
         (c-comb  (ssa-binding-combiner consumer))
         (p-nargs (length p-inputs))
         (fused-comb (compose-flat-combiners p-comb p-nargs c-comb (length c-other)))
         (base-meta (filter (lambda (kv)
                                 (not (memq (car kv) '(combiner index-fn fused?))))
                               (ssa-binding-meta consumer)))
         (fused-meta (cons `(combiner . ,fused-comb)
                           (cons `(fused? . #t) base-meta))))
    (make-ssa-binding (ssa-binding-name consumer)
                      (ssa-binding-op consumer)
                      fused-inputs
                      (ssa-binding-shape consumer)
                      (ssa-binding-dtype consumer)
                      fused-meta)))

(define (ssa-element-wise-fusion-pass prog)
  "Apply MoA psi-composition to eligible adjacent binding pairs.
   Greedy left-to-right scan: whenever a producer P has use-count=1 and
   its only consumer C immediately follows, fuse them into one binding F.
   Chains of length > 2 are handled by re-presenting F as the new producer.
   Safety constraint: fusion is only applied when ALL of producer's inputs
   have the same shape as the producer's output (flat/non-broadcasting ops),
   ensuring identity index functions are valid in the fused binding."
  (let* ((bindings   (ssa-program-bindings prog))
         (use-counts (ssa-compute-use-counts bindings))
         ;; Shape lookup: id-sym -> shape vector (bindings + constants)
         (shape-of   (make-hash-table)))
    (for-each (lambda (b)
                (hash-table-set! shape-of (ssa-binding-name b) (ssa-binding-shape b)))
              bindings)
    (hash-table-walk (ssa-program-constants prog)
      (lambda (k v) (hash-table-set! shape-of k (morph-shape v))))

    ;; True when all inputs of b have the same shape as b's output.
    (define (flat-producer? b)
      (let ((b-shape (ssa-binding-shape b)))
        (every (lambda (inp)
                 (let ((id (cases ssa-value inp
                             (binding-ref (bid) bid)
                             (const-ref   (cid) cid))))
                   (equal? (hash-table-ref/default shape-of id #f) b-shape)))
               (ssa-binding-inputs b))))

    (let loop ((bs bindings) (result '()) (eliminated (make-hash-table)))
      (cond
        ((null? bs)
         (make-ssa-program
          (ssa-program-constants prog)
          (ssa-program-morph-to-val prog)
          (reverse result)
          (ssa-program-outputs prog)
          (ssa-program-n-params prog)
          #f #f #f))
        ((hash-table-ref/default eliminated (ssa-binding-name (car bs)) #f)
         (loop (cdr bs) result eliminated))
        ((and (pair? (cdr bs))
              (not (hash-table-ref/default eliminated
                     (ssa-binding-name (cadr bs)) #f))
              (ssa-fusion-eligible? (car bs) (cadr bs) use-counts)
              (flat-producer? (car bs)))
         (let ((fused (fuse-two-ssa-bindings (car bs) (cadr bs))))
           (hash-table-set! eliminated (ssa-binding-name (car bs)) #t)
           ;; Update shape-of for the fused binding (inherits consumer's name+shape)
           (hash-table-set! shape-of (ssa-binding-name fused) (ssa-binding-shape fused))
           (loop (cons fused (cddr bs)) result eliminated)))
        (else
         (loop (cdr bs) (cons (car bs) result) eliminated))))))


;;; ============================================================
;;; Phase 1: morphism-to-ssa
;;;
;;; Post-order DFS over the morphism-expr tree rooted at loss-mv.
;;; concrete-array leaves become constants; morphism-expr / reduction-morphism
;;; nodes become SSA bindings.
;;; ============================================================

(define (morphism-to-ssa loss-mv)
  "Compile the morph-variable graph rooted at loss-mv into an SSA program.
   Returns an ssa-program with outputs = (list loss-binding-val) and n-params = 0."
  (let* ((constants   (make-hash-table))  ; cid-symbol -> concrete-array
         (visited-eb  (make-env-builder)) ; stable-id (symbol) -> ssa-value
         (bindings    '()))               ; accumulated in reverse topo order

    (define (emit-binding! op inputs shape dtype meta)
      (let* ((bid (gensym 'bid-))
             (b   (make-ssa-binding bid op inputs shape dtype meta)))
        (set! bindings (cons b bindings))  ; O(1); reversed at end
        (binding-ref bid)))

    (define (visit m)
      (or (env-builder-lookup visited-eb (morph-stable-id m))
          (cases array-morphism m

            (concrete-array (data shape strides offset dtype alloc-id batch-axis)
              (let* ((cid (gensym 'cid-))
                     (v   (const-ref cid)))
                (hash-table-set! constants cid m)
                (env-builder-extend! visited-eb (morph-stable-id m) v)
                v))

            (morphism-expr (morph-id op operands index-fn shape dtype metadata batch-axis)
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
                (env-builder-extend! visited-eb (morph-stable-id m) v)
                v))

            (reduction-morphism (morph-id rop operand reduce-axes index-fn shape dtype batch-axis)
              (let* ((src-val (visit operand))
                     (meta    (list (cons 'axes reduce-axes)
                                    (cons 'index-fn index-fn)
                                    (cons 'src-shape (morph-shape operand))
                                    (cons 'keepdims? (reduction-index-fn-keepdims? index-fn))))
                     (v       (emit-binding! (list 'reduce rop) (list src-val) shape dtype meta)))
                (env-builder-extend! visited-eb (morph-stable-id m) v)
                v)))))

    (let* ((loss-m   (am:var-value loss-mv))
           (loss-val (visit loss-m)))
      (make-ssa-program constants (env-builder-env visited-eb)
                        (reverse bindings) (list loss-val) 0 #f #f #f))))


;;; ============================================================
;;; Accessors into a compiled program
;;; ============================================================

(define (ssa-constant-id prog m)
  "Return (const-ref cid) for morphism m if it is a constant in prog, else #f."
  (let ((v (morph-env-lookup (ssa-program-morph-to-val prog) (morph-stable-id m))))
    (and v (ssa-const-ref? v) v)))

(define (ssa-loss-binding-val prog)
  "Return the ssa-value for the loss node (first output)."
  (car (ssa-program-outputs prog)))


;;; ============================================================
;;; Phase 2: ssa-vjp
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

  (let* (;; Build shape/dtype lookup from forward bindings (single env-builder)
         (fwd-bindings  (ssa-program-bindings fwd-prog))
         (binding-sd-eb (make-env-builder)) ; sym -> (shape . dtype)
         ;; Accumulate backward bindings in reverse order; reverse at end
         (bwd-bindings  '())
         ;; Adjoint table: bid-symbol -> ssa-value (the accumulated dL/d(bid))
         (adjoint-val   (make-hash-table))
         ;; Param gradient table: cid-symbol -> ssa-value
         (param-grad-val (make-hash-table))
         ;; Set of trainable param cid-symbols for fast membership test
         (param-cid-set  (make-hash-table))
         ;; Loss shape/dtype
         (loss-bid-sym  (ssa-value-id loss-binding-val)))

    ;; Index forward binding shapes and dtypes
    (for-each (lambda (b)
                (env-builder-extend! binding-sd-eb
                                     (ssa-binding-name b)
                                     (cons (ssa-binding-shape b) (ssa-binding-dtype b))))
              fwd-bindings)

    ;; Also index constant shapes/dtypes
    (hash-table-walk (ssa-program-constants fwd-prog)
      (lambda (cid m)
        (env-builder-extend! binding-sd-eb cid (cons (morph-shape m) (morph-dtype m)))))

    ;; Register trainable param cid-symbols
    (for-each (lambda (v)
                (hash-table-set! param-cid-set (ssa-value-id v) #t))
              param-const-vals)

    ;; Lookup helpers
    (define (val-shape v)
      (let ((sd (env-builder-lookup binding-sd-eb (ssa-value-id v))))
        (and sd (car sd))))
    (define (val-dtype v)
      (let ((sd (env-builder-lookup binding-sd-eb (ssa-value-id v))))
        (and sd (cdr sd))))

    ;; emit! -- create a new backward SSA binding (O(1) cons; reversed at end)
    (define (emit! op inputs shape dtype meta)
      (let* ((adj-id (gensym 'adj-))
             (b      (make-ssa-binding adj-id op inputs shape dtype meta)))
        (set! bwd-bindings (cons b bwd-bindings))
        (env-builder-extend! binding-sd-eb adj-id (cons shape dtype))
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
             (_ (env-builder-extend! binding-sd-eb ones-cid (cons src-shape dtype)))
             (ones-val (const-ref ones-cid)))
        ;; morph* broadcasts: g-kd * ones -> src-shape
        (emit! 'mul (list g-kd ones-val) src-shape dtype `((combiner . ,*)))))

    ;; emit-scalar-const! -- emit a scalar constant of given value
    (define (emit-scalar-const! value dtype)
      (let* ((data (allocate-typed-vector dtype 1))
             (_ (typed-vector-set! data dtype 0 value))
             (m   (make-morphism data '(1) dtype))
             (cid (gensym 'cid-)))
        (hash-table-set! (ssa-program-constants fwd-prog) cid m)
        (env-builder-extend! binding-sd-eb cid (cons (vector 1) dtype))
        (const-ref cid)))

    ;; Seed: dL/dL = ones-like(loss) stored as a constant
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
           (_ (env-builder-extend! binding-sd-eb seed-cid (cons loss-shape loss-dtype)))
           (seed-val    (const-ref seed-cid)))
      (hash-table-set! adjoint-val loss-bid-sym seed-val))

    ;; Backward sweep: iterate bindings in reverse order
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
                  (let* ((neg-g (emit! 'negate (list g-val) g-shape dtype
                                        `((combiner . ,(lambda (x) (- x))))))
                         (dy    (emit-reduce-sum-to! neg-g g-shape y-shape dtype)))
                    (accumulate-input-adjoint! y-val dy))))

               ;; mul(x, y) -- dx = reduce-sum-to(g*y, shape-x); dy = reduce-sum-to(g*x, shape-y)
               ((eq? op 'mul)
                (let* ((x-val (list-ref inputs 0))
                       (y-val (list-ref inputs 1))
                       (x-shape (val-shape x-val))
                       (y-shape (val-shape y-val)))
                  (let* ((gy    (emit! 'mul (list g-val y-val) g-shape dtype
                                        `((combiner . ,*))))
                         (dx    (emit-reduce-sum-to! gy g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))
                  (let* ((gx    (emit! 'mul (list g-val x-val) g-shape dtype
                                        `((combiner . ,*))))
                         (dy    (emit-reduce-sum-to! gx g-shape y-shape dtype)))
                    (accumulate-input-adjoint! y-val dy))))

               ;; div(x, y) -- dx = g/y; dy = -g*x/y^2
               ((eq? op 'div)
                (let* ((x-val (list-ref inputs 0))
                       (y-val (list-ref inputs 1))
                       (x-shape (val-shape x-val))
                       (y-shape (val-shape y-val)))
                  (let* ((g-over-y (emit! 'div (list g-val y-val) g-shape dtype
                                           `((combiner . ,/))))
                         (dx       (emit-reduce-sum-to! g-over-y g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))
                  (let* ((y2      (emit! 'mul (list y-val y-val) y-shape dtype
                                          `((combiner . ,*))))
                         (x-over-y2 (emit! 'div (list x-val y2) g-shape dtype
                                            `((combiner . ,/))))
                         (neg-dy  (emit! 'mul (list g-val x-over-y2) g-shape dtype
                                          `((combiner . ,*))))
                         (neg-dy2 (emit! 'negate (list neg-dy) g-shape dtype
                                          `((combiner . ,(lambda (x) (- x))))))
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
                                         (val-shape n-val) dtype
                                         `((combiner . ,(lambda (a b) (- a b))))))
                         (xpow    (emit! 'pow (list x-val nm1) g-shape dtype
                                          `((combiner . ,expt))))
                         (n-xpow  (emit! 'mul (list n-val xpow) g-shape dtype
                                          `((combiner . ,*))))
                         (dx-raw  (emit! 'mul (list g-val n-xpow) g-shape dtype
                                          `((combiner . ,*))))
                         (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))))

               ;; negate(x) -- dx = -g
               ((eq? op 'negate)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (neg-g   (emit! 'negate (list g-val) g-shape dtype
                                        `((combiner . ,(lambda (x) (- x))))))
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
                       (dx-raw  (emit! 'mul (list g-val sign-x) g-shape dtype
                                        `((combiner . ,*))))
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
                       (two-out (emit! 'mul (list two-val fwd-out) g-shape dtype
                                        `((combiner . ,*))))
                       (dx-raw  (emit! 'div (list g-val two-out) g-shape dtype
                                        `((combiner . ,/))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; exp(x) -- dx = g * exp(x) = g * fwd-output
               ((eq? op 'exp)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (fwd-out (binding-ref bid-sym))
                       (dx-raw  (emit! 'mul (list g-val fwd-out) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; log(x) -- dx = g / x
               ((eq? op 'log)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (dx-raw  (emit! 'div (list g-val x-val) g-shape dtype
                                        `((combiner . ,/))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; sin(x) -- dx = g * cos(x)
               ((eq? op 'sin)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (cos-x   (emit! 'cos (list x-val) x-shape dtype
                                        `((combiner . ,cos))))
                       (dx-raw  (emit! 'mul (list g-val cos-x) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; cos(x) -- dx = -g * sin(x)
               ((eq? op 'cos)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (sin-x   (emit! 'sin (list x-val) x-shape dtype
                                        `((combiner . ,sin))))
                       (neg-sin (emit! 'negate (list sin-x) x-shape dtype
                                        `((combiner . ,(lambda (x) (- x))))))
                       (dx-raw  (emit! 'mul (list g-val neg-sin) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; relu(x) -- dx = g * heaviside(x)
               ;; Decomposed as: hx = map(heaviside, x); dx = mul(g, hx)
               ;; (abs-style decomposition keeps rebuild-morphism simple)
               ((eq? op 'relu)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (hx      (emit! 'map (list x-val) x-shape dtype
                                        `((fn . ,(lambda (xv) (if (> xv 0.0) 1.0 0.0)))
                                          (combiner . ,(lambda (xv) (if (> xv 0.0) 1.0 0.0))))))
                       (dx-raw  (emit! 'mul (list g-val hx) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; sigmoid(x) -- dx = g * deriv where deriv = s*(1-s), s = forward output
               ((eq? op 'sigmoid)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (fwd-out (binding-ref bid-sym))
                       (deriv   (emit! 'map (list fwd-out) g-shape dtype
                                        `((fn . ,(lambda (sv) (* sv (- 1.0 sv))))
                                          (combiner . ,(lambda (sv) (* sv (- 1.0 sv)))))))
                       (dx-raw  (emit! 'mul (list g-val deriv) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; tanh(x) -- dx = g * (1-t^2) where t = tanh(x) = forward output
               ((eq? op 'tanh)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (fwd-out (binding-ref bid-sym))
                       (deriv   (emit! 'map (list fwd-out) g-shape dtype
                                        `((fn . ,(lambda (tv) (- 1.0 (* tv tv))))
                                          (combiner . ,(lambda (tv) (- 1.0 (* tv tv)))))))
                       (dx-raw  (emit! 'mul (list g-val deriv) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; matmul(A, B) -- dA = g @ B^T; dB = A^T @ g
               ((eq? op 'matmul)
                (let* ((a-val    (list-ref inputs 0))
                       (b-val    (list-ref inputs 1))
                       (a-shape  (val-shape a-val))
                       (b-shape  (val-shape b-val))
                       ;; Determine 2D shapes; shape = [M K] [K N] -> [M N]
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

    ;; Emit add(loss, zero-const) as the final backward binding.
    ;;
    ;; The context's compute-last-uses! extends a buffer's lifetime only when
    ;; it appears as an INPUT to a later allocation-rec.  The loss forward binding
    ;; has no backward binding that references it directly (the seed is stored as
    ;; a pre-computed constant), so loss.alloc-id's last-use stays at its birth
    ;; step and backward bindings can reuse its buffer slot in replay mode.
    ;;
    ;; add(loss, zero) is a two-operand op — can-zero-copy? returns #f so a real
    ;; allocation is recorded.  input-ids includes loss.alloc-id, which forces
    ;; compute-last-uses! to extend loss's lifetime to this final step, preventing
    ;; any backward binding from aliasing its buffer.
    (let* ((loss-bid-b  (find-fwd-binding fwd-bindings loss-bid-sym))
           (loss-shape  (ssa-binding-shape loss-bid-b))
           (loss-dtype  (ssa-binding-dtype loss-bid-b))
           ;; Zero constant with same shape as loss (alloc-id=-1, not context-tracked)
           (zero-data   (allocate-typed-vector loss-dtype (shape-size loss-shape)))
           (_ (let loop ((i 0) (n (shape-size loss-shape)))
                (when (< i n)
                  (typed-vector-set! zero-data loss-dtype i 0.0)
                  (loop (+ i 1) n))))
           (zero-const  (make-morphism zero-data (vector->list loss-shape) loss-dtype))
           (zero-cid    (gensym 'cid-))
           (_ (hash-table-set! (ssa-program-constants fwd-prog) zero-cid zero-const))
           (_ (env-builder-extend! binding-sd-eb zero-cid (cons loss-shape loss-dtype)))
           (zero-val    (const-ref zero-cid))
           (loss-out-val (emit! 'add (list loss-binding-val zero-val) loss-shape loss-dtype
                                      `((combiner . ,+))))
           ;; Collect param grad outputs in order of param-const-vals
           (grad-vals   (filter-map
                         (lambda (pcv)
                           (let ((cid-sym (ssa-value-id pcv)))
                             (hash-table-ref/default param-grad-val cid-sym #f)))
                         param-const-vals))
           (all-outputs  (cons loss-out-val grad-vals))
           (all-bindings (append fwd-bindings (reverse bwd-bindings))))
      ;; Apply MoA psi-composition fusion to the full joint forward+backward program.
      ;; Cross-AD-boundary fusion is automatic: backward element-wise bindings
      ;; with use-count=1 are eligible under the same rules as forward ones.
      (ssa-element-wise-fusion-pass
       (make-ssa-program
        (ssa-program-constants fwd-prog)
        (ssa-program-morph-to-val fwd-prog)
        all-bindings
        all-outputs
        (length param-const-vals)
        #f #f #f)))))


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
;;; Phase 3: rebuild-morphism
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
      ;; Fused binding: op is consumer's op, inputs are the fused (producer's) inputs.
      ;; Dispatch on the stored combiner instead of rebuilding from op name.
      ((and (assq 'fused? meta) (assq 'combiner meta))
       (let* ((comb   (cdr (assq 'combiner meta)))
              (shapes (map morph-shape inputs))
              (ifn    (make-compute-index-fn
                        (map (lambda (_) (lambda (idx) idx)) inputs)
                        comb
                        shapes)))
         (morphism-expr (gensym 'morph-) op inputs ifn shp (ssa-binding-dtype b) '() -1)))
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
      ((eq? op 'relu)    (morph-relu    (car inputs)))
      ((eq? op 'sigmoid) (morph-sigmoid (car inputs)))
      ((eq? op 'tanh)    (morph-tanh-am (car inputs)))
      ((eq? op 'map)
       (let ((fn (cdr (assq 'fn meta))))
         (morph-map fn (car inputs))))
      (else
       (error "rebuild-morphism: unknown op" op)))))


;;; ============================================================
;;; Phase 3: ssa-realize
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
;;; Phase 4: ssa-realize/ctx
;;;
;;; Like ssa-realize but routes each allocation through a
;;; morphism context for buffer pooling.
;;; ============================================================

;; True iff m is a concrete-array with row-major strides and zero offset.
;; Transposed zero-copy views have permuted strides and fail this check.
;; Used to distinguish pool-allocated row-major results (which can be pinned
;; and returned directly) from zero-copy transposed views (which must be copied).
(define (concrete-row-major? m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (and (= offset 0) (equal? strides (compute-strides shape))))
    (else #f)))

(define (concrete-alloc-id m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis) alloc-id)
    (else -1)))

;; De-transpose a concrete-array to a fresh row-major non-pooled buffer (alloc-id=-1).
;; Used for output bindings that are zero-copy transposed views (no pool slot of their own).
(define (copy-concrete-array m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (let* ((size     (shape-size shape))
             (new-data (allocate-typed-vector dtype size))
             (new-strs (compute-strides shape)))
        (do ((i 0 (+ i 1)))
            ((= i size))
          (let* ((multi (linear-to-multi-index i shape))
                 (phys  (multi-to-linear-index multi strides offset))
                 (val   (typed-vector-ref data dtype phys)))
            (typed-vector-set! new-data dtype i val)))
        (concrete-array new-data shape new-strs 0 dtype -1 batch-axis)))
    (else (error "copy-concrete-array: not a concrete-array" m))))

;;; ============================================================
;;; Replay Plan: compile-time classification helpers
;;;
;;; These predicates inspect trace-time concrete arrays.  They run once
;;; inside compile-one-instruction and never during replay.
;;; ============================================================

(define (trace-arr-row-major? m)
  (cases array-morphism m
    (concrete-array (_ shape strides offset _ _ _)
      (and (= offset 0) (equal? strides (compute-strides shape))))
    (else #f)))

(define (trace-arr-shape m)
  (cases array-morphism m
    (concrete-array (_ shape _ _ _ _ _) shape)
    (else (error "trace-arr-shape: not concrete" m))))

(define (flat-unary-eligible? out-shape in-traces)
  (and (= (length in-traces) 1)
       (trace-arr-row-major? (car in-traces))
       (equal? (trace-arr-shape (car in-traces)) out-shape)))

(define (flat-binary-eligible? out-shape in-traces)
  (and (= (length in-traces) 2)
       (trace-arr-row-major? (car  in-traces))
       (trace-arr-row-major? (cadr in-traces))
       (equal? (trace-arr-shape (car  in-traces)) out-shape)
       (equal? (trace-arr-shape (cadr in-traces)) out-shape)))

(define (flat-bias-broadcast-eligible? out-shape in-traces)
  (and (= (length in-traces) 2)
       (> (vector-length out-shape) 0)  ; bias-broadcast requires rank >= 1
       (trace-arr-row-major? (car  in-traces))
       (trace-arr-row-major? (cadr in-traces))
       (equal? (trace-arr-shape (car in-traces)) out-shape)
       (let* ((bias-shape (trace-arr-shape (cadr in-traces)))
              (rank       (vector-length out-shape))
              (N          (vector-ref out-shape (- rank 1))))
         (and (= (vector-length bias-shape) 1)
              (= (vector-ref bias-shape 0) N)))))


;;; ============================================================
;;; Replay Plan: compile-replay-ref
;;; ============================================================

(define (compile-replay-ref v name->pos)
  "Compile an ssa-value to a replay-ref.
   v: ssa-value (binding-ref or const-ref)
   name->pos: hash-table bid-sym -> integer (0-based binding position)"
  (cases ssa-value v
    (const-ref   (cid) (rr-const cid))
    (binding-ref (bid) (rr-val (hash-table-ref name->pos bid)))))


;;; ============================================================
;;; Replay Plan: compile-one-instruction
;;; ============================================================

(define (compile-one-instruction b in-refs pool-idx trace-info-table constants)
  "Compile one SSA binding into a replay-instruction.
   b:                ssa-binding
   in-refs:          list of replay-ref? (already compiled from inputs)
   pool-idx:         integer (physical buffer slot) or -1 (zero-copy)
   trace-info-table: hash-table bid-sym -> (concrete-array . is-pool?)
   constants:        hash-table cid-sym -> concrete-array"
  (let* ((op        (ssa-binding-op b))
         (meta      (ssa-binding-meta b))
         (shape     (ssa-binding-shape b))
         (dtype     (ssa-binding-dtype b))
         (strides   (if (vector? shape) (compute-strides shape) '#()))
         (info      (hash-table-ref trace-info-table (ssa-binding-name b)))
         (trace-arr (car info))
         (is-pool?  (cdr info)))

    ;; Look up trace-time concrete-array for any ssa-value input:
    ;; binding-ref -> trace-info-table; const-ref -> constants.
    (define (trace-arr-of v)
      (cases ssa-value v
        (const-ref   (cid) (hash-table-ref constants cid))
        (binding-ref (bid) (car (hash-table-ref trace-info-table bid)))))

    ;; Extract index-fn from meta when present (morphism-to-ssa bindings),
    ;; or rebuild the morphism with trace-time inputs to obtain it
    ;; (VJP-emitted bindings whose meta never carries 'index-fn).
    ;; rebuild-morphism can return morphism-expr (most ops) or reduction-morphism.
    (define (get-index-fn)
      (let ((ip (assq 'index-fn meta)))
        (if ip
            (cdr ip)
            ;; Fused binding: (fused? . #t) + (combiner . proc); build compute-index-fn directly.
            (let ((fp (assq 'fused? meta)))
              (if fp
                  (let ((cp (assq 'combiner meta)))
                    (make-compute-index-fn
                     (map (lambda (_) (lambda (idx) idx)) (ssa-binding-inputs b))
                     (cdr cp)
                     (map (lambda (_) shape) (ssa-binding-inputs b))))
                  (let ((trace-inputs (map trace-arr-of (ssa-binding-inputs b))))
                    (cases array-morphism (rebuild-morphism b trace-inputs)
                      (morphism-expr (_ _ _ ifn _ _ _ _) ifn)
                      (reduction-morphism (_ _ _ _ ifn _ _ _) ifn)
                      (else (error "compile-one-instruction: rebuild-morphism returned unexpected type"
                                   op meta)))))))))

    (cond
      ;; Zero-copy view: any op where the trace did not allocate (counter not incremented).
      ;; Only transpose and reshape produce zero-copy views in practice.
      ((not is-pool?)
       (let* ((index-fn   (get-index-fn))
              (batch-axis (cases array-morphism trace-arr
                            (concrete-array (_ _ _ _ _ _ ba) ba)
                            (else -1)))
              (view-fn    (lambda (src) (create-view src index-fn shape batch-axis))))
         (ri-view view-fn (car in-refs))))

      ;; matmul: use trace-time input arrays to determine BLAS variant once.
      ;; morph-matmul uses (identity-fn) as a placeholder, so matmul can never
      ;; use the ri-index path — it always needs the BLAS/Scheme kernel dispatch.
      ;; execute-blas-gemm/into! handles the row-major case; execute-blas-gemm-strided/into!
      ;; handles transposed/non-contiguous inputs via a stride-aware Scheme triple loop.
      ((eq? op 'matmul)
       (let* ((A-tr      (trace-arr-of (car  (ssa-binding-inputs b))))
              (B-tr      (trace-arr-of (cadr (ssa-binding-inputs b))))
              (blas-info (blas-compatible-operation? (morph-matmul A-tr B-tr))))
         (if blas-info
             (case (car blas-info)
               ((gemm)         (ri-gemm pool-idx shape strides dtype
                                        (car in-refs) (cadr in-refs)))
               ((gemm-strided) (ri-gemm-strided pool-idx shape strides dtype
                                                (car in-refs) (cadr in-refs)))
               (else           (ri-gemm pool-idx shape strides dtype
                                        (car in-refs) (cadr in-refs))))
             ;; Not compatible (shapes mismatch etc.) — shouldn't reach here in practice.
             (error "compile-one-instruction: matmul not BLAS-compatible" (ssa-binding-name b)))))

      ;; Reductions: op is (list 'reduce rop).
      ;; VJP-emitted reduce has 'axes and 'keepdims? in meta but no 'index-fn;
      ;; get-index-fn handles both via rebuild-morphism.
      ((and (pair? op) (eq? (car op) 'reduce))
       (let* ((rop       (cadr op))
              (axes      (cdr (assq 'axes meta)))
              (keepdims? (cdr (assq 'keepdims? meta)))
              (rfn       (get-index-fn))
              (reducer   (reduction-index-fn-reducer rfn))
              (src-arr   (trace-arr-of (car (ssa-binding-inputs b))))
              (src-dtype (cases array-morphism src-arr
                           (concrete-array (_ _ _ _ d _ _) d)
                           (else dtype))))
         (ri-reduce pool-idx shape strides dtype src-dtype rop axes reducer keepdims?
                    (car in-refs))))

      ;; Element-wise and all other pool-backed ops.
      ;; VJP-emitted bindings have empty meta; get-index-fn rebuilds via trace inputs.
      ;; classify once at compile time into flat fast-paths or generic ri-index fallback.
      (else
       (let* ((index-fn  (get-index-fn))
              (in-traces (map trace-arr-of (ssa-binding-inputs b))))
         (cond
           ;; Non-compute index-fns (affine, window, etc.): generic path
           ((not (compute-index-fn? index-fn))
            (ri-index pool-idx shape strides dtype index-fn in-refs))

           ;; Fast path 1: unary, row-major, same shape
           ((flat-unary-eligible? shape in-traces)
            (ri-flat-unary pool-idx shape strides dtype
                           (compute-index-fn-combiner index-fn)
                           (car in-refs)))

           ;; Fast path 2: binary, both row-major, same shape
           ((flat-binary-eligible? shape in-traces)
            (ri-flat-binary pool-idx shape strides dtype
                            (compute-index-fn-combiner index-fn)
                            (car in-refs) (cadr in-refs)))

           ;; Fast path 3: bias broadcast [B*N] + [N]
           ((flat-bias-broadcast-eligible? shape in-traces)
            (let* ((rank (vector-length shape))
                   (N    (vector-ref shape (- rank 1))))
              (ri-flat-bias-broadcast pool-idx shape strides dtype
                                      (compute-index-fn-combiner index-fn)
                                      N
                                      (car in-refs) (cadr in-refs))))

           ;; Generic fallback: compute-index-fn with non-trivial layout
           (else
            (ri-index pool-idx shape strides dtype index-fn in-refs))))))))


;;; ============================================================
;;; Replay Plan: compile-replay-plan
;;; ============================================================

(define (compile-replay-plan prog ctx)
  "Compile the SSA program's bindings into a vector of replay-instructions.
   ctx must be in replay mode (finalize-context! already called).
   Reads ssa-program-trace-info to determine pool-idx and BLAS variant.
   Returns a vector of replay-instruction (one per binding, in binding order)."
  (let* ((bindings    (ssa-program-bindings prog))
         (n           (length bindings))
         (constants   (ssa-program-constants prog))
         (trace-info  (ssa-program-trace-info prog))
         (plan-vec    (make-vector n #f))
         ;; Map binding name -> 0-based position in bindings list
         (name->pos   (let ((ht (make-hash-table)))
                        (let loop ((bs bindings) (i 0))
                          (unless (null? bs)
                            (hash-table-set! ht (ssa-binding-name (car bs)) i)
                            (loop (cdr bs) (+ i 1))))
                        ht)))
    ;; Detect GEMM bindings whose only consumer is an adjacent element-wise
    ;; binding that compiles to a flat fast-path instruction.
    ;; When found, pre-compile ri-gemm-epilogue for the GEMM position and
    ;; arrange for ri-alias at the epilogue position.
    (define use-counts (ssa-compute-use-counts bindings))
    (define gemm-epilogue-instr (make-hash-table))  ;; g.name -> ri-gemm-epilogue
    (define epilogue-gemm-pos   (make-hash-table))  ;; e.name -> g's position index
    (let lp ((bs bindings) (i 0))
      (when (and (pair? bs) (pair? (cdr bs)))
        (let ((g (car bs)) (e (cadr bs)))
          (when (and (eq? (ssa-binding-op g) 'matmul)
                     (= 1 (hash-table-ref/default use-counts (ssa-binding-name g) 0))
                     (let ((in0 (and (pair? (ssa-binding-inputs e))
                                     (car (ssa-binding-inputs e)))))
                       (cases ssa-value in0
                         (binding-ref (bid) (eq? bid (ssa-binding-name g)))
                         (else #f)))
                     (ssa-binding-elementwise? e))
            ;; Try to compile e as a flat instruction
            (let* ((e-in-refs  (map (lambda (v) (compile-replay-ref v name->pos))
                                    (ssa-binding-inputs e)))
                   (e-info     (hash-table-ref trace-info (ssa-binding-name e)))
                   (e-pool-idx (if (cdr e-info)
                                   (context-alloc->pool-idx ctx (concrete-alloc-id (car e-info)))
                                   -1))
                   (e-instr    (compile-one-instruction e e-in-refs e-pool-idx
                                                        trace-info constants))
                   ;; g's pool-idx and refs for the GEMM part
                   (g-info     (hash-table-ref trace-info (ssa-binding-name g)))
                   (g-pool-idx (if (cdr g-info)
                                   (context-alloc->pool-idx ctx (concrete-alloc-id (car g-info)))
                                   -1))
                   (g-in-refs  (map (lambda (v) (compile-replay-ref v name->pos))
                                    (ssa-binding-inputs g)))
                   (g-shape    (ssa-binding-shape g))
                   (g-strides  (compute-strides g-shape))
                   (g-dtype    (ssa-binding-dtype g)))
              (cases replay-instruction e-instr
                (ri-flat-unary (_ _ _ _ e-comb _)
                  (hash-table-set! gemm-epilogue-instr (ssa-binding-name g)
                    (ri-gemm-epilogue e-pool-idx g-shape g-strides g-dtype
                                      (car g-in-refs) (cadr g-in-refs)
                                      'unary e-comb 0 (rr-val 0)))
                  (hash-table-set! epilogue-gemm-pos (ssa-binding-name e) i))
                (ri-flat-bias-broadcast (_ _ _ _ e-comb N _ e-in-B)
                  (hash-table-set! gemm-epilogue-instr (ssa-binding-name g)
                    (ri-gemm-epilogue e-pool-idx g-shape g-strides g-dtype
                                      (car g-in-refs) (cadr g-in-refs)
                                      'bias-broadcast e-comb N e-in-B))
                  (hash-table-set! epilogue-gemm-pos (ssa-binding-name e) i))
                (else #f)))))
        (lp (cdr bs) (+ i 1))))

    (let loop ((bs bindings) (i 0))
      (unless (null? bs)
        (let* ((b     (car bs))
               (bname (ssa-binding-name b))
               (instr
                (cond
                  ;; GEMM with epilogue: use pre-compiled ri-gemm-epilogue
                  ((hash-table-ref/default gemm-epilogue-instr bname #f) => (lambda (ri) ri))
                  ;; Epilogue position: the GEMM already applied it in-place
                  ((hash-table-ref/default epilogue-gemm-pos bname #f) =>
                   (lambda (g-pos)
                     (let* ((info     (hash-table-ref trace-info bname))
                            (pool-idx (if (cdr info)
                                          (context-alloc->pool-idx ctx (concrete-alloc-id (car info)))
                                          -1))
                            (shape    (ssa-binding-shape b))
                            (strides  (compute-strides shape))
                            (dtype    (ssa-binding-dtype b)))
                       (ri-alias pool-idx shape strides dtype (rr-val g-pos)))))
                  ;; Normal path
                  (else
                   (let* ((in-refs  (map (lambda (v) (compile-replay-ref v name->pos))
                                         (ssa-binding-inputs b)))
                          (info     (hash-table-ref trace-info bname))
                          (is-pool? (cdr info))
                          (pool-idx (if is-pool?
                                        (context-alloc->pool-idx ctx (concrete-alloc-id (car info)))
                                        -1)))
                     (compile-one-instruction b in-refs pool-idx trace-info constants))))))
          (vector-set! plan-vec i instr)
          (loop (cdr bs) (+ i 1)))))

    ;; Pre-compute output specs to eliminate name->idx hash-table rebuild per replay step.
    ;; Each spec is either an integer (0-based position in vals for binding-refs)
    ;; or a concrete-array (the constant value itself for const-refs).
    (let ((output-specs
           (map (lambda (ov)
                  (cases ssa-value ov
                    (binding-ref (bid) (hash-table-ref name->pos bid))
                    (const-ref   (cid) (hash-table-ref constants cid))))
                (ssa-program-outputs prog))))
      (ssa-program-output-specs-set! prog output-specs))

    plan-vec))


;;; ============================================================
;;; Replay Plan: execute-replay-plan
;;; ============================================================

(define (execute-replay-plan plan pool constants)
  "Execute a pre-compiled replay-plan.
   plan:      vector of replay-instruction (from compile-replay-plan)
   pool:      buffer-pool record (from morphism-context-pool ctx)
   constants: hash-table cid-sym -> concrete-array (from ssa-program-constants)
   Returns:   vector of concrete-array, one per binding position."
  (let* ((n         (vector-length plan))
         (pool-bufs (buffer-pool-buffers pool))
         (vals      (make-vector n #f))
         ;; alloc-ctr mirrors the trace-time context counter so that pool-backed
         ;; bindings get the same alloc-id here as they did during the trace run.
         ;; Zero-copy (ri-view) bindings don't increment this, matching the trace.
         (alloc-ctr 0))

    (define (deref ref)
      (cases replay-ref ref
        (rr-val   (i)   (vector-ref vals i))
        (rr-const (cid) (hash-table-ref constants cid))))

    ;; Build a concrete-array for a pool-backed instruction using the next
    ;; trace-aligned alloc-id (not pool-idx) so callers that inspect alloc-id
    ;; see the same value as they did in the trace run.
    ;; Strides are pre-computed at compile time; no compute-strides call per step.
    (define (make-pool-arr pool-idx shape strides dtype)
      (let ((aid alloc-ctr))
        (set! alloc-ctr (+ alloc-ctr 1))
        (concrete-array (vector-ref pool-bufs pool-idx)
                        shape strides 0 dtype aid -1)))

    (do ((i 0 (+ i 1)))
        ((= i n))
      (let* ((instr (vector-ref plan i))
             (result
              (cases replay-instruction instr

                (ri-gemm (pool-idx shape strides dtype A-ref B-ref)
                  (let ((buf (vector-ref pool-bufs pool-idx)))
                    (execute-blas-gemm/into! (deref A-ref) (deref B-ref) buf)
                    (make-pool-arr pool-idx shape strides dtype)))

                (ri-gemm-strided (pool-idx shape strides dtype A-ref B-ref)
                  (let ((buf (vector-ref pool-bufs pool-idx)))
                    (execute-blas-gemm-strided/into! (deref A-ref) (deref B-ref) buf)
                    (make-pool-arr pool-idx shape strides dtype)))

                (ri-index (pool-idx shape strides dtype index-fn in-refs)
                  (let ((buf (vector-ref pool-bufs pool-idx)))
                    (execute-index-fn index-fn buf shape (map deref in-refs) dtype)
                    (make-pool-arr pool-idx shape strides dtype)))

                (ri-reduce (pool-idx shape strides dtype src-dtype rop axes reducer keepdims? in-ref)
                  (let* ((src (deref in-ref))
                         (buf (vector-ref pool-bufs pool-idx)))
                    (cases array-morphism src
                      (concrete-array (src-data src-shape src-strides src-offset _ _ _)
                        (execute-reduction-morphism
                         rop buf shape
                         src-data src-shape src-strides src-offset
                         axes reducer keepdims? dtype src-dtype)
                        (make-pool-arr pool-idx shape strides dtype))
                      (else
                       (error "execute-replay-plan ri-reduce: source not concrete" src)))))

                (ri-view (view-fn in-ref)
                  (view-fn (deref in-ref)))

                (ri-flat-unary (pool-idx shape strides dtype combiner in-A)
                  (let* ((src  (deref in-A))
                         (buf  (vector-ref pool-bufs pool-idx))
                         (size (shape-size shape)))
                    (cases array-morphism src
                      (concrete-array (data _ _ _ _ _ _)
                        (execute-flat-unary-compute combiner data buf size dtype)
                        (make-pool-arr pool-idx shape strides dtype))
                      (else (error "ri-flat-unary: source not concrete" src)))))

                (ri-flat-binary (pool-idx shape strides dtype combiner in-A in-B)
                  (let* ((A    (deref in-A))
                         (B    (deref in-B))
                         (buf  (vector-ref pool-bufs pool-idx))
                         (size (shape-size shape)))
                    (cases array-morphism A
                      (concrete-array (data1 _ _ _ _ _ _)
                        (cases array-morphism B
                          (concrete-array (data2 _ _ _ _ _ _)
                            (execute-flat-binary-compute combiner data1 data2 buf size dtype)
                            (make-pool-arr pool-idx shape strides dtype))
                          (else (error "ri-flat-binary: B not concrete" B))))
                      (else (error "ri-flat-binary: A not concrete" A)))))

                (ri-flat-bias-broadcast (pool-idx shape strides dtype combiner N in-A in-B)
                  (let* ((A    (deref in-A))
                         (B    (deref in-B))
                         (buf  (vector-ref pool-bufs pool-idx))
                         (size (shape-size shape)))
                    (cases array-morphism A
                      (concrete-array (data1 _ _ _ _ _ _)
                        (cases array-morphism B
                          (concrete-array (data2 _ _ _ _ _ _)
                            (execute-flat-bias-broadcast-compute combiner data1 data2 buf size N dtype)
                            (make-pool-arr pool-idx shape strides dtype))
                          (else (error "ri-flat-bias-broadcast: B not concrete" B))))
                      (else (error "ri-flat-bias-broadcast: A not concrete" A)))))

                (ri-gemm-epilogue (pool-idx shape strides dtype A-ref B-ref
                                   epilogue-kind epilogue-comb epilogue-N bias-ref)
                  (let* ((buf (vector-ref pool-bufs pool-idx))
                         (sz  (shape-size shape)))
                    (execute-blas-gemm-strided/into! (deref A-ref) (deref B-ref) buf)
                    (case epilogue-kind
                      ((unary)
                       (execute-flat-unary-compute-inplace! epilogue-comb buf sz dtype))
                      ((bias-broadcast)
                       (cases array-morphism (deref bias-ref)
                         (concrete-array (bias-data _ _ _ _ _ _)
                           (execute-flat-bias-broadcast-inplace!
                            epilogue-comb buf bias-data sz epilogue-N dtype))
                         (else (error "ri-gemm-epilogue: bias not concrete"))))
                      (else (error "ri-gemm-epilogue: unknown epilogue-kind" epilogue-kind)))
                    (make-pool-arr pool-idx shape strides dtype)))

                (ri-alias (pool-idx shape strides dtype in-ref)
                  ;; The epilogue was applied in-place by the preceding ri-gemm-epilogue.
                  ;; Advance alloc-ctr to stay in sync with trace-time pool allocation,
                  ;; then return the GEMM result (already epilogue-applied) with the
                  ;; correct alloc-id for this binding position.
                  (let* ((src (deref in-ref))
                         (aid alloc-ctr))
                    (set! alloc-ctr (+ alloc-ctr 1))
                    (cases array-morphism src
                      (concrete-array (src-data _ _ _ _ _ _)
                        (concrete-array src-data shape strides 0 dtype aid -1))
                      (else (error "ri-alias: source not concrete" src))))))))
        (vector-set! vals i result)))

    vals))


;;; ============================================================
;;; Phase 4: ssa-realize/ctx (with lazy replay-plan compilation)
;;; ============================================================

(define (ssa-realize/ctx ctx prog)
  "Execute the SSA program through a morphism context.

   Trace run (first call, context in trace mode):
     Executes each binding via realize/ctx, recording per-binding trace-info
     (concrete-array + is-pool? flag). Output bindings with pool allocations
     are pinned via context-pin-output! so the greedy allocator never reuses
     their slot within a replay run.

   First replay call (context in replay mode, no plan yet):
     Lazily compiles the replay-plan from trace-info + finalized pool, then
     executes it via execute-replay-plan.

   Subsequent replay calls:
     Executes the pre-compiled replay-plan directly.  No morphism rebuilding,
     no BLAS-compat re-evaluation, no context-vector allocation per binding."
  (let* ((mode        (context-mode ctx))
         (trace-mode? (eq? mode 'trace))
         (constants   (ssa-program-constants prog))
         (bindings    (ssa-program-bindings prog)))

    (cond
      ;; ---- BINDING-LOOP PATH ----
      ;; Used for the trace run AND for any replay call where trace-info is
      ;; missing (e.g., when the caller's joint-program state table missed a
      ;; lookup due to GC-invalidated eq?-hash keys and created a fresh
      ;; program).  In both cases we run every binding via realize/ctx, collect
      ;; trace-info, and return the outputs.  Output pinning via
      ;; context-pin-output! is only performed in true trace mode.
      ((or trace-mode? (not (ssa-program-trace-info prog)))
       (let* ((values      (make-hash-table))
              (trace-info  (make-hash-table))
              (output-bids (make-hash-table eq? eq?-hash)))
         ;; Index output binding names for O(1) lookup
         (for-each (lambda (ov)
                     (when (ssa-binding-ref? ov)
                       (hash-table-set! output-bids (ssa-value-id ov) #t)))
                   (ssa-program-outputs prog))
         ;; Pre-populate constants
         (hash-table-walk constants
           (lambda (k v) (hash-table-set! values k v)))
         ;; Execute each binding; record trace-info
         (for-each
          (lambda (b)
            (let* ((inputs     (map (lambda (v) (hash-table-ref values (ssa-value-id v)))
                                    (ssa-binding-inputs b)))
                   (is-output? (hash-table-ref/default output-bids (ssa-binding-name b) #f))
                   (ctr-before (context-counter ctx))
                   (result     (realize/ctx ctx (rebuild-morphism b inputs)))
                   (is-pool?   (> (context-counter ctx) ctr-before))
                   (stored
                    (if is-output?
                        (if (concrete-row-major? result)
                            (begin
                              ;; context-pin-output! only valid in trace mode
                              (when (and trace-mode? is-pool?)
                                (context-pin-output! ctx (concrete-alloc-id result)))
                              result)
                            (copy-concrete-array result))
                        result)))
              (hash-table-set! values (ssa-binding-name b) stored)
              ;; Record per-binding (concrete-array . is-pool?) for replay-plan compilation
              (hash-table-set! trace-info (ssa-binding-name b) (cons result is-pool?))))
          bindings)
         ;; Save trace-info for lazy replay-plan compilation
         (ssa-program-trace-info-set! prog trace-info)
         ;; Return outputs
         (map (lambda (ov) (hash-table-ref values (ssa-value-id ov)))
              (ssa-program-outputs prog))))

      ;; ---- REPLAY RUN ----
      (else
       ;; Lazily compile replay-plan on first replay call (pool is now available)
       (unless (ssa-program-replay-plan prog)
         (ssa-program-replay-plan-set! prog
           (compile-replay-plan prog ctx)))
       ;; Execute the pre-compiled plan directly
       (let* ((plan  (ssa-program-replay-plan prog))
              (pool  (morphism-context-pool ctx))
              (vals  (execute-replay-plan plan pool constants))
              ;; output-specs pre-computed by compile-replay-plan: list of
              ;; (integer | concrete-array) — no hash-table rebuild per step.
              (specs (ssa-program-output-specs prog)))
         (map (lambda (spec)
                (if (integer? spec)
                    (let ((v (vector-ref vals spec)))
                      (if (concrete-row-major? v) v (copy-concrete-array v)))
                    spec))   ; const-ref: direct concrete-array from constants
              specs))))))

) ; end module array-morphisms-ssa
