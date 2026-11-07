;;; test-ssa.scm
;;;
;;; Unit tests for array-morphisms-ssa
;;;
;;; Covers:
;;;   1. morphism-to-ssa compilation (structure, deduplication, constant lookup)
;;;   2. ssa-realize output correctness (all ops)
;;;   3. ssa-vjp gradient correctness (analytical verification)
;;;   4. ssa-realize/ctx trace and replay (context integration)
;;;   5. Aliasing regression tests:
;;;        - Transposed-gradient layout bug (copy-concrete-array)
;;;        - Pool-slot aliasing bug (output overwritten within same replay run)
;;;        - Multi-step replay stability
;;;

(import scheme (chicken base) (chicken format)
        test
        (only srfi-1 iota every any map filter filter-map take drop fold delete-duplicates list-index
              append-map)
        (only srfi-4 f64vector f64vector-ref f64vector-length f64vector-set! f32vector
              make-f32vector f32vector-ref f32vector-set! f32vector->list f64vector->list
              list->s32vector s32vector->list make-s32vector make-f64vector
              s32vector-set! s32vector-ref)
        srfi-69
        datatype
        array-morphisms-core
        array-morphisms-index-fn
        array-morphisms-basic-ops
        array-morphisms-structural-ops
        array-morphisms-blas-exec
        array-morphisms-realization
        array-morphisms-context
        (prefix array-morphisms-grad am:)
        array-morphisms-segop
        array-morphisms-activation-exec
        (only array-morphisms-scalar-ops scalar-op-combiner)
        array-morphisms-ssa)


;;;; ============================================================
;;;; Helpers
;;;; ============================================================

(define tol 1e-9)

(define (approx= a b)
  (< (abs (- a b)) tol))

(define (lists-approx= l1 l2)
  (and (= (length l1) (length l2))
       (every approx= l1 l2)))

;;; Read all elements of a concrete-array respecting source strides.
;;; This correctly handles zero-copy transposed views (permuted strides).
(define (concrete->list m)
  (cases array-morphism (realize m)
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (map (lambda (i)
             (let* ((multi (linear-to-multi-index i shape))
                    (phys  (multi-to-linear-index multi strides offset)))
               (exact->inexact (typed-vector-ref data dtype phys))))
           (iota (shape-size shape))))
    (else (error "concrete->list: not a concrete-array after realize"))))

;;; True iff m is a concrete-array with row-major strides and zero offset.
;;; Transposed zero-copy views fail this check.
(define (concrete-row-major? m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (and (= offset 0) (equal? strides (compute-strides shape))))
    (else #f)))

;;; Return ssa-realize results as lists of f64 values (stride-aware).
(define (ssa-results->lists results)
  (map concrete->list results))

;;; Build a requires-grad morph-variable from a flat list + shape.
(define (param-var lst shape)
  (am:make-var (morph-from-list lst (list->vector shape) 'f64) #t))

;;; Build a no-grad morph-variable (input / constant).
(define (input-var lst shape)
  (am:make-var (morph-from-list lst (list->vector shape) 'f64) #f))

;;; Compile, run VJP, realize with ssa-realize.
;;; Returns (list loss-val grad1 grad2 ...) as lists of f64.
(define (compile-and-realize loss-mv params)
  (let* ((fwd-prog  (morphism-to-ssa loss-mv))
         (p-vals    (filter-map
                     (lambda (p)
                       (ssa-constant-id fwd-prog (am:var-value p)))
                     params))
         (joint     (ssa-vjp fwd-prog p-vals (ssa-loss-binding-val fwd-prog)))
         (results   (ssa-realize joint)))
    (ssa-results->lists results)))

;;; Same but with ssa-realize/ctx (trace mode only; caller provides ctx).
(define (compile-and-realize/ctx ctx loss-mv params)
  (let* ((fwd-prog  (morphism-to-ssa loss-mv))
         (p-vals    (filter-map
                     (lambda (p)
                       (ssa-constant-id fwd-prog (am:var-value p)))
                     params))
         (joint     (ssa-vjp fwd-prog p-vals (ssa-loss-binding-val fwd-prog)))
         (results   (ssa-realize/ctx ctx joint)))
    (values joint results)))

;;; Return the raw concrete-array for gradient N (0-indexed, after loss).
(define (grad-concrete results n)
  (list-ref results (+ 1 n)))


;;;; ============================================================
;;;; Group 1: morphism-to-ssa compilation
;;;; ============================================================

(test-group "morphism-to-ssa compilation"

  (test-assert "single concrete-array leaf becomes a constant, no bindings"
    (let* ((W  (morph-from-list '(1.0 2.0) #(2) 'f64))
           (mv (am:make-var W #t))
           (prog (morphism-to-ssa mv)))
      (and (null? (ssa-program-bindings prog))
           (= 1 (let ((n 0))
                  (hash-table-walk (ssa-program-constants prog)
                                   (lambda (k v) (set! n (+ n 1))))
                  n)))))

  (test-assert "single elementwise op produces exactly one binding"
    (let* ((a  (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b  (morph-from-list '(3.0 4.0) #(2) 'f64))
           (mv (am:make-var (morph+ a b) #f))
           (prog (morphism-to-ssa mv)))
      (= 1 (length (ssa-program-bindings prog)))))

  (test-assert "chain of two ops produces exactly two bindings in topo order"
    (let* ((a  (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b  (morph+ a a))           ; binding 0
           (c  (morph* b b))           ; binding 1
           (mv (am:make-var c #f))
           (prog (morphism-to-ssa mv))
           (bs  (ssa-program-bindings prog)))
      (and (= 2 (length bs))
           ;; binding 0 should be 'add, binding 1 should be 'mul
           (eq? 'add (ssa-binding-op (list-ref bs 0)))
           (eq? 'mul (ssa-binding-op (list-ref bs 1))))))

  (test-assert "shared subexpression (diamond) is deduplicated to one binding"
    ;; b = a + a;  c = b * b  : 'b' is shared, should emit only 2 total bindings
    (let* ((a  (morph-from-list '(2.0) #(1) 'f64))
           (b  (morph+ a a))
           (c  (morph* b b))
           (mv (am:make-var c #f))
           (prog (morphism-to-ssa mv)))
      (= 2 (length (ssa-program-bindings prog)))))

  (test-assert "reduction op produces a binding with reduce op"
    (let* ((a  (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (r  (morph-reduce 'mean a '(0) #f))
           (mv (am:make-var r #f))
           (prog (morphism-to-ssa mv))
           (bs  (ssa-program-bindings prog)))
      (and (= 1 (length bs))
           (equal? '(reduce mean) (ssa-binding-op (car bs))))))

  (test-assert "transpose op produces a binding with 'transpose op"
    (let* ((a  (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
           (t  (morph-transpose a))
           (mv (am:make-var t #f))
           (prog (morphism-to-ssa mv))
           (bs  (ssa-program-bindings prog)))
      (and (= 1 (length bs))
           (eq? 'transpose (ssa-binding-op (car bs))))))

  (test-assert "ssa-constant-id finds a concrete-array constant"
    (let* ((W  (morph-from-list '(1.0 2.0) #(2) 'f64))
           (mv (am:make-var W #t))
           (prog (morphism-to-ssa mv))
           (v  (ssa-constant-id prog W)))
      (and v (ssa-const-ref? v))))

  (test-assert "ssa-constant-id returns #f for unknown morphism"
    (let* ((W  (morph-from-list '(1.0) #(1) 'f64))
           (mv (am:make-var W #t))
           (prog (morphism-to-ssa mv))
           (other (morph-from-list '(1.0) #(1) 'f64)))
      (not (ssa-constant-id prog other))))

  (test-assert "ssa-loss-binding-val is a binding-ref when loss is computed"
    (let* ((a  (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b  (morph-from-list '(3.0 4.0) #(2) 'f64))
           (mv (am:make-var (morph+ a b) #f))
           (prog (morphism-to-ssa mv)))
      (ssa-binding-ref? (ssa-loss-binding-val prog))))

  (test-assert "ssa-vjp extends outputs: first is loss, rest are param grads"
    (let* ((W  (morph-from-list '(1.0 2.0) #(2) 'f64))
           (x  (morph-from-list '(3.0 4.0) #(2) 'f64))
           (mv (am:make-var (morph+ (morph* W x) W) #f))
           (p-mv (am:make-var W #t))
           (prog (morphism-to-ssa mv))
           (p-val (ssa-constant-id prog W))
           (joint (ssa-vjp prog (list p-val) (ssa-loss-binding-val prog))))
      ;; outputs: (loss . (dW))
      (= 2 (length (ssa-program-outputs joint))))))


;;;; ============================================================
;;;; Group 2: ssa-realize output correctness
;;;; ============================================================

(test-group "ssa-realize output correctness"

  (test-assert "add: [1,2,3] + [4,5,6] = [5,7,9]"
    (let* ((a  (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b  (morph-from-list '(4.0 5.0 6.0) #(3) 'f64))
           (mv (am:make-var (morph+ a b) #f))
           (prog (morphism-to-ssa mv))
           (out  (car (ssa-results->lists (ssa-realize prog)))))
      (lists-approx= out '(5.0 7.0 9.0))))

  (test-assert "mul: [1,2,3] * [4,5,6] = [4,10,18]"
    (let* ((a  (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b  (morph-from-list '(4.0 5.0 6.0) #(3) 'f64))
           (mv (am:make-var (morph* a b) #f))
           (prog (morphism-to-ssa mv))
           (out  (car (ssa-results->lists (ssa-realize prog)))))
      (lists-approx= out '(4.0 10.0 18.0))))

  (test-assert "matmul: 2x2 @ 2x2 = correct product"
    ;; A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    ;; A@B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    (let* ((A  (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
           (B  (morph-from-list '(5.0 6.0 7.0 8.0) #(2 2) 'f64))
           (mv (am:make-var (morph-matmul A B) #f))
           (prog (morphism-to-ssa mv))
           (out  (car (ssa-results->lists (ssa-realize prog)))))
      (lists-approx= out '(19.0 22.0 43.0 50.0))))

  (test-assert "reduce-mean over all: mean([1,2,3,4]) = 2.5"
    (let* ((a  (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64))
           (mv (am:make-var (morph-reduce 'mean a '(0) #f) #f))
           (prog (morphism-to-ssa mv))
           (out  (car (ssa-results->lists (ssa-realize prog)))))
      (and (= 1 (length out))
           (approx= (car out) 2.5))))

  (test-assert "reduce-sum over all: sum([1,2,3,4]) = 10"
    (let* ((a  (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64))
           (mv (am:make-var (morph-reduce 'sum a '(0) #f) #f))
           (prog (morphism-to-ssa mv))
           (out  (car (ssa-results->lists (ssa-realize prog)))))
      (and (= 1 (length out))
           (approx= (car out) 10.0))))

  (test-assert "transpose 2x2: [[1,2],[3,4]] -> [[1,3],[2,4]]"
    ;; Element-wise check via concrete->list (stride-aware)
    (let* ((A  (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
           (At (morph-transpose A))
           (mv (am:make-var At #f))
           (prog (morphism-to-ssa mv))
           (out  (car (ssa-results->lists (ssa-realize prog)))))
      ;; At in row-major: A[0,0]=1, A[1,0]=3, A[0,1]=2, A[1,1]=4 → [1,3,2,4]
      (lists-approx= out '(1.0 3.0 2.0 4.0))))

  (test-assert "matmul followed by add (bias): Z = A@B + c gives correct result"
    ;; A = [[1,0],[0,1]], B = [[2,3],[4,5]], c = [[1,1],[1,1]]
    ;; A@B = [[2,3],[4,5]], A@B+c = [[3,4],[5,6]]
    (let* ((A  (morph-from-list '(1.0 0.0 0.0 1.0) #(2 2) 'f64))
           (B  (morph-from-list '(2.0 3.0 4.0 5.0) #(2 2) 'f64))
           (c  (morph-from-list '(1.0 1.0 1.0 1.0) #(2 2) 'f64))
           (mv (am:make-var (morph+ (morph-matmul A B) c) #f))
           (prog (morphism-to-ssa mv))
           (out  (car (ssa-results->lists (ssa-realize prog)))))
      (lists-approx= out '(3.0 4.0 5.0 6.0)))))


;;;; ============================================================
;;;; Group 3: ssa-vjp gradient correctness
;;;; ============================================================

(test-group "ssa-vjp gradient correctness"

  (test-assert "dW for loss = sum(W * x): dW = x"
    ;; loss = sum(W .* x) → dL/dW[i] = x[i]
    (let* ((W-data '(0.5 1.0 1.5 2.0))
           (x-data '(1.0 2.0 3.0 4.0))
           (W-mv (param-var W-data '(4)))
           (x-mv (input-var x-data '(4)))
           (prod-mv  (am:var* W-mv x-mv))
           (loss-mv  (am:var-sum prod-mv))
           (res      (compile-and-realize loss-mv (list W-mv)))
           (dW       (cadr res)))
      (lists-approx= dW x-data)))

  (test-assert "dW for loss = sum(W): dW = ones"
    (let* ((W-mv  (param-var '(0.1 0.2 0.3) '(3)))
           (loss-mv (am:var-sum W-mv))
           (res   (compile-and-realize loss-mv (list W-mv)))
           (dW    (cadr res)))
      (lists-approx= dW '(1.0 1.0 1.0))))

  (test-assert "db for loss = sum(Z + b) where Z is constant: db = ones"
    (let* ((Z-mv  (input-var '(1.0 2.0 3.0 4.0) '(4)))
           (b-mv  (param-var '(0.0 0.0 0.0 0.0) '(4)))
           (out-mv (am:var+ Z-mv b-mv))
           (loss-mv (am:var-sum out-mv))
           (res   (compile-and-realize loss-mv (list b-mv)))
           (db    (cadr res)))
      (lists-approx= db '(1.0 1.0 1.0 1.0))))

  (test-assert "dW for loss = mean(Z = X @ W^T): correct values (stride-aware read)"
    ;; X = [[1,2],[3,4],[5,6]] (3x2), W = [[0.1,0.2],[0.3,0.4]] (2x2)
    ;; Z = X @ W^T  (3x2), loss = mean(Z)
    ;; Analytical: dW = [[1.5, 2.0],[1.5, 2.0]] (row-major: 1.5 2.0 1.5 2.0)
    (let* ((X-mv (input-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(3 2)))
           (W-mv (param-var '(0.1 0.2 0.3 0.4) '(2 2)))
           (Wt-mv   (am:var-transpose W-mv '(1 0)))
           (Z-mv    (am:var-matmul X-mv Wt-mv))
           (loss-mv (am:var-mean Z-mv))
           (res   (compile-and-realize loss-mv (list W-mv)))
           (dW    (cadr res)))        ; stride-aware read
      (lists-approx= dW '(1.5 2.0 1.5 2.0))))

  (test-assert "two independent params yield independent gradients"
    ;; loss = sum(W1) + sum(W2); dW1 = ones, dW2 = ones, values independent
    (let* ((W1-mv (param-var '(1.0 2.0 3.0) '(3)))
           (W2-mv (param-var '(4.0 5.0 6.0) '(3)))
           (loss-mv (am:var+ (am:var-sum W1-mv) (am:var-sum W2-mv)))
           (res (compile-and-realize loss-mv (list W1-mv W2-mv)))
           (dW1 (cadr res))
           (dW2 (caddr res)))
      (and (lists-approx= dW1 '(1.0 1.0 1.0))
           (lists-approx= dW2 '(1.0 1.0 1.0)))))

  (test-assert "loss scalar value is computed correctly alongside gradients"
    ;; loss = mean(W * x): loss value = mean([1*1, 2*2, 3*3]) = (1+4+9)/3 = 14/3
    (let* ((W-mv (param-var '(1.0 2.0 3.0) '(3)))
           (x-mv (input-var '(1.0 2.0 3.0) '(3)))
           (loss-mv (am:var-mean (am:var* W-mv x-mv)))
           (res (compile-and-realize loss-mv (list W-mv)))
           (loss-val (caar res)))
      (approx= loss-val (/ 14.0 3.0)))))


;;;; ============================================================
;;;; Group 4: ssa-realize/ctx trace and replay
;;;; ============================================================

(test-group "ssa-realize/ctx trace and replay"

  (test-assert "trace run records at least one allocation"
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (x-mv (input-var '(0.5 0.5 0.5 0.5) '(4))))
      (compile-and-realize/ctx ctx
        (am:var-sum (am:var* W-mv x-mv))
        (list W-mv))
      (> (cdr (assq 'allocations (context-stats ctx))) 0)))

  (test-assert "replay gives same loss as trace"
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (x-mv (input-var '(1.0 1.0 1.0 1.0) '(4)))
           (loss-mv (am:var-sum (am:var* W-mv x-mv))))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (let ((trace-loss (caar (ssa-results->lists trace-results))))
          (finalize-context! ctx)
          (reset-context! ctx)
          (let* ((replay-results (ssa-realize/ctx ctx joint))
                 (replay-loss (f64vector-ref
                               (cases array-morphism (car replay-results)
                                 (concrete-array (data shape strides offset dtype ai ba) data)
                                 (else (error "not concrete")))
                               0)))
            (approx= trace-loss replay-loss))))))

  (test-assert "replay gradient matches trace gradient"
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (x-mv (input-var '(0.1 0.2 0.3 0.4) '(4)))
           (loss-mv (am:var-sum (am:var* W-mv x-mv))))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (let ((trace-dW (concrete->list (cadr trace-results))))
          (finalize-context! ctx)
          (reset-context! ctx)
          (let* ((replay-results (ssa-realize/ctx ctx joint))
                 (replay-dW (concrete->list (cadr replay-results))))
            (lists-approx= trace-dW replay-dW))))))

  (test-assert "three consecutive replay runs give identical results"
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(0.5 1.0 1.5 2.0) '(4)))
           (x-mv (input-var '(2.0 1.0 0.5 0.25) '(4)))
           (loss-mv (am:var-sum (am:var* W-mv x-mv))))
      (let-values (((joint _)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (finalize-context! ctx)
        (let run ((i 0) (prev-dW #f))
          (if (= i 3)
              #t
              (begin
                (reset-context! ctx)
                (let* ((results (ssa-realize/ctx ctx joint))
                       (dW      (concrete->list (cadr results))))
                  (if (and prev-dW (not (lists-approx= dW prev-dW)))
                      #f
                      (run (+ i 1) dW)))))))))

  (test-assert "context switches to replay mode after finalize"
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(1.0 2.0) '(2)))
           (loss-mv (am:var-sum W-mv)))
      (let-values (((joint _)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (finalize-context! ctx)
        (eq? (context-mode ctx) 'replay)))))


;;;; ============================================================
;;;; Group 5: Aliasing regression tests
;;;; ============================================================

(test-group "aliasing regression: transposed gradient layout"

  ;; This test targets the bug where dW from a matmul-based model was returned
  ;; as a zero-copy transposed view (non-row-major concrete-array).  Reading the
  ;; raw SRFI-4 data buffer flat (ignoring strides) produced dW^T instead of dW.
  ;;
  ;; Setup: Z = X @ W^T, loss = mean(Z)
  ;;   X  = [[1,2],[3,4],[5,6]]  (3x2)
  ;;   W  = [[0.1,0.2],[0.3,0.4]]  (2x2)
  ;;   W^T = [[0.1,0.3],[0.2,0.4]]
  ;;   Z  = [[0.5,1.1],[1.1,2.5],[1.7,3.9]]  (3x2)
  ;;   loss = mean(Z) = 10.8/6 = 1.8
  ;;
  ;; Analytical dW = [[1.5, 2.0],[1.5, 2.0]]
  ;;   Row-major flat:  [1.5, 2.0, 1.5, 2.0]  (CORRECT)
  ;;   dW^T flat:       [1.5, 1.5, 2.0, 2.0]  (BUG: transposed read)

  (test-assert "ssa-realize dW for Z = X@W^T has correct values (stride-aware)"
    (let* ((X-mv (input-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(3 2)))
           (W-mv (param-var '(0.1 0.2 0.3 0.4) '(2 2)))
           (Wt-mv   (am:var-transpose W-mv '(1 0)))
           (Z-mv    (am:var-matmul X-mv Wt-mv))
           (loss-mv (am:var-mean Z-mv))
           (res     (compile-and-realize loss-mv (list W-mv)))
           (dW-list (cadr res)))
      ;; stride-aware read must give [1.5, 2.0, 1.5, 2.0], not [1.5, 1.5, 2.0, 2.0]
      (lists-approx= dW-list '(1.5 2.0 1.5 2.0))))

  (test-assert "ssa-realize/ctx dW is row-major (copy-concrete-array applied)"
    ;; After the fix, ssa-realize/ctx copies output transposed views to fresh
    ;; row-major buffers.  The returned concrete-array must pass concrete-row-major?.
    (let* ((ctx  (make-morphism-context))
           (X-mv (input-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(3 2)))
           (W-mv (param-var '(0.1 0.2 0.3 0.4) '(2 2)))
           (Wt-mv   (am:var-transpose W-mv '(1 0)))
           (Z-mv    (am:var-matmul X-mv Wt-mv))
           (loss-mv (am:var-mean Z-mv)))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (concrete-row-major? (cadr trace-results)))))

  (test-assert "ssa-realize/ctx dW has correct flat layout [1.5 2.0 1.5 2.0]"
    ;; After copy-concrete-array, the flat data read directly (no stride adjustment)
    ;; must equal the expected dW in row-major order.
    (let* ((ctx  (make-morphism-context))
           (X-mv (input-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(3 2)))
           (W-mv (param-var '(0.1 0.2 0.3 0.4) '(2 2)))
           (Wt-mv   (am:var-transpose W-mv '(1 0)))
           (Z-mv    (am:var-matmul X-mv Wt-mv))
           (loss-mv (am:var-mean Z-mv)))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        ;; Read flat (no stride adjustment) — valid because we verified row-major above
        (let* ((dW-arr (cadr trace-results)))
          (cases array-morphism dW-arr
            (concrete-array (data shape strides offset dtype alloc-id batch-axis)
              (let ((flat (map (lambda (i) (typed-vector-ref data dtype i))
                               (iota (shape-size shape)))))
                (lists-approx= flat '(1.5 2.0 1.5 2.0))))
            (else #f))))))

  (test-assert "replay also returns row-major dW"
    (let* ((ctx  (make-morphism-context))
           (X-mv (input-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(3 2)))
           (W-mv (param-var '(0.1 0.2 0.3 0.4) '(2 2)))
           (Wt-mv   (am:var-transpose W-mv '(1 0)))
           (Z-mv    (am:var-matmul X-mv Wt-mv))
           (loss-mv (am:var-mean Z-mv)))
      (let-values (((joint _)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (finalize-context! ctx)
        (reset-context! ctx)
        (let* ((replay-results (ssa-realize/ctx ctx joint)))
          (and (concrete-row-major? (cadr replay-results))
               (cases array-morphism (cadr replay-results)
                 (concrete-array (data shape strides offset dtype alloc-id batch-axis)
                   (lists-approx=
                    (map (lambda (i) (typed-vector-ref data dtype i))
                         (iota (shape-size shape)))
                    '(1.5 2.0 1.5 2.0)))
                 (else #f))))))))


(test-group "aliasing regression: pool-slot aliasing"

  ;; This test targets the bug where an output gradient's pool buffer slot was
  ;; assigned last-use = birth, causing the greedy allocator to immediately free
  ;; the slot and reassign it to a later binding within the SAME replay run.
  ;;
  ;; Setup: loss = sum(W1 * x) + 2 * sum(W2 * x)
  ;;   W1 = [1,2,3,4], W2 = [5,6,7,8], x = [1,1,1,1]
  ;;   dW1 = x = [1,1,1,1]  (gradient coefficient 1)
  ;;   dW2 = 2*x = [2,2,2,2]  (gradient coefficient 2)
  ;;
  ;; Since W1 and W2 have the same shape, the greedy allocator may assign their
  ;; gradient pool slots to the same physical buffer (or to each other's freed slot).
  ;; If the earlier gradient's pool slot is later reused by the second gradient,
  ;; collecting the earlier gradient after both are computed gives the WRONG value.
  ;;
  ;; With the fix (copy-concrete-array for output bindings), each gradient is
  ;; immediately copied to a fresh non-pooled buffer when computed, preventing
  ;; any later binding from overwriting it.

  (test-assert "both gradients have correct distinct values after trace"
    (let* ((W1-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (W2-mv (param-var '(5.0 6.0 7.0 8.0) '(4)))
           (x-mv  (input-var '(1.0 1.0 1.0 1.0) '(4)))
           (two   (morph-from-list '(2.0) #(1) 'f64))
           (sum1-mv (am:var-sum (am:var* W1-mv x-mv)))
           (sum2-mv (am:var-sum (am:var* W2-mv x-mv)))
           ;; 2 * sum(W2*x): multiply the scalar sum by 2
           (two-mv  (am:make-var two #f))
           (scaled-mv (am:var* two-mv sum2-mv))
           (loss-mv (am:var+ sum1-mv scaled-mv))
           (ctx (make-morphism-context)))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W1-mv W2-mv))))
        (let ((dW1 (concrete->list (cadr  trace-results)))
              (dW2 (concrete->list (caddr trace-results))))
          ;; dW1 must be x = [1,1,1,1]; dW2 must be 2*x = [2,2,2,2]
          ;; If aliasing: dW1 or dW2 would be overwritten by the other
          (and (lists-approx= dW1 '(1.0 1.0 1.0 1.0))
               (lists-approx= dW2 '(2.0 2.0 2.0 2.0)))))))

  (test-assert "both gradients have correct distinct values after replay"
    (let* ((W1-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (W2-mv (param-var '(5.0 6.0 7.0 8.0) '(4)))
           (x-mv  (input-var '(1.0 1.0 1.0 1.0) '(4)))
           (two   (morph-from-list '(2.0) #(1) 'f64))
           (sum1-mv (am:var-sum (am:var* W1-mv x-mv)))
           (sum2-mv (am:var-sum (am:var* W2-mv x-mv)))
           (two-mv  (am:make-var two #f))
           (scaled-mv (am:var* two-mv sum2-mv))
           (loss-mv (am:var+ sum1-mv scaled-mv))
           (ctx (make-morphism-context)))
      (let-values (((joint _)
                    (compile-and-realize/ctx ctx loss-mv (list W1-mv W2-mv))))
        (finalize-context! ctx)
        (reset-context! ctx)
        (let* ((replay-results (ssa-realize/ctx ctx joint))
               (dW1 (concrete->list (cadr  replay-results)))
               (dW2 (concrete->list (caddr replay-results))))
          (and (lists-approx= dW1 '(1.0 1.0 1.0 1.0))
               (lists-approx= dW2 '(2.0 2.0 2.0 2.0)))))))

  (test-assert "ssa-realize (no context) also gives correct distinct gradients"
    ;; Sanity check: the pure ssa-realize path is also correct.
    (let* ((W1-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (W2-mv (param-var '(5.0 6.0 7.0 8.0) '(4)))
           (x-mv  (input-var '(1.0 1.0 1.0 1.0) '(4)))
           (two   (morph-from-list '(2.0) #(1) 'f64))
           (sum1-mv (am:var-sum (am:var* W1-mv x-mv)))
           (sum2-mv (am:var-sum (am:var* W2-mv x-mv)))
           (two-mv  (am:make-var two #f))
           (scaled-mv (am:var* two-mv sum2-mv))
           (loss-mv (am:var+ sum1-mv scaled-mv))
           (res (compile-and-realize loss-mv (list W1-mv W2-mv)))
           (dW1 (cadr res))
           (dW2 (caddr res)))
      (and (lists-approx= dW1 '(1.0 1.0 1.0 1.0))
           (lists-approx= dW2 '(2.0 2.0 2.0 2.0))))))


(test-group "aliasing regression: multi-step replay stability"

  ;; This test verifies that running the same program N times in replay mode
  ;; produces numerically identical results each time.
  ;;
  ;; If pool-slot aliasing corrupted outputs, the values would differ between
  ;; steps (because in each step, different intermediate computations write
  ;; different values into the recycled pool slots, changing what the aliased
  ;; output gradient reads).
  ;;
  ;; We use a model with a non-trivial forward pass so the pool has multiple
  ;; reuse opportunities.

  (test-assert "loss value is identical across 5 replay steps"
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(0.1 0.2 0.3 0.4 0.5 0.6) '(2 3)))
           (x-mv (input-var '(1.0 2.0 3.0) '(3 1)))
           ;; Z = W @ x (shape [2,1]), loss = mean(Z)
           (Z-mv    (am:var-matmul W-mv x-mv))
           (loss-mv (am:var-mean Z-mv)))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (let* ((trace-loss (concrete->list (car trace-results)))
               (ref-loss (car trace-loss)))
          (finalize-context! ctx)
          (let loop ((i 0))
            (if (= i 5)
                #t
                (begin
                  (reset-context! ctx)
                  (let* ((results (ssa-realize/ctx ctx joint))
                         (loss-val (car (concrete->list (car results)))))
                    (if (approx= loss-val ref-loss)
                        (loop (+ i 1))
                        #f)))))))))

  (test-assert "gradient is identical across 5 replay steps"
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(0.1 0.2 0.3 0.4 0.5 0.6) '(2 3)))
           (x-mv (input-var '(1.0 2.0 3.0) '(3 1)))
           (Z-mv    (am:var-matmul W-mv x-mv))
           (loss-mv (am:var-mean Z-mv)))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (let ((ref-dW (concrete->list (cadr trace-results))))
          (finalize-context! ctx)
          (let loop ((i 0))
            (if (= i 5)
                #t
                (begin
                  (reset-context! ctx)
                  (let* ((results (ssa-realize/ctx ctx joint))
                         (dW      (concrete->list (cadr results))))
                    (if (lists-approx= dW ref-dW)
                        (loop (+ i 1))
                        #f)))))))))

  (test-assert "two-layer model: gradients stable across 5 replay steps"
    ;; Simple 2-layer MLP: A1 = X @ W1^T, A2 = A1 @ W2^T, loss = mean(A2)
    ;; Tests pool reuse across the full joint forward+backward program.
    (let* ((ctx   (make-morphism-context))
           (X-mv  (input-var '(1.0 0.0 0.0 1.0) '(2 2)))    ; 2x2 batch
           (W1-mv (param-var '(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8) '(4 2)))  ; 4x2
           (W2-mv (param-var '(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2) '(3 4))) ; 3x4
           (Z1-mv    (am:var-matmul X-mv  (am:var-transpose W1-mv '(1 0))))  ; 2x4
           (Z2-mv    (am:var-matmul Z1-mv (am:var-transpose W2-mv '(1 0))))  ; 2x3
           (loss-mv  (am:var-mean Z2-mv)))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W1-mv W2-mv))))
        (let ((ref-dW1 (concrete->list (cadr  trace-results)))
              (ref-dW2 (concrete->list (caddr trace-results))))
          (finalize-context! ctx)
          (let loop ((i 0))
            (if (= i 5)
                #t
                (begin
                  (reset-context! ctx)
                  (let* ((results (ssa-realize/ctx ctx joint))
                         (dW1    (concrete->list (cadr  results)))
                         (dW2    (concrete->list (caddr results))))
                    (if (and (lists-approx= dW1 ref-dW1)
                             (lists-approx= dW2 ref-dW2))
                        (loop (+ i 1))
                        #f))))))))))


;;;; ============================================================
;;;; Group 6: Output pinning integration
;;;;
;;;; Verifies that after the context-pinning fix, ssa-realize/ctx
;;;; returns pool buffers (alloc-id >= 0) for row-major outputs in
;;;; replay mode, rather than fresh copies (alloc-id = -1).
;;;;
;;;; The key invariant: concrete-array alloc-id >= 0 means the value
;;;; lives in a pre-allocated pool buffer; alloc-id = -1 means a fresh
;;;; allocation was made (which is what the old copy path produced).
;;;; ============================================================

(define (output-alloc-id result)
  "Extract alloc-id from a concrete-array."
  (cases array-morphism result
    (concrete-array (data shape strides offset dtype alloc-id batch-axis) alloc-id)
    (else (error "output-alloc-id: not a concrete-array"))))

(test-group "output pinning integration"

  (test-assert "trace: row-major output has alloc-id >= 0 (pool buffer)"
    ;; After the pinning fix, ssa-realize/ctx returns the pool buffer directly
    ;; for row-major outputs instead of copying to alloc-id=-1.
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (x-mv (input-var '(0.5 0.5 0.5 0.5) '(4)))
           (loss-mv (am:var-sum (am:var* W-mv x-mv))))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        ;; dW is a reduce-sum output -> row-major, pool-allocated, pinned
        (>= (output-alloc-id (cadr trace-results)) 0))))

  (test-assert "replay: row-major output has alloc-id >= 0 (pool buffer, no copy)"
    ;; In replay mode, the same pinned pool buffer is returned directly.
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (x-mv (input-var '(0.5 0.5 0.5 0.5) '(4)))
           (loss-mv (am:var-sum (am:var* W-mv x-mv))))
      (let-values (((joint _)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (finalize-context! ctx)
        (reset-context! ctx)
        (let* ((results (ssa-realize/ctx ctx joint)))
          (>= (output-alloc-id (cadr results)) 0)))))

  (test-assert "replay output alloc-id matches trace output alloc-id"
    ;; The same physical pool buffer slot is reused each replay run.
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (x-mv (input-var '(0.5 0.5 0.5 0.5) '(4)))
           (loss-mv (am:var-sum (am:var* W-mv x-mv))))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (let ((trace-id (output-alloc-id (cadr trace-results))))
          (finalize-context! ctx)
          (reset-context! ctx)
          (let* ((r1 (ssa-realize/ctx ctx joint))
                 (r1-id (output-alloc-id (cadr r1))))
            (reset-context! ctx)
            (let* ((r2 (ssa-realize/ctx ctx joint))
                   (r2-id (output-alloc-id (cadr r2))))
              ;; All three runs return pool buffers with the same slot assignment
              (and (= trace-id r1-id) (= r1-id r2-id))))))))

  (test-assert "transposed gradient output retains alloc-id = -1 (copy-path)"
    ;; dW from Z = X @ W^T comes through a transpose VJP step, producing a
    ;; non-row-major view.  copy-concrete-array is still required -> alloc-id = -1.
    (let* ((ctx  (make-morphism-context))
           (X-mv (input-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(3 2)))
           (W-mv (param-var '(0.1 0.2 0.3 0.4) '(2 2)))
           (Wt-mv   (am:var-transpose W-mv '(1 0)))
           (Z-mv    (am:var-matmul X-mv Wt-mv))
           (loss-mv (am:var-mean Z-mv)))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        ;; dW gradient for W comes through transpose -> non-row-major -> must copy
        (= (output-alloc-id (cadr trace-results)) -1))))

  ;; A strided output is copied, at every replay, into one buffer that the
  ;; plan holds, rather than into a fresh vector per run.
  (let* ((ctx  (make-morphism-context))
         (X-mv (input-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(3 2)))
         (W-mv (param-var '(0.1 0.2 0.3 0.4) '(2 2)))
         (loss-mv (am:var-mean (am:var-matmul X-mv (am:var-transpose W-mv '(1 0))))))
    (let-values (((joint trace-results)
                  (compile-and-realize/ctx ctx loss-mv (list W-mv))))
      (let ((trace-dW (concrete->list (cadr trace-results)))
            (data-of  (lambda (a) (cases array-morphism a
                                    (concrete-array (data _ _ _ _ _ _) data)
                                    (else #f)))))
        (finalize-context! ctx)
        (reset-context! ctx)
        (let* ((r1 (ssa-realize/ctx ctx joint))
               (dW1 (concrete->list (cadr r1)))
               (data1 (data-of (cadr r1))))
          (reset-context! ctx)
          (let* ((r2 (ssa-realize/ctx ctx joint)))
            (test-assert "replay: strided output is row-major with the trace values"
              (and (concrete-row-major? (cadr r1)) (lists-approx= trace-dW dW1)))
            (test-assert "replay: strided output reuses the same plan buffer each run"
              (and (eq? data1 (data-of (cadr r2)))
                   (lists-approx= trace-dW (concrete->list (cadr r2))))))))))

  ;; Once the plan is compiled the trace arrays are released; later calls
  ;; still run the plan (each plan run counts as a timing step) and give
  ;; the trace values.
  (test-assert "replay keeps running the plan after the trace arrays are released"
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (x-mv (input-var '(0.1 0.2 0.3 0.4) '(4)))
           (loss-mv (am:var-sum (am:var* W-mv x-mv))))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (let ((trace-dW (concrete->list (cadr trace-results))))
          (finalize-context! ctx)
          (replay-timing-reset!)
          (let loop ((i 0) (ok #t))
            (if (= i 3)
                (and ok (= 3 (cdr (assq 'steps (replay-timing-results)))))
                (begin
                  (reset-context! ctx)
                  (loop (+ i 1)
                        (and ok (lists-approx= trace-dW
                                               (concrete->list (cadr (ssa-realize/ctx ctx joint)))))))))))))

  (test-assert "pinning: numerical values identical trace vs replay (row-major case)"
    (let* ((ctx  (make-morphism-context))
           (W-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (x-mv (input-var '(0.1 0.2 0.3 0.4) '(4)))
           (loss-mv (am:var-sum (am:var* W-mv x-mv))))
      (let-values (((joint trace-results)
                    (compile-and-realize/ctx ctx loss-mv (list W-mv))))
        (let ((trace-dW (concrete->list (cadr trace-results))))
          (finalize-context! ctx)
          (reset-context! ctx)
          (let* ((replay-results (ssa-realize/ctx ctx joint))
                 (replay-dW (concrete->list (cadr replay-results))))
            (lists-approx= trace-dW replay-dW))))))

  (test-assert "pinning: two-param model replay outputs both have alloc-id >= 0"
    (let* ((ctx   (make-morphism-context))
           (W1-mv (param-var '(1.0 2.0 3.0 4.0) '(4)))
           (W2-mv (param-var '(0.5 0.5 0.5 0.5) '(4)))
           (x-mv  (input-var '(1.0 1.0 1.0 1.0) '(4)))
           (loss-mv (am:var+ (am:var-sum (am:var* W1-mv x-mv))
                             (am:var-sum (am:var* W2-mv x-mv)))))
      (let-values (((joint _)
                    (compile-and-realize/ctx ctx loss-mv (list W1-mv W2-mv))))
        (finalize-context! ctx)
        (reset-context! ctx)
        (let* ((results (ssa-realize/ctx ctx joint)))
          (and (>= (output-alloc-id (cadr  results)) 0)
               (>= (output-alloc-id (caddr results)) 0)))))))


;;;; ============================================================
;;;; Group E2: gradients through fused bindings
;;;; ============================================================

(test-group "E2: gradients through fused bindings"

  (test "the gradient of sum(negate(x)) is -1 everywhere"
    #t
    (let* ((xs   '(1.0 2.0 3.0))
           (x-m  (morph-from-list xs #(3) 'f64))
           (xv   (am:make-var x-m #t))
           (loss (am:var-sum (am:var-negate xv)))
           (res  (compile-and-realize loss (list xv)))
           (got-dx (cadr res)))
      ;; dL/dx[i] = -1 for all i
      (lists-approx= got-dx '(-1.0 -1.0 -1.0)))))


;;;; ============================================================
;;;; Group E1: single-op activations -- forward and backward
;;;; ============================================================

(test-group "E1: morph-relu forward"
  (test "relu: positive values unchanged"
    '(1.0 2.0 3.0)
    (let* ((x  (morph-from-list '(1.0 2.0 3.0) #(3) 'f64)))
      (map exact->inexact (concrete->list (realize (morph-relu x))))))

  (test "relu: negative values clamped to zero"
    '(0.0 0.0 0.0)
    (let* ((x  (morph-from-list '(-1.0 -2.0 -3.0) #(3) 'f64)))
      (map exact->inexact (concrete->list (realize (morph-relu x))))))

  (test "relu: mixed signs"
    '(0.0 2.0 0.0 4.0)
    (let* ((x  (morph-from-list '(-1.0 2.0 -3.0 4.0) #(4) 'f64)))
      (map exact->inexact (concrete->list (realize (morph-relu x)))))))


(test-group "E1: morph-sigmoid forward"
  (define (sigmoid x) (/ 1.0 (+ 1.0 (exp (- x)))))

  (test "sigmoid: known values approx correct"
    #t
    (let* ((xs  '(-2.0 -1.0 0.0 1.0 2.0))
           (x-m (morph-from-list xs #(5) 'f64))
           (got (map exact->inexact (concrete->list (realize (morph-sigmoid x-m)))))
           (exp (map sigmoid xs)))
      (lists-approx= got exp))))


(test-group "E1: morph-tanh-am forward"
  (define (my-tanh x)
    (let ((e2 (exp (* 2.0 (exact->inexact x)))))
      (/ (- e2 1.0) (+ e2 1.0))))

  (test "tanh: known values approx correct"
    #t
    (let* ((xs  '(-1.0 0.0 1.0))
           (x-m (morph-from-list xs #(3) 'f64))
           (got (map exact->inexact (concrete->list (realize (morph-tanh-am x-m)))))
           (exp (map my-tanh xs)))
      (lists-approx= got exp)))

  (test "tanh: saturates to +-1.0 for large |x| instead of overflowing"
    '(1.0 -1.0 1.0 -1.0)
    (let ((x-m (morph-from-list '(400.0 -400.0 710.0 -710.0) #(4) 'f64)))
      (map exact->inexact (concrete->list (realize (morph-tanh-am x-m))))))

  (test "tanh: f32 saturates to +-1.0 for large |x|"
    '(1.0 -1.0)
    (let ((x-m (morph-from-list '(400.0 -400.0) #(2) 'f32)))
      (map exact->inexact (concrete->list (realize (morph-tanh-am x-m)))))))


(test-group "E1: var-relu SSA gradient"
  (test "relu: gradient correct via SSA VJP"
    #t
    ;; loss = mean(relu(x)), x = [-2 -1 0 1 2]
    ;; dL/dx[i] = (1/5) * (if x[i]>0 1 0) = [0 0 0 0.2 0.2]
    (let* ((xs  '(-2.0 -1.0 0.0 1.0 2.0))
           (x-m (morph-from-list xs #(5) 'f64))
           (xv  (am:make-var x-m #t))
           (out (am:var-relu xv))
           (loss (am:var-mean out))
           (res  (compile-and-realize loss (list xv)))
           (got  (cadr res)))
      (lists-approx= got '(0.0 0.0 0.0 0.2 0.2)))))


(test-group "E1: var-sigmoid SSA gradient"
  (define (sigmoid x) (/ 1.0 (+ 1.0 (exp (- (exact->inexact x))))))

  (test "sigmoid: gradient correct via SSA VJP"
    #t
    ;; loss = mean(sigmoid(x)), x = [1 2]
    ;; dL/dx[i] = (1/2) * sigmoid(x[i]) * (1 - sigmoid(x[i]))
    (let* ((xs  '(1.0 2.0))
           (x-m (morph-from-list xs #(2) 'f64))
           (xv  (am:make-var x-m #t))
           (out (am:var-sigmoid xv))
           (loss (am:var-mean out))
           (res  (compile-and-realize loss (list xv)))
           (got  (cadr res))
           (exp  (map (lambda (x)
                        (let ((s (sigmoid x)))
                          (* 0.5 s (- 1.0 s))))
                      xs)))
      (lists-approx= got exp))))


(test-group "E1: var-tanh SSA gradient"
  (define (my-tanh x)
    (let ((e2 (exp (* 2.0 (exact->inexact x)))))
      (/ (- e2 1.0) (+ e2 1.0))))

  (test "tanh: gradient correct via SSA VJP"
    #t
    ;; loss = mean(tanh(x)), x = [0.5 -0.5]
    ;; dL/dx[i] = (1/2) * (1 - tanh(x[i])^2)
    (let* ((xs  '(0.5 -0.5))
           (x-m (morph-from-list xs #(2) 'f64))
           (xv  (am:make-var x-m #t))
           (out (am:var-tanh xv))
           (loss (am:var-mean out))
           (res  (compile-and-realize loss (list xv)))
           (got  (cadr res))
           (exp  (map (lambda (x)
                        (let ((t (my-tanh x)))
                          (* 0.5 (- 1.0 (* t t)))))
                      xs)))
      (lists-approx= got exp))))


(test-group "E4: cross-AD-boundary fusion with relu"
  (test "relu: gradient correct + joint program binding count lower than decomposed"
    #t
    ;; Compare binding count: var-relu vs old abs+add+mul decomposition
    (let* ((xs  '(-1.0 0.5 1.0 2.0))
           (x-m (morph-from-list xs #(4) 'f64))
           ;; New single-op relu path
           (xv1  (am:make-var x-m #t))
           (fwd1 (morphism-to-ssa (am:var-mean (am:var-relu xv1))))
           (p1   (filter-map (lambda (v)
                               (ssa-constant-id fwd1 (am:var-value v)))
                             (list xv1)))
           (jnt1 (ssa-vjp fwd1 p1 (ssa-loss-binding-val fwd1)))
           (n1   (length (ssa-program-bindings jnt1)))
           ;; Old 3-op decomposition: relu = 0.5*(x + |x|)
           (xv2    (am:make-var x-m #t))
           (abs-x  (am:var-abs xv2))
           (sum-x  (am:var+ xv2 abs-x))
           (half-m (morph-from-list '(0.5) #(1) 'f64))
           (half-v (am:make-var half-m #f))
           (relu2  (am:var* sum-x half-v))
           (fwd2   (morphism-to-ssa (am:var-mean relu2)))
           (p2     (filter-map (lambda (v)
                                 (ssa-constant-id fwd2 (am:var-value v)))
                               (list xv2)))
           (jnt2   (ssa-vjp fwd2 p2 (ssa-loss-binding-val fwd2)))
           (n2     (length (ssa-program-bindings jnt2))))
      (< n1 n2))))


;;;; ============================================================
;;;; Group E3: GEMM epilogue fusion
;;;; ============================================================

(test-group "E3: in-place epilogue kernels"

  (test "execute-flat-unary-compute-inplace! relu clamps negatives"
    '(0.0 0.0 1.0 2.0)
    (let* ((buf (f64vector -1.0 -0.5 1.0 2.0))
           (_   (execute-flat-unary-compute-inplace!
                 (lambda (x) (max 0.0 x)) buf 4 'f64)))
      (map (lambda (i) (f64vector-ref buf i)) (iota 4))))

  (test "execute-flat-unary-compute-inplace! negate"
    '(1.0 -2.0 3.0)
    (let* ((buf (f64vector -1.0 2.0 -3.0))
           (_   (execute-flat-unary-compute-inplace! - buf 3 'f64)))
      (map (lambda (i) (f64vector-ref buf i)) (iota 3))))

  (test "execute-flat-bias-broadcast-inplace! add-bias relu"
    '(0.0 2.0 0.0 3.0)
    ;; buf = [-1, 1, -2, 2], bias = [1, 1] (N=2)
    ;; combiner = max(0, x+b): -1+1=0->0, 1+1=2->2, -2+1=-1->0, 2+1=3->3
    (let* ((buf  (f64vector -1.0 1.0 -2.0 2.0))
           (bias (f64vector 1.0 1.0))
           (_    (execute-flat-bias-broadcast-inplace!
                  (lambda (x b) (max 0.0 (+ x b))) buf bias 4 2 'f64)))
      (map (lambda (i) (f64vector-ref buf i)) (iota 4)))))


(define (plan-has-epilogue? plan)
  (let loop ((i 0))
    (cond
      ((>= i (vector-length plan)) #f)
      (else
       (cases replay-instruction (vector-ref plan i)
         (ri-segop (_ _ _ _ _ _ tag _)
           (or (eq? tag 'segop/gemm-epilogue) (loop (+ i 1))))
         (else (loop (+ i 1))))))))

(test-group "E3: GEMM epilogue in replay plan"

  (test "Dense+relu: replay plan contains a GEMM epilogue segop after ctx replay"
    #t
    ;; x[2,3] @ W[3,4] + b[4], then relu, then mean
    (let* ((x-data (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(2 3) 'f64))
           (W-data (morph-from-list '(0.1 0.1 0.1 0.1 0.1 0.1
                                     0.1 0.1 0.1 0.1 0.1 0.1) #(3 4) 'f64))
           (b-data (morph-from-list '(0.1 0.1 0.1 0.1) #(4) 'f64))
           (xv (am:make-var x-data #f))
           (Wv (am:make-var W-data #t))
           (bv (am:make-var b-data #t))
           (pre  (am:var+ (am:var-matmul xv Wv) bv))
           (act  (am:var-relu pre))
           (loss (am:var-mean act))
           (ctx  (make-morphism-context)))
      ;; trace run (compile-and-realize/ctx returns (values joint results))
      (let-values (((jnt _trace) (compile-and-realize/ctx ctx loss (list Wv bv))))
        ;; finalize + replay (replay call compiles the plan)
        (finalize-context! ctx)
        (ssa-realize/ctx ctx jnt)
        (plan-has-epilogue? (ssa-program-replay-plan jnt)))))

  (test "Dense+relu: ssa-realize/ctx result matches direct realize"
    #t
    (let* ((x-data (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(2 3) 'f64))
           (W-data (morph-from-list '(0.1 0.1 0.1 0.1 0.1 0.1
                                     0.1 0.1 0.1 0.1 0.1 0.1) #(3 4) 'f64))
           (b-data (morph-from-list '(0.1 0.1 0.1 0.1) #(4) 'f64))
           ;; Reference: direct realize without SSA
           (pre-ref  (morph+ (morph-matmul x-data W-data) b-data))
           (act-ref  (morph-relu pre-ref))
           (ref-vals (concrete->list (realize act-ref)))
           ;; SSA replay path
           (xv (am:make-var x-data #f))
           (Wv (am:make-var W-data #t))
           (bv (am:make-var b-data #t))
           (pre  (am:var+ (am:var-matmul xv Wv) bv))
           (act  (am:var-relu pre))
           (loss (am:var-mean act))
           (fwd  (morphism-to-ssa loss))
           (pv   (filter-map (lambda (v) (ssa-constant-id fwd (am:var-value v)))
                             (list Wv bv)))
           (jnt  (ssa-vjp fwd pv (ssa-loss-binding-val fwd)))
           (ctx  (make-morphism-context))
           (res  (ssa-realize/ctx ctx jnt))
           (ssa-loss (car (concrete->list (car res))))
           (ref-loss (/ (apply + ref-vals) (length ref-vals))))
      (approx= ssa-loss ref-loss)))

  (test "Dense+relu: gradient w.r.t. W is correct"
    #t
    ;; dL/dW numerically via finite differences vs analytical SSA gradient
    (let* ((x-data  (morph-from-list '(1.0 0.0) #(1 2) 'f64))
           (W-data  (morph-from-list '(0.5 0.5 0.5 0.5) #(2 2) 'f64))
           (b-data  (morph-from-list '(0.0 0.0) #(2) 'f64))
           (xv (am:make-var x-data #f))
           (Wv (am:make-var W-data #t))
           (bv (am:make-var b-data #t))
           (pre  (am:var+ (am:var-matmul xv Wv) bv))
           (act  (am:var-relu pre))
           (loss (am:var-mean act))
           ;; SSA gradient
           (fwd  (morphism-to-ssa loss))
           (pv   (filter-map (lambda (v) (ssa-constant-id fwd (am:var-value v)))
                             (list Wv bv)))
           (jnt  (ssa-vjp fwd pv (ssa-loss-binding-val fwd)))
           (ctx  (make-morphism-context))
           (res  (ssa-realize/ctx ctx jnt))
           ;; Gradient for W is second output (index 1)
           (dW   (concrete->list (list-ref res 1)))
           ;; x = [1, 0], W = [[0.5,0.5],[0.5,0.5]], b = [0,0]
           ;; pre = [0.5, 0.5], relu -> [0.5, 0.5], loss = 0.5
           ;; d loss/d W[i,j] = (1/2) * x[i] * (if pre[j]>0 1 0)
           ;; dW = x^T * (heaviside(pre) / N) = [[0.5,0.5],[0,0]]
           (expected-dW '(0.5 0.5 0.0 0.0)))
      (lists-approx= dW expected-dW))))


(test-group "aliasing regression: GEMM epilogue output slot"

  ;; y = sigmoid(relu(x) @ W) as a forward-only program.  relu's output is
  ;; last read by the matmul, so the trace may give sigmoid's output the
  ;; same pool slot.  The replay fuses the matmul and sigmoid into one
  ;; instruction that writes into sigmoid's slot while still reading
  ;; relu's output, so the pair must stay unfused when the slots coincide.
  (test-assert "replayed forward program equals the reference"
    (let* ((x-mv (input-var '(0.5 -1.0 2.0 -0.3 1.2 0.7
                              -0.8 0.1 1.5 -2.0 0.9 0.4
                              1.1 -0.6 -0.2 0.3 2.2 -1.4) '(3 6)))
           (W-mv (param-var '(0.1 -0.2 0.3 0.4
                              -0.5 0.6 0.7 -0.8
                              0.9 1.0 -1.1 1.2
                              1.3 -1.4 1.5 1.6
                              -1.7 1.8 1.9 -2.0
                              2.1 -2.2 0.2 0.5) '(6 4)))
           (y-mv (am:var-sigmoid (am:var-matmul (am:var-relu x-mv) W-mv)))
           (prog (morphism-to-ssa y-mv))
           (reference (concrete->list (car (ssa-realize prog))))
           (ctx (make-morphism-context)))
      (ssa-realize/ctx ctx prog)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((first  (concrete->list (car (ssa-realize/ctx ctx prog)))))
        (reset-context! ctx)
        (let ((second (concrete->list (car (ssa-realize/ctx ctx prog)))))
          (and (lists-approx= first reference)
               (lists-approx= second reference)))))))


;;;; ============================================================
;;;; Broadcast replay: the flat-broadcast segop pattern
;;;;
;;;; Row broadcasts ([M,N] op [M,1]) and scalar broadcasts ([M,N] op [1])
;;;; compile to the flat-broadcast segop pattern.  Replay must reproduce the gradients of
;;;; the trace run, which evaluates every binding through realize.
;;;; ============================================================

(test-group "broadcast replay: flat-broadcast segops"
  (let* ((ctx  (make-morphism-context))
         (X-mv (param-var '(0.5 -1.0 2.0 0.25 1.5 -0.5 3.0 1.0 -2.0 0.75 0.1 -0.3) '(3 4)))
         (g-mv (param-var '(1.0 0.5 -1.5 2.0) '(4)))
         (eps  (input-var '(1e-5) '(1)))
         (half (input-var '(0.5) '(1)))
         ;; RMSNorm rows of X, scale by g, add 0.5 * X, then mean of squares
         (ms   (am:var-mean (am:var* X-mv X-mv) '(1) #t))            ; [3,1]
         (rms  (am:var-sqrt (am:var+ ms eps)))
         (Y    (am:var+ (am:var* (am:var/ X-mv rms) g-mv) (am:var* half X-mv)))
         (loss (am:var-mean (am:var* Y Y))))
    (let-values (((joint trace-results)
                  (compile-and-realize/ctx ctx loss (list X-mv g-mv))))
      (let ((ref (map concrete->list trace-results)))
        (finalize-context! ctx)
        (reset-context! ctx)
        (let ((first (map concrete->list (ssa-realize/ctx ctx joint))))
          (test-assert "the replay plan uses the flat-broadcast pattern"
            (let ((counts (cdr (assq 'counts (replay-plan-stats joint)))))
              (> (or (let ((e (assq 'segop/flat-broadcast counts))) (and e (cdr e))) 0) 0)))
          (test-assert "replay reproduces the trace loss and gradients"
            (every lists-approx= first ref))
          (test-assert "5 further replay steps are identical"
            (let loop ((i 0))
              (or (= i 5)
                  (begin
                    (reset-context! ctx)
                    (and (every lists-approx= (map concrete->list (ssa-realize/ctx ctx joint)) ref)
                         (loop (+ i 1))))))))))))

;;;; ============================================================
;;;; Segop lowering of element-wise bindings
;;;;
;;;; Element-wise bindings compile to ri-segop instructions.  A checking
;;;; backend, registered ahead of the built-in ones, wraps every executor
;;;; they select: after the executor runs, the segop is evaluated again by
;;;; the reference interpreter on the same input arrays, and the two
;;;; outputs must agree bit for bit.  This checks the lowered index maps,
;;;; layouts and bodies against what replay actually computes.
;;;; ============================================================

(define (seg-var lst shape dtype grad?)
  (am:make-var (morph-from-list lst (list->vector shape) dtype) grad?))

(define (seg-same? a b)
  (if (and (flonum? a) (flonum? b))
      (or (and (nan? a) (nan? b)) (eqv? a b))
      (eqv? a b)))

;; Runs thunk with the checking backend in front of the backends in use,
;; through a parameterize of segop-backends.  The checker asks the other
;; backends for the executor it wraps.  Returns
;; (values result checks mismatches), the numbers of segop executions
;; checked and of output elements that differed.
(define (with-segop-checker thunk)
  (let ((checks 0) (mismatches 0) (others (segop-backends)))
    (define (array-data a)
      (cases array-morphism a
        (concrete-array (data _ _ _ _ _ _) data)
        (else (error "with-segop-checker: not concrete" a))))
    ;; A seg-red whose combine order the backend chooses (as the BLAS dot
    ;; product of a rank-1 sum, or a GEMM kernel) is compared up to
    ;; rounding: the relative tolerance is 8 n eps, with n the number of
    ;; reduced values and eps the machine epsilon of the accumulator dtype.
    ;; Returns that tolerance, or #f for an exact comparison.
    (define (unspecified-order-tolerance sg)
      (cases segop sg
        (seg-red (space nred ins results map-outs fallback)
          (let ((specs (filter (lambda (spec) (eq? 'unspecified (combine-spec-order spec)))
                               (map red-result-combine results))))
            (and (pair? specs)
                 (let ((n (fold * 1 (drop (vector->list space) (- (vector-length space) nred))))
                       (eps (if (any (lambda (spec) (eq? (combine-spec-acc-dtype spec) 'f32)) specs)
                                1.2e-7
                                2.3e-16)))
                   (* 8 n eps)))))
        (else #f)))
    (define (agree? a b tol)
      (or (seg-same? a b)
          (and tol (<= (abs (- a b)) (* tol (max 1.0 (abs a)))))))
    (define (select sg)
      (let ((sel (parameterize ((segop-backends others))
                   (select-segop-executor sg))))
             (and sel
                  (cons (car sel)
                        (if (eq? (segop-kind sg) 'seg-lib)
                            (cdr sel)
                            ;; extra-bufs holds the buffers of the outputs
                            ;; after the first, for a segop with several
                            (lambda (deref buf #!optional (extra-bufs '()))
                              (if (null? extra-bufs)
                                  ((cdr sel) deref buf)
                                  ((cdr sel) deref buf extra-bufs))
                              (let* ((outs (segop-outputs sg))
                                     (bufs (cons buf extra-bufs))
                                     (refs (map (lambda (o)
                                                  (allocate-typed-vector
                                                   (segop-operand-dtype o)
                                                   (shape-size (segop-operand-shape o))))
                                                outs))
                                     (loose? (unspecified-order-tolerance sg)))
                                (segop-interpret sg
                                                 (map (lambda (o) (array-data (deref (segop-operand-slot o))))
                                                      (segop-inputs sg))
                                                 refs)
                                (set! checks (+ checks 1))
                                (for-each
                                 (lambda (o ref buf)
                                   (let ((dt (segop-operand-dtype o)))
                                     (do ((i 0 (+ i 1))) ((= i (shape-size (segop-operand-shape o))))
                                       (unless (agree? (typed-vector-ref ref dt i)
                                                       (typed-vector-ref buf dt i) loose?)
                                         (set! mismatches (+ mismatches 1))))))
                                 outs refs bufs))))))))
    (parameterize ((segop-backends (cons (make-segop-backend 'test-checker select) others)))
      (let ((r (thunk))) (values r checks mismatches)))))

;; Placeholder ssa-values for bindings built by hand.
(define (const-ref* cid) (const-ref cid))
(define (binding-ref* bid) (binding-ref bid))

;; An element-wise binding with n inputs (placeholders) and the given meta.
(define (make-ssa-binding* op n meta)
  (make-ssa-binding (gensym 'bid-) op (map (lambda (i) #f) (iota n)) #(2) 'f64 meta))

;; The ri-segop instructions of a compiled plan.
(define (plan-segops joint)
  (let ((plan (ssa-program-replay-plan joint)))
    (filter-map (lambda (instr)
                  (cases replay-instruction instr
                    (ri-segop (_ _ _ _ sg _ tag _) (cons tag sg))
                    (else #f)))
                (vector->list plan))))

;; A two-layer network: h = relu(x W1 + b1); y = sigmoid(tanh(h W2));
;; loss = mean((y - t)^2) + mean(h * c) + mean(exp(-y)) with a per-column
;; constant c.  negate has a single use, so it is fused into exp.
(define (seg-mlp-loss dtype)
  (let* ((x  (seg-var (map (lambda (i) (- (* 0.37 i) 2.0)) (iota 12)) '(4 3) dtype #f))
         (W1 (seg-var (map (lambda (i) (* 0.21 (sin (+ i 1.0)))) (iota 15)) '(3 5) dtype #t))
         (b1 (seg-var (map (lambda (i) (- (* 0.1 i) 0.2)) (iota 5)) '(5) dtype #t))
         (W2 (seg-var (map (lambda (i) (* 0.3 (cos (+ i 0.5)))) (iota 10)) '(5 2) dtype #t))
         (t  (seg-var '(0.1 0.9 0.4 0.6 0.8 0.2 0.5 0.5) '(4 2) dtype #f))
         (c  (seg-var '(1.0 -0.5 0.25 2.0 -1.5) '(5) dtype #f))
         (h  (am:var-relu (am:var+ (am:var-matmul x W1) b1)))
         (y  (am:var-sigmoid (am:var-tanh (am:var-matmul h W2))))
         (d  (am:var- y t)))
    (values (am:var+ (am:var+ (am:var-mean (am:var* d d)) (am:var-mean (am:var* h c)))
                     (am:var-mean (am:var-exp (am:var-negate y))))
            (list W1 b1 W2))))

;; Traces the loss, then replays it twice.  Returns (values joint trace
;; replay) with the outputs of the trace and of the second replay as lists.
(define (seg-trace-and-replay make-loss)
  (let-values (((loss params) (make-loss)))
    (let ((ctx (make-morphism-context)))
      (let-values (((joint trace) (compile-and-realize/ctx ctx loss params)))
        (let ((trace-lists (map concrete->list trace)))
          (finalize-context! ctx)
          (reset-context! ctx)
          (ssa-realize/ctx ctx joint)
          (reset-context! ctx)
          (values joint trace-lists (map concrete->list (ssa-realize/ctx ctx joint))))))))

;; The bodies of segop sg: those of a seg-map's outputs, or a seg-red's
;; result and map-output bodies.
(define (segop-bodies sg)
  (cases segop sg
    (seg-map (space ins outs bodies fallback) bodies)
    (seg-red (space nred ins results map-outs fallback)
      (append (map red-result-body results) (map cdr map-outs)))
    (else '())))

;; True when body b applies op somewhere.
(define (body-has-op? b op)
  (and (pair? b) (not (eq? (car b) 'opaque))
       (or (eq? (car b) op) (any (lambda (e) (body-has-op? e op)) (cdr b)))))

;; True when body b contains a subexpression of the form pattern, in which
;; (in _) matches any input.
(define (body-has-form? b pattern)
  (define (matches? e p)
    (cond ((equal? p '(in _)) (and (pair? e) (eq? (car e) 'in)))
          ((and (pair? p) (pair? e)) (and (= (length p) (length e)) (every matches? e p)))
          (else (equal? e p))))
  (let walk ((e b))
    (or (matches? e pattern)
        (and (pair? e) (not (eq? (car e) 'opaque)) (any walk (cdr e))))))

(test-group "segop lowering of element-wise bindings"
  (for-each
   (lambda (dtype)
     (let-values (((r checks mismatches)
                   (with-segop-checker
                    (lambda ()
                      ;; the policy all, so that negate is fused into exp
                      (let-values (((joint trace replay)
                                    (parameterize ((segop-fusion 'all))
                                      (seg-trace-and-replay (lambda () (seg-mlp-loss dtype))))))
                        (list joint trace replay))))))
       (let* ((joint (car r)) (trace (cadr r)) (replay (caddr r))
              (segs (plan-segops joint))
              (tags (map car segs)))
         (test-assert (sprintf "~A: element-wise bindings compile to ri-segop" dtype)
           (> (length segs) 5))
         (test-assert (sprintf "~A: every tag is a known pattern" dtype)
           (every (lambda (t)
                    (memq t '(segop/activation-unary segop/flat-unary segop/activation-binary
                              segop/flat-binary segop/bias-broadcast segop/flat-ternary
                              segop/flat-broadcast segop/reduce-1d segop/reduce-axis0
                              segop/gemm segop/gemm-strided segop/gemm-epilogue
                              segop/reduce-axis1 segop/reduce-general segop/gather
                              segop/strided-map segop/strided-red segop/strided-multi-map)))
                  tags))
         (test-assert (sprintf "~A: relu keeps its symbolic body" dtype)
           (any (lambda (ts) (any (lambda (b) (body-has-op? b 'relu)) (segop-bodies (cdr ts))))
                segs))
         ;; an f32 producer's value is rounded to f32 inside the fused body
         (test-assert (sprintf "~A: the fused negate and exp have a composed body" dtype)
           (any (lambda (ts)
                  (any (lambda (b)
                         (body-has-form? b (if (eq? dtype 'f32)
                                               '(exp (cast f32 (negate (in _))))
                                               '(exp (negate (in _))))))
                       (segop-bodies (cdr ts))))
                segs))
         (test-assert (sprintf "~A: the checker ran on every segop execution" dtype)
           (>= checks (* 2 (length segs))))
         (test (sprintf "~A: executors and the interpreter agree bit for bit" dtype)
           0 mismatches)
         (test-assert (sprintf "~A: replay reproduces the trace" dtype)
           (every (lambda (a b)
                    (every (lambda (x y) (< (abs (- x y)) (if (eq? dtype 'f32) 1e-6 1e-12))) a b))
                  trace replay)))))
   '(f64 f32))

  (let-values (((r checks mismatches)
                (with-segop-checker
                 (lambda ()
                   (let-values (((joint trace replay)
                                 (seg-trace-and-replay
                                  (lambda ()
                                    (let* ((X    (param-var '(0.5 -1.0 2.0 0.25 1.5 -0.5 3.0 1.0 -2.0 0.75 0.1 -0.3) '(3 4)))
                                           (g    (param-var '(1.0 0.5 -1.5 2.0) '(4)))
                                           (eps  (input-var '(1e-5) '(1)))
                                           (ms   (am:var-mean (am:var* X X) '(1) #t))
                                           (rms  (am:var-sqrt (am:var+ ms eps)))
                                           (Y    (am:var* (am:var/ X rms) g)))
                                      (values (am:var-mean (am:var* Y Y)) (list X g)))))))
                     (list joint trace replay))))))
    (let ((tags (map car (plan-segops (car r)))))
      (test-assert "row normalization uses the broadcast patterns"
        (and (memq 'segop/flat-broadcast tags) (memq 'segop/bias-broadcast tags) #t))
      (test "broadcast executors and the interpreter agree bit for bit" 0 mismatches)
      (test-assert "broadcast segops were checked" (> checks 0))))

  (test "the body of a plain binding names its op" '(add (in 0) (in 1))
    (binding-scalar-body (make-ssa-binding* 'add 2 '()) (scalar-op-combiner 'add)))
  (test-assert "a binding whose combiner is not its op's gets an opaque body"
    (let ((body (binding-scalar-body (make-ssa-binding* 'add 2 '()) -)))
      (and (eq? (car body) 'opaque) (eq? (cadr body) -))))
  (test-assert "a map binding gets an opaque body"
    (eq? 'opaque (car (binding-scalar-body (make-ssa-binding* 'map 1 '()) abs))))
  (test "a binding with a body in its meta keeps that body" '(mul (exp (in 0)) (in 1))
    (binding-scalar-body (make-ssa-binding* 'mul 2 '((body . (mul (exp (in 0)) (in 1)))))
                         *)))


;;;; ============================================================
;;;; Fusion legality and gradient dtypes
;;;;
;;;; Fused bindings must give the values of the unfused program when
;;;; inputs broadcast, when producer and consumer differ in dtype, and when
;;;; a fused body reads three or more inputs.  Gradients of parameters have
;;;; the parameter's dtype, through a cast when the backward pass computes
;;;; them in another dtype.
;;;; ============================================================

;; Loss and gradients of make-loss computed by the eager autodiff of
;; array-morphisms-grad, as lists of flonums.
(define (eager-results make-loss)
  (let-values (((loss params) (make-loss)))
    (am:backward! loss)
    (cons (concrete->list (am:var-value loss))
          (map (lambda (p) (concrete->list (am:var-grad p))) params))))

;; Loss and gradients from SSA: (values joint trace replay) as lists.
(define (ssa-results make-loss)
  (seg-trace-and-replay make-loss))

(define (results-close? a b tol)
  (and (= (length a) (length b))
       (every (lambda (x y)
                (and (= (length x) (length y))
                     (every (lambda (u v) (<= (abs (- u v)) (* tol (max 1.0 (abs v))))) x y)))
              a b)))

;; True when b is a GEMM fused with its epilogue.
(define (gemm-epilogue-binding? b)
  (and (fused-red-binding? b) (eq? 'gemm (alist-ref 'path (ssa-binding-meta b)))))

(define (fused-bindings joint)
  (filter (lambda (b) (eq? (ssa-binding-op b) 'fused))
          (ssa-program-bindings joint)))

;; True when every input of every fused binding of joint has the binding's
;; shape: fused bindings read their inputs at the iteration point.
(define (fused-inputs-have-shape? joint)
  (let ((shape-of (lambda (v)
                    (let ((c (hash-table-ref/default (ssa-program-constants joint) (ssa-value-id v) #f)))
                      (if c
                          (morph-shape c)
                          (ssa-binding-shape (car (filter (lambda (b) (eq? (ssa-binding-name b)
                                                                           (ssa-value-id v)))
                                                          (ssa-program-bindings joint)))))))))
    (every (lambda (b)
             (every (lambda (v) (equal? (shape-of v) (ssa-binding-shape b))) (ssa-binding-inputs b)))
           (fused-bindings joint))))

(define (output-dtypes results)
  (map (lambda (m) (cases array-morphism m
                     (concrete-array (_ _ _ _ dt _ _) dt)
                     (else #f)))
       results))

(test-group "fusion legality and gradient dtypes"
  ;; u = x*y + b with a bias b of shape [3]: x*y is fused into the add,
  ;; which reads b through its broadcast map.
  (let* ((make (lambda ()
                 (let* ((x (param-var '(1 2 3 4 5 6) '(2 3)))
                        (y (param-var '(1 1 1 2 2 2) '(2 3)))
                        (b (param-var '(100 200 300) '(3)))
                        (u (am:var+ (am:var* x y) b)))
                   (values (am:var-sum (am:var* u u)) (list x y b))))))
    (let-values (((joint trace replay) (parameterize ((segop-fusion 'all)) (ssa-results make))))
      (test-assert "a product is fused into an add with a broadcast input"
        (and (pair? (fused-bindings joint)) (not (fused-inputs-have-shape? joint))))
      (test-assert "loss and gradients with a broadcast bias (trace)"
        (equal? '((295922.0) (202.0 404.0 606.0 432.0 840.0 1248.0)
                  (202.0 808.0 1818.0 864.0 2100.0 3744.0) (418.0 824.0 1230.0))
                trace))
      (test-assert "replay gives the same values" (equal? trace replay))))

  ;; Under the policy all, tanh(x*y + z) becomes one fused binding of
  ;; three inputs, which runs on the flat three-input executor.
  (let* ((make (lambda ()
                 (let* ((x (param-var '(-0.7 -0.4 -0.1 0.2 0.5 0.8) '(2 3)))
                        (y (param-var '(0.2 0.3 0.4 0.5 0.6 0.7) '(2 3)))
                        (z (param-var '(1.0 0.75 0.5 0.25 0.0 -0.25) '(2 3)))
                        (v (am:var-tanh (am:var+ (am:var* x y) z))))
                   (values (am:var-sum (am:var* v v)) (list x y z))))))
    (let-values (((joint trace replay) (parameterize ((segop-fusion 'all)) (ssa-results make))))
      (test-assert "the chain compiles, with a three-input fused binding"
        (any (lambda (b) (= 3 (length (ssa-binding-inputs b)))) (fused-bindings joint)))
      (test-assert "no fused binding reads an input twice"
        (every (lambda (b) (= (length (ssa-binding-inputs b))
                              (length (delete-duplicates (map ssa-value-id (ssa-binding-inputs b))))))
               (fused-bindings joint)))
      (test-assert "trace agrees with the eager autodiff"
        (results-close? trace (eager-results make) 1e-12))
      (test-assert "replay gives the same values as the trace" (equal? trace replay))))

  ;; loss = mean(sigmoid(x) * (x + y)) with x f32 and y f64.
  (let* ((xs '(-0.7 -0.4 -0.1 0.2 0.5 0.8))
         (ys '(0.2 0.3 0.4 0.5 0.6 0.7))
         (make (lambda ()
                 (let ((x (seg-var xs '(2 3) 'f32 #t))
                       (y (seg-var ys '(2 3) 'f64 #t)))
                   (values (am:var-mean (am:var* (am:var-sigmoid x) (am:var+ x y))) (list x y))))))
    (let-values (((loss params) (make)))
      (let* ((ctx (make-morphism-context)))
        (let-values (((joint trace) (compile-and-realize/ctx ctx loss params)))
          (test "mixed f32/f64: the trace gives each gradient its parameter's dtype"
            '(f64 f32 f64) (output-dtypes trace))
          (finalize-context! ctx)
          (reset-context! ctx)
          (ssa-realize/ctx ctx joint)
          (reset-context! ctx)
          (let ((replay (ssa-realize/ctx ctx joint)))
            (test "mixed f32/f64: replay gives the same dtypes" '(f64 f32 f64) (output-dtypes replay))
            (test "mixed f32/f64: replay reproduces the trace"
              (map concrete->list trace) (map concrete->list replay))
            (let* ((x32 (map (lambda (v) (exact->inexact v))
                             (concrete->list (am:var-value (car params)))))
                   ;; sigmoid(x) is an f32 binding: its values are rounded to f32
                   (s   (map (lambda (x)
                               (let ((v (make-f32vector 1)))
                                 (f32vector-set! v 0 (/ 1.0 (+ 1.0 (exp (- x)))))
                                 (f32vector-ref v 0)))
                             x32)))
              (test-assert "mixed f32/f64: gradient of the f32 parameter"
                (every (lambda (g si xi yi)
                         (< (abs (- g (/ (+ (* si (- 1.0 si) (+ xi yi)) si) 6.0))) 1e-6))
                       (concrete->list (cadr replay)) s x32 ys))
              (test-assert "mixed f32/f64: gradient of the f64 parameter"
                (every (lambda (g si) (< (abs (- g (/ si 6.0))) 1e-12))
                       (concrete->list (caddr replay)) s))))))))

  (test "a cast to f32 rounds" (list (exact->inexact 0.10000000149011612))
    (concrete->list (realize (morph-cast (morph-from-list '(0.1) #(1) 'f64) 'f32))))
  (test "a cast to s32 truncates" '(2.0 -2.0)
    (concrete->list (realize (morph-cast (morph-from-list '(2.7 -2.7) #(2) 'f64) 's32))))
  (test "a cast keeps its dtype metadata" 'f32
    (get-morphism-dtype (morph-cast (morph-from-list '(1.0) #(1) 'f64) 'f32)))
  (test "the body of a cast binding is (in 0)" '(in 0)
    (binding-scalar-body (make-ssa-binding 'bid-x 'cast '(#f) #(2) 'f32 '((dtype . f32)))
                         (cast-combiner 'f32)))

  ;; A cast in the forward graph: loss = mean(cast(x, f64) * y) with x f32.
  (let* ((make (lambda ()
                 (let* ((x (seg-var '(0.5 -1.5 2.25) '(3) 'f32 #t))
                        (y (seg-var '(2.0 4.0 -1.0) '(3) 'f64 #t))
                        (xc (am:make-var (morph-cast (am:var-value x) 'f64) #t)))
                   (values (am:var-mean (am:var* xc y)) (list x y))))))
    (let-values (((loss params) (make)))
      (let* ((xv  (am:var-value (car params)))
             (yv  (am:var-value (cadr params)))
             (fwd (morphism-to-ssa loss))
             (p   (filter-map (lambda (m) (ssa-constant-id fwd m)) (list xv yv)))
             (joint (ssa-vjp fwd p (ssa-loss-binding-val fwd)))
             (res   (ssa-realize joint)))
        (test "a forward cast: the gradient of x is f32, that of y f64"
          '(f64 f32 f64) (output-dtypes res))
        (test-assert "a forward cast: gradient of x (f32)"
          (results-close? (list (concrete->list (cadr res)))
                          (list (list (/ 2.0 3.0) (/ 4.0 3.0) (/ -1.0 3.0))) 1e-7))
        (test-assert "a forward cast: gradient of y (f64)"
          (results-close? (list (concrete->list (caddr res)))
                          (list (list (/ 0.5 3.0) (/ -1.5 3.0) (/ 2.25 3.0))) 1e-15))))))


;;;; ============================================================
;;;; Reductions, gathers and strided element-wise ops as segops
;;;;
;;;; Forward programs are traced and replayed with the checking backend in
;;;; place, so every reduction, gather and strided map is also evaluated by
;;;; the reference interpreter.  Replay must reproduce the trace bit for
;;;; bit.  The executors for host index functions, integer outputs and the
;;;; interpreter fallback are also called directly on hand-built segops.
;;;; ============================================================

;; Element values of a concrete array, exactly as stored, in row-major order.
(define (s2-values m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype _ _)
      (map (lambda (i) (typed-vector-ref data dtype
                                         (multi-to-linear-index (linear-to-multi-index i shape)
                                                                strides offset)))
           (iota (shape-size shape))))
    (else (error "s2-values: not concrete"))))

(define (s2-same-lists? a b)
  (and (= (length a) (length b)) (every seg-same? a b)))

;; Traces the forward program of morphism m and replays it twice with the
;; checking backend.  Returns (list prog trace replay checks mismatches),
;; trace and replay as value lists of the output.
(define (s2-run make-m)
  (let-values (((r checks mismatches)
                (with-segop-checker
                 (lambda ()
                   (let* ((ctx  (make-morphism-context))
                          (prog (morphism-to-ssa (am:make-var (make-m) #f)))
                          (tr   (s2-values (car (ssa-realize/ctx ctx prog)))))
                     (finalize-context! ctx)
                     (reset-context! ctx)
                     (ssa-realize/ctx ctx prog)
                     (reset-context! ctx)
                     (list prog tr (s2-values (car (ssa-realize/ctx ctx prog)))))))))
    (append r (list checks mismatches))))

(define (s2-tags prog)
  (map car (plan-segops prog)))

;; A computing node, so that the reduced array is a pool buffer.
(define (s2-src dt shape f)
  (morph* (morph-from-list (map f (iota (apply * shape))) (list->vector shape) dt)
          (morph-from-list '(1.0) #(1) dt)))

(define (s2-val i) (case (modulo i 7) ((0) -0.0) ((3) 1e8) (else (- (* 0.37 i) 2.0))))

(test-group "reductions as seg-red"
  (for-each
   (lambda (dt)
     (for-each
      (lambda (op)
        (for-each
         (lambda (spec)
           (let* ((axes (car spec)) (kd (cadr spec)) (expect (caddr spec))
                  (r (s2-run (lambda () (morph-reduce op (s2-src dt '(3 4) s2-val) axes kd))))
                  (name (sprintf "~A ~A over ~A~A" dt op axes (if kd " keepdims" ""))))
             (test-assert (string-append name ": replay equals the trace")
               (s2-same-lists? (cadr r) (caddr r)))
             (test (string-append name ": executors agree with the interpreter") 0 (list-ref r 4))
             (test-assert (string-append name ": " (symbol->string (if (eq? op 'prod) 'segop/reduce-general expect)))
               (memq (if (eq? op 'prod) 'segop/reduce-general expect) (s2-tags (car r))))))
         '(((0) #f segop/reduce-axis0) ((1) #t segop/reduce-axis1)
           ((0 1) #f segop/reduce-general) ((1) #f segop/reduce-axis1))))
      '(sum mean max min prod)))
   '(f64 f32))

  (for-each
   (lambda (case-spec)
     (let* ((name (car case-spec)) (tag (cadr case-spec))
            (r (s2-run (caddr case-spec))))
       (test-assert (string-append name ": replay equals the trace") (s2-same-lists? (cadr r) (caddr r)))
       (test (string-append name ": executors agree with the interpreter") 0 (list-ref r 4))
       (test-assert (string-append name ": uses " (symbol->string tag)) (memq tag (s2-tags (car r))))))
   (list
    (list "rank-1 f64 sum" 'segop/reduce-1d
          (lambda () (morph-reduce 'sum (s2-src 'f64 '(7) s2-val) '(0))))
    (list "rank-1 f32 max" 'segop/reduce-1d
          (lambda () (morph-reduce 'max (s2-src 'f32 '(7) s2-val) '(0))))
    (list "sum of a transposed source" 'segop/reduce-general
          (lambda () (morph-reduce 'sum (morph-transpose (s2-src 'f64 '(3 4) s2-val) '(1 0)) '(0))))
    (list "mean over axes 0 and 2 of a rank-3 array, keepdims" 'segop/reduce-general
          (lambda () (morph-reduce 'mean (s2-src 'f32 '(2 3 4) s2-val) '(0 2) #t)))
    (list "max with NaN over axis 0" 'segop/reduce-axis0
          (lambda () (morph-reduce 'max (s2-src 'f64 '(2 3) (lambda (i) (list-ref '(1.0 +nan.0 3.0 -0.0 0.0 -1.0) i))) '(0))))
    (list "max with NaN over both axes" 'segop/reduce-general
          (lambda () (morph-reduce 'max (s2-src 'f64 '(2 3) (lambda (i) (list-ref '(1.0 +nan.0 3.0 -0.0 0.0 -1.0) i))))))))

  (let ((r (s2-run (lambda () (morph-reduce 'sum (morph+ (morph-from-list '(3 -1 4 1 -5 9) #(2 3) 's32)
                                                        (morph-from-list '(0) #(1) 's32))
                                            '(0))))))
    (test "an s32 sum over axis 0 replays correctly" '(4 -6 13) (caddr r))
    (test-assert "the s32 sum takes the general path" (memq 'segop/reduce-general (s2-tags (car r)))))
  (let ((r (s2-run (lambda () (morph-reduce 'max (morph+ (morph-from-list (list (expt 2 62) (+ (expt 2 62) 1) 7) #(3) 's64)
                                                        (morph-from-list '(0) #(1) 's64)))))))
    (test "an s64 max near 2^62 replays exactly" (list (+ (expt 2 62) 1)) (caddr r)))

  (let* ((r (s2-run (lambda () (morph-reduce 'mean (s2-src 'f64 '(2 3 4) s2-val) '(2 0) #t))))
         (prog (car r))
         (reds (filter (lambda (ts) (eq? 'seg-red (segop-kind (cdr ts)))) (plan-segops prog))))
    (test "standard-reduction recovers the reduction" '((mean (0 2) #t))
      (map (lambda (ts) (let ((p (standard-reduction (cdr ts)))) (list (car p) (cadr p) (caddr p))))
           reds))
    (test "replay-plan-stats lists the reduction" '(((0 2) #t #(1 3 1)))
      (map cdr (cdr (assq 'ri-reduce-specs (replay-plan-stats prog)))))))

(test-group "gathers and strided element-wise ops"
  (let ((r (s2-run (lambda () (morph-reshape (morph-transpose (s2-src 'f64 '(3 4) s2-val) '(1 0)) #(2 6))))))
    (test-assert "a reshape of a transposed view is a gather" (memq 'segop/gather (s2-tags (car r))))
    (test-assert "the gather replays the trace" (s2-same-lists? (cadr r) (caddr r)))
    (test "the gather agrees with the interpreter" 0 (list-ref r 4))
    (test "replay-plan-stats lists the gather" '(#(2 6))
      (map cdr (cdr (assq 'ri-index-shapes (replay-plan-stats (car r)))))))

  ;; With a copy kernel registered, gathers call it and give the same values.
  (let* ((calls 0)
         (kernel (lambda (src offset shape strides dst)
                   (set! calls (+ calls 1))
                   (do ((i 0 (+ i 1))) ((= i (shape-size shape)))
                     (f32vector-set! dst i
                       (f32vector-ref src (multi-to-linear-index (linear-to-multi-index i shape)
                                                                 strides offset))))))
         (bkend (make-activation-backend 'test-copy))
         (saved (active-activation-backend))
         (make (lambda () (morph-reshape (morph-transpose (s2-src 'f32 '(2 3 4) s2-val) '(2 0 1)) #(24)))))
    (activation-backend-add-copy-kernel! bkend 'f32 kernel)
    (let ((without (s2-run make)))
      (register-activation-backend! bkend)
      (let ((with (s2-run make)))
        (register-activation-backend! saved)
        (test-assert "the copy kernel is called by the gather" (> calls 0))
        (test-assert "the copy kernel gives the same values"
          (s2-same-lists? (caddr without) (caddr with))))))

  (for-each
   (lambda (case-spec)
     (let ((r (s2-run (cadr case-spec))))
       (test-assert (string-append (car case-spec) ": a strided map")
         (memq 'segop/strided-map (s2-tags (car r))))
       (test-assert (string-append (car case-spec) ": replay equals the trace")
         (s2-same-lists? (cadr r) (caddr r)))
       (test (string-append (car case-spec) ": agrees with the interpreter") 0 (list-ref r 4))))
   (list
    (list "exp of a transposed view"
          (lambda () (morph-exp (morph-transpose (s2-src 'f64 '(3 4) s2-val) '(1 0)))))
    (list "product of a transposed and a row-major array"
          (lambda () (morph* (morph-transpose (s2-src 'f32 '(3 4) s2-val) '(1 0))
                             (morph-from-list (map (lambda (i) (- i 5.0)) (iota 12)) #(4 3) 'f32))))
    (list "s32 sum of a transposed view and a row-major array"
          (lambda () (morph+ (morph-transpose (morph+ (morph-from-list (iota 6) #(2 3) 's32)
                                                      (morph-from-list '(1) #(1) 's32))
                                              '(1 0))
                             (morph-from-list '(10 20 30 40 50 60) #(3 2) 's32)))))))

(test-group "segop executors called directly"
  (let* ((deref-of (lambda (alist) (lambda (slot) (cdr (assq slot alist)))))
         (a (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(2 3) 'f64))
         (b (morph-from-list '(10.0 20.0 30.0 40.0 50.0 60.0) #(2 3) 'f64))
         (sel (lambda (sg) (select-segop-executor sg))))
    ;; host index function: a stack of two arrays
    (let* ((sl (make-seg-lib 'host-index-fn
                             (list (make-segop-operand 'a 'f64 #(2 3) #(3 1) 0 (opaque-index-map values))
                                   (make-segop-operand 'b 'f64 #(2 3) #(3 1) 0 (opaque-index-map values)))
                             (list (make-segop-operand 'out 'f64 #(2 2 3) #(6 3 1) 0 (opaque-index-map values)))
                             `((index-fn . ,(make-stack-index-fn 0)))))
           (ex  (sel sl))
           (buf (make-f64vector 12 0.0)))
      (test "a host index function is selected" 'segop/host-index-fn (car ex))
      ((cdr ex) (deref-of `((a . ,a) (b . ,b))) buf)
      (test "the host executor stacks the inputs"
        (s2-values (realize (morph-stack (list a b) 0)))
        (f64vector->list buf)))
    ;; strided map with an s64 output over a rank-0 space
    (let* ((x  (morph-from-list '(7) #(1) 's64))
           (sm (make-seg-map #() (list (make-segop-operand 'x 's64 #(1) #(1) 0 (broadcast-index-map #(1) 0)))
                             (list (make-segop-operand 'out 's64 #() #() 0 (identity-index-map 0)))
                             '((negate (in 0))) -))
           (ex (sel sm))
           (buf (allocate-typed-vector 's64 1)))
      (test "a rank-0 s64 map is a strided map" 'segop/strided-map (car ex))
      ((cdr ex) (deref-of `((x . ,x))) buf)
      (test "the rank-0 s64 result" -7 (typed-vector-ref buf 's64 0)))
    ;; an opaque input map falls back to the interpreter
    (let* ((sm (make-seg-map #(3) (list (make-segop-operand 'a 'f64 #(2 3) #(3 1) 0
                                                            (opaque-index-map (lambda (c) (vector 1 (vector-ref c 0))))))
                             (list (make-segop-operand 'out 'f64 #(3) #(1) 0 (identity-index-map 1)))
                             '((in 0))))
           (ex (sel sm))
           (buf (make-f64vector 3 0.0)))
      (test "an opaque map is interpreted" 'segop/interpreted (car ex))
      ((cdr ex) (deref-of `((a . ,a))) buf)
      (test "the interpreted gather reads the second row" '(4.0 5.0 6.0) (f64vector->list buf)))))

;;;; ============================================================
;;;; The strided seg-red executor
;;;;
;;;; Seg-reds that are not standard reductions (two inputs, a body other
;;;; than (in 0), map outputs, finalize bodies with inputs) run on the
;;;; strided reduction executor, which must write exactly what
;;;; segop-interpret writes.
;;;; ============================================================

(define (e4-map . es) (make-index-map (list->vector es) #f 0.0 #f))

;; A concrete array over data with the given shape and strides.
(define (e4-arr data shape strides dtype) (concrete-array data shape strides 0 dtype -1 -1))

(define (e4-data arr)
  (cases array-morphism arr
    (concrete-array (data _ _ _ _ _ _) data)
    (else (error "e4-data: not concrete" arr))))

;; An operand at slot reading array arr (of the given dtype) through map m.
(define (e4-op slot arr dtype m)
  (cases array-morphism arr
    (concrete-array (_ s st off _ _ _) (make-segop-operand slot dtype s st off m))
    (else (error "e4-op: not concrete" arr))))

;; A row-major output operand of the given shape, written through m.
(define (e4-out dtype shape m) (make-segop-operand #f dtype shape (compute-strides shape) 0 m))

;; Runs seg-red sg on the arrays of arrs (an alist from slot to array)
;; with its selected executor and with segop-interpret, each into fresh
;; output buffers of the given (dtype size) specs.  Returns the tag and
;; whether every output element is the same in both, NaNs included.
(define (e4-run sg arrs outs)
  (let* ((ex    (select-segop-executor sg))
         (mk    (lambda () (map (lambda (o) (allocate-typed-vector (car o) (cadr o))) outs)))
         (b1    (mk))
         (b2    (mk))
         (deref (lambda (s) (cdr (assq s arrs)))))
    ((cdr ex) deref (car b1) (cdr b1))
    (segop-interpret sg (map (lambda (o) (e4-data (deref (segop-operand-slot o)))) (segop-inputs sg)) b2)
    (list (car ex)
          (every (lambda (o x y)
                   (every (lambda (i)
                            (let ((u (typed-vector-ref x (car o) i)) (v (typed-vector-ref y (car o) i)))
                              (if (and (flonum? u) (flonum? v) (nan? u) (nan? v)) #t (eqv? u v))))
                          (iota (cadr o))))
                 outs b1 b2)
          b1)))

(define e4-specials (list +nan.0 +inf.0 -inf.0 -0.0 0.0 1e8 1.0 -2.5 3.25e-3 0.1 7.0 -1e-7))

(define (e4-values n dtype)
  (let ((v (allocate-typed-vector dtype n)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (typed-vector-set! v dtype i (list-ref e4-specials (modulo (* 5 i) (length e4-specials)))))))

(define (e4-ints n)
  (let ((v (make-s32vector n)))
    (do ((i 0 (+ i 1))) ((= i n) v) (s32vector-set! v i (- (* 7 (modulo (* 3 i) 11)) 30)))))

(test-group "strided seg-red executor"
  ;; Sum over axis 0 of x*y, with x*y kept as a map output sharing the body.
  (let* ((x (e4-arr (e4-values 15 'f32) #(5 3) #(3 1) 'f32))
         (y (e4-arr (e4-values 15 'f32) #(5 3) #(3 1) 'f32))
         (t (e4-map '(dim 1) '(dim 0)))
         (body '(cast f32 (mul (in 0) (in 1))))
         (sg (make-seg-red #(3 5) 1 (list (e4-op 'x x 'f32 t) (e4-op 'y y 'f32 t))
                           (list (make-red-result (e4-out 'f32 #(3) (e4-map '(dim 0))) body
                                                  (make-combine-spec 'add 0.0 'f32) #f))
                           (list (cons (e4-out 'f32 #(5 3) t) body))))
         (r (e4-run sg `((x . ,x) (y . ,y)) '((f32 3) (f32 15)))))
    (test "a fused column sum runs on the strided reduction executor" 'segop/strided-red (car r))
    (test-assert "column sum and kept products equal the interpreter's" (cadr r)))
  ;; Sum over axis 1 accumulated in f64, a mean finalize and keepdims.
  (let* ((x (e4-arr (e4-values 24 'f32) #(4 6) #(6 1) 'f32))
         (sg (make-seg-red #(4 6) 1 (list (e4-op 'x x 'f32 (identity-index-map 2)))
                           (list (make-red-result (e4-out 'f32 #(4 1) (e4-map '(dim 0) 0))
                                                  '(cast f32 (exp (in 0)))
                                                  (make-combine-spec 'add 0.0 'f64) '(/ (reduced) 6)))))
         (r (e4-run sg `((x . ,x)) '((f32 4)))))
    (test "a row mean with keepdims is a strided reduction" 'segop/strided-red (car r))
    (test-assert "the row mean equals the interpreter's" (cadr r)))
  ;; Max and min over axes 0 and 2 of a 3-D array, as two results.
  (let* ((x (e4-arr (e4-values 24 'f64) #(2 3 4) #(12 4 1) 'f64))
         (m (e4-map '(dim 1) '(dim 0) '(dim 2)))
         (sg (make-seg-red #(3 2 4) 2 (list (e4-op 'x x 'f64 m))
                           (list (make-red-result (e4-out 'f64 #(3) (e4-map '(dim 0))) '(add (in 0) 0.5)
                                                  (make-combine-spec 'max -inf.0 'f64) #f)
                                 (make-red-result (e4-out 'f64 #(3) (e4-map '(dim 0))) '(add (in 0) 0.5)
                                                  (make-combine-spec 'min +inf.0 'f64) #f))))
         (r (e4-run sg `((x . ,x)) '((f64 3) (f64 3)))))
    (test "two results over two reduced axes" 'segop/strided-red (car r))
    (test-assert "max and min with NaN, infinities and -0.0 equal the interpreter's" (cadr r)))
  ;; A product over axis 1 of a transposed (column-major) input.
  (let* ((x (e4-arr (e4-values 12 'f64) #(3 4) #(1 3) 'f64))
         (sg (make-seg-red #(3 4) 1 (list (e4-op 'x x 'f64 (identity-index-map 2)))
                           (list (make-red-result (e4-out 'f64 #(3) (e4-map '(dim 0))) '(add (in 0) 1.0)
                                                  (make-combine-spec 'mul 1.0 'f64) #f))))
         (r (e4-run sg `((x . ,x)) '((f64 3)))))
    (test "a strided input" 'segop/strided-red (car r))
    (test-assert "the product of a transposed input equals the interpreter's" (cadr r)))
  ;; Everything reduced to a rank-0 result, with a finalize.
  (let* ((x (e4-arr (e4-values 6 'f64) #(2 3) #(3 1) 'f64))
         (sg (make-seg-red #(2 3) 2 (list (e4-op 'x x 'f64 (identity-index-map 2)))
                           (list (make-red-result (e4-out 'f64 #() (e4-map)) '(mul (in 0) (in 0))
                                                  (make-combine-spec 'add 0.0 'f64) '(sqrt (reduced))))))
         (r (e4-run sg `((x . ,x)) '((f64 1)))))
    (test "a rank-0 result" 'segop/strided-red (car r))
    (test-assert "the rank-0 result equals the interpreter's" (cadr r)))
  ;; Integer sums of squares, rounded to s32 after every step.
  (let* ((x (e4-arr (e4-ints 12) #(3 4) #(4 1) 's32))
         (sg (make-seg-red #(3 4) 1 (list (e4-op 'x x 's32 (identity-index-map 2)))
                           (list (make-red-result (e4-out 's32 #(3) (e4-map '(dim 0))) '(mul (in 0) (in 0))
                                                  (make-combine-spec 'add 0 's32) #f))))
         (r (e4-run sg `((x . ,x)) '((s32 3)))))
    (test "an s32 reduction" 'segop/strided-red (car r))
    (test-assert "the s32 sums equal the interpreter's" (cadr r))
    (test "the s32 sums are the sums of squares"
      (map (lambda (row) (fold + 0 (map (lambda (j) (let ((v (s32vector-ref (e4-data x) (+ (* 4 row) j)))) (* v v)))
                                       (iota 4))))
           (iota 3))
      (s32vector->list (car (caddr r)))))
  ;; A blocked sum in single precision: blocks of 3 from the neutral element.
  (let* ((d (make-f32vector 10 1.0))
         (_ (f32vector-set! d 0 1e8))
         (x (e4-arr d #(1 10) #(10 1) 'f32))
         (sg (make-seg-red #(1 10) 1 (list (e4-op 'x x 'f32 (identity-index-map 2)))
                           (list (make-red-result (e4-out 'f32 #(1) (e4-map '(dim 0))) '(mul (in 0) 1.0)
                                                  (make-combine-spec 'add 0.0 'f32 'blocked 3) #f))))
         (r (e4-run sg `((x . ,x)) '((f32 1)))))
    (test "a blocked reduction" 'segop/strided-red (car r))
    (test-assert "the blocked sum equals the interpreter's" (cadr r))
    (test "the blocks keep the ones a sequential f32 sum would lose" (+ 1e8 8.0)
      (f32vector-ref (car (caddr r)) 0)))
  ;; A finalize reading an input over the parallel dimension, and a map
  ;; output with a body of its own.
  (let* ((x (e4-arr (e4-values 15 'f64) #(5 3) #(3 1) 'f64))
         (b (e4-arr (f64vector 0.25 -1.0 +inf.0) #(3) #(1) 'f64))
         (t (e4-map '(dim 1) '(dim 0)))
         (sg (make-seg-red #(3 5) 1 (list (e4-op 'x x 'f64 t) (e4-op 'b b 'f64 (e4-map '(dim 0))))
                           (list (make-red-result (e4-out 'f64 #(3) (e4-map '(dim 0))) '(mul (in 0) 2.0)
                                                  (make-combine-spec 'add 0.0 'f64) '(add (reduced) (in 1))))
                           (list (cons (e4-out 'f64 #(5 3) t) '(add (in 0) (in 1))))))
         (r (e4-run sg `((x . ,x) (b . ,b)) '((f64 3) (f64 15)))))
    (test "a finalize with an input and a separate map output" 'segop/strided-red (car r))
    (test-assert "they equal the interpreter's" (cadr r)))
  ;; A seg-red whose input map is opaque still goes to the interpreter.
  (let* ((x (e4-arr (e4-values 6 'f64) #(2 3) #(3 1) 'f64))
         (sg (make-seg-red #(2 3) 1 (list (make-segop-operand 'x 'f64 #(2 3) #(3 1) 0 (opaque-index-map values)))
                           (list (make-red-result (e4-out 'f64 #(2) (e4-map '(dim 0))) '(mul (in 0) 3.0)
                                                  (make-combine-spec 'add 0.0 'f64) #f)))))
    (test "an opaque map is not a strided reduction" 'segop/interpreted
      (car (select-segop-executor sg)))))
;;;; ============================================================
;;;; Numerical gradient checks
;;;;
;;;; The gradients of the replayed plan are compared with central
;;;; differences of the loss, realized directly from its morphism.  The
;;;; graphs send their backward pass through transposed element-wise ops,
;;;; gathers, reductions with keepdims and casts.  Every plan runs with the
;;;; checking backend in place.
;;;; ============================================================

;; Parameters of the given (shape dtype) specs and values; build maps them
;; to the loss.  Returns a make-loss thunk for seg-trace-and-replay.
(define (gc-make-loss build specs vals)
  (lambda ()
    (let ((ps (map (lambda (spec v) (seg-var v (car spec) (cadr spec) #t)) specs vals)))
      (values (build ps) ps))))

(define (gc-loss-value build specs vals)
  (let-values (((loss ps) ((gc-make-loss build specs vals))))
    (car (concrete->list (am:var-value loss)))))

;; The value v as stored in a dtype buffer.
(define (gc-stored dtype v)
  (if (eq? dtype 'f32)
      (let ((b (make-f32vector 1))) (f32vector-set! b 0 v) (f32vector-ref b 0))
      v))

;; Central differences of the loss with respect to every parameter
;; element.  The step is the difference of the stored perturbed values.
(define (gc-numeric-grads build specs vals eps)
  (map (lambda (spec j)
         (map (lambda (i)
                (let* ((dt (cadr spec))
                       (x  (list-ref (list-ref vals j) i))
                       (xp (gc-stored dt (+ x eps)))
                       (xm (gc-stored dt (- x eps)))
                       (at (lambda (xv)
                             (gc-loss-value build specs
                                            (map (lambda (v k)
                                                   (if (= k j)
                                                       (append (take v i) (list xv) (drop v (+ i 1)))
                                                       v))
                                                 vals (iota (length vals)))))))
                  (/ (- (at xp) (at xm)) (- xp xm))))
              (iota (length (list-ref vals j)))))
       specs (iota (length specs))))

(define (gc-values n seed)
  (map (lambda (i) (* 0.8 (sin (+ (* 1.3 i) seed)))) (iota n)))

;; Runs one gradient check.  The tags of the replay plan must include
;; expect-tags.
(define (gradcheck name build specs expect-tags #!key (eps 1e-6) (tol 1e-6))
  (let* ((vals (map (lambda (spec k)
                      (map (lambda (v) (gc-stored (cadr spec) v))
                           (gc-values (apply * (car spec)) (* 0.7 k))))
                    specs (iota (length specs))))
         (numeric (gc-numeric-grads build specs vals eps)))
    (let-values (((r checks mismatches)
                  (with-segop-checker
                   (lambda ()
                     (let-values (((joint trace replay) (seg-trace-and-replay (gc-make-loss build specs vals))))
                       (list joint trace replay))))))
      (let* ((joint (car r)) (trace (cadr r)) (replay (caddr r))
             (tags  (map car (plan-segops joint))))
        (test-assert (string-append name ": the plan uses the expected executors")
          (every (lambda (t) (memq t tags)) expect-tags))
        (test-assert (string-append name ": segop executions were checked") (> checks 0))
        (test (string-append name ": executors agree with the interpreter") 0 mismatches)
        (test-assert (string-append name ": replay reproduces the trace") (equal? trace replay))
        (test-assert (string-append name ": gradients match central differences")
          (results-close? (cdr replay) numeric tol))))))

(test-group "numerical gradient checks"
  ;; transposes feeding element-wise ops
  (gradcheck "tanh of a transpose times a row-major parameter"
             (lambda (ps)
               (let ((wt (am:var-transpose (car ps) '(1 0))))
                 (am:var-sum (am:var* (am:var-tanh wt) (cadr ps)))))
             '(((3 4) f64) ((4 3) f64))
             '(segop/strided-map))
  (gradcheck "a sum of two transposed views"
             (lambda (ps)
               (let ((s (am:var+ (am:var-transpose (car ps) '(1 0))
                                 (am:var-exp (am:var-transpose (cadr ps) '(1 0))))))
                 (am:var-mean (am:var* s s))))
             '(((2 5) f64) ((2 5) f64))
             '(segop/strided-map))
  (gradcheck "a rank-3 permutation feeding sigmoid"
             (lambda (ps)
               (am:var-sum (am:var-sigmoid (am:var-transpose (car ps) '(2 0 1)))))
             '(((2 3 4) f64))
             '(segop/strided-map))
  (gradcheck "a reshape of a transpose"
             (lambda (ps)
               (let ((r (am:var-reshape (am:var-transpose (car ps) '(1 0)) '(2 6))))
                 (am:var-sum (am:var* (am:var-sin r) (cadr ps)))))
             '(((3 4) f64) ((2 6) f64))
             '(segop/gather))
  ;; reductions with keepdims
  (gradcheck "centering rows with a keepdims mean"
             (lambda (ps)
               (let* ((x (car ps))
                      (c (am:var- x (am:var-mean x '(1) #t))))
                 (am:var-sum (am:var* c c))))
             '(((4 5) f64))
             '(segop/reduce-axis1 segop/bias-broadcast))
  (gradcheck "scaling columns by a keepdims sum"
             (lambda (ps)
               (let* ((x (am:var-exp (car ps)))
                      (s (am:var-sum x '(0) #t)))
                 (am:var-sum (am:var* (am:var/ x s) (cadr ps)))))
             '(((3 4) f64) ((3 4) f64))
             '(segop/reduce-axis0))
  (gradcheck "a keepdims mean over two axes of a rank-3 array"
             (lambda (ps)
               (let* ((x (car ps))
                      (m (am:var-mean (am:var* x x) '(0 2) #t)))
                 (am:var-sum (am:var* (am:var-sqrt m) (am:var-sum x '(0 2) #t)))))
             '(((2 3 4) f64))
             '(segop/reduce-general segop/strided-map))
  (gradcheck "a keepdims sum over a transposed view"
             (lambda (ps)
               (let* ((xt (am:var-transpose (car ps) '(1 0)))
                      (s  (am:var-sum xt '(1) #t)))
                 (am:var-sum (am:var* (am:var-tanh (am:var* xt s)) (cadr ps)))))
             '(((3 4) f64) ((4 3) f64))
             '(segop/strided-map segop/reduce-axis1))
  ;; casts
  (gradcheck "a cast of an f64 parameter to f32 and back"
             (lambda (ps)
               (let* ((x   (car ps))
                      (x32 (am:make-var (morph-cast (am:var-value x) 'f32) #t))
                      (y32 (am:var-tanh (am:var* x32 x32)))
                      (y64 (am:make-var (morph-cast (am:var-value y32) 'f64) #t)))
                 (am:var-sum (am:var* y64 (cadr ps)))))
             '(((2 3) f64) ((2 3) f64))
             '(segop/reduce-general)
             eps: 1e-3 tol: 2e-4)
  (gradcheck "an f32 parameter cast to f64"
             (lambda (ps)
               (let ((xc (am:make-var (morph-cast (am:var-value (car ps)) 'f64) #t)))
                 (am:var-mean (am:var* (am:var-exp xc) (cadr ps)))))
             '(((5) f32) ((5) f64))
             '(segop/reduce-1d segop/strided-map)
             eps: 1e-3 tol: 1e-4)
  (gradcheck "a cast of a transposed view"
             (lambda (ps)
               (let* ((xt (am:var-transpose (car ps) '(1 0)))
                      (xc (am:make-var (morph-cast (am:var-value xt) 'f32) #t))
                      (yc (am:make-var (morph-cast (am:var-value (am:var-sigmoid xc)) 'f64) #t)))
                 (am:var-sum (am:var* yc (cadr ps)))))
             '(((3 2) f64) ((2 3) f64))
             '(segop/strided-map)
             eps: 1e-3 tol: 2e-4))

;;;; ============================================================
;;;; f32 eager gradients
;;;;
;;;; Eager backward! on f32 parameters must give f32 gradients, and
;;;; these must agree with the ssa-vjp gradients of the same graph and
;;;; with eager f64 gradients to within f32 precision.  Each graph is
;;;; built fresh for every evaluation so that gradients do not
;;;; accumulate across runs.
;;;; ============================================================

(define f32-tol 1e-5)

(define (f32-approx= a b)
  (<= (abs (- a b)) (* f32-tol (max 1.0 (abs b)))))

(define (lists-f32-approx= l1 l2)
  (and (= (length l1) (length l2))
       (every f32-approx= l1 l2)))

;;; build: (lambda (make-param) -> (values loss-var param-vars)), where
;;; make-param takes a flat list and a shape list.
(define (make-param-of dtype)
  (lambda (lst shape)
    (am:make-var (morph-from-list lst (list->vector shape) dtype) #t)))

;;; Eager gradients: (values grad-lists grad-dtypes).
(define (eager-grads build dtype)
  (let-values (((loss params) (build (make-param-of dtype))))
    (am:backward! loss)
    (let ((grads (map (lambda (p) (realize (am:var-grad p))) params)))
      (values (map concrete->list grads)
              (map get-morphism-dtype grads)))))

;;; SSA gradients as lists, without the loss value.
(define (ssa-grads build dtype)
  (let-values (((loss params) (build (make-param-of dtype))))
    (cdr (compile-and-realize loss params))))

(define (check-f32-eager-grads name build)
  (let-values (((g32 dtypes) (eager-grads build 'f32))
               ((g64 _)      (eager-grads build 'f64)))
    (let ((s32 (ssa-grads build 'f32)))
      (test-assert (string-append name ": eager gradients are f32")
        (every (lambda (d) (eq? d 'f32)) dtypes))
      (test-assert (string-append name ": eager f32 agrees with SSA f32")
        (every lists-f32-approx= g32 s32))
      (test-assert (string-append name ": eager f32 agrees with eager f64")
        (every lists-f32-approx= g32 g64)))))

(define A-vals '(0.5 -1.0 2.0 0.25 1.5 -0.5))
(define B-vals '(1.0 -2.0 0.5 0.75 -1.25 3.0))

(test-group "f32 eager gradients"

  (test-assert "mean of an f32 matmul has f32 gradients"
    (let* ((A (am:make-var (morph-from-list '(1 2 3 4 5 6) #(2 3) 'f32) #t))
           (B (am:make-var (morph-from-list '(1 2 3 4 5 6) #(3 2) 'f32) #t))
           (L (am:var-mean (am:var-matmul A B))))
      (am:backward! L)
      ;; dL/dA[i,k] = sum_j B[k,j] / 4, dL/dB[k,j] = sum_i A[i,k] / 4
      (and (eq? (get-morphism-dtype (realize (am:var-grad A))) 'f32)
           (lists-f32-approx= (concrete->list (am:var-grad A))
                              '(0.75 1.75 2.75 0.75 1.75 2.75))
           (lists-f32-approx= (concrete->list (am:var-grad B))
                              '(1.25 1.25 1.75 1.75 2.25 2.25)))))

  (check-f32-eager-grads "mean(A B)"
    (lambda (param)
      (let ((A (param A-vals '(2 3)))
            (B (param B-vals '(3 2))))
        (values (am:var-mean (am:var-matmul A B)) (list A B)))))

  (check-f32-eager-grads "row mean of (A B)^2"
    (lambda (param)
      (let* ((A (param A-vals '(2 3)))
             (B (param B-vals '(3 2)))
             (Y (am:var-matmul A B)))
        (values (am:var-mean (am:var-mean (am:var* Y Y) '(1) #t))
                (list A B)))))

  (check-f32-eager-grads "mean(A^T B)"
    (lambda (param)
      (let ((At (param A-vals '(3 2)))
            (B  (param B-vals '(3 2))))
        (values (am:var-mean (am:var-matmul (am:var-transpose At '(1 0)) B))
                (list At B)))))

  (check-f32-eager-grads "mean(A B^T)"
    (lambda (param)
      (let ((A  (param A-vals '(2 3)))
            (Bt (param B-vals '(2 3))))
        (values (am:var-mean (am:var-matmul A (am:var-transpose Bt '(1 0))))
                (list A Bt)))))

  (check-f32-eager-grads "two-layer mean(sigmoid(relu(X W1) W2))"
    (lambda (param)
      (let* ((X  (param A-vals '(2 3)))
             (W1 (param B-vals '(3 2)))
             (W2 (param '(0.3 -0.7 1.1 0.2) '(2 2)))
             (H  (am:var-relu (am:var-matmul X W1))))
        (values (am:var-mean (am:var-sigmoid (am:var-matmul H W2)))
                (list X W1 W2)))))

  (test-assert "f32 parameter times f64 input has an f32 gradient"
    (let* ((W (am:make-var (morph-from-list B-vals #(3 2) 'f32) #t))
           (X (am:make-var (morph-from-list A-vals #(2 3) 'f64) #f))
           (L (am:var-mean (am:var-matmul X W))))
      (am:backward! L)
      (eq? (get-morphism-dtype (realize (am:var-grad W))) 'f32))))

;;;; ============================================================
;;;; GEMM as seg-red
;;;;
;;;; Every matmul is a seg-red over (M N K) routed to the BLAS backend.
;;;; Forward programs are traced and replayed with the checking backend, so
;;;; each GEMM is also run by the reference interpreter.  Products below
;;;; the BLAS size threshold run the Scheme loop, whose combine spec is
;;;; exact (sequential, f64 accumulator), and are compared with a
;;;; reference computed here.
;;;; ============================================================

(define (gm-vals n seed) (map (lambda (i) (* 0.9 (sin (+ (* 1.37 i) seed)))) (iota n)))

(define (gm-round dt x)
  (if (eq? dt 'f32)
      (let ((b (make-f32vector 1))) (f32vector-set! b 0 x) (f32vector-ref b 0))
      x))

;; Row-major values of the [M,K] matrix seen through a transpose when t?.
(define (gm-matrix-values vals rows cols t?)
  (if t?
      (append-map* (lambda (i) (map (lambda (j) (list-ref vals (+ (* j rows) i))) (iota cols)))
                   (iota rows))
      vals))
(define (append-map* f lst) (apply append (map f lst)))

;; A[M,K] (stored transposed when tA?) times B[K,N] (stored transposed when tB?).
(define (gm-operands dt M K N tA? tB?)
  (let ((a (map (lambda (v) (gm-round dt v)) (gm-vals (* M K) 0.3)))
        (b (map (lambda (v) (gm-round dt v)) (gm-vals (* K N) 1.1))))
    (values (if tA?
                (morph-transpose (morph-from-list a (vector K M) dt) '(1 0))
                (morph-from-list a (vector M K) dt))
            (if tB?
                (morph-transpose (morph-from-list b (vector N K) dt) '(1 0))
                (morph-from-list b (vector K N) dt))
            (gm-matrix-values a M K tA?)
            (gm-matrix-values b K N tB?))))

;; The product summed over increasing k in double precision, rounded once.
(define (gm-reference dt av bv M K N)
  (append-map* (lambda (i)
                 (map (lambda (j)
                        (gm-round dt (fold (lambda (k acc) (+ acc (* (list-ref av (+ (* i K) k))
                                                                     (list-ref bv (+ (* k N) j)))))
                                           0.0 (iota K))))
                      (iota N)))
               (iota M)))

(define (gm-close? xs ys tol)
  (and (= (length xs) (length ys))
       (every (lambda (x y) (<= (abs (- x y)) (* tol (max 1.0 (abs y))))) xs ys)))

;; The GEMM segops of a plan, as (tag . segop).
(define (gm-segops prog)
  (filter (lambda (p) (memq (car p) '(segop/gemm segop/gemm-strided segop/gemm-epilogue)))
          (plan-segops prog)))

(test-group "GEMM as seg-red"
  ;; layouts, dtypes, and sizes below (3x4x5) and above (9x8x10) the threshold
  (for-each
   (lambda (dt)
     (for-each
      (lambda (layout)
        (for-each
         (lambda (mkn)
           (let* ((M (car mkn)) (K (cadr mkn)) (N (caddr mkn))
                  (tA? (car layout)) (tB? (cadr layout))
                  (name (sprintf "~A ~Ax~Ax~A~A~A" dt M K N (if tA? " A^T" "") (if tB? " B^T" ""))))
             (let-values (((A B av bv) (gm-operands dt M K N tA? tB?)))
               (let* ((r (s2-run (lambda () (morph-matmul A B))))
                      (prog (car r)) (trace (cadr r)) (replay (caddr r))
                      (gs (gm-segops prog))
                      (small? (< (* M N) 64)))
                 (test-assert (string-append name ": one GEMM segop of the expected kind")
                   (and (= (length gs) 1)
                        (eq? (caar gs) (if (or tA? tB?) 'segop/gemm-strided 'segop/gemm))))
                 (let* ((sg (cdar gs))
                        (g  (standard-gemm sg))
                        (spec (red-result-combine (car (cases segop sg
                                                         (seg-red (_ _ _ results _ _) results)
                                                         (else '()))))))
                   (test-assert (string-append name ": standard-gemm recovers it")
                     (and g (eq? (list-ref g 3) (if (or tA? tB?) 'gemm-strided 'gemm))
                          (not (list-ref g 5))))
                   (test-assert (string-append name ": the combine spec follows the plan")
                     (if small?
                         (and (eq? (combine-spec-acc-dtype spec) 'f64)
                              (eq? (combine-spec-order spec) 'sequential))
                         (and (eq? (combine-spec-acc-dtype spec) dt)
                              (eq? (combine-spec-order spec) 'unspecified)))))
                 (test (string-append name ": executors agree with the interpreter") 0 (list-ref r 4))
                 (test-assert (string-append name ": replay reproduces the trace")
                   (s2-same-lists? trace replay))
                 (test-assert (string-append name ": values")
                   (let ((ref (gm-reference dt av bv M K N)))
                     (if small?
                         (s2-same-lists? trace ref)
                         (gm-close? trace ref (if (eq? dt 'f32) 1e-5 1e-13)))))))))
         '((3 4 5) (9 8 10))))
      '((#f #f) (#t #f) (#f #t) (#t #t))))
   '(f32 f64))

  ;; degenerate shapes
  (for-each
   (lambda (mkn)
     (let* ((M (car mkn)) (K (cadr mkn)) (N (caddr mkn)))
       (let-values (((A B av bv) (gm-operands 'f64 M K N #f #f)))
         (let ((r (s2-run (lambda () (morph-matmul A B)))))
           (test-assert (sprintf "~Ax~Ax~A: replay, interpreter and reference agree" M K N)
             (and (= 0 (list-ref r 4)) (s2-same-lists? (cadr r) (caddr r))
                  (gm-close? (cadr r) (gm-reference 'f64 av bv M K N) 1e-13)))))))
   '((4 1 5) (1 6 5) (4 6 1) (1 1 1) (1 70 1) (70 1 1)))

  ;; the Scheme loop with BLAS disabled, above the threshold
  (let-values (((A B av bv) (gm-operands 'f32 9 8 10 #t #f)))
    (disable-blas!)
    (let* ((r (s2-run (lambda () (morph-matmul A B))))
           (spec (red-result-combine
                  (car (cases segop (cdar (gm-segops (car r)))
                         (seg-red (_ _ _ results _ _) results)
                         (else '()))))))
      (enable-blas!)
      (test-assert "without BLAS the combine spec is sequential with an f64 accumulator"
        (and (eq? (combine-spec-order spec) 'sequential) (eq? (combine-spec-acc-dtype spec) 'f64)))
      (test-assert "without BLAS the values are the exact reference"
        (s2-same-lists? (cadr r) (gm-reference 'f32 av bv 9 8 10)))
      (test "without BLAS the executor and the interpreter agree" 0 (list-ref r 4))))

  ;; epilogues: a table op, a fused composed body, a bias, and bias + activation
  (for-each
   (lambda (dt)
     (for-each
      (lambda (case)
        (let-values (((A B av bv) (gm-operands dt 9 8 10 #f #t)))
          (let* ((bias (morph-from-list (map (lambda (v) (gm-round dt v)) (gm-vals 10 2.3)) #(10) dt))
                 (make (lambda () ((cadr case) (morph-matmul A B) bias)))
                 (r    (s2-run make))
                 (gs   (gm-segops (car r)))
                 (name (sprintf "~A ~A" dt (car case))))
            (test-assert (string-append name ": the GEMM carries the epilogue")
              (and (= 1 (length gs)) (eq? (caar gs) 'segop/gemm-epilogue)
                   (let ((g (standard-gemm (cdar gs))))
                     (and g (eq? (car (list-ref g 5)) (caddr case))))))
            (test-assert (string-append name ": the finalize reads the stored product")
              (let ((fin (red-result-finalize
                          (car (cases segop (cdar gs)
                                 (seg-red (_ _ _ results _ _) results)
                                 (else '()))))))
                (and (pair? fin) (member `(cast ,dt (reduced)) (cdr fin)) #t)))
            (test (string-append name ": executors agree with the interpreter") 0 (list-ref r 4))
            (test-assert (string-append name ": replay reproduces the trace")
              (s2-same-lists? (cadr r) (caddr r)))
            (test-assert (string-append name ": the trace agrees with realize")
              (gm-close? (cadr r) (s2-values (realize (make))) (if (eq? dt 'f32) 1e-6 1e-13))))))
      (list (list "sigmoid" (lambda (c b) (morph-sigmoid c)) 'unary)
            (list "exp" (lambda (c b) (morph-exp c)) 'unary)
            (list "bias" (lambda (c b) (morph+ c b)) 'bias))))
   '(f32 f64))

  ;; a cast after a GEMM is not fused: its buffer has another dtype
  (let-values (((A B av bv) (gm-operands 'f32 9 8 10 #f #f)))
    (let* ((r (s2-run (lambda () (morph-cast (morph-matmul A B) 'f64))))
           (tags (s2-tags (car r))))
      (test-assert "a cast to f64 after an f32 GEMM is left unfused"
        (and (memq 'segop/gemm tags) (not (memq 'segop/gemm-epilogue tags))))
      (test-assert "the cast after a GEMM replays as traced"
        (s2-same-lists? (cadr r) (caddr r)))
      (test-assert "the cast after a GEMM has the f32 product's values"
        (gm-close? (cadr r) (gm-reference 'f32 av bv 9 8 10) 1e-5))))

  ;; with only the scheme backend, a GEMM runs on the strided reduction
  ;; executor
  (let-values (((A B av bv) (gm-operands 'f32 3 4 5 #t #f)))
    (let* ((scheme-only (filter (lambda (b) (eq? (segop-backend-name b) 'scheme)) (segop-backends)))
           (r (parameterize ((segop-backends scheme-only))
                (s2-run (lambda () (morph-matmul A B))))))
      (test-assert "with only the scheme backend a GEMM is a strided reduction"
        (and (memq 'segop/strided-red (s2-tags (car r))) (null? (gm-segops (car r)))))
      (test-assert "the host GEMM gives the exact reference"
        (and (s2-same-lists? (cadr r) (gm-reference 'f32 av bv 3 4 5))
             (s2-same-lists? (cadr r) (caddr r)))))))


;;;; ============================================================
;;;; morph-map in SSA
;;;;
;;;; A user procedure mapped over an array traces and replays like any
;;;; element-wise op, with an opaque body.  It has no derivative: ssa-vjp
;;;; rejects a map whose input depends on a trainable parameter.  A
;;;; parameter that no gradient reaches gets zeros, so the gradient outputs
;;;; stay aligned with the parameters.
;;;; ============================================================

(define (mm-square x) (* x x))
(define (mm-shift x) (- (* 0.75 x) 0.5))

;; The finalize body of the single seg-red among the plan's GEMM segops.
(define (mm-gemm-finalize prog)
  (let ((gs (gm-segops prog)))
    (and (= 1 (length gs))
         (red-result-finalize
          (car (cases segop (cdar gs)
                 (seg-red (_ _ _ results _ _) results)
                 (else '())))))))

(test-group "morph-map in SSA"
  ;; forward programs: trace, replay, interpreter and realize agree
  (for-each
   (lambda (case)
     (let* ((make (cadr case))
            (r    (s2-run make))
            (name (car case)))
       (test (string-append name ": executors agree with the interpreter") 0 (list-ref r 4))
       (test-assert (string-append name ": replay reproduces the trace")
         (s2-same-lists? (cadr r) (caddr r)))
       (test-assert (string-append name ": the trace equals realize")
         (s2-same-lists? (cadr r) (s2-values (realize (make)))))
       (when (pair? (cddr case))
         (test-assert (string-append name ": executed as " (symbol->string (caddr case)))
           (memq (caddr case) (s2-tags (car r)))))))
   (list (list "map of a computed array"
               (lambda () (morph-map mm-square (s2-src 'f64 '(3 4) s2-val))))
         (list "map of a constant"
               (lambda () (morph-map mm-shift (morph-from-list (gm-vals 12 0.4) #(3 4) 'f32))))
         (list "map of a transposed view"
               (lambda () (morph-map mm-shift (morph-transpose (s2-src 'f32 '(3 4) s2-val) '(1 0))))
               'segop/strided-map)
         (list "s32 map with non-integer results"
               (lambda () (morph-map mm-shift (morph+ (morph-from-list '(3 -1 4 1 -5 9) #(2 3) 's32)
                                                       (morph-from-list '(2) #(1) 's32)))))
         (list "f32 map after a matmul"
               (lambda () (let-values (((A B av bv) (gm-operands 'f32 9 8 10 #f #f)))
                            (morph-map mm-shift (morph-matmul A B))))
               'segop/gemm-epilogue)
         (list "f64 map after a matmul"
               (lambda () (let-values (((A B av bv) (gm-operands 'f64 9 8 10 #t #f)))
                            (morph-map mm-square (morph-matmul A B))))
               'segop/gemm-epilogue)))

  (let-values (((A B av bv) (gm-operands 'f32 9 8 10 #f #f)))
    (let ((fin (mm-gemm-finalize (car (s2-run (lambda () (morph-map mm-shift (morph-matmul A B))))))))
      (test-assert "a map epilogue is an opaque finalize over the stored product"
        (and (pair? fin) (eq? (car fin) 'opaque) (eq? (cadr fin) mm-shift)
             (equal? (cddr fin) '((cast f32 (reduced))))))))

  ;; a map off the gradient path: x is an input, so map(x) needs no gradient
  (gradcheck "a map of an input added to a product"
             (lambda (ps)
               (let* ((x  (seg-var (gm-vals 6 0.9) '(2 3) 'f64 #f))
                      (mx (am:make-var (morph-map mm-square (am:var-value x)) #f)))
                 (am:var-mean (am:var* (am:var+ (am:var* (car ps) x) mx) (cadr ps)))))
             '(((2 3) f64) ((2 3) f64))
             '())

  ;; a map on a gradient path is rejected
  (let ((W (seg-var (gm-vals 6 0.2) '(2 3) 'f64 #t)))
    (test-error "a map of a parameter has no derivative"
      (compile-and-realize (am:var-sum (am:make-var (morph-map mm-square (am:var-value W)) #t))
                           (list W)))
    (test-error "a map of a value computed from a parameter has no derivative"
      (compile-and-realize
       (am:var-sum (am:make-var (morph-map mm-square (am:var-value (am:var* W W))) #t))
       (list W))))

  ;; a parameter reached only through a max reduction (whose VJP is skipped)
  ;; gets zeros, and the parameters after it keep their own gradients
  (let* ((av (gm-vals 6 0.1)) (wv (gm-vals 6 0.5)) (bv (gm-vals 3 0.8))
         (make (lambda ()
                 (let* ((A (seg-var av '(2 3) 'f64 #t))
                        (W (seg-var wv '(2 3) 'f32 #t))
                        (B (seg-var bv '(3) 'f64 #t))
                        (mw (am:make-var (morph-reduce 'max (am:var-value W) '(1)) #t)))
                   (values (am:var+ (am:var-sum (am:var* (am:var-exp A) B))
                                    (am:var-sum (am:make-var (morph-cast (am:var-value mw) 'f64) #t)))
                           (list A W B))))))
    (let-values (((joint trace replay) (seg-trace-and-replay make)))
      (test "one output per parameter, after the loss" 4 (length replay))
      (test "the parameter behind the max reduction gets zeros" (map (lambda (i) 0.0) (iota 6))
        (caddr replay))
      (test-assert "the zero gradient has the parameter's dtype"
        (eq? 'f32 (cases array-morphism (list-ref (ssa-realize joint) 2)
                    (concrete-array (_ _ _ _ dt _ _) dt)
                    (else #f))))
      (test-assert "replay reproduces the trace" (equal? trace replay))
      (test-assert "the other gradients are those of the expression"
        (and (results-close? (list (cadr replay))
                             (list (map (lambda (a b) (* (exp a) b))
                                        av (append bv bv)))
                             1e-12)
             (results-close? (list (cadddr replay))
                             (list (map (lambda (j) (+ (exp (list-ref av j)) (exp (list-ref av (+ j 3)))))
                                        (iota 3)))
                             1e-12))))))


;;;; ============================================================
;;;; Fusion oracle
;;;;
;;;; A fused binding rounds each producer value to the producer's dtype,
;;;; as storing it would, so a program computes the same results with and
;;;; without the fusion pass.  Each graph is traced and replayed with fusion
;;;; on and off, under the checking backend.
;;;; ============================================================

;; (list trace replay mismatches fused-bodies) for make-loss under the
;; fusion policy fusion (all or #f).
(define (fo-run make-loss fusion)
  (parameterize ((segop-fusion fusion))
    (let-values (((r checks mismatches)
                  (with-segop-checker
                   (lambda ()
                     (let-values (((joint trace replay) (seg-trace-and-replay make-loss)))
                       (list joint trace replay))))))
      (list (cadr r) (caddr r) mismatches
            (map (lambda (b) (cdr (assq 'body (ssa-binding-meta b)))) (fused-bindings (car r)))))))

(define (fo-has-cast? e dt)
  (and (pair? e)
       (or (and (eq? (car e) 'cast) (eq? (cadr e) dt))
           (any (lambda (x) (fo-has-cast? x dt)) (cdr e)))))

(define (fo-vals n seed) (map (lambda (i) (* 0.7 (sin (+ (* 1.3 i) seed)))) (iota n)))

(define fo-graphs
  (list
   (list "mlp f32" 'f32 (lambda () (seg-mlp-loss 'f32)))
   (list "mlp f64" 'f64 (lambda () (seg-mlp-loss 'f64)))
   (list "row normalization f32" 'f32
         (lambda ()
           (let* ((X   (seg-var (fo-vals 12 0.1) '(3 4) 'f32 #t))
                  (g   (seg-var (fo-vals 4 0.2) '(4) 'f32 #t))
                  (eps (seg-var '(1e-5) '(1) 'f32 #f))
                  (ms  (am:var-mean (am:var* X X) '(1) #t))
                  (Y   (am:var* (am:var/ X (am:var-sqrt (am:var+ ms eps))) g)))
             (values (am:var-mean (am:var* Y Y)) (list X g)))))
   (list "tanh(x*y + z) f32" 'f32
         (lambda ()
           (let* ((x (seg-var (fo-vals 6 0.3) '(2 3) 'f32 #t))
                  (y (seg-var (fo-vals 6 0.4) '(2 3) 'f32 #t))
                  (z (seg-var (fo-vals 6 0.5) '(2 3) 'f32 #t))
                  (v (am:var-tanh (am:var+ (am:var* x y) z))))
             (values (am:var-sum (am:var* v v)) (list x y z)))))
   (list "sigmoid(tanh(x*y)) f32" 'f32
         (lambda ()
           (let* ((x (seg-var (fo-vals 8 0.6) '(2 4) 'f32 #t))
                  (y (seg-var (fo-vals 8 0.7) '(2 4) 'f32 #t))
                  (w (seg-var (fo-vals 8 0.8) '(2 4) 'f32 #f)))
             (values (am:var-sum (am:var* (am:var-sigmoid (am:var-tanh (am:var* x y))) w))
                     (list x y)))))
   (list "a long f32 chain" 'f32
         (lambda ()
           (let* ((x (seg-var (fo-vals 10 0.9) '(2 5) 'f32 #t))
                  (y (seg-var (fo-vals 10 1.0) '(2 5) 'f32 #t))
                  (c (am:var-exp (am:var-negate (am:var-sigmoid (am:var-tanh (am:var* x y)))))))
             (values (am:var-mean (am:var* c c)) (list x y)))))
   ;; the f32 sigmoid(x) is fused into the f64 product, through a cast
   ;; to f32
   (list "mixed f32/f64" 'f32
         (lambda ()
           (let ((x (seg-var '(-0.7 -0.4 -0.1 0.2 0.5 0.8) '(2 3) 'f32 #t))
                 (y (seg-var '(0.2 0.3 0.4 0.5 0.6 0.7) '(2 3) 'f64 #t)))
             (values (am:var-mean (am:var* (am:var-sigmoid x) (am:var+ x y))) (list x y)))))))

(test-group "fusion oracle"
  (for-each
   (lambda (g)
     (let* ((name (car g)) (dt (cadr g)) (make (caddr g))
            (on  (fo-run make 'all))
            (off (fo-run make #f)))
       (unless (eq? dt 'none)
         (test-assert (string-append name ": the fusion pass fuses bindings") (pair? (list-ref on 3))))
       (test-assert (string-append name ": no fusion without the pass") (null? (list-ref off 3)))
       (test-assert (string-append name ": the trace is the same with and without fusion")
         (equal? (car on) (car off)))
       (test-assert (string-append name ": the replay is the same with and without fusion")
         (equal? (cadr on) (cadr off)))
       (test (string-append name ": fused executors agree with the interpreter") 0 (caddr on))
       (case dt
         ((f32) (test-assert (string-append name ": a fused body rounds to f32")
                  (any (lambda (b) (fo-has-cast? b 'f32)) (list-ref on 3))))
         ((f64) (test-assert (string-append name ": f64 bodies have no casts")
                  (not (any (lambda (b) (fo-has-cast? b 'f64)) (list-ref on 3))))))))
   fo-graphs)

  ;; x*y rounds to f32 before z is added.  With x = y = 1 + 2^-12 the exact
  ;; product 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11, so x*y - 1 is 2^-11;
  ;; adding in double precision without that rounding would give
  ;; 2^-11 + 2^-24.
  (let* ((x0 (+ 1.0 (expt 2.0 -12)))
         (make (lambda ()
                 (let ((x (seg-var (list x0) '(1) 'f32 #t))
                       (y (seg-var (list x0) '(1) 'f32 #t))
                       (z (seg-var '(-1.0) '(1) 'f32 #t)))
                   (values (am:var-sum (am:var+ (am:var* x y) z)) (list x y z)))))
         (on  (fo-run make 'all))
         (off (fo-run make #f)))
    (test-assert "x*y + z fuses with a cast at the boundary"
      (any (lambda (b) (fo-has-cast? b 'f32)) (list-ref on 3)))
    (test "the fused x*y + z rounds the product to f32" (expt 2.0 -11) (car (car (cadr on))))
    (test-assert "the fused x*y + z equals the unfused result"
      (and (equal? (car on) (car off)) (equal? (cadr on) (cadr off))))))


;;;; ============================================================
;;;; Segop fusion
;;;;
;;;; The fusion pass builds a graph of uses with fusible, infusible and
;;;; result edges, fuses a map binding into its only consumer wherever the
;;;; consumer is (not only the next binding), and realizes fused bindings
;;;; through their segop executor, with one context allocation per output.
;;;; ============================================================

(define (sf-var lst shape) (am:make-var (morph-from-list lst (list->vector shape) 'f64) #f))

(define (sf-binding prog op)
  (car (filter (lambda (b) (equal? (ssa-binding-op b) op)) (ssa-program-bindings prog))))

(define (sf-uses graph b)
  (map (lambda (u) (list (if (car u) 'binding #f) (caddr u)))
       (cdr (assq (ssa-binding-name b) graph))))

;; Values of a forward program as lists: with ssa-realize, then traced and
;; replayed through a context.
;; prog as it is: prog itself when the fusion policy has been applied to
;; it, otherwise a copy that ssa-realize/ctx does not fuse at its first
;; trace.
(define (sf-as-given prog)
  (if (ssa-program-fused? prog)
      prog
      (make-ssa-program (ssa-program-constants prog) (ssa-program-morph-to-val prog)
                        (ssa-program-bindings prog) (ssa-program-outputs prog)
                        (ssa-program-n-params prog) #f #f #f #t)))

;; The results of prog, as it is, from ssa-realize, the trace and two
;; replays.
(define (sf-run prog*)
  (let ((ctx (make-morphism-context)) (prog (sf-as-given prog*)))
    (let ((plain (map concrete->list (ssa-realize prog)))
          (trace (map concrete->list (ssa-realize/ctx ctx prog))))
      (finalize-context! ctx)
      (reset-context! ctx)
      (ssa-realize/ctx ctx prog)
      (reset-context! ctx)
      (list plain trace (map concrete->list (ssa-realize/ctx ctx prog))))))

(define (sf-fused prog)
  (filter (lambda (b) (eq? (ssa-binding-op b) 'fused)) (ssa-program-bindings prog)))

(define (sf-members b) (cdr (assq 'members (ssa-binding-meta b))))

(define sf-x '(0.5 -1.25 2.0 0.75 -0.5 1.5))

(test-group "segop fusion"
 ;; Under the default policy nothing is fused on the host, whose fused
 ;; executors are slower than the separate bindings'; GEMM epilogues of
 ;; one op run as flat kernels on the stored product.
 (let-values (((loss params) (seg-mlp-loss 'f32)))
   (let* ((fwd (morphism-to-ssa loss))
          (pv  (filter-map (lambda (p) (ssa-constant-id fwd (am:var-value p))) params)))
     (test "the default policy fuses only GEMM epilogues" '()
       (filter (lambda (b) (not (gemm-epilogue-binding? b)))
               (fused-bindings (ssa-vjp fwd pv (ssa-loss-binding-val fwd)))))
     (test "the policy all fuses" #t
       (pair? (fused-bindings (parameterize ((segop-fusion 'all))
                                (ssa-vjp fwd pv (ssa-loss-binding-val fwd))))))))
 (test "segop-fusion takes #t, all and #f" '(#t all #f)
   (map (lambda (v) (parameterize ((segop-fusion v)) (segop-fusion))) '(yes all #f)))

 (parameterize ((segop-fusion 'all))
  ;; e = exp(x) * y; u = e + q with q = exp(b) of shape [3], read through a
  ;; broadcast; m = u W, a barrier; the loss sums m.
  (let* ((x (sf-var sf-x '(2 3)))
         (y (sf-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(2 3)))
         (b (sf-var '(0.1 0.2 0.3) '(3)))
         (W (sf-var '(1.0 0.0 0.5 2.0 -1.0 1.0) '(3 2)))
         (prog (morphism-to-ssa (am:make-var
                                 (morph-reduce 'sum (morph-matmul (morph+ (morph* (morph-exp (am:var-value x))
                                                                                  (am:var-value y))
                                                                          (morph-exp (am:var-value b)))
                                                                  (am:var-value W))
                                               '(0 1) #f)
                                 #f)))
         (graph (ssa-fusion-graph prog))
         (exps  (filter (lambda (b) (eq? (ssa-binding-op b) 'exp)) (ssa-program-bindings prog)))
         (big   (car (filter (lambda (b) (equal? (ssa-binding-shape b) #(2 3))) exps)))
         (small (car (filter (lambda (b) (equal? (ssa-binding-shape b) #(3))) exps))))
    (test "a same-shape read by a map binding is fusible" '((binding fusible))
      (sf-uses graph big))
    (test "a broadcast read is infusible" '((binding infusible)) (sf-uses graph small))
    (test "a matmul operand is infusible" '((binding infusible))
      (sf-uses graph (sf-binding prog 'add)))
    (test "a reduction source is fusible" '((binding fusible))
      (sf-uses graph (sf-binding prog 'matmul)))
    (test "a program output has a result use" '((#f result))
      (sf-uses graph (sf-binding prog '(reduce sum))))
    (let ((fused (ssa-segop-fusion-pass prog)))
      (test "exp(x), the product and the add are fused, the broadcast exp(b) is not" 1
        (length (sf-fused fused)))
      (test-assert "the fused binding computes exp(x), the product and the add"
        (= 3 (length (sf-members (car (sf-fused fused))))))
      (test "the fused program computes the same values" (sf-run prog) (sf-run fused))))

  ;; A producer that is also a program output is fused, and its value kept
  ;; as a second output.
  (let* ((x  (sf-var sf-x '(2 3)))
         (fwd (morphism-to-ssa (am:make-var (morph-negate (morph-exp (am:var-value x))) #f)))
         (e   (sf-binding fwd 'exp))
         (two (make-ssa-program (ssa-program-constants fwd) (ssa-program-morph-to-val fwd)
                                (ssa-program-bindings fwd)
                                (append (ssa-program-outputs fwd) (list (binding-ref (ssa-binding-name e))))
                                0 #f #f #f))
         (fused (ssa-segop-fusion-pass two)))
    (test "a producer that is a program output is fused with its value kept" '(2)
      (map (lambda (b) (length (binding-out-names b))) (sf-fused fused)))
    (test "both outputs are computed" (sf-run two) (sf-run fused))
    (test "without the second output, the producer is fused" 1
      (length (sf-fused (ssa-segop-fusion-pass fwd)))))

  ;; c = negate(e) + e with e = exp(x): negate is fused into the add, whose
  ;; body then reads e twice, so e is not fused (it would be computed twice).
  (let* ((x  (sf-var sf-x '(2 3)))
         (ex (morph-exp (am:var-value x)))
         (prog (morphism-to-ssa (am:make-var (morph+ (morph-negate ex) ex) #f)))
         (fused (ssa-segop-fusion-pass prog))
         (fb (sf-fused fused)))
    (test "the negate is fused into the add" 1 (length fb))
    (test-assert "a producer read twice by the fused body stays a binding"
      (any (lambda (b) (eq? (ssa-binding-op b) 'exp)) (ssa-program-bindings fused)))
    (test "the fused binding reads exp(x) once" 1 (length (ssa-binding-inputs (car fb))))
    (test "the values are unchanged" (sf-run prog) (sf-run fused)))

  ;; mul(e, e): one consumer reading e twice.
  (let* ((x  (sf-var sf-x '(2 3)))
         (ex (morph-exp (am:var-value x)))
         (prog (morphism-to-ssa (am:make-var (morph* ex ex) #f))))
    (test "a producer read twice by its consumer is not fused" '()
      (sf-fused (ssa-segop-fusion-pass prog))))

  ;; A matmul consumer is a barrier.
  (let* ((x (sf-var sf-x '(2 3)))
         (W (sf-var '(1.0 0.0 0.5 2.0 -1.0 1.0) '(3 2)))
         (prog (morphism-to-ssa (am:make-var (morph-matmul (morph-exp (am:var-value x))
                                                           (am:var-value W))
                                             #f))))
    (test "a matmul operand is not fused" '() (sf-fused (ssa-segop-fusion-pass prog))))

  ;; e = exp(x) is consumed by a product after a matmul that does not read
  ;; it: the two need not be adjacent to fuse.
  (let* ((x (sf-var sf-x '(2 3)))
         (z (sf-var '(1.0 -1.0 0.5 0.25 2.0 -0.5) '(2 3)))
         (W (sf-var '(1.0 0.0 0.5 0.0 1.0 -1.0 2.0 0.5 1.0) '(3 3)))
         (prog (morphism-to-ssa (am:make-var (morph* (morph-exp (am:var-value x))
                                                     (morph-matmul (am:var-value z) (am:var-value W)))
                                             #f)))
         (fused (ssa-segop-fusion-pass prog)))
    (test "exp(x) is fused into the product past the matmul" 1 (length (sf-fused fused)))
    (let-values (((r checks mismatches) (with-segop-checker (lambda () (sf-run fused)))))
      (test-assert "non-adjacent fusion keeps the values" (equal? (sf-run prog) r))
      (test "the fused executors agree with the interpreter" 0 mismatches)))

  ;; x*y + x reads x once.
  (let* ((x (sf-var sf-x '(2 3)))
         (y (sf-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(2 3)))
         (prog (morphism-to-ssa (am:make-var (morph+ (morph* (am:var-value x) (am:var-value y))
                                                     (am:var-value x))
                                             #f)))
         (fused (ssa-segop-fusion-pass prog)))
    (test "x*y + x is one binding" 1 (length (ssa-program-bindings fused)))
    (test "with two inputs" 2 (length (ssa-binding-inputs (car (ssa-program-bindings fused)))))
    (test "and the same values" (sf-run prog) (sf-run fused)))

  ;; A chain of five unary ops.
  (let* ((x (sf-var sf-x '(2 3)))
         (prog (morphism-to-ssa
                (am:make-var (morph-negate (morph-exp (morph-sin (morph-cos (morph-abs (am:var-value x))))))
                             #f))))
    (let ((fused (ssa-segop-fusion-pass prog)))
      (test "a chain of five collapses into one binding" 1 (length (ssa-program-bindings fused)))
      (test "in one round, confirmed by a second"
        '((rounds . 2) (fusions . 4) (bound-reached? . #f)) (take (ssa-fusion-pass-stats) 3))
      (test "the chain keeps its values" (sf-run prog) (sf-run fused)))
    (let ((fused (parameterize ((segop-fusion-max-rounds 1)) (ssa-segop-fusion-pass prog))))
      (test-assert "a bound of one round is reported as reached"
        (cdr (assq 'bound-reached? (ssa-fusion-pass-stats))))
      (test "the values are still correct" (sf-run prog) (sf-run fused))))

  ;; Allocation counts and pool size on the two-layer network.
  (let* ((run (lambda (fusion?)
                (parameterize ((segop-fusion (and fusion? 'all)))
                  (let-values (((loss params) (seg-mlp-loss 'f32)))
                    (let ((ctx (make-morphism-context)))
                      (let-values (((joint trace) (compile-and-realize/ctx ctx loss params)))
                        ;; the number of values the program computes
                        (let ((n-fused (length (append-map binding-out-names (ssa-program-bindings joint))))
                              (allocs (cdr (assq 'allocations (context-stats ctx)))))
                          (finalize-context! ctx)
                          (reset-context! ctx)
                          (let ((replay (ssa-realize/ctx ctx joint)))
                            (list allocs n-fused
                                  (cdr (assq 'bytes (context-stats ctx)))
                                  (map get-allocation-id trace)
                                  (map get-allocation-id replay))))))))))
         (on (run #t)) (off (run #f)))
    (test "each value fused away saves one trace allocation"
      (- (car off) (- (cadr off) (cadr on))) (car on))
    (test "replay outputs keep their trace allocation ids" (list-ref on 3) (list-ref on 4))
    ;; the greedy allocator does not guarantee a smaller pool for fewer
    ;; allocations on every graph; the size is reported for benchmarks
    (test-assert "context-stats reports the pool size in bytes"
      (and (> (caddr on) 0) (> (caddr off) 0))))

  ;; A fused binding with two outputs, built by hand: a = exp(x) and
  ;; b = sin(x); d = b * a; the program outputs are d and b.
  (let* ((xm  (morph-from-list sf-x #(2 3) 'f64))
         (cid (gensym 'cid-))
         (consts (let ((h (make-hash-table))) (hash-table-set! h cid xm) h))
         (fb  (make-fused-ssa-binding (list (const-ref cid)) #(2 3) (list (identity-index-map 2))
                                      '((bid-a #(2 3) f64 (exp (in 0)))
                                        (bid-b #(2 3) f64 (sin (in 0))))
                                      '(bid-a bid-b)))
         (db  (make-ssa-binding 'bid-d 'mul (list (binding-ref 'bid-b) (binding-ref 'bid-a))
                                #(2 3) 'f64 `((combiner . ,*))))
         (mk  (lambda () (make-ssa-program consts #f (list fb db)
                                           (list (binding-ref 'bid-d) (binding-ref 'bid-b))
                                           0 #f #f #f)))
         (expect (list (map (lambda (v) (* (sin v) (exp v))) sf-x) (map sin sf-x)))
         (prog (mk))
         (ctx  (make-morphism-context)))
    (let-values (((r checks mismatches)
                  (with-segop-checker
                   (lambda ()
                     (let ((trace (ssa-realize/ctx ctx prog)))
                       (let ((n-allocs (context-counter ctx)))
                         (finalize-context! ctx)
                         (reset-context! ctx)
                         (let* ((r1 (map concrete->list (ssa-realize/ctx ctx prog)))
                                (_  (reset-context! ctx))
                                (r2 (map concrete->list (ssa-realize/ctx ctx prog)))
                                (_  (reset-context! ctx))
                                ;; a program without trace-info runs the
                                ;; binding loop on the finalized context
                                (r3 (map concrete->list (ssa-realize/ctx ctx (mk)))))
                           (list (map concrete->list trace) r1 r2 r3 n-allocs))))))))
      (test "ssa-realize computes both outputs" expect (map concrete->list (ssa-realize prog)))
      (test "the trace computes both outputs" expect (car r))
      (test "replay reproduces the trace" expect (cadr r))
      (test "a second replay too" expect (caddr r))
      (test "the binding loop on a finalized context too" expect (cadddr r))
      (test "one allocation per output, and one for d" 3 (list-ref r 4))
      (test "the executors agree with the interpreter" 0 mismatches)
      (test-assert "the checker compared the two-output segop" (> checks 0)))
    (let* ((plan (ssa-program-replay-plan prog))
           (asg  (buffer-pool-assignment (morphism-context-pool ctx))))
      (test "the fused binding compiles to a segop with one extra output" 1
        (cases replay-instruction (vector-ref plan 0)
          (ri-segop (_ _ _ _ _ _ _ extra-outs) (length extra-outs))
          (else #f)))
      (test-assert "its outputs have distinct pool buffers"
        (not (= (vector-ref asg 0) (vector-ref asg 1))))
      (test-assert "the extra output, a program output, is not reused by d"
        (not (= (vector-ref asg 1) (vector-ref asg 2)))))))
)

;;;; ============================================================
;;;; Fused-kernel catalogues
;;;;
;;;; Under the default policy a fusion is made when a segop backend reports
;;;; that it would run the fused segop natively.  The registry backend
;;;; answers from the fused kernels of the active activation backend.  The
;;;; mock catalogue below holds Scheme kernels for y * exp(x), which count
;;;; their calls.
;;;; ============================================================

(define fk-calls 0)

;; A fused kernel computing canonical body over n inputs on dtype arrays,
;; storing as the flat executors do.
(define (fk-kernel body n dtype)
  (let ((proc (scalar-expr->procedure body n)))
    (lambda (size ins out)
      (set! fk-calls (+ fk-calls 1))
      (do ((i 0 (+ i 1))) ((= i size))
        (typed-vector-set! out dtype i
                           (exact->inexact
                            (apply proc (map (lambda (v) (typed-vector-ref v dtype i)) ins))))))))

(define fk-f32-body '(mul (in 0) (cast f32 (exp (in 1)))))
(define fk-f64-body '(mul (in 0) (exp (in 1))))

(define (make-mock-catalogue dtypes)
  (let ((be (make-activation-backend 'mock-catalogue)))
    (when (memq 'f32 dtypes) (activation-backend-add-fused-kernel! be fk-f32-body 'f32 (fk-kernel fk-f32-body 2 'f32)))
    (when (memq 'f64 dtypes) (activation-backend-add-fused-kernel! be fk-f64-body 'f64 (fk-kernel fk-f64-body 2 'f64)))
    be))

;; The forward program of m.
(define (fk-prog m) (morphism-to-ssa (am:make-var m #f)))

(define (fk-arr lst shape dtype) (morph-from-list lst (list->vector shape) dtype))

(define fk-x '(0.5 -1.25 2.0 0.75 -0.5 1.5))
(define fk-y '(1.0 -2.0 3.0 0.25 4.0 -0.75))

;; (values fused-program results tags fused-placement) for the forward
;; program of m under the default policy with catalogue be active.
(define (fk-run m be)
  (register-activation-backend! be)
  (let* ((prog  (fk-prog m))
         (fused (ssa-segop-fusion-pass prog))
         (res   (sf-run fused))
         (tags  (map car (plan-segops fused)))
         (place (cdr (assq 'fused (replay-plan-stats fused)))))
    (register-activation-backend! #f)
    (values fused res tags place)))

(test-group "fused-kernel catalogues"
  (for-each
   (lambda (dtype)
     (let ((x (fk-arr fk-x '(2 3) dtype)) (y (fk-arr fk-y '(2 3) dtype)))
       (let-values (((fused res tags place) (fk-run (morph* y (morph-exp x)) #f)))
         (test (sprintf "~A: without a catalogue the default fuses nothing" dtype) '()
           (sf-fused fused)))
       (set! fk-calls 0)
       (let-values (((fused res tags place) (fk-run (morph* y (morph-exp x)) (make-mock-catalogue '(f32 f64)))))
         (test (sprintf "~A: with the catalogue, y * exp(x) is fused" dtype) 1 (length (sf-fused fused)))
         (test-assert (sprintf "~A: and runs as the fused kernel" dtype) (memq 'segop/fused-kernel tags))
         (test-assert (sprintf "~A: the kernel was called" dtype) (> fk-calls 0))
         (test (sprintf "~A: the plan counts it as native" dtype) '((native . 1) (host . 0)) (values place))
         (test (sprintf "~A: the values are those of the unfused program" dtype)
           (sf-run (fk-prog (morph* y (morph-exp x)))) (values res)))
       (let-values (((fused res tags place) (fk-run (morph* (morph-exp x) y) (make-mock-catalogue '(f32 f64)))))
         (test-assert (sprintf "~A: exp(x) * y matches the same entry" dtype)
           (memq 'segop/fused-kernel tags))
         (test (sprintf "~A: with the same values" dtype)
           (sf-run (fk-prog (morph* (morph-exp x) y))) (values res)))
       (let-values (((fused res tags place) (fk-run (morph* y (morph-negate x)) (make-mock-catalogue '(f32 f64)))))
         (test (sprintf "~A: a body without an entry is not fused" dtype) '() (sf-fused fused)))))
   '(f32 f64))

  (let ((x (fk-arr fk-x '(2 3) 'f64)) (y (fk-arr fk-y '(2 3) 'f64)))
    (let-values (((fused res tags place) (fk-run (morph* y (morph-exp x)) (make-mock-catalogue '(f32)))))
      (test "an f32 entry is not used for f64" '() (sf-fused fused))))

  ;; y is a transposed view: its layout is not flat, so the fused segop
  ;; could not run as the kernel.
  (let* ((x  (fk-arr fk-x '(2 3) 'f32))
         (yt (morph-transpose (fk-arr fk-y '(3 2) 'f32) '(1 0))))
    (let-values (((fused res tags place) (fk-run (morph* yt (morph-exp x)) (make-mock-catalogue '(f32)))))
      (test "an input that is a view is not expected flat, so nothing is fused" '() (sf-fused fused))))

  ;; y is a reshape of a flat value, which keeps its row-major layout.
  (let* ((x  (fk-arr fk-x '(2 3) 'f32))
         (yr (morph-reshape (fk-arr fk-y '(3 2) 'f32) (vector 2 3)))
         (m  (lambda () (morph* yr (morph-exp x)))))
    (let-values (((fused res tags place) (fk-run (m) (make-mock-catalogue '(f32)))))
      (test "an input that is a reshape of a flat value is fused" 1 (length (sf-fused fused)))
      (test "and runs as the fused kernel" '((native . 1) (host . 0)) (values place))
      (test "with the values of the unfused program" (sf-run (fk-prog (m))) (values res))))

  ;; The pairs left unfused, by reason.
  (let* ((x (fk-arr fk-x '(2 3) 'f32)) (y (fk-arr fk-y '(2 3) 'f32))
         (b (fk-arr '(0.5 -1.0 2.0) '(1 3) 'f32))
         (missed (lambda (m policy)
                   (parameterize ((segop-fusion policy))
                     (ssa-segop-fusion-pass (fk-prog m)))
                   (cdr (assq 'missed (ssa-fusion-pass-stats))))))
    (test "a pair with no native backend is missed as not-native" '((not-native . 1))
      (missed (morph* y (morph-exp x)) #t))
    (test "under all it is fused" '() (missed (morph* y (morph-exp x)) 'all))
    (test "a fusion reading a broadcast input is missed as broadcast-input"
      '((not-native . 1) (broadcast-input . 1))
      (missed (morph* y (morph-exp (morph+ x b))) #t))
    (test "under all it is fused" '() (missed (morph* y (morph-exp (morph+ x b))) 'all))
    (test "a producer read through a view is missed as view" '((view . 1))
      (missed (morph* y (morph-reshape (morph-exp (fk-arr fk-x '(3 2) 'f32)) (vector 2 3))) 'all)))

  ;; The catalogue is consulted again when the step runs: without it, the
  ;; fused segop runs on the host and computes the same values.
  (let* ((x (fk-arr fk-x '(2 3) 'f32)) (y (fk-arr fk-y '(2 3) 'f32))
         (m (morph* y (morph-exp x)))
         (be (make-mock-catalogue '(f32))))
    (register-activation-backend! be)
    (let* ((fused (ssa-segop-fusion-pass (fk-prog m)))
           (ctx   (make-morphism-context)))
      (ssa-realize/ctx ctx fused)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((with-kernel (map concrete->list (ssa-realize/ctx ctx fused))))
        (register-activation-backend! #f)
        (reset-context! ctx)
        (test "a missing kernel at run time falls back to the host, with the same values"
          with-kernel (map concrete->list (ssa-realize/ctx ctx fused))))))

  ;; The executors agree with the interpreter, and the candidates are
  ;; reported.
  (let* ((x (fk-arr fk-x '(2 3) 'f32)) (y (fk-arr fk-y '(2 3) 'f32)))
    (let-values (((r checks mismatches)
                  (with-segop-checker
                   (lambda ()
                     (let-values (((fused res tags place)
                                   (fk-run (morph* y (morph-exp x)) (make-mock-catalogue '(f32)))))
                       tags)))))
      (test-assert "the checker saw the fused kernel" (memq 'segop/fused-kernel r))
      (test "the fused kernel agrees with the interpreter" 0 mismatches))
    (parameterize ((segop-fusion 'all))
      (ssa-segop-fusion-pass (fk-prog (morph* y (morph-negate (morph-exp x))))))
    (test "the pass reports the bodies it formed"
      '(((mul (in 0) (cast f32 (negate (cast f32 (exp (in 1)))))) f32 1 6))
      (cdr (assq 'candidates (ssa-fusion-pass-stats))))
    (parameterize ((segop-fusion 'all))
      (let ((fused (ssa-segop-fusion-pass (fk-prog (morph* y (morph-negate x))))))
        (sf-run fused)
        (test "a fused binding without a kernel is counted as host" '((native . 0) (host . 1))
          (cdr (assq 'fused (replay-plan-stats fused))))))))

;;;; ============================================================
;;;; Broadcast inputs
;;;;
;;;; A producer or a consumer may read an input that broadcasts to its
;;;; shape; the fused segop reads it through its broadcast map.
;;;; ============================================================

(test-group "fusion with broadcast inputs"
  (for-each
   (lambda (dtype)
     (let ((x (fk-arr fk-x '(2 3) dtype)) (y (fk-arr fk-y '(2 3) dtype)))
       (for-each
        (lambda (name b)
          (let* ((m (lambda () (morph* y (morph-exp (morph- x b)))))
                 (fused (parameterize ((segop-fusion 'all)) (ssa-segop-fusion-pass (fk-prog (m))))))
            (let-values (((r checks mismatches) (with-segop-checker (lambda () (sf-run fused)))))
              (test (sprintf "~A, ~A input: the chain is one fused binding" dtype name)
                1 (length (sf-fused fused)))
              (test (sprintf "~A, ~A input: the values are those of the unfused program" dtype name)
                (sf-run (fk-prog (m))) (values r))
              (test (sprintf "~A, ~A input: the executors agree with the interpreter" dtype name)
                0 mismatches)
              (test-assert (sprintf "~A, ~A input: not run by the interpreter" dtype name)
                (not (memq 'segop/interpreted (map car (plan-segops fused))))))))
        '("scalar" "row" "column" "rank-1")
        (list (fk-arr '(0.25) '(1 1) dtype)
              (fk-arr '(0.5 -1.0 2.0) '(1 3) dtype)
              (fk-arr '(1.5 -0.25) '(2 1) dtype)
              (fk-arr '(0.5 -1.0 2.0) '(3) dtype)))))
   '(f32 f64))

  ;; A broadcast read of the producer itself is not fused.
  (let* ((x (fk-arr '(0.5 -1.25 2.0) '(1 3) 'f32)) (y (fk-arr fk-y '(2 3) 'f32))
         (fused (parameterize ((segop-fusion 'all))
                  (ssa-segop-fusion-pass (fk-prog (morph* y (morph-exp x)))))))
    (test "a producer read through a broadcast is not fused" '() (sf-fused fused))))

;;;; ============================================================
;;;; Forward programs
;;;;
;;;; A program from morphism-to-ssa is fused by ssa-realize/ctx at its first
;;;; trace, in place, under the policy and the backends in effect then.
;;;; ============================================================

;; The results of the trace and two replays of prog through a fresh
;; context, prog itself being realized.
(define (fp-run prog)
  (let* ((ctx (make-morphism-context))
         (trace (map concrete->list (ssa-realize/ctx ctx prog))))
    (finalize-context! ctx)
    (reset-context! ctx)
    (let ((r1 (map concrete->list (ssa-realize/ctx ctx prog))))
      (reset-context! ctx)
      (list trace r1 (map concrete->list (ssa-realize/ctx ctx prog))))))

(test-group "forward programs"
  (let* ((x (fk-arr fk-x '(2 3) 'f32)) (y (fk-arr fk-y '(2 3) 'f32))
         (m (lambda () (morph* y (morph-negate (morph-exp x)))))
         (expect (map concrete->list (ssa-realize (fk-prog (m))))))
    (let ((prog (fk-prog (m))))
      (test-assert "a new forward program is not marked fused" (not (ssa-program-fused? prog)))
      (let ((r (parameterize ((segop-fusion 'all)) (fp-run prog))))
        (test-assert "it is marked fused after the first trace" (ssa-program-fused? prog))
        (test "its bindings were fused in place" 1 (length (sf-fused prog)))
        (test "the trace and the replays give the unfused values"
          (list expect expect expect) (values r))
        (let ((bs (ssa-program-bindings prog)))
          (parameterize ((segop-fusion 'all)) (fp-run prog))
          (test-assert "a second trace does not fuse again" (eq? bs (ssa-program-bindings prog))))))
    (let ((prog (fk-prog (m))))
      (let ((bs (ssa-program-bindings prog)))
        (parameterize ((segop-fusion #f)) (fp-run prog))
        (test-assert "with fusion off the bindings are kept" (eq? bs (ssa-program-bindings prog)))
        (test-assert "and the program is marked" (ssa-program-fused? prog))))
    (let ((prog (fk-prog (m))))
      (fp-run prog)
      (test "without a native backend the default fuses nothing" '() (sf-fused prog))))

  ;; With the mock catalogue active at the first trace, the default fuses
  ;; y * exp(x) and runs it as the fused kernel.
  (let* ((x (fk-arr fk-x '(2 3) 'f32)) (y (fk-arr fk-y '(2 3) 'f32))
         (prog (fk-prog (morph* y (morph-exp x)))))
    (register-activation-backend! (make-mock-catalogue '(f32)))
    (let ((r (fp-run prog)))
      (test "the default fuses the catalogue pattern at the first trace" 1 (length (sf-fused prog)))
      (test "it runs natively" '((native . 1) (host . 0))
        (cdr (assq 'fused (replay-plan-stats prog))))
      (register-activation-backend! #f)
      (test "with the unfused values"
        (let ((v (map concrete->list (ssa-realize (fk-prog (morph* y (morph-exp x))))))) (list v v v))
        (values r))))

  ;; Joint programs are fused by ssa-vjp, and differentiating a fused
  ;; forward program is refused.
  (let* ((x (fk-arr fk-x '(2 3) 'f64)) (y (fk-arr fk-y '(2 3) 'f64))
         (prog (fk-prog (morph-reduce 'sum (morph* y (morph-negate (morph-exp x))) '(0 1) #f))))
    (let* ((fwd (fk-prog (morph-reduce 'sum (morph* y (morph-negate (morph-exp x))) '(0 1) #f)))
           (joint (ssa-vjp fwd '() (ssa-loss-binding-val fwd))))
      (test-assert "a joint program from ssa-vjp is marked fused" (ssa-program-fused? joint)))
    (parameterize ((segop-fusion 'all)) (fp-run prog))
    (test-error "ssa-vjp refuses a forward program with fused bindings"
      (ssa-vjp prog '() (ssa-loss-binding-val prog)))))

;;;; ============================================================
;;;; Fusion with reductions
;;;;
;;;; Maps fuse into the reductions that read them (with the map kept as an
;;;; extra output when it has other uses), and maps that read a
;;;; reduction's result fuse into it as its finalize.  Under the policy
;;;; all, every program gives the values of its unfused form, bit for bit,
;;;; from ssa-realize, the trace and the replays.
;;;; ============================================================

;; The fused reductions of prog other than GEMM epilogues.
(define (rd-fused-reds prog)
  (filter (lambda (b) (and (fused-red-binding? b) (not (gemm-epilogue-binding? b))))
          (ssa-program-bindings prog)))

;; The names of the outputs of binding b, its own first.
(define fused-outputs binding-out-names)

(define (rd-position prog name)
  (list-index (lambda (b) (memq name (binding-out-names b))) (ssa-program-bindings prog)))

;; Fuses the forward program of m under all (with at most rounds rounds),
;; and returns the fused program, the pass statistics, and whether its
;; values, run with the checking backend, are those of the unfused
;; program, with no executor disagreeing with the interpreter.
(define (rd-check m #!optional (rounds 16))
  (let* ((fused (parameterize ((segop-fusion 'all) (segop-fusion-max-rounds rounds))
                  (ssa-segop-fusion-pass (fk-prog m))))
         (stats (ssa-fusion-pass-stats)))
    (let-values (((r checks mismatches) (with-segop-checker (lambda () (sf-run fused)))))
      (values fused stats (and (rd-same? r (sf-run (fk-prog m))) (= mismatches 0))))))

;; True when a and b, nested lists of numbers, are equal, NaNs included.
(define (rd-same? a b)
  (cond ((and (pair? a) (pair? b)) (and (rd-same? (car a) (car b)) (rd-same? (cdr a) (cdr b))))
        ((and (number? a) (number? b)) (seg-same? a b))
        (else (equal? a b))))

;; The numbers of a nested list, in order.
(define (rd-flatten x)
  (cond ((null? x) '())
        ((pair? x) (append (rd-flatten (car x)) (rd-flatten (cdr x))))
        (else (list x))))

(define (rd-missed stats) (cdr (assq 'missed stats)))

(define rd-a '(0.5 -1.25 2.0 0.75 -0.5 1.5 0.1 -3.0 2.5 0.0 -0.0 1e-3))
(define rd-b '(1.0 -2.0 3.0 0.25 4.0 -0.75 0.3 5.0 -1.5 2.0 7.0 -0.2))

(test-group "fusion with reductions"
  (for-each
   (lambda (dtype)
     (let ((x (fk-arr rd-a '(4 3) dtype)) (y (fk-arr rd-b '(4 3) dtype)))
       (for-each
        (lambda (case)
          (let-values (((fused stats same?) (rd-check ((cadr case)))))
            (test (sprintf "~A, ~A: one fused reduction" dtype (car case))
              1 (length (rd-fused-reds fused)))
            (test-assert (sprintf "~A, ~A: the values of the unfused program" dtype (car case)) same?)))
        (list (list "sum over axis 0 of x*y" (lambda () (morph-reduce 'sum (morph* x y) '(0) #f)))
              (list "keepdims mean over axis 1 of exp(x)"
                    (lambda () (morph-reduce 'mean (morph-exp x) '(1) #t)))
              (list "max over axis 1 of x - y" (lambda () (morph-reduce 'max (morph- x y) '(1) #f)))
              (list "min over axis 0 of y / x" (lambda () (morph-reduce 'min (morph/ y x) '(0) #t)))
              (list "sum of x + a broadcast row"
                    (lambda () (morph-reduce 'sum (morph+ x (fk-arr '(0.5 -1.0 2.0) '(1 3) dtype)) '(0) #f)))
              (list "sum of a chain of maps" (lambda () (morph-reduce 'sum (morph* y (morph-exp (morph-negate x))) '(1) #f)))
              (list "everything summed to a scalar" (lambda () (morph-reduce 'sum (morph* x x) '(0 1) #f)))))))
   '(f32 f64))

  ;; NaN, infinities and -0.0 through max, min and sum, with the values
  ;; doubled by a product.
  (let ((x (fk-arr (list +nan.0 1.0 -0.0 0.0 +inf.0 -inf.0 2.0 +nan.0 -1.0 0.5 -0.0 3.0) '(4 3) 'f64))
        (two (fk-arr '(2.0) '(1 1) 'f64)))
    (for-each
     (lambda (rop)
       (let-values (((fused stats same?) (rd-check (morph-reduce rop (morph* x two) '(1) #f))))
         (test-assert (sprintf "~A of values with NaN, infinities and -0.0" rop) same?)))
     '(max min sum)))

  ;; -0.0 + -0.0 is -0.0 in the fused body as in the unfused loops, so a
  ;; min over sums of negative zeros keeps the sign.  No row holds 0.0,
  ;; since min may return either zero.
  (for-each
   (lambda (dtype)
     (let ((x (fk-arr '(-0.0 -0.0 -0.0 -0.0 1.0 2.0) '(2 3) dtype)))
       (let-values (((fused stats same?) (rd-check (morph-reduce 'min (morph+ x x) '(1) #f))))
         (test (sprintf "~A, min of x + x with -0.0: one fused reduction" dtype)
           1 (length (rd-fused-reds fused)))
         (test-assert (sprintf "~A, min of x + x with -0.0: the values of the unfused program" dtype)
           same?)
         (test-assert (sprintf "~A, min of x + x with -0.0 is -0.0 in each row" dtype)
           (every (lambda (v) (eqv? v -0.0)) (rd-flatten (sf-run fused)))))
       ;; The same for a fused element-wise add: negate(y) + x with y = -x.
       (let* ((y (fk-arr '(0.0 0.0 0.0 0.0 -1.0 -2.0) '(2 3) dtype))
              (m (morph+ (morph-negate y) x)))
         (let-values (((fused stats same?) (rd-check m)))
           (test-assert (sprintf "~A, negate(y) + x with -0.0 is fused" dtype)
             (pair? (fused-bindings fused)))
           (test-assert (sprintf "~A, negate(y) + x with -0.0: the values of the unfused program" dtype)
             same?)
           (test-assert (sprintf "~A, negate(y) + x keeps -0.0 + -0.0 as -0.0" dtype)
             (every (lambda (run)
                      (every eqv? '(-0.0 -0.0 -0.0 -0.0 2.0 4.0) (rd-flatten run)))
                    (sf-run fused)))))))
   '(f32 f64))

  ;; The general path, over axes 0 and 2 of a rank-3 value.
  (let-values (((fused stats same?)
                (rd-check (morph-reduce 'min (morph-exp (fk-arr (map (lambda (i) (* 0.1 i)) (iota 24)) '(2 3 4) 'f64))
                                        '(0 2) #f))))
    (test "a reduction over two axes of a rank-3 value is fused" 1 (length (rd-fused-reds fused)))
    (test-assert "with the values of the unfused program" same?))

  ;; f32 values into a reduction accumulated in f64 (axis 1), and s32 sums.
  (let-values (((fused stats same?)
                (rd-check (morph-reduce 'sum (morph* (fk-arr rd-a '(4 3) 'f32) (fk-arr rd-b '(4 3) 'f32)) '(1) #f))))
    (test-assert "an f32 product summed in f64 keeps its value" same?))
  (let-values (((fused stats same?)
                (rd-check (morph-reduce 'sum (morph* (fk-arr '(1 2 3 4 5 6) '(2 3) 's32)
                                                     (fk-arr '(7 -8 9 1 2 3) '(2 3) 's32))
                                        '(1) #f))))
    (test "an s32 product sum is fused" 1 (length (rd-fused-reds fused)))
    (test-assert "with the values of the unfused program" same?))

  ;; The 1d sum may use the BLAS dot product, whose order is not known.
  (let ((v (fk-arr '(1.0 2.0 3.0) '(3) 'f64)) (w (fk-arr '(4.0 5.0 6.0) '(3) 'f64)))
    (let-values (((fused stats same?) (rd-check (morph-reduce 'sum (morph* v w) '(0) #f))))
      (test "a 1d sum is not fused" '() (rd-fused-reds fused))
      (test "and is reported as reduction-order" 1 (alist-ref 'reduction-order (rd-missed stats)))
      (test-assert "its values are unchanged" same?))
    (let-values (((fused stats same?) (rd-check (morph-reduce 'max (morph* v w) '(0) #f))))
      (test "a 1d max, accumulated in order, is fused" 1 (length (rd-fused-reds fused)))
      (test-assert "with the values of the unfused program" same?)))

  ;; A reduction of a reshape of P reads a view: P is not fused into it.
  (let-values (((fused stats same?)
                (rd-check (morph-reduce 'sum (morph-reshape (morph* (fk-arr rd-a '(4 3) 'f64) (fk-arr rd-b '(4 3) 'f64))
                                                            (vector 3 4))
                                        '(0) #f))))
    (test "a reduction of a reshape of P is not fused" '() (rd-fused-reds fused))
    (test-assert "its values are unchanged" same?))

  ;; P read by the reduction and by a later map is kept as a map output,
  ;; computed at P's position.
  (let* ((x (fk-arr rd-a '(4 3) 'f64)) (y (fk-arr rd-b '(4 3) 'f64)))
    (let-values (((fused stats same?)
                  (rd-check (let ((p (morph* x y))) (morph+ (morph-reduce 'sum p '(0) #t) p)))))
      (let ((f (car (rd-fused-reds fused))))
        (test "a kept producer is a second output" 2 (length (fused-outputs f)))
        (test-assert "the later map reads the kept output"
          (any (lambda (b) (any (lambda (v) (and (ssa-binding-ref? v) (eq? (ssa-value-id v) (cadr (fused-outputs f)))))
                                (ssa-binding-inputs b)))
               (ssa-program-bindings fused)))
        (test-assert "the values are those of the unfused program" same?))))

  ;; Epilogues.
  (let ((x (fk-arr rd-a '(4 3) 'f32)) (w (fk-arr rd-b '(4 3) 'f32)))
    (let-values (((fused stats same?)
                  (rd-check (morph/ (morph-reduce 'sum (morph* x x) '(1) #t) (fk-arr '(4.0) '(1 1) 'f32)))))
      (test "a map of a keepdims sum is its finalize" 1 (length (ssa-program-bindings fused)))
      (test-assert "with the values of the unfused program" same?))
    ;; rmsnorm: x * (mean(x*x) + eps)^-0.5 * w
    (let-values (((fused stats same?)
                  (rd-check (morph* (morph* x (morph-pow (morph+ (morph-reduce 'mean (morph* x x) '(1) #t)
                                                                 (fk-arr '(1e-5) '(1 1) 'f32))
                                                         (fk-arr '(-0.5) '(1 1) 'f32)))
                                    w))))
      (let ((f (car (rd-fused-reds fused))))
        (test "rmsnorm's mean, add and pow are one fused reduction" 4 (length (sf-members f)))
        (test-assert "with the values of the unfused program" same?)))
    ;; A mean rounded to f32 before the map reads it.
    (let-values (((fused stats same?)
                  (rd-check (morph+ (morph-reduce 'mean (morph* x w) '(1) #t) (fk-arr '(0.3) '(1 1) 'f32)))))
      (test-assert "a finalize reads the mean as stored in f32" same?)))

  (let ((x (fk-arr rd-a '(4 3) 'f64)))
    ;; x - max(x) reads the reduction through a broadcast.
    (let-values (((fused stats same?)
                  (rd-check (morph- (morph-exp x) (morph-reduce 'max (morph-exp x) '(1) #t)))))
      (test-assert "a broadcast read of a reduction is not an epilogue"
        (every (lambda (b) (not (any (lambda (m) (eq? m (ssa-binding-name (car (reverse (ssa-program-bindings fused))))))
                                     (sf-members b))))
               (rd-fused-reds fused)))
      (test-assert "its values are unchanged" same?))
    ;; a reduction result read by two maps
    (let-values (((fused stats same?)
                  (rd-check (let ((s (morph-reduce 'sum (morph-exp x) '(1) #t))) (morph+ (morph-log s) (morph-exp s))) 1)))
      (test "a reduction with two readers is reported" 1 (alist-ref 'multi-use-reduction (rd-missed stats)))
      (test-assert "its values are unchanged" same?))
    ;; r = sum(P) keeps P, which u = exp(P) reads after it, so r is at P's
    ;; position, before t = sum(u): r * t cannot be r's finalize.
    (let-values (((fused stats same?)
                  (rd-check (let* ((p (morph* x x)) (r (morph-reduce 'sum p '(1) #t))
                                   (t (morph-reduce 'sum (morph-exp p) '(1) #t)))
                              (morph* r t))
                            1)))
      (test "an epilogue whose inputs come too late is reported" 1 (alist-ref 'placement (rd-missed stats)))
      (test-assert "its values are unchanged" same?))
    (let-values (((fused stats same?)
                  (rd-check (let* ((p (morph* x x)) (r (morph-reduce 'sum p '(1) #t))
                                   (t (morph-reduce 'sum (morph-exp p) '(1) #t)))
                              (morph* r t)))))
      (test-assert "the product fuses into the later reduction instead" same?)))

  ;; A forward program fused at its first trace.
  (let* ((x (fk-arr rd-a '(4 3) 'f32))
         (m (lambda () (morph-sqrt (morph-reduce 'mean (morph* x x) '(1) #t))))
         (expect (map concrete->list (ssa-realize (fk-prog (m)))))
         (prog (fk-prog (m)))
         (r (parameterize ((segop-fusion 'all)) (fp-run prog))))
    (test "a forward program fuses its reduction at the first trace" 1 (length (rd-fused-reds prog)))
    (test "the trace and the replays give the unfused values" (list expect expect expect) (values r)))

  ;; Under the default policy, with no backend running fused reductions
  ;; natively, nothing is fused into a reduction.
  (let ((fused (ssa-segop-fusion-pass (fk-prog (morph-reduce 'sum (morph* (fk-arr rd-a '(4 3) 'f32)
                                                                         (fk-arr rd-b '(4 3) 'f32))
                                                              '(0) #f)))))
    (test "the default fuses no reduction without a native kernel" '() (rd-fused-reds fused))
    (test-assert "and reports it as not native" (alist-ref 'not-native (rd-missed (ssa-fusion-pass-stats)))))
  (let ((fused (parameterize ((segop-fusion #f))
                 (ssa-segop-fusion-pass (fk-prog (morph-reduce 'sum (morph* (fk-arr rd-a '(4 3) 'f32)
                                                                            (fk-arr rd-b '(4 3) 'f32))
                                                               '(0) #f))))))
    (test "the policy #f does not change the program" 2 (length (ssa-program-bindings fused)))))

;; The joint program of a dense layer, loss sum(relu(x W + b) * c), whose
;; bias gradient sums g * relu'(z) over the batch, and the product's other
;; readers, the GEMMs, come before that sum.
(define (rd-dense-joint policy dtype)
  (let* ((x (seg-var (map (lambda (i) (- (* 0.37 i) 2.0)) (iota 12)) '(4 3) dtype #f))
         (W (seg-var (map (lambda (i) (* 0.21 (sin (+ i 1.0)))) (iota 15)) '(3 5) dtype #t))
         (b (seg-var (map (lambda (i) (- (* 0.1 i) 0.2)) (iota 5)) '(5) dtype #t))
         (c (seg-var (map (lambda (i) (cos (* 0.5 i))) (iota 20)) '(4 5) dtype #f))
         (loss (am:var-sum (am:var* (am:var-relu (am:var+ (am:var-matmul x W) b)) c)))
         (fwd (morphism-to-ssa loss))
         (pv (map (lambda (p) (ssa-constant-id fwd (am:var-value p))) (list W b))))
    (parameterize ((segop-fusion policy)) (ssa-vjp fwd pv (ssa-loss-binding-val fwd)))))

;; The results of the trace and of three replays of joint, and the pool's
;; bytes.
(define (rd-replays joint)
  (let* ((ctx (make-morphism-context))
         (trace (map concrete->list (ssa-realize/ctx ctx joint))))
    (finalize-context! ctx)
    (let loop ((k 0) (rs (list trace)))
      (if (= k 3)
          (values (reverse rs) (cdr (assq 'bytes (context-stats ctx))))
          (begin
            (reset-context! ctx)
            (loop (+ k 1) (cons (map concrete->list (ssa-realize/ctx ctx joint)) rs)))))))

(test-group "fusion of bias gradients"
  (for-each
   (lambda (dtype)
     (let* ((off (rd-dense-joint #f dtype))
            (all (rd-dense-joint 'all dtype))
            (kept (filter (lambda (f) (> (length (fused-outputs f)) 1)) (rd-fused-reds all))))
       (test (sprintf "~A: the bias gradient keeps g * relu' as a map output" dtype) 1 (length kept))
       (let* ((f (car kept)) (mo (cadr (fused-outputs f))))
         (test-assert (sprintf "~A: a GEMM reads the kept output after the fused reduction" dtype)
           (let ((fpos (rd-position all (ssa-binding-name f))))
             (any (lambda (b i) (and (eq? (ssa-binding-op b) 'matmul) (> i fpos)
                                     (any (lambda (v) (and (ssa-binding-ref? v) (eq? (ssa-value-id v) mo)))
                                          (ssa-binding-inputs b))))
                  (ssa-program-bindings all) (iota (length (ssa-program-bindings all))))))
         (test-assert (sprintf "~A: the bias gradient is computed by the fused reduction" dtype)
           (any (lambda (ov) (eq? (ssa-value-id ov) (ssa-binding-name f))) (ssa-program-outputs all))))
       (let-values (((r-all bytes-all) (rd-replays all))
                    ((r-off bytes-off) (rd-replays off)))
         (test (sprintf "~A: the trace and the replays give the unfused gradients" dtype) r-off r-all)
         (test-assert (sprintf "~A: the replays give the trace's values" dtype)
           (every (lambda (r) (equal? r (car r-all))) (cdr r-all)))
         (test-assert (sprintf "~A: the pool is not larger" dtype) (<= bytes-all bytes-off)))
       (let-values (((r checks mismatches)
                     (with-segop-checker
                      (lambda () (call-with-values (lambda () (rd-replays (rd-dense-joint 'all dtype))) list)))))
         (test (sprintf "~A: the executors agree with the interpreter" dtype) 0 mismatches))))
   '(f32 f64)))

;; A fused column-sum kernel computing canonical body over n inputs on
;; dtype arrays, adding the rows in order with the sum rounded to dtype.
(define rd-colsum-calls 0)

(define (rd-colsum-kernel body n dtype)
  (let ((proc (scalar-expr->procedure body n)))
    (lambda (rows cols ins mo out)
      (set! rd-colsum-calls (+ rd-colsum-calls 1))
      (do ((j 0 (+ j 1))) ((= j cols)) (typed-vector-set! out dtype j 0.0))
      (do ((i 0 (+ i 1))) ((= i rows))
        (do ((j 0 (+ j 1))) ((= j cols))
          (let ((k (+ (* i cols) j)))
            (typed-vector-set! mo dtype k
                               (exact->inexact (apply proc (map (lambda (v) (typed-vector-ref v dtype k)) ins))))
            (typed-vector-set! out dtype j (+ (typed-vector-ref out dtype j) (typed-vector-ref mo dtype k)))))))))

(define rd-relu-map-body '((f32 . (mul (in 0) (cast f32 (relu-deriv (in 1)))))
                           (f64 . (mul (in 0) (relu-deriv (in 1))))))
(define rd-relu-colsum-body '((f32 . (cast f32 (mul (in 0) (cast f32 (relu-deriv (in 1))))))
                              (f64 . (mul (in 0) (relu-deriv (in 1))))))

;; An activation backend with the fused g * relu'(z) kernel and its
;; column-sum kernel, for the given dtypes.
(define (rd-mock-catalogue dtypes)
  (let ((be (make-activation-backend 'mock-colsum)))
    (for-each (lambda (dt)
                (let ((mb (cdr (assq dt rd-relu-map-body))) (cb (cdr (assq dt rd-relu-colsum-body))))
                  (activation-backend-add-fused-kernel! be mb dt (fk-kernel mb 2 dt))
                  (activation-backend-add-colsum-kernel! be cb dt (rd-colsum-kernel cb 2 dt))))
              dtypes)
    be))

(test-group "fused column-sum kernels"
  (for-each
   (lambda (dtype)
     (let ((off (rd-dense-joint #f dtype)))
       (register-activation-backend! (rd-mock-catalogue '(f32 f64)))
       (set! rd-colsum-calls 0)
       (let ((on (rd-dense-joint #t dtype)))
         (let-values (((r-on bytes-on) (rd-replays on))
                      ((r-off bytes-off) (rd-replays off)))
           (let ((tags (map car (plan-segops on))))
             (test-assert (sprintf "~A: the default fuses the bias gradient into a column-sum kernel" dtype)
               (memq 'segop/colsum-kernel tags))
             (test-assert (sprintf "~A: the kernel was called" dtype) (> rd-colsum-calls 0))
             (test (sprintf "~A: the plan counts it as native, with nothing on the host" dtype)
               0 (cdr (assq 'host (cdr (assq 'fused (replay-plan-stats on))))))
             (test (sprintf "~A: the gradients are those of the unfused program" dtype) r-off r-on))))
       (register-activation-backend! #f)))
   '(f32 f64))
  (register-activation-backend! (rd-mock-catalogue '(f32)))
  (let ((on (rd-dense-joint #t 'f64)))
    (test "an f32 column-sum entry is not used for f64" '() (rd-fused-reds on)))
  (register-activation-backend! #f)
  ;; The kernel missing when the step runs: the host executor computes it.
  (let* ((be (rd-mock-catalogue '(f32)))
         (on (begin (register-activation-backend! be) (rd-dense-joint #t 'f32)))
         (off (rd-dense-joint #f 'f32)))
    (let-values (((r-on b1) (rd-replays on)))
      (register-activation-backend! (make-activation-backend 'empty))
      (let-values (((r checks mismatches)
                    (with-segop-checker (lambda () (call-with-values (lambda () (rd-replays on)) list))))
                   ((r-off b3) (rd-replays off)))
        (test "without its kernel at run time, the fused reduction still gives the unfused values"
          r-off (car r))
        (test "and agrees with the interpreter" 0 mismatches)))
    (register-activation-backend! #f)))


;;;; ============================================================
;;;; Multi-output and horizontal fusion
;;;;
;;;; A map with other uses is fused into a map that reads it, with its value
;;;; kept as a second output; maps of the same shape that read a common
;;;; input are fused into one seg-map; outputs left without uses are
;;;; removed.  Under the policy all, the values are those of the unfused
;;;; program, bit for bit.
;;;; ============================================================

(define mo-x '(0.5 -1.25 2.0 0.75 -0.5 1.5))
(define mo-y '(1.0 -2.0 3.0 0.25 4.0 -0.75))
(define mo-w '(0.3 -0.2 0.5 0.1 -0.4 0.6))

(define (mo-stat stats key) (cdr (assq key stats)))

;; The number of outputs of each fused seg-map of prog.
(define (mo-outputs prog)
  (map (lambda (b) (length (binding-out-names b)))
       (filter (lambda (b) (and (eq? (ssa-binding-op b) 'fused) (not (fused-red-binding? b))))
               (ssa-program-bindings prog))))

(test-group "multi-output seg-map executor"
  ;; Three outputs sharing exp(x), with x read transposed and b broadcast.
  (let* ((x  (e4-arr (e4-values 6 'f32) #(3 2) #(1 3) 'f32))
         (b  (e4-arr (f32vector 0.5 -1.0) #(1 2) #(2 1) 'f32))
         (e  '(cast f32 (exp (in 0))))
         (sg (make-seg-map #(3 2) (list (e4-op 'x x 'f32 (identity-index-map 2)) (e4-op 'b b 'f32 (e4-map 0 '(dim 1))))
                           (list (e4-out 'f32 #(3 2) (identity-index-map 2)) (e4-out 'f32 #(3 2) (identity-index-map 2))
                                 (e4-out 'f64 #(3 2) (identity-index-map 2)))
                           (list `(add ,e (in 1)) e `(mul ,e 2.0))))
         (r  (e4-run sg `((x . ,x) (b . ,b)) '((f32 6) (f32 6) (f64 6)))))
    (test "a seg-map with several outputs runs on the strided multi-output executor" 'segop/strided-multi-map (car r))
    (test-assert "its outputs equal the interpreter's, special values included" (cadr r)))
  ;; s32 outputs over a rank-0 space.
  (let* ((x  (e4-arr (e4-ints 1) #() #() 's32))
         (sg (make-seg-map #() (list (e4-op 'x x 's32 (identity-index-map 0)))
                           (list (e4-out 's32 #() (identity-index-map 0)) (e4-out 's32 #() (identity-index-map 0)))
                           (list '(mul (in 0) (in 0)) '(negate (in 0)))))
         (r  (e4-run sg `((x . ,x)) '((s32 1) (s32 1)))))
    (test "a rank-0 s32 seg-map with two outputs" 'segop/strided-multi-map (car r))
    (test-assert "equals the interpreter's" (cadr r)))
  ;; A value shared by both outputs is computed once per point.
  (let* ((calls 0)
         (f  (lambda (v) (set! calls (+ calls 1)) (* v 3.0)))
         (sh (list 'opaque f '(in 0)))
         (x  (e4-arr (e4-values 6 'f64) #(2 3) #(3 1) 'f64))
         (sg (make-seg-map #(2 3) (list (e4-op 'x x 'f64 (identity-index-map 2)))
                           (list (e4-out 'f64 #(2 3) (identity-index-map 2)) (e4-out 'f64 #(2 3) (identity-index-map 2)))
                           (list `(add ,sh 1.0) sh)))
         (ex (select-segop-executor sg)))
    ((cdr ex) (lambda (s) x) (make-f64vector 6) (list (make-f64vector 6)))
    (test "a shared value is computed once per point" 6 calls)))

(test-group "multi-output and horizontal fusion"
  (for-each
   (lambda (dtype)
     (let* ((x  (fk-arr mo-x '(2 3) dtype))
            (y  (fk-arr mo-y '(2 3) dtype))
            (W  (fk-arr mo-w '(3 2) dtype))
            (W2 (fk-arr (reverse mo-w) '(3 2) dtype))
            (A  (fk-arr '(1.0 0.5 -0.5 2.0) '(2 2) dtype))
            (Bm (fk-arr mo-y '(2 3) dtype)))
       ;; P = exp(x) read by C = P*y and by a GEMM: C's other input is a
       ;; constant, so the fused binding takes P's position.
       (let-values (((fused stats same?)
                     (rd-check (let ((p (morph-exp x)))
                                 (morph+ (morph-matmul (morph* p y) W2) (morph-matmul p W))))))
         (test (sprintf "~A: a producer read by a GEMM is kept as a second output" dtype) 1 (mo-stat stats 'kept))
         (test (sprintf "~A: one fused seg-map with two outputs" dtype) '(2) (mo-outputs fused))
         (test-assert (sprintf "~A: the values are those of the unfused program" dtype) same?))
       ;; A chain, exp(negate(x)), fused first and then kept.
       (let-values (((fused stats same?)
                     (rd-check (let ((p (morph-exp (morph-negate x))))
                                 (morph+ (morph-matmul (morph* p y) W2) (morph-matmul p W))))))
         (test (sprintf "~A: a fused producer is kept too" dtype) 1 (mo-stat stats 'kept))
         (test-assert (sprintf "~A: with the values of the unfused program" dtype) same?))
       ;; C = P*z with z a GEMM after P: the fused binding takes C's
       ;; position, since P's other use comes after C.  z has a second use,
       ;; so that C does not become z's epilogue.
       (let-values (((fused stats same?)
                     (rd-check (let ((p (morph-exp x)) (z (morph-matmul A Bm)))
                                 (morph+ (morph+ (morph-matmul (morph* p z) W2) (morph-matmul p W))
                                         (morph-matmul z W))))))
         (test (sprintf "~A: a kept producer may take the consumer's position" dtype) 1 (mo-stat stats 'kept))
         (test-assert (sprintf "~A: with the values of the unfused program" dtype) same?))
       ;; P's other use comes before C, and C's other input after P.  The
       ;; GEMMs reading P and z have second uses, so that neither takes the
       ;; adds as an epilogue and moves.
       (let-values (((fused stats same?)
                     (rd-check (let* ((p (morph-exp x)) (z (morph-matmul A Bm)) (g (morph-matmul p W)))
                                 (morph+ (morph+ (morph+ g (morph-matmul (morph* p z) W2))
                                                 (morph-matmul z W))
                                         g))
                               1)))
         (test (sprintf "~A: no position fits: the producer is not kept" dtype) 0 (mo-stat stats 'kept))
         (test-assert (sprintf "~A: and it is reported" dtype) (alist-ref 'placement (rd-missed stats)))
         (test-assert (sprintf "~A: the values are unchanged" dtype) same?))
       ;; Siblings exp(x) and sin(x), each read by a GEMM.
       (let-values (((fused stats same?)
                     (rd-check (morph+ (morph-matmul (morph-exp x) W) (morph-matmul (morph-sin x) W)))))
         (test (sprintf "~A: siblings that read x are fused horizontally" dtype) 1 (mo-stat stats 'horizontal))
         (test (sprintf "~A: into one seg-map with two outputs" dtype) '(2) (mo-outputs fused))
         (test-assert (sprintf "~A: with the values of the unfused program" dtype) same?))
       ;; Siblings read by the same later GEMM: the fused binding takes the
       ;; later sibling's position.
       (let-values (((fused stats same?)
                     (rd-check (morph-matmul (morph-exp x) (morph-transpose (morph-sin x) '(1 0))))))
         (test (sprintf "~A: siblings read by one GEMM are fused" dtype) 1 (mo-stat stats 'horizontal))
         (test-assert (sprintf "~A: with the values of the unfused program" dtype) same?))
       ;; b = x + matmul(exp(x), I) depends on a = exp(x) through the GEMM.
       (let-values (((fused stats same?)
                     (rd-check (let* ((a (morph-exp x))
                                      (b (morph+ x (morph-matmul a (fk-arr '(1 0 0 0 1 0 0 0 1) '(3 3) dtype)))))
                                 (morph+ (morph-matmul a W) (morph-matmul b W))))))
         (test (sprintf "~A: a sibling reachable from the other is not fused" dtype) 0 (mo-stat stats 'horizontal))
         (test-assert (sprintf "~A: the values are unchanged" dtype) same?))
       ;; Five siblings: no fused binding has more than four outputs.
       (let-values (((fused stats same?)
                     (rd-check (fold (lambda (f acc) (morph+ acc (morph-matmul (f x) W)))
                                     (morph-matmul (morph-exp x) W)
                                     (list morph-sin morph-cos morph-negate morph-abs)))))
         (test-assert (sprintf "~A: at most four outputs per fused binding" dtype)
           (and (pair? (mo-outputs fused)) (every (lambda (k) (<= k 4)) (mo-outputs fused))))
         (test-assert (sprintf "~A: with the values of the unfused program" dtype) same?))))
   '(f32 f64))
  ;; Siblings of different dtypes: an f64 cast of x and exp(x).
  (let* ((x (fk-arr mo-x '(2 3) 'f32))
         (W64 (fk-arr mo-w '(3 2) 'f64))
         (W32 (fk-arr mo-w '(3 2) 'f32)))
    (let-values (((fused stats same?)
                  (rd-check (morph+ (morph-cast (morph-matmul (morph-cast x 'f64) W64) 'f32)
                                    (morph-matmul (morph-exp x) W32)))))
      (test "siblings of different dtypes are fused" 1 (mo-stat stats 'horizontal))
      (test-assert "with the values of the unfused program" same?))))

;; The forward program of negate(exp(x)) with the exp binding replaced by a
;; fused binding that also computes sin(x), under the name extra; when
;; extra-output? is true, extra is also a program output.
(define (mo-dead-prog extra-output?)
  (let* ((x    (fk-arr mo-x '(2 3) 'f64))
         (prog (fk-prog (morph-negate (morph-exp x))))
         (e    (car (filter (lambda (b) (eq? (ssa-binding-op b) 'exp)) (ssa-program-bindings prog))))
         (f    (make-fused-ssa-binding (ssa-binding-inputs e) #(2 3) (list (identity-index-map 2))
                                       (list (list (ssa-binding-name e) #(2 3) 'f64 '(exp (in 0)))
                                             (list 'extra #(2 3) 'f64 '(sin (in 0))))
                                       (list (ssa-binding-name e)))))
    (make-ssa-program (ssa-program-constants prog) (ssa-program-morph-to-val prog)
                      (map (lambda (b) (if (eq? b e) f b)) (ssa-program-bindings prog))
                      (if extra-output?
                          (append (ssa-program-outputs prog) (list (binding-ref 'extra)))
                          (ssa-program-outputs prog))
                      0 #f #f #f #f)))

(test-group "dead-output removal"
  (let* ((prog  (mo-dead-prog #f))
         (fused (parameterize ((segop-fusion 'all)) (ssa-segop-fusion-pass prog)))
         (stats (ssa-fusion-pass-stats)))
    (test "an output without uses is removed" 1 (mo-stat stats 'dead-outputs))
    (test "the binding left with one output then fuses with its reader" 1 (length (ssa-program-bindings fused)))
    (test "the values are unchanged" (sf-run prog) (sf-run fused)))
  (let* ((prog  (mo-dead-prog #t))
         (fused (parameterize ((segop-fusion 'all)) (ssa-segop-fusion-pass prog)))
         (stats (ssa-fusion-pass-stats)))
    (test "an output that is a program output is kept" 0 (mo-stat stats 'dead-outputs))
    (test "both program outputs are computed" (sf-run prog) (sf-run fused)))
  (let ((fused (ssa-segop-fusion-pass (fk-prog (morph+ (morph-matmul (morph-exp (fk-arr mo-x '(2 3) 'f32))
                                                                     (fk-arr mo-w '(3 2) 'f32))
                                                       (morph-matmul (morph-sin (fk-arr mo-x '(2 3) 'f32))
                                                                     (fk-arr mo-w '(3 2) 'f32)))))))
    (test "the default policy makes no horizontal fusion without a native backend" 0
      (mo-stat (ssa-fusion-pass-stats) 'horizontal))))

;;;; ============================================================
;;;; GEMM epilogues
;;;;
;;;; Maps that read a GEMM's product are fused into the GEMM as its
;;;; epilogue, which runs in place on the stored product.  The default
;;;; policy fuses one op applied to the product, or to the product and a
;;;; row bias; under all, chains and other inputs are fused too.
;;;; ============================================================

;; The forward program of m fused under policy, or unfused under #f, as
;; ssa-realize/ctx would run it.
(define (ge-fused policy m)
  (if policy
      (parameterize ((segop-fusion policy)) (ssa-segop-fusion-pass (fk-prog m)))
      (sf-as-given (fk-prog m))))

(define (ge-epilogues prog) (filter gemm-epilogue-binding? (ssa-program-bindings prog)))

;; The segop tags of the replay plan of fused forward program prog.
(define (ge-tags prog)
  (let ((ctx (make-morphism-context)))
    (ssa-realize/ctx ctx prog)
    (finalize-context! ctx)
    (reset-context! ctx)
    (ssa-realize/ctx ctx prog)
    (map car (plan-segops prog))))

(define (ge-gemm-tags prog)
  (filter (lambda (t) (memq t '(segop/gemm segop/gemm-strided segop/gemm-epilogue
                                segop/gemm-epilogue-host)))
          (ge-tags prog)))

(test-group "GEMM epilogues"
  (for-each
   (lambda (dtype)
     (let* ((x  (fk-arr mo-x '(2 3) dtype))
            (xt (fk-arr mo-x '(3 2) dtype))
            (W  (fk-arr mo-w '(3 2) dtype))
            (W2 (fk-arr (reverse mo-w) '(3 2) dtype))
            (b  (fk-arr '(0.25 -0.5) '(2) dtype))
            (bc (fk-arr '(0.25 -0.5) '(2 1) dtype))
            (y  (fk-arr '(1.0 -2.0 0.5 3.0) '(2 2) dtype))
            (other (if (eq? dtype 'f32) 'f64 'f32))
            (name (lambda (s) (sprintf "~A: ~A" dtype s))))
       ;; A row bias: fused by default, and run by the flat bias kernel.
       (let* ((m   (morph+ (morph-matmul x W) b))
              (on  (ge-fused #t m))
              (off (ge-fused #f m)))
         (test (name "a row bias is fused by default") 1 (length (ge-epilogues on)))
         (test (name "the fused binding takes the GEMM's position") 0 (rd-position on (ssa-binding-name (car (ge-epilogues on)))))
         (test (name "it runs as one native GEMM epilogue") '(segop/gemm-epilogue) (ge-tags on))
         (test (name "the plan counts it as native") '((native . 1) (host . 0))
           (cdr (assq 'fused (replay-plan-stats on))))
         (test (name "unfused, the GEMM and the bias are separate") '(segop/gemm) (ge-gemm-tags off))
         (test (name "the values are those of the unfused program") (sf-run off) (sf-run on)))
       ;; One op on the product alone.
       (let ((on (ge-fused #t (morph-exp (morph-matmul x W)))))
         (test (name "a unary op is fused by default") '(segop/gemm-epilogue) (ge-tags on)))
       ;; relu(x W + b): the default fuses the bias only; all fuses the chain.
       (let* ((m (morph-relu (morph+ (morph-matmul x W) b))))
         (test (name "by default a chain is not the epilogue") 1
           (length (filter (lambda (b) (eq? (ssa-binding-op b) 'relu)) (ssa-program-bindings (ge-fused #t m)))))
         (let-values (((fused stats same?) (rd-check m)))
           (test (name "under all the chain is the epilogue") 1 (length (ge-epilogues fused)))
           (test (name "with no other binding") 1 (length (ssa-program-bindings fused)))
           (test-assert (name "and the values of the unfused program") same?)))
       ;; A transposed operand: the GEMM runs with the strided plan.
       (let-values (((fused stats same?) (rd-check (morph+ (morph-matmul (morph-transpose xt '(1 0)) W) b))))
         (test (name "a GEMM of a transposed operand takes an epilogue") 1 (length (ge-epilogues fused)))
         (test-assert (name "with the values of the unfused program") same?))
       ;; A second input of the product's shape: on the host, under all only.
       (let ((m (morph* (morph-matmul x W) y)))
         (test (name "by default a full-shape input is not fused") '() (ge-epilogues (ge-fused #t m)))
         (test-assert (name "and it is reported as not native")
           (alist-ref 'not-native (rd-missed (ssa-fusion-pass-stats))))
         (let-values (((fused stats same?) (rd-check m)))
           (test (name "under all it is fused") 1 (length (ge-epilogues fused)))
           (test (name "and runs on the host") '(segop/gemm-epilogue-host) (ge-gemm-tags fused))
           (test-assert (name "with the values of the unfused program") same?)))
       ;; The product as the second operand of the map.
       (let-values (((fused stats same?) (rd-check (morph- y (morph-matmul x W)))))
         (test (name "the product may be the map's second operand") 1 (length (ge-epilogues fused)))
         (test-assert (name "with the values of the unfused program") same?))
       ;; A column bias, read along the rows.
       (let ((m (morph+ (morph-matmul x W) bc)))
         (test (name "by default a column bias is not fused") '() (ge-epilogues (ge-fused #t m)))
         (let-values (((fused stats same?) (rd-check m)))
           (test (name "under all it is fused") 1 (length (ge-epilogues fused)))
           (test-assert (name "with the values of the unfused program") same?)))
       ;; Two GEMMs read by one add: the later one is the earlier's input,
       ;; so the fused binding takes the add's position.
       (let-values (((fused stats same?) (rd-check (morph+ (morph-matmul x W) (morph-matmul x W2)))))
         (test (name "two GEMMs added: one takes the add as epilogue") 1 (length (ge-epilogues fused)))
         (test (name "placed after the other GEMM") 1
           (rd-position fused (ssa-binding-name (car (ge-epilogues fused)))))
         (test-assert (name "with the values of the unfused program") same?))
       ;; Refusals.
       (let-values (((fused stats same?) (rd-check (morph-cast (morph-matmul x W) other))))
         (test (name "a change of dtype is not an epilogue") '() (ge-epilogues fused))
         (test-assert (name "and it is reported") (alist-ref 'gemm-epilogue-form (rd-missed stats)))
         (test-assert (name "the values are unchanged") same?))
       (let-values (((fused stats same?)
                     (rd-check (let ((g (morph-matmul x W))) (morph+ (morph+ g y) (morph-exp g))))))
         (test (name "a product with two uses is not fused") '() (ge-epilogues fused))
         (test-assert (name "and it is reported") (alist-ref 'multi-use-reduction (rd-missed stats)))
         (test-assert (name "the values are unchanged") same?))
       (let-values (((fused stats same?) (rd-check (morph+ (morph-matmul (fk-arr '(1.0 2.0 3.0) '(1 3) dtype) W) y))))
         (test (name "a broadcast read of the product is not fused") '() (ge-epilogues fused))
         (test-assert (name "the values are unchanged") same?))
       ;; The joint program of a dense layer, replayed: the same gradients,
       ;; and no more pool memory than unfused.
       (let-values (((r-on bytes-on) (rd-replays (rd-dense-joint #t dtype)))
                    ((r-all bytes-all) (rd-replays (rd-dense-joint 'all dtype)))
                    ((r-off bytes-off) (rd-replays (rd-dense-joint #f dtype))))
         (test (name "dense layer: default gradients are those of the unfused program") r-off r-on)
         (test (name "dense layer: so are those under all") r-off r-all)
         (test-assert (name "dense layer: the default pool is not larger than unfused") (<= bytes-on bytes-off)))))
   '(f32 f64)))

;; A mock activation backend with a counting relu kernel and a counting add
;; broadcast kernel.  calls is a list (relu add in-place), whose elements
;; count the calls of each kernel and the calls that wrote over an operand.
(define (ge-mock-backend calls)
  (let ((be (make-activation-backend 'mock-epilogue))
        (relu (scalar-op-combiner 'relu))
        (add  (scalar-op-combiner 'add))
        (bump! (lambda (k) (set-car! (list-tail calls k) (+ 1 (list-ref calls k))))))
    (for-each
     (lambda (dt)
       (activation-backend-add-kernel!
        be 'relu dt
        (lambda (n in out)
          (bump! 0)
          (when (eq? in out) (bump! 2))
          (do ((i 0 (+ i 1))) ((= i n))
            (typed-vector-set! out dt i (relu (typed-vector-ref in dt i))))))
       (activation-backend-add-broadcast-kernel!
        be 'add dt
        (lambda (rows cols a ma b mb out)
          (bump! 1)
          (when (or (eq? a out) (eq? b out)) (bump! 2))
          (let ((at (lambda (v mode i j)
                      (typed-vector-ref v dt (case mode
                                               ((0) (+ (* i cols) j)) ((1) i) ((2) j) (else 0))))))
            (do ((i 0 (+ i 1))) ((= i rows))
              (do ((j 0 (+ j 1))) ((= j cols))
                (typed-vector-set! out dt (+ (* i cols) j)
                                   (add (at a ma i j) (at b mb i j)))))))))
     '(f32 f64))
    be))

(test-group "GEMM epilogues through activation kernels"
  (for-each
   (lambda (dtype)
     (let* ((x (fk-arr mo-x '(2 3) dtype))
            (W (fk-arr mo-w '(3 2) dtype))
            (b (fk-arr '(0.25 -0.5) '(2) dtype))
            (name (lambda (s) (sprintf "~A: ~A" dtype s))))
       (for-each
        (lambda (case)
          (let* ((m    ((cadr case) (morph-matmul x W) b))
                 (off  (sf-run (ge-fused #f m)))
                 (on   (ge-fused #t m))
                 (all  (ge-fused 'all m))
                 (what (lambda (s) (name (string-append (car case) ": " s)))))
            (test (what "fused, it runs as one native GEMM epilogue") '(segop/gemm-epilogue) (ge-tags on))
            (test (what "without a backend the values are those of the unfused program") off (sf-run on))
            (test (what "under all too") off (sf-run all))
            (let ((calls (list 0 0 0)))
              (register-activation-backend! (ge-mock-backend calls))
              (let ((off-k (sf-run (ge-fused #f m))))
                (set-car! calls 0) (set-car! (cdr calls) 0) (set-car! (cddr calls) 0)
                (let ((on-k (sf-run on)))
                  (register-activation-backend! #f)
                  (test-assert (what "the epilogue calls the backend's kernel")
                    (> (list-ref calls (caddr case)) 0))
                  (test-assert (what "in place on the product") (> (list-ref calls 2) 0))
                  (test (what "with the values of the unfused program run by the kernel") off-k on-k)
                  (test (what "which are those without a backend") off on-k))))))
        (list (list "relu(x W)" (lambda (p b) (morph-relu p)) 0)
              (list "x W + b" (lambda (p b) (morph+ p b)) 1)))))
   '(f32 f64)))

;;;; ============================================================
;;;; Convolution segops
;;;;
;;;; The convolution bindings lower to seg-maps and seg-reds read through
;;;; guarded maps, which conv-select recognizes and runs with the
;;;; convolution executors.  im2col and col2im must write exactly what
;;;; segop-interpret writes; the contractions, whose order BLAS chooses,
;;;; agree within a tolerance.
;;;; ============================================================

(define (cv-values n dtype seed)
  (let ((v (allocate-typed-vector dtype n)))
    (do ((i 0 (+ i 1))) ((= i n) v)
      (typed-vector-set! v dtype i (* 0.9 (sin (+ (* 1.37 i) seed)))))))

;; The meta of a convolution binding with kernel k, stride s, padding p,
;; output size so (or #f) and layout.
(define (cv-meta k s p so layout)
  `((kernel-size . ,k) (stride . ,s) (padding . ,p)
    ,@(if so `((spatial-output . ,so)) '())
    ,@(if (eq? layout 'nhwc) '((layout . nhwc)) '())))

;; Runs the segop of a convolution binding with its selected executor and
;; with segop-interpret.  ins is a list of (slot . array).  Returns
;; (tag kind executor-output interpreter-output).
(define (cv-run op meta ins shape dtype)
  (let* ((sg   (lower-conv-binding op meta (map cdr ins) (map car ins) #f
                                   shape (compute-strides shape) dtype))
         (ex   (select-segop-executor sg))
         (n    (shape-size shape))
         (b1   (allocate-typed-vector dtype n))
         (b2   (allocate-typed-vector dtype n))
         (geom (conv-segop-geom sg)))
    ((cdr ex) (lambda (s) (cdr (assq s ins))) b1)
    (segop-interpret sg (map (lambda (o) (e4-data (cdr (assq (segop-operand-slot o) ins))))
                             (segop-inputs sg))
                     (list b2))
    (list (car ex) (and geom (conv-geom-kind geom)) b1 b2)))

(define (cv-same? a b dtype n)
  (every (lambda (i) (seg-same? (typed-vector-ref a dtype i) (typed-vector-ref b dtype i))) (iota n)))

(define (cv-close? a b dtype n)
  (let ((tol (if (eq? dtype 'f32) 1e-5 1e-12)))
    (every (lambda (i)
             (let ((u (typed-vector-ref a dtype i)) (v (typed-vector-ref b dtype i)))
               (<= (abs (- u v)) (* tol (max 1.0 (abs u) (abs v))))))
           (iota n))))

(define (cv-out n k s p) (+ 1 (quotient (+ n (* 2 p) (- k)) s)))

(test-group "convolution segops"
  (for-each
   (lambda (dtype)
     (for-each
      (lambda (cs)
        (let* ((layout (list-ref cs 0)) (xs (list-ref cs 1)) (C (list-ref cs 2))
               (k (list-ref cs 3)) (s (list-ref cs 4)) (p (list-ref cs 5))
               (rank (length xs)) (nhwc? (eq? layout 'nhwc))
               (N (if (= rank 4) (car xs) 1))
               (H (cond (nhwc? (list-ref xs 1)) ((= rank 4) (list-ref xs 2)) (else (list-ref xs 1))))
               (W (cond (nhwc? (list-ref xs 2)) ((= rank 4) (list-ref xs 3)) (else (list-ref xs 2))))
               (OH (cv-out H (car k) (car s) (car p))) (OW (cv-out W (cadr k) (cadr s) (cadr p)))
               (fan (* C (car k) (cadr k))) (M (* N OH OW)) (CO 3)
               (xshape (list->vector xs))
               (col-shape (if (= rank 4) (vector N fan (* OH OW)) (vector fan (* OH OW))))
               (name (lambda (what) (sprintf "~A ~A ~A k~A s~A p~A: ~A" dtype layout xs k s p what)))
               (arr (lambda (shape seed)
                      (e4-arr (cv-values (shape-size shape) dtype seed) shape (compute-strides shape) dtype)))
               (x (arr xshape 0.1))
               (meta (cv-meta k s p #f layout)))
          ;; im2col and col2im: exact, special values included.
          (let* ((xsp (e4-arr (e4-values (shape-size xshape) dtype) xshape (compute-strides xshape) dtype))
                 (r (cv-run 'im2col meta (list (cons 'x xsp)) col-shape dtype)))
            (test (name "im2col runs on its executor") 'segop/im2col (car r))
            (test (name "im2col is recognized") 'im2col (cadr r))
            (test-assert (name "im2col writes what the interpreter writes")
              (cv-same? (caddr r) (cadddr r) dtype (shape-size col-shape))))
          (let* ((col (e4-arr (e4-values (shape-size col-shape) dtype) col-shape
                              (compute-strides col-shape) dtype))
                 (r (cv-run 'col2im meta (list (cons 'c col)) xshape dtype)))
            (test (name "col2im runs on its executor") 'segop/col2im (car r))
            (test (name "col2im is recognized") 'col2im (cadr r))
            (test-assert (name "col2im writes what the interpreter writes")
              (cv-same? (caddr r) (cadddr r) dtype (shape-size xshape))))
          ;; The contractions take rank-4 images.
          (when (= rank 4)
            (let* ((cmeta (cv-meta k s p (list OH OW) layout))
                   (wt (arr (vector fan CO) 0.2)) (b (arr (vector CO) 0.3)) (g (arr (vector M CO) 0.4)))
              (for-each
               (lambda (c)
                 (let ((r (cv-run (car c) cmeta (cadr c) (caddr c) dtype)))
                   (test (name (sprintf "~A runs on its executor" (car c))) (cadddr c) (car r))
                   (test-assert (name (sprintf "~A agrees with the interpreter" (car c)))
                     (cv-close? (caddr r) (cadddr r) dtype (shape-size (caddr c))))))
               (list (list 'conv2d-fwd (list (cons 'x x) (cons 'w wt) (cons 'b b)) (vector M CO)
                           'segop/conv-fwd)
                     (list 'conv2d-bwd-data (list (cons 'g g) (cons 'w wt)) xshape
                           'segop/conv-bwd-data)
                     (list 'conv2d-bwd-weights (list (cons 'g g) (cons 'x x)) (vector fan CO)
                           'segop/conv-bwd-weights)))))))
      '((nchw (2 3 6 6) 3 (3 3) (1 1) (1 1))
        (nchw (2 3 6 6) 3 (3 3) (2 2) (1 1))
        (nchw (2 3 7 5) 3 (2 2) (2 2) (0 0))
        (nchw (2 2 6 5) 2 (3 2) (2 1) (1 0))
        (nchw (3 5 5)   3 (3 3) (1 1) (0 0))
        (nhwc (2 6 6 3) 3 (3 3) (2 2) (1 1))
        (nhwc (2 6 5 2) 2 (3 2) (2 1) (1 0)))))
   '(f32 f64))

  ;; A strided image: the executor reads it as the interpreter does.
  (let* ((dtype 'f64) (shape #(2 3 5 5))
         (data (cv-values 150 dtype 0.5))
         ;; [N C H W] indices over NHWC storage
         (x (e4-arr data shape (vector 75 1 15 3) dtype))
         (r (cv-run 'conv2d-fwd (cv-meta '(3 3) '(1 1) '(1 1) '(5 5) 'nchw)
                    (list (cons 'x x) (cons 'w (e4-arr (cv-values 108 dtype 0.6) #(27 4) #(4 1) dtype))
                          (cons 'b (e4-arr (cv-values 4 dtype 0.7) #(4) #(1) dtype)))
                    #(50 4) dtype)))
    (test-assert "conv2d-fwd of a strided image agrees with the interpreter"
      (cv-close? (caddr r) (cadddr r) dtype 200)))

  ;; A seg-map with the maps of im2col but another body is not im2col.
  (let* ((x (e4-arr (cv-values 72 'f64 0.1) #(2 1 6 6) #(36 36 6 1) 'f64))
         (sg (lower-conv-binding 'im2col (cv-meta '(3 3) '(1 1) '(1 1) #f 'nchw) (list x) '(x) #f
                                 #(2 9 36) (compute-strides #(2 9 36)) 'f64))
         (neg (make-seg-map (segop-space sg) (segop-inputs sg) (segop-outputs sg) '((mul (in 0) (in 0))))))
    (test-assert "im2col is recognized" (conv-segop-geom sg))
    (test "a seg-map with another body is not" #f (conv-segop-geom neg)))

  ;; A two-layer conv program: replays agree, the plan runs every
  ;; convolution on its executor, and the convolutions share one scratch
  ;; buffer.
  (let* ((dtype 'f32) (N 2)
         (conv (lambda (x wt b C H)
                 (am:make-var
                  (morphism-expr (gensym 'conv2d-) 'conv2d-fwd
                                 (list (am:var-value x) (am:var-value wt) (am:var-value b))
                                 (lambda (i) i) (vector (* N H H) (vector-ref (morph-shape (am:var-value wt)) 1))
                                 dtype
                                 `((kernel-size . (3 3)) (stride . (1 1)) (padding . (1 1))
                                   (spatial-output . (,H ,H)))
                                 0)
                  #f)))
         (vals (lambda (n seed) (map (lambda (i) (* 0.5 (sin (+ (* 1.37 i) seed)))) (iota n))))
         (x  (seg-var (vals (* N 1 8 8) 0.1) (list N 1 8 8) dtype #f))
         (w1 (seg-var (vals (* 9 4) 0.2) '(9 4) dtype #t))
         (b1 (seg-var (vals 4 0.3) '(4) dtype #t))
         (w2 (seg-var (vals (* 36 6) 0.4) '(36 6) dtype #t))
         (b2 (seg-var (vals 6 0.5) '(6) dtype #t))
         (h  (am:var-relu (conv x w1 b1 1 8)))
         ;; [N*8*8, 4] back to NCHW for the second layer
         (h4 (am:var-transpose (am:var-reshape h (list N 8 8 4)) '(0 3 1 2)))
         (y  (conv h4 w2 b2 4 8))
         (loss (am:var-sum (am:var* y (seg-var (vals (* N 64 6) 0.6) (list (* N 64) 6) dtype #f))))
         (fwd (morphism-to-ssa loss))
         (pv (map (lambda (p) (ssa-constant-id fwd (am:var-value p))) (list w1 b1 w2 b2)))
         (joint (ssa-vjp fwd pv (ssa-loss-binding-val fwd))))
    (let-values (((rs bytes) (rd-replays joint)))
      (test-assert "two conv layers: the replays agree" (every (lambda (r) (equal? r (cadr rs))) (cddr rs)))
      (let ((tags (map car (plan-segops joint)))
            (s1 (* N 64 9)) (s2 (* N 64 36)))
        (test "two conv layers: two forward convolutions" 2
          (length (filter (lambda (t) (eq? t 'segop/conv-fwd)) tags)))
        (test "two conv layers: two weight gradients" 2
          (length (filter (lambda (t) (eq? t 'segop/conv-bwd-weights)) tags)))
        (test-assert "two conv layers: a data gradient"
          (memq 'segop/conv-bwd-data tags))
        (test "two conv layers: the convolutions share one column buffer"
          (* 4 (max s1 s2))
          (cdr (assq 'conv-scratch-bytes (replay-plan-stats joint))))))))

(test-end)
