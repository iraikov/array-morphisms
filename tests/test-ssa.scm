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
        (only srfi-1 iota every any map filter filter-map take drop fold)
        (only srfi-4 f64vector f64vector-ref f64vector-length f64vector-set!
              make-f32vector f32vector-ref f32vector-set! f32vector->list f64vector->list
              list->s32vector s32vector->list make-s32vector make-f64vector)
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
;;;; Group E2: compose-flat-combiners and SSA element-wise fusion pass
;;;; ============================================================

(test-group "E2: compose-flat-combiners"

  (test "unary+unary: abs then negate"
    -3.0
    (let ((fused (compose-flat-combiners abs 1 - 0)))
      (fused -3.0)))

  (test "unary+unary: negate then abs"
    3.0
    (let ((fused (compose-flat-combiners - 1 abs 0)))
      (fused -3.0)))

  (test "binary+unary: add then relu passes negative to zero"
    0.0
    (let ((fused (compose-flat-combiners + 2 (lambda (x) (max 0.0 x)) 0)))
      (fused 1.0 -3.0)))

  (test "binary+unary: add then relu passes positive through"
    1.5
    (let ((fused (compose-flat-combiners + 2 (lambda (x) (max 0.0 x)) 0)))
      (fused 1.0 0.5)))

  (test "unary chain of 3 composes correctly"
    3.0
    ;; abs(-3) = 3; negate(3) = -3; abs(-3) = 3
    (let* ((f01  (compose-flat-combiners abs 1 (lambda (x) (- x)) 0))
           (f012 (compose-flat-combiners f01 1 abs 0)))
      (f012 -3.0))))


(test-group "E2: ssa-element-wise-fusion-pass"

  (test "unary chain abs+negate: fusion reduces binding count by 1"
    1
    (let* ((a     (morph-from-list '(-3.0 2.0) #(2) 'f64))
           (prog0 (morphism-to-ssa (am:make-var (morph-negate (morph-abs a)) #f)))
           (n-before (length (ssa-program-bindings prog0)))
           (prog1 (ssa-element-wise-fusion-pass prog0))
           (n-after  (length (ssa-program-bindings prog1))))
      (- n-before n-after)))

  (test "fused abs+negate produces correct numerics"
    '(-3 -2)
    ;; abs(-3)=3 -> negate(3)=-3; abs(2)=2 -> negate(2)=-2
    (let* ((a     (morph-from-list '(-3.0 2.0) #(2) 'f64))
           (prog0 (morphism-to-ssa (am:make-var (morph-negate (morph-abs a)) #f)))
           (prog1 (ssa-element-wise-fusion-pass prog0))
           (res   (ssa-realize prog1)))
      (map inexact->exact (map exact->inexact (concrete->list (car res))))))

  (test "multi-consumer binding is not fused (b has 2 consumers)"
    3
    ;; b = abs(a); c = negate(b); d = b + c  => b has use-count=2 -> no fusion of b into c
    (let* ((a     (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b     (morph-abs a))
           (c     (morph-negate b))
           (d     (morph+ b c))
           (prog0 (morphism-to-ssa (am:make-var d #f)))
           (n-before (length (ssa-program-bindings prog0)))
           (prog1 (ssa-element-wise-fusion-pass prog0))
           (n-after  (length (ssa-program-bindings prog1))))
      n-after))

  (test "reduction is not fused (non-elementwise shape change)"
    2
    ;; abs on [4] -> reduce-sum to scalar: abs has 1 consumer but reduce is not elementwise
    (let* ((a     (morph-from-list '(1.0 -2.0 3.0 -4.0) #(4) 'f64))
           (b     (morph-abs a))
           (c     (morph-reduce 'sum b '(0) #f))
           (prog0 (morphism-to-ssa (am:make-var c #f)))
           (prog1 (ssa-element-wise-fusion-pass prog0)))
      (length (ssa-program-bindings prog1))))

  (test "ssa-vjp fusion: negate+mul VJP bindings fuse when eligible"
    #t
    ;; loss = sum(negate(x)) for leaf x
    ;; Forward: negate binding (1)
    ;; Backward: negate-of-g (dx=-g), no reduction needed since x same shape as loss
    ;; With E2, backward negate-of-g may fuse with the add(loss,zero) trailing binding
    ;; The key invariant: gradient values must be correct after fusion
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
         (ri-segop (_ _ _ _ _ _ tag)
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
                            (lambda (deref buf)
                              ((cdr sel) deref buf)
                              (let* ((out (car (segop-outputs sg)))
                                     (dt  (segop-operand-dtype out))
                                     (n   (shape-size (segop-operand-shape out)))
                                     (ref (allocate-typed-vector dt n))
                                     (loose? (unspecified-order-tolerance sg)))
                                (segop-interpret sg
                                                 (map (lambda (o) (array-data (deref (segop-operand-slot o))))
                                                      (segop-inputs sg))
                                                 (list ref))
                                (set! checks (+ checks 1))
                                (do ((i 0 (+ i 1))) ((= i n))
                                  (unless (agree? (typed-vector-ref ref dt i)
                                                  (typed-vector-ref buf dt i) loose?)
                                    (set! mismatches (+ mismatches 1)))))))))))
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
                    (ri-segop (_ _ _ _ sg _ tag) (cons tag sg))
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

(test-group "segop lowering of element-wise bindings"
  (for-each
   (lambda (dtype)
     (let-values (((r checks mismatches)
                   (with-segop-checker
                    (lambda ()
                      (let-values (((joint trace replay)
                                    (seg-trace-and-replay (lambda () (seg-mlp-loss dtype)))))
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
                              segop/strided-map)))
                  tags))
         (test-assert (sprintf "~A: relu keeps its symbolic body" dtype)
           (any (lambda (ts) (equal? (cases segop (cdr ts)
                                       (seg-map (space ins outs bodies fallback) (car bodies))
                                       (else #f))
                                     '(relu (in 0))))
                segs))
         (test-assert (sprintf "~A: the fused negate and exp have a composed body" dtype)
           (any (lambda (ts) (equal? (cases segop (cdr ts)
                                       (seg-map (space ins outs bodies fallback) (car bodies))
                                       (else #f))
                                     '(exp (negate (in 0)))))
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
    (binding-scalar-body (make-ssa-binding* 'add 2 '()) +))
  (test-assert "a binding whose combiner is not its op's gets an opaque body"
    (let ((body (binding-scalar-body (make-ssa-binding* 'add 2 '()) -)))
      (and (eq? (car body) 'opaque) (eq? (cadr body) -))))
  (test-assert "a map binding gets an opaque body"
    (eq? 'opaque (car (binding-scalar-body (make-ssa-binding* 'map 1 '()) abs))))
  (test "a fused binding keeps its composed body" '(mul (exp (in 0)) (in 1))
    (binding-scalar-body (make-ssa-binding* 'mul 2 '((body . (mul (exp (in 0)) (in 1)))
                                                      (fused? . #t)))
                         *)))


;;;; ============================================================
;;;; Fusion legality and gradient dtypes
;;;;
;;;; The element-wise fusion pass reads every input of a fused binding at
;;;; the output's index, so it may fuse only when all those inputs have the
;;;; output's shape, when producer and consumer share a dtype, and when the
;;;; input counts are ones compose-flat-combiners supports.  Gradients of
;;;; parameters have the parameter's dtype, through a cast when the
;;;; backward pass computes them in another dtype.
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

(define (fused-bindings joint)
  (filter (lambda (b) (assq 'fused? (ssa-binding-meta b))) (ssa-program-bindings joint)))

(define (output-dtypes results)
  (map (lambda (m) (cases array-morphism m
                     (concrete-array (_ _ _ _ dt _ _) dt)
                     (else #f)))
       results))

(test-group "fusion legality and gradient dtypes"
  ;; u = x*y + b with a bias b of shape [3]: x*y must not be fused into the
  ;; add, whose other input is broadcast.
  (let* ((make (lambda ()
                 (let* ((x (param-var '(1 2 3 4 5 6) '(2 3)))
                        (y (param-var '(1 1 1 2 2 2) '(2 3)))
                        (b (param-var '(100 200 300) '(3)))
                        (u (am:var+ (am:var* x y) b)))
                   (values (am:var-sum (am:var* u u)) (list x y b))))))
    (let-values (((joint trace replay) (ssa-results make)))
      (test "a product is not fused into an add with a broadcast input" '()
        (filter (lambda (b) (memq 'add (list (ssa-binding-op b)))) (fused-bindings joint)))
      (test-assert "loss and gradients with a broadcast bias (trace)"
        (equal? '((295922.0) (202.0 404.0 606.0 432.0 840.0 1248.0)
                  (202.0 808.0 1818.0 864.0 2100.0 3744.0) (418.0 824.0 1230.0))
                trace))
      (test-assert "replay gives the same values" (equal? trace replay))))

  ;; tanh(x*y + z): the three-input fused binding x*y + z is not fused
  ;; again into tanh, which compose-flat-combiners cannot express.
  (let* ((make (lambda ()
                 (let* ((x (param-var '(-0.7 -0.4 -0.1 0.2 0.5 0.8) '(2 3)))
                        (y (param-var '(0.2 0.3 0.4 0.5 0.6 0.7) '(2 3)))
                        (z (param-var '(1.0 0.75 0.5 0.25 0.0 -0.25) '(2 3)))
                        (v (am:var-tanh (am:var+ (am:var* x y) z))))
                   (values (am:var-sum (am:var* v v)) (list x y z))))))
    (let-values (((joint trace replay) (ssa-results make)))
      (test-assert "the chain compiles, with a three-input fused binding"
        (any (lambda (b) (= 3 (length (ssa-binding-inputs b)))) (fused-bindings joint)))
      (test-assert "no fused binding has more than three inputs"
        (every (lambda (b) (<= (length (ssa-binding-inputs b)) 3)) (fused-bindings joint)))
      (test-assert "trace agrees with the eager autodiff"
        (results-close? trace (eager-results make) 1e-12))
      (test-assert "replay gives the same values as the trace" (equal? trace replay))))

  (let* ((mk (lambda (name op inputs dtype)
               (make-ssa-binding name op inputs #(2 3) dtype `((combiner . ,+)))))
         (p   (mk 'bid-p 'add (list (const-ref* 'c1) (const-ref* 'c2)) 'f32))
         (c32 (mk 'bid-c 'add (list (binding-ref* 'bid-p) (const-ref* 'c3)) 'f32))
         (c64 (mk 'bid-c 'add (list (binding-ref* 'bid-p) (const-ref* 'c3)) 'f64))
         (p3  (mk 'bid-p 'add (list (const-ref* 'c1) (const-ref* 'c2) (const-ref* 'c4)) 'f32))
         (counts (let ((h (make-hash-table))) (hash-table-set! h 'bid-p 1) h)))
    (test-assert "same dtype, two inputs: eligible" (ssa-fusion-eligible? p c32 counts))
    (test-assert "different dtypes: not eligible" (not (ssa-fusion-eligible? p c64 counts)))
    (test-assert "a three-input producer: not eligible" (not (ssa-fusion-eligible? p3 c32 counts))))

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
                (and (pair? fin) (member `(cast ,dt (acc)) (cdr fin)) #t)))
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

  ;; with only the scheme backend, a GEMM is interpreted
  (let-values (((A B av bv) (gm-operands 'f32 3 4 5 #t #f)))
    (let* ((scheme-only (filter (lambda (b) (eq? (segop-backend-name b) 'scheme)) (segop-backends)))
           (r (parameterize ((segop-backends scheme-only))
                (s2-run (lambda () (morph-matmul A B))))))
      (test-assert "with only the scheme backend a GEMM is interpreted"
        (and (memq 'segop/interpreted (s2-tags (car r))) (null? (gm-segops (car r)))))
      (test-assert "the interpreted GEMM gives the exact reference"
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
             (equal? (cddr fin) '((cast f32 (acc))))))))

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


(test-end)
