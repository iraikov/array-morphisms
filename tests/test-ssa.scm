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

(import scheme (chicken base))
(import test)
(import (only srfi-1 iota every map filter filter-map))
(import (only srfi-4 f64vector-ref f64vector-length))
(import srfi-69)
(import datatype)
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-blas-exec)
(import array-morphisms-realization)
(import array-morphisms-context)
(import (prefix array-morphisms-grad am:))
(import array-morphisms-ssa)


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
