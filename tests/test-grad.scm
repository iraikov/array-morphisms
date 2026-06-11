;;; test-grad.scm
;;; Test Suite for array-morphisms-grad
;;;
;;; Tests the morph-variable type, backward rules for all operations,
;;; chain rule, multi-use variable gradient accumulation, and the
;;; no-grad short-circuit path.
;;;
;;; All backward rules are verified by:
;;;   1. Calling backward! to compute analytical gradients (lazy morphisms)
;;;   2. Calling realize on var-grad to materialize
;;;   3. Comparing element-wise to known analytical values

(import scheme (chicken base))
(import test)
(import (only srfi-1 iota every map))
(import (only srfi-4 f64vector-ref f64vector-length f64vector))
(import datatype matchable)
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-blas-exec)
(import array-morphisms-realization)
(import array-morphisms-grad)


;;;; ============================================================
;;;; Test Utilities
;;;; ============================================================

(define tol 1e-5)

(define (approx= a b)
  (< (abs (- a b)) tol))

(define (realized-data v)
  "Realize var-grad v and return its concrete data vector."
  (let ((g (var-grad v)))
    (unless g (error "realized-data: var-grad is #f"))
    (let ((c (realize g)))
      (cases array-morphism c
        (concrete-array (data shape strides offset dtype alloc-id batch-axis)
          data)
        (else (error "realized-data: not a concrete-array after realize"))))))

(define (grad-list v)
  "Return var-grad of v as a flat list of f64 values."
  (let ((data (realized-data v))
        (n    (morph-size (var-value v))))
    (map (lambda (i) (exact->inexact (typed-vector-ref data 'f64 i)))
         (iota n))))

(define (value-list m)
  "Realize morphism m and return flat list of f64 values."
  (let ((c (realize m)))
    (cases array-morphism c
      (concrete-array (data shape strides offset dtype alloc-id batch-axis)
        (let ((n (shape-size shape)))
          (map (lambda (i) (exact->inexact (typed-vector-ref data dtype i)))
               (iota n))))
      (else (error "value-list: not concrete")))))

(define (make-f64-var lst shape)
  "Convenience: make a requires-grad=#t variable from a flat f64 list."
  (make-var (morph-from-list lst (list->vector shape) 'f64) #t))

(define (lists-approx= l1 l2)
  (and (= (length l1) (length l2))
       (every approx= l1 l2)))


;;;; ============================================================
;;;; Group 1: make-var and accessors
;;;; ============================================================

(test-group "make-var and accessors"

  (let* ((m (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
         (v (make-var m #t)))
    (test-assert "morph-variable? predicate"
      (morph-variable? v))
    (test-assert "var-value returns the morphism"
      (eq? m (var-value v)))
    (test-assert "var-grad is #f initially"
      (not (var-grad v)))
    (test-assert "var-requires-grad? returns #t when requested"
      (var-requires-grad? v)))

  (let* ((m (morph-from-list '(1.0) #(1) 'f64))
         (v (make-var m)))
    (test-assert "var-requires-grad? defaults to #f"
      (not (var-requires-grad? v))))

  (let* ((m (morph-from-list '(1.0 2.0) #(2) 'f64))
         (v (make-var m #t))
         (g (morph-from-list '(0.5 0.5) #(2) 'f64))
         (dummy (accumulate-grad! v g)))
    (test-assert "accumulate-grad! sets var-grad"
      (not (not (var-grad v)))))

  (let* ((m (morph-from-list '(1.0) #(1) 'f64))
         (v (make-var m #t))
         (g (morph-from-list '(3.0) #(1) 'f64))
         (dummy (accumulate-grad! v g))
         (dummy2 (zero-grad! v)))
    (test-assert "zero-grad! resets var-grad to #f"
      (not (var-grad v)))))


;;;; ============================================================
;;;; Group 2: accumulate-grad! with multiple calls
;;;; ============================================================

(test-group "accumulate-grad! accumulation"

  ;; Single call: gradient set directly
  (let* ((v   (make-f64-var '(1.0 2.0) '(2)))
         (g1  (morph-from-list '(1.0 1.0) #(2) 'f64))
         (dummy (accumulate-grad! v g1))
         (result (value-list (var-grad v))))
    (test-assert "single accumulate-grad! sets grad"
      (lists-approx= result '(1.0 1.0))))

  ;; Double call: morph+ combines them (lazy, then realized)
  (let* ((v   (make-f64-var '(1.0 2.0) '(2)))
         (g1  (morph-from-list '(1.0 1.0) #(2) 'f64))
         (g2  (morph-from-list '(0.5 0.5) #(2) 'f64))
         (dummy1 (accumulate-grad! v g1))
         (dummy2 (accumulate-grad! v g2))
         (result (value-list (var-grad v))))
    (test-assert "double accumulate-grad! sums contributions"
      (lists-approx= result '(1.5 1.5)))))


;;;; ============================================================
;;;; Group 3: var+ backward
;;;; ============================================================

(test-group "var+ backward"

  ;; Simple vector addition: d/dx1 = g, d/dx2 = g
  (let* ((v1  (make-f64-var '(1.0 2.0 3.0) '(3)))
         (v2  (make-f64-var '(4.0 5.0 6.0) '(3)))
         (out (var+ v1 v2))
         (dummy (backward! out)))
    (test-assert "var+ grad v1 = ones (seed)"
      (lists-approx= (grad-list v1) '(1.0 1.0 1.0)))
    (test-assert "var+ grad v2 = ones (seed)"
      (lists-approx= (grad-list v2) '(1.0 1.0 1.0))))

  ;; Broadcast: [1] + [3] -> [3], grad of [1] is summed
  (let* ((v1  (make-f64-var '(2.0) '(1)))
         (v2  (make-f64-var '(1.0 2.0 3.0) '(3)))
         (out (var+ v1 v2))
         (dummy (backward! out)))
    (test-assert "var+ broadcast: grad of scalar v1 = sum(g) = 3.0"
      (lists-approx= (grad-list v1) '(3.0)))))


;;;; ============================================================
;;;; Group 4: var- backward
;;;; ============================================================

(test-group "var- backward"

  (let* ((v1  (make-f64-var '(5.0 3.0) '(2)))
         (v2  (make-f64-var '(1.0 1.0) '(2)))
         (out (var- v1 v2))
         (dummy (backward! out)))
    (test-assert "var- grad v1 = ones"
      (lists-approx= (grad-list v1) '(1.0 1.0)))
    (test-assert "var- grad v2 = -ones"
      (lists-approx= (grad-list v2) '(-1.0 -1.0)))))


;;;; ============================================================
;;;; Group 5: var* backward (product rule)
;;;; ============================================================

(test-group "var* backward"

  ;; f = x1 * x2,  df/dx1 = x2,  df/dx2 = x1  (with seed g=1)
  (let* ((x1-vals '(2.0 3.0))
         (x2-vals '(4.0 5.0))
         (v1  (make-f64-var x1-vals '(2)))
         (v2  (make-f64-var x2-vals '(2)))
         (out (var* v1 v2))
         (dummy (backward! out)))
    (test-assert "var* grad v1 = x2 * seed = x2"
      (lists-approx= (grad-list v1) x2-vals))
    (test-assert "var* grad v2 = x1 * seed = x1"
      (lists-approx= (grad-list v2) x1-vals))))


;;;; ============================================================
;;;; Group 6: var/ backward (quotient rule)
;;;; ============================================================

(test-group "var/ backward"

  ;; f = x1 / x2,  df/dx1 = 1/x2,  df/dx2 = -x1/x2^2
  (let* ((v1  (make-f64-var '(6.0 8.0) '(2)))
         (v2  (make-f64-var '(2.0 4.0) '(2)))
         (out (var/ v1 v2))
         (dummy (backward! out)))
    (test-assert "var/ grad v1 = 1/x2"
      (lists-approx= (grad-list v1) '(0.5 0.25)))
    (test-assert "var/ grad v2 = -x1/x2^2"
      (lists-approx= (grad-list v2) '(-1.5 -0.5)))))


;;;; ============================================================
;;;; Group 7: var-exp backward
;;;; ============================================================

(test-group "var-exp backward"

  ;; f = exp(x),  df/dx = exp(x)  (seed g=1 per element)
  (let* ((x-vals '(0.0 1.0 2.0))
         (v   (make-f64-var x-vals '(3)))
         (out (var-exp v))
         (dummy (backward! out))
         (expected (map exp x-vals)))
    (test-assert "var-exp grad = exp(x)"
      (lists-approx= (grad-list v) expected))))


;;;; ============================================================
;;;; Group 8: var-log backward
;;;; ============================================================

(test-group "var-log backward"

  ;; f = log(x),  df/dx = 1/x
  (let* ((x-vals '(1.0 2.0 4.0))
         (v   (make-f64-var x-vals '(3)))
         (out (var-log v))
         (dummy (backward! out))
         (expected (map (lambda (x) (/ 1.0 x)) x-vals)))
    (test-assert "var-log grad = 1/x"
      (lists-approx= (grad-list v) expected))))


;;;; ============================================================
;;;; Group 9: var-sqrt backward
;;;; ============================================================

(test-group "var-sqrt backward"

  ;; f = sqrt(x),  df/dx = 1/(2*sqrt(x))
  (let* ((x-vals '(1.0 4.0 9.0))
         (v   (make-f64-var x-vals '(3)))
         (out (var-sqrt v))
         (dummy (backward! out))
         (expected (map (lambda (x) (/ 1.0 (* 2.0 (sqrt x)))) x-vals)))
    (test-assert "var-sqrt grad = 1/(2*sqrt(x))"
      (lists-approx= (grad-list v) expected))))


;;;; ============================================================
;;;; Group 10: var-sin and var-cos backward
;;;; ============================================================

(test-group "var-sin backward"

  ;; f = sin(x),  df/dx = cos(x)
  (let* ((x-vals '(0.0 1.0 2.0))
         (v   (make-f64-var x-vals '(3)))
         (out (var-sin v))
         (dummy (backward! out))
         (expected (map cos x-vals)))
    (test-assert "var-sin grad = cos(x)"
      (lists-approx= (grad-list v) expected))))

(test-group "var-cos backward"

  ;; f = cos(x),  df/dx = -sin(x)
  (let* ((x-vals '(0.0 1.0 2.0))
         (v   (make-f64-var x-vals '(3)))
         (out (var-cos v))
         (dummy (backward! out))
         (expected (map (lambda (x) (- (sin x))) x-vals)))
    (test-assert "var-cos grad = -sin(x)"
      (lists-approx= (grad-list v) expected))))


;;;; ============================================================
;;;; Group 11: var-negate backward
;;;; ============================================================

(test-group "var-negate backward"

  (let* ((v   (make-f64-var '(1.0 -2.0 3.0) '(3)))
         (out (var-negate v))
         (dummy (backward! out)))
    (test-assert "var-negate grad = -ones"
      (lists-approx= (grad-list v) '(-1.0 -1.0 -1.0)))))


;;;; ============================================================
;;;; Group 12: var-sum backward
;;;; ============================================================

(test-group "var-sum backward"

  ;; Sum all elements of a vector: gradient broadcasts back to input shape
  (let* ((v   (make-f64-var '(1.0 2.0 3.0) '(3)))
         (out (var-sum v))
         (dummy (backward! out)))
    (test-assert "var-sum (all axes) grad = ones"
      (lists-approx= (grad-list v) '(1.0 1.0 1.0))))

  ;; Sum along axis 0 of a matrix [2,3]:  grad broadcasts along axis 0
  (let* ((v   (make-f64-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(2 3)))
         (out (var-sum v '(0)))
         (dummy (backward! out)))
    (test-assert "var-sum along axis 0: grad shape [2,3] all ones"
      (lists-approx= (grad-list v)
                     '(1.0 1.0 1.0 1.0 1.0 1.0)))))


;;;; ============================================================
;;;; Group 13: var-mean backward
;;;; ============================================================

(test-group "var-mean backward"

  ;; Mean of 4 elements: gradient is 1/4 for each
  (let* ((v   (make-f64-var '(1.0 2.0 3.0 4.0) '(4)))
         (out (var-mean v))
         (dummy (backward! out)))
    (test-assert "var-mean (all axes) grad = 1/n"
      (lists-approx= (grad-list v) '(0.25 0.25 0.25 0.25))))

  ;; Mean along axis 0 of [2,3]: grad = 1/2 per element
  (let* ((v   (make-f64-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(2 3)))
         (out (var-mean v '(0)))
         (dummy (backward! out)))
    (test-assert "var-mean along axis 0: grad = 0.5 per element"
      (lists-approx= (grad-list v)
                     '(0.5 0.5 0.5 0.5 0.5 0.5)))))


;;;; ============================================================
;;;; Group 14: var-reshape backward
;;;; ============================================================

(test-group "var-reshape backward"

  ;; Reshape [6] -> [2,3]: gradient is reshaped back to [6]
  ;; With ones seed, gradient should be all ones with shape [6]
  (let* ((v   (make-f64-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(6)))
         (out (var-reshape v '(2 3)))
         (dummy (backward! out)))
    (test-assert "var-reshape grad has shape [6]"
      (equal? (morph-shape (var-grad v)) #(6)))
    (test-assert "var-reshape grad values are ones"
      (lists-approx= (grad-list v) '(1.0 1.0 1.0 1.0 1.0 1.0)))))


;;;; ============================================================
;;;; Group 15: var-transpose backward
;;;; ============================================================

(test-group "var-transpose backward"

  ;; Transpose [2,3] with perm (1 0): gradient transposed back (0 1)
  ;; Seed is ones [3,2]. After backward, grad should be ones [2,3].
  (let* ((v   (make-f64-var '(1.0 2.0 3.0 4.0 5.0 6.0) '(2 3)))
         (out (var-transpose v '(1 0)))
         (dummy (backward! out)))
    (test-assert "var-transpose grad has original shape [2,3]"
      (equal? (morph-shape (var-grad v)) #(2 3)))
    (test-assert "var-transpose grad values are ones"
      (lists-approx= (grad-list v) '(1.0 1.0 1.0 1.0 1.0 1.0)))))


;;;; ============================================================
;;;; Group 16: var-matmul backward
;;;; ============================================================

(test-group "var-matmul backward"

  ;; A: [2,3]  B: [3,4]  out: [2,4]
  ;; seed g: ones[2,4]
  ;; dA = g @ B^T = ones[2,4] @ B^T[4,3]
  ;;   dA[i,k] = sum_j B[k,j]  (row sums of B)
  ;; dB = A^T @ g = A^T[3,2] @ ones[2,4]
  ;;   dB[k,j] = sum_i A[i,k]  (col sums of A)
  (let* ((a-data '(1.0 2.0 3.0
                   4.0 5.0 6.0))         ; A [2,3]
         (b-data '(1.0 2.0 3.0 4.0
                   5.0 6.0 7.0 8.0
                   9.0 10.0 11.0 12.0))  ; B [3,4]
         (vA  (make-var (morph-from-list a-data #(2 3) 'f64) #t))
         (vB  (make-var (morph-from-list b-data #(3 4) 'f64) #t))
         (out (var-matmul vA vB))
         (dummy (backward! out))
         (da-list (grad-list vA))
         (db-list (grad-list vB))
         ;; dA[i,k] = sum_j B[k,j]: row sums of B = [10, 26, 42]
         ;; dA has shape [2,3]; each row is [10,26,42]
         (expected-dA '(10.0 26.0 42.0
                        10.0 26.0 42.0))
         ;; dB[k,j] = sum_i A[i,k]: col sums of A = [5, 7, 9]
         ;; dB has shape [3,4]; each col is [5,7,9], so each row repeats the col-sum for that k
         ;; dB[0,j]=5 dB[1,j]=7 dB[2,j]=9 for all j
         (expected-dB '(5.0 5.0 5.0 5.0
                        7.0 7.0 7.0 7.0
                        9.0 9.0 9.0 9.0)))
    (test-assert "var-matmul dA has shape [2,3]"
      (equal? (morph-shape (var-grad vA)) #(2 3)))
    (test-assert "var-matmul dB has shape [3,4]"
      (equal? (morph-shape (var-grad vB)) #(3 4)))
    (test-assert "var-matmul dA values correct"
      (lists-approx= da-list expected-dA))
    (test-assert "var-matmul dB values correct"
      (lists-approx= db-list expected-dB)))

  ;; Lazy transpose: realize(matmul(A, transpose(B))) must not crash
  ;; and must return correct values via the gemm-strided path.
  (let* ((a-data '(1.0 2.0 3.0
                   4.0 5.0 6.0))   ; A [2,3]
         (b-data '(1.0 2.0 3.0
                   4.0 5.0 6.0
                   7.0 8.0 9.0
                   10.0 11.0 12.0)); B [4,3] -> B^T [3,4]
         (A (morph-from-list a-data #(2 3) 'f64))
         (B (morph-from-list b-data #(4 3) 'f64))
         ;; morph-matmul(A [2,3], morph-transpose(B [4,3]) [3,4]) -> [2,4]
         (result (realize (morph-matmul A (morph-transpose B '(1 0)))))
         ;; A @ B^T: result[i,j] = sum_k A[i,k]*B[j,k]
         ;; row 0 of A: [1,2,3]
         ;; B rows: [1,2,3],[4,5,6],[7,8,9],[10,11,12]
         ;; result[0,j] = dot([1,2,3], B[j]) = 14,32,50,68
         ;; result[1,j] = dot([4,5,6], B[j]) = 32,77,122,167
         (expected '(14.0 32.0 50.0 68.0
                     32.0 77.0 122.0 167.0)))
    (test-assert "realize(matmul(A, transpose(B))): concrete result"
      (concrete-array? result))
    (test-assert "realize(matmul(A, transpose(B))): correct shape"
      (equal? (get-morphism-shape result) #(2 4)))
    (test-assert "realize(matmul(A, transpose(B))): correct values"
      (lists-approx= (value-list result) expected))))


;;;; ============================================================
;;;; Group 17: chain rule
;;;; ============================================================

(test-group "chain rule: log(exp(x) + 1.0)"

  ;; f(x) = sum(log(exp(x) + 1))  (softplus sum)
  ;; df/dx_i = exp(x_i) / (exp(x_i) + 1) = sigmoid(x_i)
  (let* ((x-vals '(0.0 1.0 -1.0))
         (v   (make-f64-var x-vals '(3)))
         (one (make-var (morph-from-list '(1.0) '(1) 'f64) #f))
         (e   (var-exp v))
         (ep1 (var+ e one))
         (lg  (var-log ep1))
         (out (var-sum lg))
         (dummy (backward! out))
         (expected (map (lambda (x)
                          (/ (exp x) (+ (exp x) 1.0)))
                        x-vals)))
    (test-assert "softplus sum grad = sigmoid(x)"
      (lists-approx= (grad-list v) expected))))


;;;; ============================================================
;;;; Group 18: multi-use variable gradient accumulation
;;;; ============================================================

(test-group "multi-use variable"

  ;; f(x) = x * x  (using same variable twice)
  ;; df/dx = 2x (gradient accumulates from both uses)
  (let* ((x-vals '(1.0 2.0 3.0))
         (v   (make-f64-var x-vals '(3)))
         (out (var* v v))   ; v used as both v1 and v2
         (dummy (backward! out))
         ;; d(x*x)/dx = x (from v1 side) + x (from v2 side) = 2x
         (expected (map (lambda (x) (* 2.0 x)) x-vals)))
    (test-assert "multi-use: grad accumulates from both uses => 2x"
      (lists-approx= (grad-list v) expected))))


;;;; ============================================================
;;;; Group 19: no-grad path
;;;; ============================================================

(test-group "no-grad path"

  ;; Variables with requires-grad=#f should not accumulate gradients
  (let* ((v1  (make-var (morph-from-list '(1.0 2.0) #(2) 'f64) #t))
         (v2  (make-var (morph-from-list '(3.0 4.0) #(2) 'f64) #f)) ; no grad
         (out (var* v1 v2))
         (dummy (backward! out)))
    (test-assert "requires-grad=#f variable: var-grad remains #f"
      (not (var-grad v2)))
    (test-assert "requires-grad=#t variable: var-grad is computed"
      (not (not (var-grad v1)))))

  ;; If neither input requires grad, output has no grad-fn
  (let* ((v1  (make-var (morph-from-list '(1.0) #(1) 'f64) #f))
         (v2  (make-var (morph-from-list '(2.0) #(1) 'f64) #f))
         (out (var+ v1 v2))
         (dummy (backward! out)))
    (test-assert "no-grad operation: output requires-grad is #f"
      (not (var-requires-grad? out)))))


(test-exit)
