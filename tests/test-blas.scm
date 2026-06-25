;;; test-array-morphisms-blas.scm
;;; Test suite for BLAS integration - Phase 1 (compatibility predicates)
;;; and Phase 2 (execution kernels).
;;;
;;; Organisation:
;;;   Group 1  - Layout predicates          (Phase 1)
;;;   Group 2  - Compatibility checks       (Phase 1)
;;;   Group 3  - Function name mapping      (Phase 1)
;;;   Group 4  - Pure Scheme GEMM           (Phase 2)
;;;   Group 5  - Pure Scheme GEMV           (Phase 2)
;;;   Group 6  - Pure Scheme DOT            (Phase 2)
;;;   Group 7  - Pure Scheme AXPY           (Phase 2)
;;;   Group 8  - Morphism constructors      (Phase 2)
;;;   Group 9  - Execute-or-fallback dispatch(Phase 2)
;;;   Group 10 - Configuration              (Phase 2)
;;;
;;; All tests pass without a BLAS library; the pure Scheme fallback is
;;; exercised throughout.  BLAS-specific behaviour (Group 9 "with BLAS")
;;; is guarded by (blas-available?) so those assertions are no-ops in the
;;; default environment.

(import scheme (chicken base))
(import test)
(import (only srfi-1 every iota make-list fold))
(import datatype matchable)
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-blas-compat)
(import array-morphisms-blas-exec)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (approx= a b #!optional (tol 1e-6))
  "Approximate numeric equality within tolerance."
  (< (abs (- a b)) tol))

(define (concrete-values-approx? m expected-list #!optional (tol 1e-6))
  "True when every element of concrete array m is close to the
  corresponding element in expected-list (after flattening both)."
  (let* ((actual-nested (morph->list m))
         (actual   (flatten-nested-list actual-nested))
         (expected (flatten-nested-list expected-list)))
    (and (= (length actual) (length expected))
         (every (lambda (a e) (approx= a e tol)) actual expected))))

(define (make-matrix rows cols init)
  "Create a flat list of (rows * cols) elements filled with init."
  (make-list (* rows cols) init))

(define (iota-matrix rows cols #!optional (start 1))
  "Create a nested list for a (rows x cols) matrix with consecutive integers."
  (let ((flat (iota (* rows cols) start)))
    (nest-list flat (list rows cols))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1: Layout Predicates
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 1 - Layout Predicates"

  (test-assert "contiguous-row-major? 1-D array"
    (contiguous-row-major? (morph-from-list '(1 2 3) #(3) 'f64)))

  (test-assert "contiguous-row-major? 2-D array"
    (contiguous-row-major? (morph-from-list '((1 2) (3 4)) #(2 2) 'f64)))

  (test-assert "contiguous-row-major? 3-D array"
    (contiguous-row-major?
     (morph-from-list (make-list 24 1.0) #(2 3 4) 'f64)))

  (test-assert "contiguous-row-major? f32 array"
    (contiguous-row-major? (morph-from-list '(1 2 3 4) #(4) 'f32)))

  (test-assert "contiguous-row-major? returns #f for stepped slice"
    ;; After realize, slice with step=2 has stride 2 -> not contiguous
    (let* ((m  (morph-from-list '(0 1 2 3 4 5 6 7 8 9) #(10) 'f64))
           (sl (realize (morph-slice m '(0) '(10) 2))))
      (not (contiguous-row-major? sl))))

  (test-assert "contiguous-row-major? returns #f for 2-D transpose"
    ;; Transposed (2,3) has strides (1,3) vs expected (3,1)
    (let* ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (t (realize (morph-transpose m))))
      (not (contiguous-row-major? t))))

  (test-assert "contiguous-row-major? returns #f for abstract morphism"
    (let ((m (morph-from-list '(1 2 3) #(3) 'f64)))
      (not (contiguous-row-major? (morph+ m m)))))

  (test-assert "contiguous-column-major? 2-D: transposed row-major matrix"
    ;; source (2,3) strides (3,1); after transpose: shape (3,2) strides (1,3).
    ;; Column-major for (3,2) requires strides[0]=1, strides[1]=3.
    (let* ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (t (realize (morph-transpose m))))
      (contiguous-column-major? t)))

  (test-assert "contiguous-column-major? returns #f for standard row-major"
    ;; Row-major (2,3) has strides (3,1), not (1,2) -> not column-major.
    (let ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64)))
      (not (contiguous-column-major? m))))

  (test-assert "contiguous-column-major? 1-D array"
    ;; 1-D array: strides=(1), expected column-major strides=(1).
    (contiguous-column-major? (morph-from-list '(1 2 3 4) #(4) 'f64)))

  (test-assert "non-zero offset makes array non-contiguous"
    ;; A slice starting mid-array has offset != 0
    (let* ((m  (morph-from-list '(0 1 2 3 4) #(5) 'f64))
           (sl (realize (morph-slice m '(2) '(5)))))
      (not (contiguous-row-major? sl))))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2: Compatibility Checks
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 1 - blas-compatible-matmul?"

  (test-assert "accepts square f64 matrices"
    (let ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
          (m2 (morph-from-list '((5 6) (7 8)) #(2 2) 'f64)))
      (blas-compatible-matmul? m1 m2)))

  (test-assert "accepts rectangular f64 matrices"
    (let ((m1 (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
          (m2 (morph-from-list '((1 2) (3 4) (5 6)) #(3 2) 'f64)))
      (blas-compatible-matmul? m1 m2)))

  (test-assert "accepts square f32 matrices"
    (let ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f32))
          (m2 (morph-from-list '((5 6) (7 8)) #(2 2) 'f32)))
      (blas-compatible-matmul? m1 m2)))

  (test-assert "rejects mixed f32/f64"
    (let ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
          (m2 (morph-from-list '((5 6) (7 8)) #(2 2) 'f32)))
      (not (blas-compatible-matmul? m1 m2))))

  (test-assert "rejects integer dtypes"
    (let ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 's32))
          (m2 (morph-from-list '((5 6) (7 8)) #(2 2) 's32)))
      (not (blas-compatible-matmul? m1 m2))))

  (test-assert "rejects incompatible inner dimensions"
    ;; (1,2) x (3,2): inner dims 2 != 3
    (let ((m1 (morph-from-list '((1 2)) #(1 2) 'f64))
          (m2 (morph-from-list '((3 4) (5 6) (7 8)) #(3 2) 'f64)))
      (not (blas-compatible-matmul? m1 m2))))

  (test-assert "rejects 1-D vectors"
    (let ((v1 (morph-from-list '(1 2 3) #(3) 'f64))
          (v2 (morph-from-list '(4 5 6) #(3) 'f64)))
      (not (blas-compatible-matmul? v1 v2))))

  (test-assert "rejects 3-D tensors"
    (let ((t1 (morph-from-list (make-list 8 1.0) #(2 2 2) 'f64))
          (t2 (morph-from-list (make-list 8 1.0) #(2 2 2) 'f64)))
      (not (blas-compatible-matmul? t1 t2))))

  (test-assert "rejects non-contiguous operand (stride-2 slice)"
    (let* ((src (morph-from-list (make-list 16 1.0) #(4 4) 'f64))
           ;; Slice every other row -> stride on axis-0 = 2 -> not contiguous
           (sl  (realize (morph-slice src '(0 0) '(4 4) '(2 1))))
           (m2  (morph-from-list (make-list 4 1.0) #(2 2) 'f64)))
      (not (blas-compatible-matmul? sl m2))))

  (test-assert "rejects abstract (non-concrete) morphism"
    (let* ((a  (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (abs (morph+ a a))            ; abstract morphism-expr
           (b  (morph-from-list '((1 0) (0 1)) #(2 2) 'f64)))
      (not (blas-compatible-matmul? abs b))))
)

(test-group "Phase 1 - blas-compatible-matvec?"

  (test-assert "accepts 2-D matrix and 1-D vector"
    (let ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
          (v (morph-from-list '(1 2 3) #(3) 'f64)))
      (blas-compatible-matvec? m v)))

  (test-assert "rejects shape mismatch (N=3 vs N=2)"
    (let ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
          (v (morph-from-list '(1 2) #(2) 'f64)))
      (not (blas-compatible-matvec? m v))))

  (test-assert "rejects wrong argument order (vec, mat)"
    (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
          (v (morph-from-list '(1 2) #(2) 'f64)))
      (not (blas-compatible-matvec? v m))))

  (test-assert "rejects mixed dtypes"
    (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
          (v (morph-from-list '(1 2) #(2) 'f32)))
      (not (blas-compatible-matvec? m v))))
)

(test-group "Phase 1 - blas-compatible-dot?"

  (test-assert "accepts equal-length f64 vectors"
    (let ((v1 (morph-from-list '(1 2 3) #(3) 'f64))
          (v2 (morph-from-list '(4 5 6) #(3) 'f64)))
      (blas-compatible-dot? v1 v2)))

  (test-assert "accepts equal-length f32 vectors"
    (let ((v1 (morph-from-list '(1 2 3) #(3) 'f32))
          (v2 (morph-from-list '(4 5 6) #(3) 'f32)))
      (blas-compatible-dot? v1 v2)))

  (test-assert "rejects different lengths"
    (let ((v1 (morph-from-list '(1 2 3) #(3) 'f64))
          (v2 (morph-from-list '(1 2)   #(2) 'f64)))
      (not (blas-compatible-dot? v1 v2))))

  (test-assert "rejects integer dtype"
    (let ((v1 (morph-from-list '(1 2 3) #(3) 's32))
          (v2 (morph-from-list '(4 5 6) #(3) 's32)))
      (not (blas-compatible-dot? v1 v2))))

  (test-assert "rejects 2-D matrices"
    (let ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
          (m2 (morph-from-list '((5 6) (7 8)) #(2 2) 'f64)))
      (not (blas-compatible-dot? m1 m2))))
)

(test-group "Phase 1 - blas-compatible-operation?"

  (test-assert "detects matmul -> gemm with concrete operands"
    ;; morph-from-list creates concrete arrays, so operands are already concrete.
    (let* ((m1   (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (m2   (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (expr (morph-matmul m1 m2))
           (info (blas-compatible-operation? expr)))
      (and info (eq? 'gemm (car info)))))

  (test-assert "detects matvec -> gemv with concrete operands"
    (let* ((m    (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (v    (morph-from-list '(1 2 3) #(3) 'f64))
           (expr (morph-matvec m v))
           (info (blas-compatible-operation? expr)))
      (and info (eq? 'gemv (car info)))))

  (test-assert "detects dot -> dot with concrete operands"
    (let* ((v1   (morph-from-list '(1 2 3) #(3) 'f64))
           (v2   (morph-from-list '(4 5 6) #(3) 'f64))
           (expr (morph-dot v1 v2))
           (info (blas-compatible-operation? expr)))
      (and info (eq? 'dot (car info)))))

  (test-assert "returns #f for element-wise add"
    (let* ((m1   (morph-from-list '(1 2 3) #(3) 'f64))
           (m2   (morph-from-list '(4 5 6) #(3) 'f64))
           (expr (morph+ m1 m2)))
      (not (blas-compatible-operation? expr))))

  (test-assert "returns #f when operands are abstract"
    ;; Wrapping in morph+ produces an abstract morphism-expr as operand.
    (let* ((a    (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (b    (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (a-abs (morph+ a (morph-from-list '((0 0) (0 0)) #(2 2) 'f64)))
           (expr  (morph-matmul a-abs b)))
      (not (blas-compatible-operation? expr))))

  (test-assert "returns #f for reduction morphism"
    (let* ((m    (morph-from-list '(1 2 3 4) #(4) 'f64))
           (expr (morph-reduce 'sum m)))
      (not (blas-compatible-operation? expr))))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3: Function Name Mapping
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 1 - get-blas-function-name"

  (test-assert "gemm f64 -> dgemm!"
    (eq? 'dgemm! (get-blas-function-name 'gemm 'f64)))

  (test-assert "gemm f32 -> sgemm!"
    (eq? 'sgemm! (get-blas-function-name 'gemm 'f32)))

  (test-assert "gemv f64 -> dgemv!"
    (eq? 'dgemv! (get-blas-function-name 'gemv 'f64)))

  (test-assert "gemv f32 -> sgemv!"
    (eq? 'sgemv! (get-blas-function-name 'gemv 'f32)))

  (test-assert "dot f64 -> ddot"
    (eq? 'ddot (get-blas-function-name 'dot 'f64)))

  (test-assert "dot f32 -> sdot"
    (eq? 'sdot (get-blas-function-name 'dot 'f32)))

  (test-assert "axpy f64 -> daxpy!"
    (eq? 'daxpy! (get-blas-function-name 'axpy 'f64)))

  (test-assert "scal f64 -> dscal!"
    (eq? 'dscal! (get-blas-function-name 'scal 'f64)))

  (test-assert "unknown operation returns #f"
    (not (get-blas-function-name 'frobnicate 'f64)))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4: Pure Scheme GEMM
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 2 - execute-scheme-gemm"

  (test-assert "2x2 square product"
    ;; [[1 2][3 4]] * [[5 6][7 8]] = [[19 22][43 50]]
    (let* ((A (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (B (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (C (execute-scheme-gemm A B)))
      (and (concrete-array? C)
           (equal? (get-morphism-shape C) #(2 2))
           (concrete-values-approx? C '((19 22) (43 50))))))

  (test-assert "rectangular (2,3) x (3,2) product"
    ;; Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
    ;; Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
    (let* ((A (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (B (morph-from-list '((7 8) (9 10) (11 12)) #(3 2) 'f64))
           (C (execute-scheme-gemm A B)))
      (and (equal? (get-morphism-shape C) #(2 2))
           (concrete-values-approx? C '((58 64) (139 154))))))

  (test-assert "product with identity matrix returns A unchanged"
    (let* ((A (morph-from-list '((1 2 3) (4 5 6) (7 8 9)) #(3 3) 'f64))
           (I (morph-from-list '((1 0 0) (0 1 0) (0 0 1)) #(3 3) 'f64))
           (C (execute-scheme-gemm A I)))
      (concrete-values-approx? C '((1 2 3) (4 5 6) (7 8 9)))))

  (test-assert "product with zero matrix yields zero"
    (let* ((A (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (Z (morph-from-list '((0 0) (0 0)) #(2 2) 'f64))
           (C (execute-scheme-gemm A Z)))
      (concrete-values-approx? C '((0 0) (0 0)))))

  (test-assert "preserves f32 dtype"
    (let* ((A (morph-from-list '((1 2) (3 4)) #(2 2) 'f32))
           (B (morph-from-list '((5 6) (7 8)) #(2 2) 'f32))
           (C (execute-scheme-gemm A B)))
      (and (eq? (get-morphism-dtype C) 'f32)
           (concrete-values-approx? C '((19 22) (43 50)) 1e-4))))

  (test-assert "correct result shape for non-square (3,4) x (4,2)"
    (let* ((A (morph-from-list (iota 12 1) #(3 4) 'f64))
           (B (morph-from-list (iota 8  1) #(4 2) 'f64))
           (C (execute-scheme-gemm A B)))
      (equal? (get-morphism-shape C) #(3 2))))

  (test-assert "handles non-contiguous (transposed) operand correctly"
    ;; source: (3,2) = [[1 2][3 4][5 6]], row-major
    ;; after transpose: shape (2,3), strides (1,2) -> rows = [1 3 5] [2 4 6]
    ;; A_T * I_3 should return A_T unchanged
    (let* ((src (morph-from-list '((1 2) (3 4) (5 6)) #(3 2) 'f64))
           (A-T (realize (morph-transpose src)))     ; shape (2,3), non-contiguous
           (I3  (morph-from-list '((1 0 0) (0 1 0) (0 0 1)) #(3 3) 'f64))
           (C   (execute-scheme-gemm A-T I3)))
      (and (equal? (get-morphism-shape C) #(2 3))
           (concrete-values-approx? C '((1 3 5) (2 4 6))))))

  (test-assert "handles offset from column slice"
    ;; Slice columns 1..3 of a (3,4) matrix -> shape (3,3), offset 0 but stride ok
    (let* ((src (morph-from-list '((1 2 3 4) (5 6 7 8) (9 10 11 12)) #(3 4) 'f64))
           (sl  (realize (morph-slice src '(0 1) '(3 4))))  ; shape (3,3)
           (I3  (morph-from-list '((1 0 0) (0 1 0) (0 0 1)) #(3 3) 'f64))
           (C   (execute-scheme-gemm sl I3)))
      ;; sl = [[2 3 4][6 7 8][10 11 12]]
      (concrete-values-approx? C '((2 3 4) (6 7 8) (10 11 12)))))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5: Pure Scheme GEMV
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 2 - execute-scheme-gemv"

  (test-assert "(2,3) matrix times (3,) vector"
    ;; [1 2 3]*[1 2 3]^T = 14,  [4 5 6]*[1 2 3]^T = 32
    (let* ((A (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (v (morph-from-list '(1 2 3) #(3) 'f64))
           (y (execute-scheme-gemv A v)))
      (and (concrete-array? y)
           (equal? (get-morphism-shape y) #(2))
           (concrete-values-approx? y '(14 32)))))

  (test-assert "identity matrix times vector returns vector unchanged"
    (let* ((I (morph-from-list '((1 0 0) (0 1 0) (0 0 1)) #(3 3) 'f64))
           (v (morph-from-list '(2 5 7) #(3) 'f64))
           (y (execute-scheme-gemv I v)))
      (concrete-values-approx? y '(2 5 7))))

  (test-assert "zero matrix times any vector yields zero vector"
    (let* ((Z (morph-from-list '((0 0 0) (0 0 0)) #(2 3) 'f64))
           (v (morph-from-list '(9 9 9) #(3) 'f64))
           (y (execute-scheme-gemv Z v)))
      (concrete-values-approx? y '(0 0))))

  (test-assert "(1,N) row vector contracted with (N,) vector is a scalar"
    ;; 1*1 + 2*2 + 3*3 + 4*4 = 30
    (let* ((A (morph-from-list '((1 2 3 4)) #(1 4) 'f64))
           (v (morph-from-list '(1 2 3 4) #(4) 'f64))
           (y (execute-scheme-gemv A v)))
      (and (equal? (get-morphism-shape y) #(1))
           (concrete-values-approx? y '(30)))))

  (test-assert "handles transposed (non-contiguous) matrix"
    ;; A^T of [[1 2][3 4]] is [[1 3][2 4]] (shape 2,2, non-contiguous strides)
    ;; [[1 3][2 4]] * [1 0]^T = [1 2]
    (let* ((m   (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (A-T (realize (morph-transpose m)))
           (v   (morph-from-list '(1 0) #(2) 'f64))
           (y   (execute-scheme-gemv A-T v)))
      (concrete-values-approx? y '(1 2))))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6: Pure Scheme DOT
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 2 - execute-scheme-dot"

  (test-assert "basic dot product"
    ;; 1*4 + 2*5 + 3*6 = 32
    (let* ((v1 (morph-from-list '(1 2 3) #(3) 'f64))
           (v2 (morph-from-list '(4 5 6) #(3) 'f64))
           (r  (execute-scheme-dot v1 v2)))
      (and (concrete-array? r)
           (equal? (get-morphism-shape r) #())   ; scalar shape
           (concrete-values-approx? r '(32)))))

  (test-assert "dot product of orthogonal unit vectors is zero"
    (let* ((v1 (morph-from-list '(1 0 0) #(3) 'f64))
           (v2 (morph-from-list '(0 1 0) #(3) 'f64))
           (r  (execute-scheme-dot v1 v2)))
      (concrete-values-approx? r '(0))))

  (test-assert "dot product of vector with itself is squared norm"
    ;; ||[3 4]||^2 = 9 + 16 = 25
    (let* ((v  (morph-from-list '(3 4) #(2) 'f64))
           (r  (execute-scheme-dot v v)))
      (concrete-values-approx? r '(25))))

  (test-assert "single-element dot product"
    (let* ((v1 (morph-from-list '(3) #(1) 'f64))
           (v2 (morph-from-list '(4) #(1) 'f64))
           (r  (execute-scheme-dot v1 v2)))
      (concrete-values-approx? r '(12))))

  (test-assert "f32 dot product preserves dtype"
    (let* ((v1 (morph-from-list '(1 2 3) #(3) 'f32))
           (v2 (morph-from-list '(4 5 6) #(3) 'f32))
           (r  (execute-scheme-dot v1 v2)))
      (and (eq? (get-morphism-dtype r) 'f32)
           (concrete-values-approx? r '(32) 1e-4))))

  (test-assert "dot with stride-2 slice operand"
    ;; [0 2 4 6 8] even elements of (0..9)
    (let* ((src (morph-from-list '(0 1 2 3 4 5 6 7 8 9) #(10) 'f64))
           (ev  (realize (morph-slice src '(0) '(10) 2)))  ; [0 2 4 6 8] stride=2
           (v   (morph-from-list '(1 1 1 1 1) #(5) 'f64))
           (r   (execute-scheme-dot ev v)))
      ;; 0+2+4+6+8 = 20
      (concrete-values-approx? r '(20))))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 7: Pure Scheme AXPY
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 2 - execute-scheme-axpy"

  (test-assert "alpha=1: result = x + y"
    (let* ((x (morph-from-list '(1 2 3) #(3) 'f64))
           (y (morph-from-list '(4 5 6) #(3) 'f64))
           (r (execute-scheme-axpy 1.0 x y)))
      (concrete-values-approx? r '(5 7 9))))

  (test-assert "alpha=2: result = 2*x + y"
    (let* ((x (morph-from-list '(1 2 3) #(3) 'f64))
           (y (morph-from-list '(4 5 6) #(3) 'f64))
           (r (execute-scheme-axpy 2.0 x y)))
      (concrete-values-approx? r '(6 9 12))))

  (test-assert "alpha=0: result equals y"
    (let* ((x (morph-from-list '(100 200 300) #(3) 'f64))
           (y (morph-from-list '(4 5 6) #(3) 'f64))
           (r (execute-scheme-axpy 0.0 x y)))
      (concrete-values-approx? r '(4 5 6))))

  (test-assert "alpha=-1: result = y - x"
    (let* ((x (morph-from-list '(1 2 3) #(3) 'f64))
           (y (morph-from-list '(4 5 6) #(3) 'f64))
           (r (execute-scheme-axpy -1.0 x y)))
      (concrete-values-approx? r '(3 3 3))))

  (test-assert "does not modify the y operand"
    (let* ((x (morph-from-list '(1 1 1) #(3) 'f64))
           (y (morph-from-list '(0 0 0) #(3) 'f64))
           (_ (execute-scheme-axpy 1.0 x y)))
      ;; y must be unchanged after the call
      (concrete-values-approx? y '(0 0 0))))

  (test-assert "does not modify the x operand"
    (let* ((x (morph-from-list '(1 2 3) #(3) 'f64))
           (y (morph-from-list '(4 5 6) #(3) 'f64))
           (_ (execute-scheme-axpy 5.0 x y)))
      (concrete-values-approx? x '(1 2 3))))

  (test-assert "handles stride-2 slice for x"
    ;; x = [0 2 4] (even elements), y = [10 10 10]
    ;; result = 1*[0 2 4] + [10 10 10] = [10 12 14]
    (let* ((src (morph-from-list '(0 1 2 3 4 5) #(6) 'f64))
           (x   (realize (morph-slice src '(0) '(6) 2)))  ; [0 2 4]
           (y   (morph-from-list '(10 10 10) #(3) 'f64))
           (r   (execute-scheme-axpy 1.0 x y)))
      (concrete-values-approx? r '(10 12 14))))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 8: Morphism Constructors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 2 - Morphism Constructors"

  (test-assert "morph-matmul produces correct output shape"
    (let* ((A (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (B (morph-from-list '((1 2) (3 4) (5 6)) #(3 2) 'f64))
           (C (morph-matmul A B)))
      (equal? (get-morphism-shape C) #(2 2))))

  (test-assert "morph-matmul stores op as 'matmul"
    (let* ((A (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (B (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (C (morph-matmul A B)))
      (cases array-morphism C
        (morphism-expr (_ op operands idx-fn shape dtype meta batch-axis)
          (eq? op 'matmul))
        (else #f))))

  (test-assert "morph-matmul promotes f32+f64 to f64"
    (let* ((A (morph-from-list '((1 2) (3 4)) #(2 2) 'f32))
           (B (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (C (morph-matmul A B)))
      (eq? (get-morphism-dtype C) 'f64)))

  (test-assert "morph-matmul stores k-dim in metadata"
    (let* ((A (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (B (morph-from-list '((1 2) (3 4) (5 6)) #(3 2) 'f64))
           (C (morph-matmul A B)))
      (cases array-morphism C
        (morphism-expr (_ op operands idx-fn shape dtype meta batch-axis)
          (= 3 (cdr (assq 'k-dim meta))))
        (else #f))))

  (test-assert "morph-matvec produces correct output shape"
    (let* ((A (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (v (morph-from-list '(1 2 3) #(3) 'f64))
           (y (morph-matvec A v)))
      (equal? (get-morphism-shape y) #(2))))

  (test-assert "morph-matvec stores op as 'matvec"
    (let* ((A (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (v (morph-from-list '(1 2) #(2) 'f64))
           (y (morph-matvec A v)))
      (cases array-morphism y
        (morphism-expr (_ op operands idx-fn shape dtype meta batch-axis)
          (eq? op 'matvec))
        (else #f))))

  (test-assert "morph-dot produces scalar shape #()"
    (let* ((v1 (morph-from-list '(1 2 3) #(3) 'f64))
           (v2 (morph-from-list '(4 5 6) #(3) 'f64))
           (r  (morph-dot v1 v2)))
      (equal? (get-morphism-shape r) #())))

  (test-assert "morph-dot stores op as 'dot"
    (let* ((v1 (morph-from-list '(1 2) #(2) 'f64))
           (v2 (morph-from-list '(3 4) #(2) 'f64))
           (r  (morph-dot v1 v2)))
      (cases array-morphism r
        (morphism-expr (_ op operands idx-fn shape dtype meta batch-axis)
          (eq? op 'dot))
        (else #f))))

  (test-assert "morph-axpy produces shape matching x and y"
    (let* ((x (morph-from-list '(1 2 3 4) #(4) 'f64))
           (y (morph-from-list '(5 6 7 8) #(4) 'f64))
           (r (morph-axpy 2.0 x y)))
      (equal? (get-morphism-shape r) #(4))))

  (test-assert "morph-axpy stores alpha in metadata"
    (let* ((x (morph-from-list '(1 2) #(2) 'f64))
           (y (morph-from-list '(3 4) #(2) 'f64))
           (r (morph-axpy 3.14 x y)))
      (cases array-morphism r
        (morphism-expr (_ op operands idx-fn shape dtype meta batch-axis)
          (approx= 3.14 (cdr (assq 'alpha meta))))
        (else #f))))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 9: Execute-or-Fallback Dispatch
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 2 - execute-blas-* (Scheme fallback, no BLAS registered)"

  (test-assert "execute-blas-gemm produces correct result"
    (let* ((A (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (B (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (C (execute-blas-gemm A B)))
      (and (concrete-array? C)
           (concrete-values-approx? C '((19 22) (43 50))))))

  (test-assert "execute-blas-gemv produces correct result"
    (let* ((A (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (v (morph-from-list '(1 2 3) #(3) 'f64))
           (y (execute-blas-gemv A v)))
      (and (concrete-array? y)
           (concrete-values-approx? y '(14 32)))))

  (test-assert "execute-blas-dot produces correct result"
    (let* ((v1 (morph-from-list '(1 2 3) #(3) 'f64))
           (v2 (morph-from-list '(4 5 6) #(3) 'f64))
           (r  (execute-blas-dot v1 v2)))
      (and (concrete-array? r)
           (concrete-values-approx? r '(32)))))

  (test-assert "execute-blas-axpy produces correct result"
    (let* ((x (morph-from-list '(1 2 3) #(3) 'f64))
           (y (morph-from-list '(4 5 6) #(3) 'f64))
           (r (execute-blas-axpy 2.0 x y)))
      (concrete-values-approx? r '(6 9 12))))

  (test-assert "execute-blas-operation with concrete matmul operands"
    ;; operands are concrete-array instances -> compat check succeeds
    (let* ((A    (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (B    (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (expr (morph-matmul A B))
           (C    (execute-blas-operation expr)))
      (and C
           (concrete-array? C)
           (concrete-values-approx? C '((19 22) (43 50))))))

  (test-assert "execute-blas-operation returns #f for abstract operands"
    ;; Wrap A in morph+ to make it an abstract morphism-expr
    (let* ((A    (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (B    (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (A-abs (morph+ A (morph-from-list '((0 0) (0 0)) #(2 2) 'f64)))
           (expr  (morph-matmul A-abs B))
           (result (execute-blas-operation expr)))
      (not result)))

  (test-assert "execute-blas-operation returns #f for non-linalg morphism"
    (let* ((m1   (morph-from-list '(1 2 3) #(3) 'f64))
           (m2   (morph-from-list '(4 5 6) #(3) 'f64))
           (expr (morph+ m1 m2)))
      (not (execute-blas-operation expr))))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 10: Configuration and Backend Record
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 2 - Configuration and Backend Record"

  (test-assert "blas-enabled? is true initially"
    (begin (enable-blas!) (blas-enabled?)))

  (test-assert "disable-blas! sets enabled to #f"
    (begin
      (disable-blas!)
      (let ((r (not (blas-enabled?))))
        (enable-blas!)
        r)))

  (test-assert "enable-blas! restores enabled after disable"
    (begin
      (disable-blas!)
      (enable-blas!)
      (blas-enabled?)))

  (test-assert "blas-available? is #f before any backend is registered"
    (not (blas-available?)))

  (test-assert "active-blas-backend is #f before registration"
    (not (active-blas-backend)))

  (test-assert "*blas-size-threshold* is a positive integer"
    (and (exact-integer? *blas-size-threshold*)
         (> *blas-size-threshold* 0)))

  (test-assert "make-blas-backend constructs a record with correct name"
    (let ((b (make-blas-backend 'test-backend
                #f #f #f #f #f #f #f #f #f #f)))
      (and (blas-backend? b)
           (eq? 'test-backend (blas-backend-name b)))))

  (test-assert "make-blas-backend stores kernel slots"
    ;; Use a dummy procedure for all slots and verify retrieval.
    (let* ((dummy (lambda args 0))
           (b (make-blas-backend 'dummy-backend
                dummy dummy dummy dummy dummy dummy dummy dummy dummy dummy)))
      (and (eq? dummy (blas-backend-gemm-f64 b))
           (eq? dummy (blas-backend-gemm-f32 b))
           (eq? dummy (blas-backend-gemv-f64 b))
           (eq? dummy (blas-backend-gemv-f32 b))
           (eq? dummy (blas-backend-dot-f64  b))
           (eq? dummy (blas-backend-dot-f32  b))
           (eq? dummy (blas-backend-axpy-f64 b))
           (eq? dummy (blas-backend-axpy-f32 b)))))

  (test-assert "register-blas-backend! makes blas-available? return #t"
    (let* ((dummy (lambda args 0))
           (b (make-blas-backend 'test-backend
                dummy dummy dummy dummy dummy dummy dummy dummy dummy dummy)))
      (register-blas-backend! b)
      (let ((r (blas-available?)))
        ;; Deregister so other tests are unaffected
        (set! *active-backend* #f)
        r)))

  (test-assert "active-blas-backend returns registered record"
    (let* ((dummy (lambda args 0))
           (b (make-blas-backend 'my-blas
                dummy dummy dummy dummy dummy dummy dummy dummy dummy dummy)))
      (register-blas-backend! b)
      (let ((r (eq? b (active-blas-backend))))
        (set! *active-backend* #f)
        r)))

  (test-assert "execute-blas-gemm falls back when BLAS disabled"
    (let* ((A (morph-from-list '((2 0) (0 2)) #(2 2) 'f64))
           (B (morph-from-list '((1 0) (0 1)) #(2 2) 'f64)))
      (disable-blas!)
      (let ((C (execute-blas-gemm A B)))
        (enable-blas!)
        (concrete-values-approx? C '((2 0) (0 2))))))

  (test-assert "execute-blas-gemm falls back when no backend registered"
    ;; Even with BLAS enabled, no active backend -> Scheme fallback.
    (let* ((A (morph-from-list '((3 0) (0 3)) #(2 2) 'f64))
           (B (morph-from-list '((1 0) (0 1)) #(2 2) 'f64))
           (saved *active-backend*))
      (set! *active-backend* #f)
      (let ((C (execute-blas-gemm A B)))
        (set! *active-backend* saved)
        (concrete-values-approx? C '((3 0) (0 3))))))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 11: Phase 3 - Realization Engine Integration
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Helper used across Phase 3 and Phase 4 test groups.
;; Realizes expr twice (BLAS on, then off) and checks results agree.
(define (blas-matches-scheme? expr #!optional (tol 1e-10))
  (enable-blas!)
  (let ((with-blas (realize expr)))
    (disable-blas!)
    (let ((without-blas (realize expr)))
      (enable-blas!)
      (let* ((a (flatten-nested-list (morph->list with-blas)))
             (b (flatten-nested-list (morph->list without-blas))))
        (and (= (length a) (length b))
             (every (lambda (x y) (approx= x y tol)) a b))))))

(test-group "Phase 3 - realize dispatches through BLAS"

  (test-assert "realize(morph-matmul) produces correct result"
    (let* ((A (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (B (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (C (realize (morph-matmul A B))))
      (and (concrete-array? C)
           (equal? (get-morphism-shape C) #(2 2))
           (concrete-values-approx? C '((19 22) (43 50))))))

  (test-assert "realize(morph-matmul) rectangular (2x3)(3x2)"
    (let* ((A (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (B (morph-from-list '((7 8) (9 10) (11 12)) #(3 2) 'f64))
           (C (realize (morph-matmul A B))))
      (concrete-values-approx? C '((58 64) (139 154)))))

  (test-assert "realize(morph-matvec) produces correct result"
    (let* ((A (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (v (morph-from-list '(1 2 3) #(3) 'f64))
           (y (realize (morph-matvec A v))))
      (and (concrete-array? y)
           (equal? (get-morphism-shape y) #(2))
           (concrete-values-approx? y '(14 32)))))

  (test-assert "realize(morph-dot) produces scalar result"
    (let* ((v1 (morph-from-list '(1 2 3) #(3) 'f64))
           (v2 (morph-from-list '(4 5 6) #(3) 'f64))
           (r  (realize (morph-dot v1 v2))))
      (and (concrete-array? r)
           (equal? (get-morphism-shape r) #())
           (concrete-values-approx? r '(32)))))

  (test-assert "realize(morph-matmul) BLAS matches Scheme fallback"
    (let* ((A (morph-from-list (iota 9 1) #(3 3) 'f64))
           (B (morph-from-list (iota 9 1) #(3 3) 'f64)))
      (blas-matches-scheme? (morph-matmul A B))))

  (test-assert "realize(morph-matvec) BLAS matches Scheme fallback"
    (let* ((A (morph-from-list (iota 6 1) #(2 3) 'f64))
           (v (morph-from-list (iota 3 1) #(3) 'f64)))
      (blas-matches-scheme? (morph-matvec A v))))

  (test-assert "realize(morph-dot) BLAS matches Scheme fallback"
    (let* ((v1 (morph-from-list (iota 5 1) #(5) 'f64))
           (v2 (morph-from-list (iota 5 1) #(5) 'f64)))
      (blas-matches-scheme? (morph-dot v1 v2))))

  (test-assert "realize with BLAS disabled uses standard path"
    (begin
      (disable-blas!)
      (let* ((A (morph-from-list '((1 0) (0 1)) #(2 2) 'f64))
             (B (morph-from-list '((3 4) (5 6)) #(2 2) 'f64))
             (C (realize (morph-matmul A B))))
        (enable-blas!)
        (concrete-values-approx? C '((3 4) (5 6))))))

  (test-assert "morph-matmul of abstract operands realized correctly"
    ;; A and B are abstract morphism-exprs; realize must recurse before BLAS
    (let* ((raw-A (morph-from-list '((1 0) (0 1)) #(2 2) 'f64))
           (raw-B (morph-from-list '((2 3) (4 5)) #(2 2) 'f64))
           (A  (morph+ raw-A (morph-from-list '((0 0) (0 0)) #(2 2) 'f64)))
           (B  (morph+ raw-B (morph-from-list '((0 0) (0 0)) #(2 2) 'f64)))
           (C  (realize (morph-matmul A B))))
      (concrete-values-approx? C '((2 3) (4 5)))))

  (test-assert "realize(morph-matmul) f32 dtype"
    (let* ((A (morph-from-list '((1 2) (3 4)) #(2 2) 'f32))
           (B (morph-from-list '((5 6) (7 8)) #(2 2) 'f32))
           (C (realize (morph-matmul A B))))
      (and (eq? (get-morphism-dtype C) 'f32)
           (concrete-values-approx? C '((19 22) (43 50)) 1e-4))))

  (test-assert "blas-enabled? accessible from realization module"
    (boolean? (blas-enabled?)))

  (test-assert "enable-blas!/disable-blas! accessible from realization module"
    (begin
      (disable-blas!)
      (let ((r (not (blas-enabled?))))
        (enable-blas!)
        r)))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 12: Phase 4 - Convolution via im2col + BLAS
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 4 - Conv2D via im2col+BLAS"

  ;; -------------------------------------------------------------------------
  ;; Pattern detection
  ;; -------------------------------------------------------------------------

  (test-assert "detect-conv2d-pattern returns list for reshape(matmul(w,im2col(x)))"
    (let* ((input  (morph-from-list (iota 16 1) #(1 4 4) 'f64))
           (weight (morph-from-list (make-list 9 1.0) #(1 1 3 3) 'f64))
           (w2d    (morph-reshape weight #(1 9)))   ; flatten to 2-D for matmul
           (col    (im2col-morph input '(3 3) 1 0))
           (mm     (morph-matmul w2d col))
           (expr   (morph-reshape mm #(1 2 2))))
      (list? (detect-conv2d-pattern expr))))

  (test-assert "detect-conv2d-pattern returns #f for plain reshape"
    (let* ((m    (morph-from-list (iota 6 1) #(2 3) 'f64))
           (expr (morph-reshape m #(3 2))))
      (not (detect-conv2d-pattern expr))))

  (test-assert "detect-conv2d-pattern returns #f for matmul (no reshape wrapper)"
    (let* ((A (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (B (morph-from-list '((5 6) (7 8)) #(2 2) 'f64)))
      (not (detect-conv2d-pattern (morph-matmul A B)))))

  (test-assert "detect-conv2d-pattern returns #f for reshape(add(...))"
    (let* ((A    (morph-from-list '(1 2 3 4) #(4) 'f64))
           (B    (morph-from-list '(1 1 1 1) #(4) 'f64))
           (expr (morph-reshape (morph+ A B) #(2 2))))
      (not (detect-conv2d-pattern expr))))

  ;; -------------------------------------------------------------------------
  ;; Non-batched correctness
  ;; Input (C=1,H=4,W=4), weight (C_out=1,C_in=1,KH=3,KW=3) all-ones,
  ;; stride=1, padding=0 -> output (1,2,2)
  ;;
  ;; Manual computation (sum of 3x3 windows):
  ;;   [0,0]: 1+2+3+5+6+7+9+10+11 = 54
  ;;   [0,1]: 2+3+4+6+7+8+10+11+12 = 63
  ;;   [1,0]: 5+6+7+9+10+11+13+14+15 = 90
  ;;   [1,1]: 6+7+8+10+11+12+14+15+16 = 99
  ;; -------------------------------------------------------------------------

  (test-assert "realize(conv2d) non-batched shape is (1,2,2)"
    (let* ((input  (morph-from-list (iota 16 1) #(1 4 4) 'f64))
           (weight (morph-from-list (make-list 9 1.0) #(1 1 3 3) 'f64))
           (w2d    (morph-reshape weight #(1 9)))
           (col    (im2col-morph input '(3 3) 1 0))
           (mm     (morph-matmul w2d col))
           (result (realize (morph-reshape mm #(1 2 2)))))
      (and (concrete-array? result)
           (equal? (get-morphism-shape result) #(1 2 2)))))

  (test-assert "realize(conv2d) non-batched values correct"
    (let* ((input  (morph-from-list (iota 16 1) #(1 4 4) 'f64))
           (weight (morph-from-list (make-list 9 1.0) #(1 1 3 3) 'f64))
           (w2d    (morph-reshape weight #(1 9)))
           (col    (im2col-morph input '(3 3) 1 0))
           (mm     (morph-matmul w2d col))
           (result (realize (morph-reshape mm #(1 2 2)))))
      (concrete-values-approx? result '(54 63 90 99))))

  (test-assert "realize(conv2d) non-batched BLAS matches Scheme fallback"
    (let* ((input  (morph-from-list (iota 16 1) #(1 4 4) 'f64))
           (weight (morph-from-list (make-list 9 1.0) #(1 1 3 3) 'f64))
           (w2d    (morph-reshape weight #(1 9)))
           (col    (im2col-morph input '(3 3) 1 0))
           (mm     (morph-matmul w2d col))
           (expr   (morph-reshape mm #(1 2 2))))
      (blas-matches-scheme? expr)))

  ;; -------------------------------------------------------------------------
  ;; Non-batched with multiple output channels
  ;; Input (C=1,H=4,W=4), weight (C_out=2,C_in=1,KH=3,KW=3):
  ;;   filter 0: all ones -> (54 63 90 99)
  ;;   filter 1: all twos -> (108 126 180 198)
  ;; output shape: (2,2,2)
  ;; -------------------------------------------------------------------------

  (test-assert "realize(conv2d) non-batched 2-channel output shape is (2,2,2)"
    (let* ((input  (morph-from-list (iota 16 1) #(1 4 4) 'f64))
           (weight (morph-from-list (append (make-list 9 1.0) (make-list 9 2.0))
                                    #(2 1 3 3) 'f64))
           (w2d    (morph-reshape weight #(2 9)))
           (col    (im2col-morph input '(3 3) 1 0))
           (mm     (morph-matmul w2d col))
           (result (realize (morph-reshape mm #(2 2 2)))))
      (and (concrete-array? result)
           (equal? (get-morphism-shape result) #(2 2 2)))))

  (test-assert "realize(conv2d) non-batched 2-channel output values correct"
    (let* ((input  (morph-from-list (iota 16 1) #(1 4 4) 'f64))
           (weight (morph-from-list (append (make-list 9 1.0) (make-list 9 2.0))
                                    #(2 1 3 3) 'f64))
           (w2d    (morph-reshape weight #(2 9)))
           (col    (im2col-morph input '(3 3) 1 0))
           (mm     (morph-matmul w2d col))
           (result (realize (morph-reshape mm #(2 2 2)))))
      (concrete-values-approx? result '(54 63 90 99 108 126 180 198))))

  ;; Note: batched conv2d (N,C,H,W input -> 3-D im2col) cannot be expressed
  ;; as reshape(matmul(w2d, col)) because morph-matmul rejects 3-D B operands.
  ;; Batched conv2d support requires a dedicated morph-conv2d constructor
  ;; (deferred to a future phase).  The execute-conv2d-blas loop is exercised
  ;; indirectly through the non-batched tests above (N=1 is a degenerate batch).

  ;; -------------------------------------------------------------------------
  ;; f32 dtype
  ;; -------------------------------------------------------------------------

  (test-assert "realize(conv2d) f32 dtype preserved"
    (let* ((input  (morph-from-list (iota 16 1) #(1 4 4) 'f32))
           (weight (morph-from-list (make-list 9 1.0) #(1 1 3 3) 'f32))
           (w2d    (morph-reshape weight #(1 9)))
           (col    (im2col-morph input '(3 3) 1 0))
           (mm     (morph-matmul w2d col))
           (result (realize (morph-reshape mm #(1 2 2)))))
      (and (eq? (get-morphism-dtype result) 'f32)
           (concrete-values-approx? result '(54 63 90 99) 1e-3))))

  ;; -------------------------------------------------------------------------
  ;; BLAS disabled: pattern is skipped, standard fallback produces correct result
  ;; -------------------------------------------------------------------------

  (test-assert "realize(conv2d) with BLAS disabled falls back to standard path"
    (begin
      (disable-blas!)
      (let* ((input  (morph-from-list (iota 16 1) #(1 4 4) 'f64))
             (weight (morph-from-list (make-list 9 1.0) #(1 1 3 3) 'f64))
             (w2d    (morph-reshape weight #(1 9)))
             (col    (im2col-morph input '(3 3) 1 0))
             (mm     (morph-matmul w2d col))
             (result (realize (morph-reshape mm #(1 2 2)))))
        (enable-blas!)
        (concrete-values-approx? result '(54 63 90 99)))))

  ;; -------------------------------------------------------------------------
  ;; detect-conv2d-pattern and execute-conv2d-blas accessible from realization
  ;; -------------------------------------------------------------------------

  (test-assert "detect-conv2d-pattern exported from array-morphisms-realization"
    (procedure? detect-conv2d-pattern))

  (test-assert "execute-conv2d-blas exported from array-morphisms-realization"
    (procedure? execute-conv2d-blas))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 11: Strided GEMM -- array->gemm-blas-params and
;;;           execute-blas-gemm-strided/execute-blas-gemm-strided/into!
;;;
;;; These tests exercise the new stride-aware BLAS path that supports
;;; transposed zero-copy views (the var-matmul backward gradient pattern).
;;; All assertions hold without a BLAS backend because execute-blas-gemm-strided
;;; falls back to the stride-aware Scheme kernel (%scheme-gemm/into!).
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 2 - Strided GEMM"

  ;; ------------------------------------------------------------------
  ;; array->gemm-blas-params: parameter extraction from stride patterns
  ;; ------------------------------------------------------------------

  (test-assert "params: row-major (2x3) f64 -> no-trans, lda=3"
    ;; Standard row-major: shape (2,3), strides (3,1)
    ;; -> s1==1 and s0==C=3 -> 'no-trans, lda=3
    (let* ((m      (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (params (array->gemm-blas-params m)))
      (and params
           (eq? (cadr params) 'no-trans)
           (= (caddr params) 3))))

  (test-assert "params: transposed (3x2) view of (2x3) -> trans, lda=3"
    ;; Transposed view: shape (3,2), strides (1,3)
    ;; -> s0==1 and s1==R=3 -> 'trans, lda=3
    (let* ((m      (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (t      (realize (morph-transpose m)))
           (params (array->gemm-blas-params t)))
      (and params
           (eq? (cadr params) 'trans)
           (= (caddr params) 3))))

  (test-assert "params: 1-D vector -> #f"
    (not (array->gemm-blas-params
          (morph-from-list '(1 2 3) #(3) 'f64))))

  (test-assert "params: 3-D array -> #f"
    ;; array->gemm-blas-params requires exactly 2 dimensions
    (not (array->gemm-blas-params
          (morph-from-list (make-list 24 1.0) #(2 3 4) 'f64))))

  ;; ------------------------------------------------------------------
  ;; execute-blas-gemm-strided: correctness with known values
  ;; All cases fall back to the stride-aware Scheme kernel when no BLAS
  ;; backend is registered.
  ;; ------------------------------------------------------------------

  (test-assert "strided gemm: both row-major matches morph-matmul reference"
    ;; A (3x2) x B (2x3) -> (3x3); both operands are contiguous row-major
    (let* ((A   (morph-from-list '((1 2) (3 4) (5 6)) #(3 2) 'f64))
           (B   (morph-from-list '((7 8 9) (10 11 12)) #(2 3) 'f64))
           (ref (realize (morph-matmul A B)))
           (res (execute-blas-gemm-strided A B)))
      (concrete-values-approx? res (morph->list ref))))

  (test-assert "strided gemm: G x B^T (var-matmul dA backward pattern)"
    ;; G (2x2) x B^T (2x3, zero-copy transposed view of B(3x2)) -> (2x3)
    ;; G = [[1,2],[4,5]], B = [[1,0],[0,1],[1,1]]
    ;; B^T = [[1,0,1],[0,1,1]]
    ;; G x B^T = [[1+0, 0+2, 1+2],[4+0, 0+5, 4+5]] = [[1,2,3],[4,5,9]]
    (let* ((G   (morph-from-list '((1 2) (4 5)) #(2 2) 'f64))
           (B   (morph-from-list '((1 0) (0 1) (1 1)) #(3 2) 'f64))
           (Bt  (realize (morph-transpose B)))   ; shape (2,3), strides (1,2)
           (res (execute-blas-gemm-strided G Bt)))
      (concrete-values-approx? res '((1 2 3) (4 5 9)))))

  (test-assert "strided gemm: A^T x G (var-matmul dB backward pattern)"
    ;; A^T (2x3, zero-copy transposed view of A(3x2)) x G (3x3) -> (2x3)
    ;; A = [[1,2],[3,4],[5,6]]
    ;; A^T = [[1,3,5],[2,4,6]]
    ;; G = [[1,0,1],[0,1,0],[1,1,0]]
    ;; A^T x G = [[1+0+5, 0+3+5, 1+0+0],[2+0+6, 0+4+6, 2+0+0]] = [[6,8,1],[8,10,2]]
    (let* ((A   (morph-from-list '((1 2) (3 4) (5 6)) #(3 2) 'f64))
           (G   (morph-from-list '((1 0 1) (0 1 0) (1 1 0)) #(3 3) 'f64))
           (At  (realize (morph-transpose A)))   ; shape (2,3), strides (1,2)
           (res (execute-blas-gemm-strided At G)))
      (concrete-values-approx? res '((6 8 1) (8 10 2)))))

) ;; end group "Phase 2 - Strided GEMM"

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Run All Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-exit)
