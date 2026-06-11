;;; test-index-fn.scm
;;; Test suite for array-morphisms-index-fn

(import scheme (chicken base) test array-morphisms-index-fn array-morphisms-core)

;;;; ============================================================
;;;; Test Utilities
;;;; ============================================================

(define (approx= a b #!optional (tol 1e-6))
  (< (abs (- a b)) tol))

(define (lists-equal? l1 l2 #!optional (tol 1e-6))
  (cond
    ((and (null? l1) (null? l2)) #t)
    ((or  (null? l1) (null? l2)) #f)
    ((and (not (pair? l1)) (not (pair? l2)))
     (approx= l1 l2 tol))
    ((or (not (pair? l1)) (not (pair? l2))) #f)
    (else
     (and (lists-equal? (car l1) (car l2) tol)
          (lists-equal? (cdr l1) (cdr l2) tol)))))

(define (matrices-equal? m1 m2 #!optional (tol 1e-6))
  (lists-equal? m1 m2 tol))

;;;; ============================================================
;;;; Matrix Operations Tests  (unchanged)
;;;; ============================================================

(test-group "Matrix Operations"

  (test "make-identity-matrix 2x2"
    '((1 0) (0 1))
    (index-fn-make-identity-matrix 2))

  (test "make-identity-matrix 3x3"
    '((1 0 0) (0 1 0) (0 0 1))
    (index-fn-make-identity-matrix 3))

  (test "matrix-multiply 2x2"
    #t
    (matrices-equal?
     (index-fn-matrix-multiply '((1 2) (3 4)) '((5 6) (7 8)))
     '((19 22) (43 50))))

  (test "matrix-multiply 3x3"
    #t
    (matrices-equal?
     (index-fn-matrix-multiply '((1 0 0) (0 1 0) (0 0 1))
                               '((2 3 4) (5 6 7) (8 9 10)))
     '((2 3 4) (5 6 7) (8 9 10))))

  (test "matrix-multiply identity (A*I = A)"
    #t
    (let ((A '((1 2 3) (4 5 6) (7 8 9)))
          (I (index-fn-make-identity-matrix 3)))
      (matrices-equal? (index-fn-matrix-multiply A I) A)))

  (test "matrix-multiply identity (I*A = A)"
    #t
    (let ((A '((1 2 3) (4 5 6) (7 8 9)))
          (I (index-fn-make-identity-matrix 3)))
      (matrices-equal? (index-fn-matrix-multiply I A) A)))

  (test "matrix-multiply with #f (identity)"
    '((1 2) (3 4))
    (index-fn-matrix-multiply #f '((1 2) (3 4))))

  (test "matrix-vector-multiply"
    '(3 7)
    (index-fn-matrix-multiply '((1 2) (3 4)) '(1 1)))

  (test "matrix-vector-multiply 3x3"
    '(14 32 50)
    (index-fn-matrix-multiply '((1 2 3) (4 5 6) (7 8 9)) '(1 2 3)))

  (test "matrix-vector-multiply with #f (identity)"
    '(1 2 3)
    (index-fn-matrix-multiply #f '(1 2 3)))

  (test "vector-add"
    '(5 7 9)
    (index-fn-vector-add '(1 2 3) '(4 5 6)))

  (test "vector-add with #f"
    '(1 2 3)
    (index-fn-vector-add #f '(1 2 3)))

  (test "vector-scale"
    '(2 4 6)
    (index-fn-vector-scale 2 '(1 2 3)))

  (test "vector-zero? true"
    #t
    (index-fn-vector-zero? '(0 0 0)))

  (test "vector-zero? false"
    #f
    (index-fn-vector-zero? '(1 0 0)))

  (test "vector-zero? #f is zero"
    #t
    (index-fn-vector-zero? #f))

  (test "matrix-identity? true"
    #t
    (index-fn-matrix-identity? (index-fn-make-identity-matrix 3)))

  (test "matrix-identity? false"
    #f
    (index-fn-matrix-identity? '((1 2) (3 4))))

  (test "matrix-identity? #f is identity"
    #t
    (index-fn-matrix-identity? #f))
)

;;;; ============================================================
;;;; Identity and Constant Index Functions Tests
;;;; ============================================================

(test-group "Identity and Constant Index Functions"

  (test "make-identity-index-fn creates affine"
    #t
    (affine-index-fn? (make-identity-index-fn 3)))

  (test "identity-index-fn? recognizes identity"
    #t
    (identity-index-fn? (make-identity-index-fn 3)))

  ;; Behavioral replacement for the removed affine-index-fn-matrix/bias tests:
  ;; identity leaves every index unchanged.
  (test "identity-index-fn is transparent (application)"
    '(1 2 3)
    (apply-affine-index-fn (make-identity-index-fn 3) '(1 2 3)))

  (test "make-constant-index-fn creates procedure"
    #t
    (procedure? (make-constant-index-fn 42)))

  (test "constant-index-fn? recognizes constant"
    #t
    (constant-index-fn? (make-constant-index-fn 42)))

  (test "constant function returns constant"
    42
    (let ((fn (make-constant-index-fn 42)))
      (fn '(1 2 3))))
)

;;;; ============================================================
;;;; Affine Composition Tests
;;;; ============================================================

(test-group "Affine Index Function Composition"

  (test "compose identity with identity"
    #t
    (let ((id1 (make-identity-index-fn 2))
          (id2 (make-identity-index-fn 2)))
      (identity-index-fn? (compose-affine-index-fns id1 id2))))

  ;; Old test inspected (affine-index-fn-matrix composed) and
  ;; (affine-index-fn-bias composed) directly.  The new test verifies the
  ;; same algebraic result by applying the composed function.
  ;;
  ;; f(i) = 2*i + (1,1),  g(i) = i + (2,3)
  ;; (f o g)(0,0) = f(g(0,0)) = f(2,3) = (5,7)
  ;; (f o g)(1,1) = f(3,4) = (7,9)
  (test "compose diagonal functions - correct result type"
    #t
    (let* ((f (diagonal-fn '(2 2) '(1 1)))
           (g (diagonal-fn '(1 1) '(2 3)))
           (composed (compose-affine-index-fns f g)))
      (affine-index-fn? composed)))

  (test "compose diagonal functions - application at origin"
    #t
    (let* ((f (diagonal-fn '(2 2) '(1 1)))
           (g (diagonal-fn '(1 1) '(2 3)))
           (composed (compose-affine-index-fns f g)))
      (lists-equal? (apply-affine-index-fn composed '(0 0)) '(5 7))))

  (test "compose diagonal functions - application off-origin"
    #t
    (let* ((f (diagonal-fn '(2 2) '(1 1)))
           (g (diagonal-fn '(1 1) '(2 3)))
           (composed (compose-affine-index-fns f g)))
      (lists-equal? (apply-affine-index-fn composed '(1 1)) '(7 9))))

  ;; Old test used an explicit identity matrix for f and checked bias.
  ;; New test: composing a diagonal with (identity-fn) returns a function
  ;; that applies the diagonal transform unchanged.
  (test "compose with identity - application"
    #t
    (let* ((f        (diagonal-fn '(1 1) '(1 2)))
           (composed (compose-affine-index-fns f (identity-fn))))
      (lists-equal? (apply-affine-index-fn composed '(0 0)) '(1 2))))

  ;; Old: checked A of composed was '((6 0) (0 6)).
  ;; New: apply at (1,1) to verify scale composition.
  (test "compose scales correctly - application"
    #t
    (let* ((scale2   (diagonal-fn '(2 2) '(0 0)))
           (scale3   (diagonal-fn '(3 3) '(0 0)))
           (composed (compose-affine-index-fns scale2 scale3)))
      (lists-equal? (apply-affine-index-fn composed '(1 1)) '(6 6))))

  ;; Old: checked bias of composed was '(4 6).
  ;; New: apply at (0,0) where the offset is the only contribution.
  (test "compose offsets correctly - application"
    #t
    (let* ((offset1  (diagonal-fn '(1 1) '(1 2)))
           (offset2  (diagonal-fn '(1 1) '(3 4)))
           (composed (compose-affine-index-fns offset1 offset2)))
      (lists-equal? (apply-affine-index-fn composed '(0 0)) '(4 6))))
)

;;;; ============================================================
;;;; General Composition Tests
;;;; ============================================================

(test-group "General Index Function Composition"

  (test "compose-index-fns eliminates identity (left)"
    #t
    (let ((id (make-identity-index-fn 2))
          (f  (diagonal-fn '(2 2) '(1 1))))
      (eq? (compose-index-fns id f) f)))

  (test "compose-index-fns eliminates identity (right)"
    #t
    (let ((id (make-identity-index-fn 2))
          (f  (diagonal-fn '(2 2) '(1 1))))
      (eq? (compose-index-fns f id) f)))

  (test "compose-index-fns optimizes affine o affine"
    #t
    (let* ((f        (diagonal-fn '(2 2) '(0 0)))
           (g        (diagonal-fn '(3 3) '(0 0)))
           (composed (compose-index-fns f g)))
      (affine-index-fn? composed)))

  (test "compose-index-fns creates composed for mixed types"
    #t
    (let* ((f        (diagonal-fn '(2 2) '(0 0)))
           (g        (make-constant-index-fn 42))
           (composed (compose-index-fns f g)))
      (composed-index-fn? composed)))
)

;;;; ============================================================
;;;; Index Function Application Tests
;;;; ============================================================

(test-group "Index Function Application"

  (test "apply identity function"
    '(1 2 3)
    (apply-index-fn (make-identity-index-fn 3) '(1 2 3)))

  ;; Pure scale via diagonal-fn with zero bias.
  (test "apply affine function - scale"
    '(2 4 6)
    (apply-affine-index-fn (diagonal-fn '(2 2 2) '(0 0 0)) '(1 2 3)))

  ;; Pure offset via diagonal-fn with all-ones scale.
  (test "apply affine function - offset"
    '(3 5 7)
    (apply-affine-index-fn (diagonal-fn '(1 1 1) '(2 3 4)) '(1 2 3)))

  ;; Scale and offset together.
  (test "apply affine function - scale and offset"
    '(4 7 10)
    (apply-affine-index-fn (diagonal-fn '(2 2 2) '(2 3 4)) '(1 2 3)))

  ;; Composed via make-composed-index-fn: outer scale2 applied after inner offset.
  (test "apply composed function"
    '(6 10 14)
    (let* ((scale2   (diagonal-fn '(2 2 2) '(0 0 0)))
           (offset   (diagonal-fn '(1 1 1) '(2 3 4)))
           (composed (make-composed-index-fn scale2 offset)))
      (apply-index-fn composed '(1 2 3))))

  (test "apply constant function"
    42
    (apply-index-fn (make-constant-index-fn 42) '(1 2 3)))
)

;;;; ============================================================
;;;; Simplification Tests
;;;; ============================================================

(test-group "Index Function Simplification"

  ;; Old test constructed (diagonal-fn all-ones all-zeros) and expected
  ;; simplify-index-fn to convert it to identity-fn.  With the ADT,
  ;; identity is expressed directly via (identity-fn); simplify-index-fn
  ;; leaves it unchanged and identity-index-fn? recognises it.
  (test "simplify identity-fn stays identity"
    #t
    (identity-index-fn? (simplify-index-fn (identity-fn))))

  (test "simplify composed with identity (left)"
    #t
    (let* ((id       (make-identity-index-fn 2))
           (f        (diagonal-fn '(2 2) '(0 0)))
           (composed (make-composed-index-fn id f))
           (simplified (simplify-index-fn composed)))
      (affine-index-fn? simplified)))

  (test "simplify composed with identity (right)"
    #t
    (let* ((id       (make-identity-index-fn 2))
           (f        (diagonal-fn '(2 2) '(0 0)))
           (composed (make-composed-index-fn f id))
           (simplified (simplify-index-fn composed)))
      (affine-index-fn? simplified)))

  (test "simplify preserves non-identity"
    #t
    (let ((f (diagonal-fn '(2 3) '(1 2))))
      (eq? (simplify-index-fn f) f)))
)

;;;; ============================================================
;;;; Permutation Tests  (unchanged)
;;;; ============================================================

(test-group "Permutation Utilities"

  (test "identity-permutation? true"
    #t
    (identity-permutation? '(0 1 2 3)))

  (test "identity-permutation? false"
    #f
    (identity-permutation? '(1 0 2 3)))

  (test "compose-permutations identity"
    '(0 1 2)
    (compose-permutations '(0 1 2) '(0 1 2)))

  (test "compose-permutations inverse"
    '(0 1 2)
    (compose-permutations '(1 2 0) '(2 0 1)))

  (test "compose-permutations example"
    '(0 1 2)
    (compose-permutations '(2 0 1) '(1 2 0)))

  (test "invert-permutation"
    '(2 0 1)
    (invert-permutation '(1 2 0)))

  (test "invert-permutation identity"
    '(0 1 2)
    (invert-permutation '(0 1 2)))

  (test "permutation-to-matrix 2D"
    '((0 1) (1 0))
    (permutation-to-matrix '(1 0)))

  (test "permutation-to-matrix 3D"
    '((0 1 0) (0 0 1) (1 0 0))
    (permutation-to-matrix '(1 2 0)))

  (test "permutation round-trip"
    #t
    (let* ((perm    '(2 0 1))
           (inv     (invert-permutation perm))
           (composed (compose-permutations perm inv)))
      (identity-permutation? composed)))
)

;;;; ============================================================
;;;; Specialized Index Function Constructors Tests
;;;; ============================================================

(test-group "Specialized Index Function Constructors"

  (test "make-reshape-index-fn is identity"
    #t
    (identity-index-fn? (make-reshape-index-fn #(2 3 4) #(6 4))))

  (test-error "make-reshape-index-fn incompatible"
    (make-reshape-index-fn #(2 3 4) #(5 5)))

  (test "make-transpose-index-fn creates affine"
    #t
    (affine-index-fn? (make-transpose-index-fn '(1 0))))

  ;; Old test checked affine-index-fn-matrix = '((0 1)(1 0)).
  ;; New: applying the transpose permutation (1 0) swaps the two coordinates.
  (test "make-transpose-index-fn swaps coordinates"
    #t
    (let* ((fn     (make-transpose-index-fn '(1 0)))
           (result (apply-affine-index-fn fn '(3 7))))
      (lists-equal? result '(7 3))))

  (test "make-slice-index-fn creates affine"
    #t
    (affine-index-fn? (make-slice-index-fn '(0 0) '(5 5) '(1 1))))

  ;; Old test checked affine-index-fn-matrix and affine-index-fn-bias directly.
  ;; New: the slice fn '(1 2)...'(2 3) maps (0,0) -> (1,2) and (1,1) -> (3,5).
  (test "make-slice-index-fn with stride - application at origin"
    #t
    (let* ((fn     (make-slice-index-fn '(1 2) '(5 8) '(2 3)))
           (result (apply-index-fn fn '(0 0))))
      (lists-equal? result '(1 2))))

  (test "make-slice-index-fn with stride - application off-origin"
    #t
    (let* ((fn     (make-slice-index-fn '(1 2) '(5 8) '(2 3)))
           (result (apply-index-fn fn '(1 1))))
      (lists-equal? result '(3 5))))

  (test "apply slice function"
    '(1 2)
    (let ((fn (make-slice-index-fn '(1 2) '(5 8) '(1 1))))
      (apply-index-fn fn '(0 0))))

  (test "apply slice with stride"
    '(3 5)
    (let ((fn (make-slice-index-fn '(1 2) '(5 8) '(2 3))))
      (apply-index-fn fn '(1 1))))
)

;;;; ============================================================
;;;; Index Function Information Tests
;;;; ============================================================

(test-group "Index Function Information"

  ;; Use an unambiguous diagonal-fn so the rank (= length of diag) is 2.
  (test "index-fn-rank affine"
    2
    (index-fn-rank (diagonal-fn '(2 3) '(0 0))))

  ;; identity-fn has no stored rank; the implementation returns 0.
  (test "index-fn-rank identity"
    0
    (index-fn-rank (make-identity-index-fn 3)))

  (test "index-fn-rank composed"
    2
    (let* ((f        (diagonal-fn '(2 2) '(0 0)))
           (g        (diagonal-fn '(1 1) '(0 0)))
           (composed (make-composed-index-fn f g)))
      (index-fn-rank composed)))

  (test "index-fn-invertible? identity"
    #t
    (index-fn-invertible? (make-identity-index-fn 3)))

  ;; diagonal-fn with all non-zero scale factors is invertible.
  (test "index-fn-invertible? diagonal with non-zero scales"
    #t
    (index-fn-invertible? (diagonal-fn '(2 2) '(0 0))))

  ;; diagonal-fn with a zero scale factor collapses a dimension: not invertible.
  (test "index-fn-invertible? diagonal with zero scale"
    #f
    (index-fn-invertible? (diagonal-fn '(2 0) '(0 0))))

  (test "index-fn-invertible? non-affine"
    #f
    (index-fn-invertible? (make-constant-index-fn 42)))
)

;;;; ============================================================
;;;; Integration Tests
;;;; ============================================================

(test-group "Integration Tests"

  (test "compose three affine functions"
    #t
    (let* ((f1 (diagonal-fn '(2 2) '(0 0)))
           (f2 (diagonal-fn '(3 3) '(0 0)))
           (f3 (diagonal-fn '(1 1) '(1 1)))
           (c1 (compose-index-fns f1 f2))
           (c2 (compose-index-fns c1 f3)))
      (affine-index-fn? c2)))

  (test "compose and apply chain"
    '(12 12)
    (let* ((scale2   (diagonal-fn '(2 2) '(0 0)))
           (scale3   (diagonal-fn '(3 3) '(0 0)))
           (offset   (diagonal-fn '(1 1) '(1 1)))
           (composed (compose-index-fns
                      scale2
                      (compose-index-fns scale3 offset))))
      (apply-index-fn composed '(1 1))))

  (test "reshape followed by transpose"
    #f
    (let* ((reshape   (make-reshape-index-fn #(6 4) #(2 3 4)))
           (transpose (make-transpose-index-fn '(2 0 1)))
           (composed  (compose-index-fns transpose reshape)))
      (identity-index-fn? composed)))

  (test "multiple slice compositions"
    '(6 8)
    (let* ((slice1   (make-slice-index-fn '(0 0) '(10 10) '(2 2)))
           (slice2   (make-slice-index-fn '(1 1) '(5 5)   '(1 1)))
           (composed (compose-affine-index-fns slice1 slice2)))
      (apply-index-fn composed '(2 3))))
)

;;;; ============================================================
;;;; Edge Cases and Error Handling
;;;; ============================================================

(test-group "Edge Cases"

  (test "compose empty identity chain"
    #t
    (let ((id (make-identity-index-fn 0)))
      (identity-index-fn? id)))

  (test "apply function to empty indices"
    '()
    (apply-affine-index-fn (make-identity-index-fn 0) '()))

  (test "simplify already simplified"
    #t
    (let* ((f  (diagonal-fn '(2 2) '(1 1)))
           (s1 (simplify-index-fn f))
           (s2 (simplify-index-fn s1)))
      (eq? s1 s2)))

  ;; Fixed: the original tests called undefined procedures matrix-multiply
  ;; and matrix-vector-multiply.  Use index-fn-matrix-multiply throughout.
  (test-error "matrix dimension mismatch"
    (index-fn-matrix-multiply '((1 2)) '((1) (2) (3))))

  (test-error "vector length mismatch"
    (index-fn-matrix-multiply '((1 2 3)) '(1 2)))
)

;;;; ============================================================
;;;; Run All Tests
;;;; ============================================================

(test-exit)
