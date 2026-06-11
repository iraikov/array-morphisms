;;; tests/test-attention.scm
;;; Test suite for fused attention kernel.
;;; (FUSED_ATTENTION_IMPLEMENTATION_PROPOSAL, Items 1-4)
;;;
;;; Organisation:
;;;   Group 1  - Static cost annotation (Item 1): estimated-materialization-bytes
;;;   Group 2  - Constructor validation (Item 4a): attention-morphism shapes/errors
;;;   Group 3  - Non-batched 2-D attention: uniform inputs (known exact result)
;;;   Group 4  - Non-batched 2-D attention: cross-validation vs pure-Scheme reference
;;;   Group 5  - Batched attention (rank 3): B=2 wrapping the 2-D instance
;;;   Group 6  - Transpose-into-reduction verification (Item 2)
;;;   Group 7  - Scale default (1/sqrt(dk))
;;;   Group 8  - Larger cross-validation (n=8, dk=4, dv=6)
;;;
;;; The pure-Scheme reference implementation (ref-attention) is independent
;;; of attention-morphism and exercises exactly the same algorithm so that
;;; discrepancies localise to the CHICKEN-typed-vector loops, not to the
;;; mathematical formulation.

(import scheme (chicken base))
(import test)
(import (only srfi-1 every iota make-list fold append-map))
(import datatype matchable)
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-blas-exec)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (approx= a b #!optional (tol 1e-9))
  (< (abs (- a b)) tol))

(define (flat-values m)
  "Extract all elements of concrete-array m as a flat list."
  (flatten-nested-list (morph->list m)))

(define (arrays-approx? m expected-flat #!optional (tol 1e-9))
  "True when flat elements of m match expected-flat within tol."
  (let ((actual (flat-values m)))
    (and (= (length actual) (length expected-flat))
         (every (lambda (a e) (approx= a e tol)) actual expected-flat))))

;;; Reference attention (plain Scheme, no morphisms).
;;; Computes scaled dot-product attention on flat lists:
;;;   Q-flat, K-flat : n*dk elements  (row-major)
;;;   V-flat         : n*dv elements  (row-major)
;;; Returns flat list of n*dv output elements.
(define (ref-attention Q-flat K-flat V-flat n dk dv scale)
  (define (row lst i cols)
    (let ((start (* i cols)))
      (let loop ((j 0) (acc '()))
        (if (= j cols) (reverse acc)
            (loop (+ j 1) (cons (list-ref lst (+ start j)) acc))))))
  (define (dot v1 v2)
    (apply + (map * v1 v2)))
  (apply append
    (map (lambda (i)
           (let* ((qi     (row Q-flat i dk))
                  ;; Pass 1: scores and row max
                  (scores (map (lambda (k)
                                 (* scale (dot qi (row K-flat k dk))))
                               (iota n)))
                  (rmax   (apply max scores))
                  ;; Pass 2: exp(s - rmax), Z
                  (exps   (map (lambda (s) (exp (- s rmax))) scores))
                  (Z      (apply + exps))
                  (w      (map (lambda (e) (/ e Z)) exps)))
             ;; Pass 3: Out[i,p] = sum_k w[k] * V[k,p]
             (map (lambda (p)
                    (apply + (map (lambda (k wk)
                                    (* wk (list-ref V-flat (+ (* k dv) p))))
                                  (iota n) w)))
                  (iota dv))))
         (iota n))))

(define (ref-attention-weights Q-flat K-flat n dk scale)
  "Return the n*n attention weight matrix (softmax rows) as a flat list."
  (define (row lst i cols)
    (let ((start (* i cols)))
      (let loop ((j 0) (acc '()))
        (if (= j cols) (reverse acc)
            (loop (+ j 1) (cons (list-ref lst (+ start j)) acc))))))
  (define (dot v1 v2) (apply + (map * v1 v2)))
  (apply append
    (map (lambda (i)
           (let* ((qi    (row Q-flat i dk))
                  (scores (map (lambda (k) (* scale (dot qi (row K-flat k dk)))) (iota n)))
                  (rmax  (apply max scores))
                  (exps  (map (lambda (s) (exp (- s rmax))) scores))
                  (Z     (apply + exps)))
             (map (lambda (e) (/ e Z)) exps)))
         (iota n))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Fixed test matrices (n=3, dk=4, dv=4)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define n3  3)
(define dk4 4)
(define dv4 4)
(define scale-default (/ 1.0 (sqrt 4.0)))   ; 0.5

;; Distinct non-trivial values chosen to exercise all code paths.
(define Q3x4-flat '( 1.0  2.0  0.5 -1.0
                    -0.5  1.0  2.0  0.0
                     0.0 -1.0  1.5  2.0))
(define K3x4-flat '( 1.0  0.0  1.0  0.0
                     0.0  1.0  0.0  1.0
                    -1.0  1.0 -1.0  1.0))
(define V3x4-flat '( 1.0  0.0  0.0  0.0
                     0.0  1.0  0.0  0.0
                     0.0  0.0  1.0  0.0))

(define Q3-morph (morph-from-list Q3x4-flat (vector n3 dk4) 'f64))
(define K3-morph (morph-from-list K3x4-flat (vector n3 dk4) 'f64))
(define V3-morph (morph-from-list V3x4-flat (vector n3 dv4) 'f64))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1: Static cost annotation (Item 1)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Item 1 - estimated-materialization-bytes"

  (test "f64 (3,4) array: 3*4*8 = 96 bytes"
    96
    (estimated-materialization-bytes Q3-morph))

  (test "f32 (2,3) array: 2*3*4 = 24 bytes"
    24
    (estimated-materialization-bytes (morph-from-list (make-list 6 1.0) (vector 2 3) 'f32)))

  (test "f64 abstract morphism-expr: shape-size * 8"
    (* n3 dv4 8)
    (estimated-materialization-bytes (attention-morphism Q3-morph K3-morph V3-morph)))

  (test "estimated bytes proportional to element count"
    (* 2 (estimated-materialization-bytes Q3-morph))
    (estimated-materialization-bytes (morph-from-list (make-list (* 2 n3 dk4) 1.0)
                                                       (vector (* 2 n3) dk4) 'f64)))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2: Constructor validation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "attention-morphism constructor"

  (test-assert "returns a morphism"
    (array-morphism? (attention-morphism Q3-morph K3-morph V3-morph)))

  (test "output shape (n, dv)"
    (vector n3 dv4)
    (get-morphism-shape (attention-morphism Q3-morph K3-morph V3-morph)))

  (test "output shape uses V's dv (not Q's dk)"
    (vector n3 6)
    (let ((V6 (morph-from-list (make-list (* n3 6) 0.0) (vector n3 6) 'f64)))
      (get-morphism-shape (attention-morphism Q3-morph K3-morph V6))))

  (test "output dtype is promoted f64"
    'f64
    (get-morphism-dtype (attention-morphism Q3-morph K3-morph V3-morph)))

  (test-assert "explicit scale stored in metadata"
    (let ((m (attention-morphism Q3-morph K3-morph V3-morph 0.25)))
      (cases array-morphism m
        (morphism-expr (_ _ _ _ _ meta _)
          (approx= (cdr (assq 'scale meta)) 0.25))
        (else #f))))

  (test-assert "default scale = 1/sqrt(dk)"
    (let ((m (attention-morphism Q3-morph K3-morph V3-morph)))
      (cases array-morphism m
        (morphism-expr (_ _ _ _ _ meta _)
          (approx= (cdr (assq 'scale meta)) scale-default))
        (else #f))))

  (test-error "mismatched Q/K shapes raises error"
    (let ((Kwrong (morph-from-list (make-list (* n3 5) 0.0) (vector n3 5) 'f64)))
      (attention-morphism Q3-morph Kwrong V3-morph)))

  (test-error "mismatched Q/V sequence length raises error"
    (let ((Vwrong (morph-from-list (make-list (* 2 dv4) 0.0) (vector 2 dv4) 'f64)))
      (attention-morphism Q3-morph K3-morph Vwrong)))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3: Non-batched 2-D attention -- uniform inputs (exact known result)
;;;
;;; When Q = K = V = all-ones(n, dk):
;;;   scores[i,k] = scale * dk  (same for all i,k)
;;;   softmax row = uniform: w[k] = 1/n  for all k
;;;   Out[i,p]   = (1/n) * sum_k V[k,p] = (1/n) * n * 1.0 = 1.0
;;; So the output is all-ones regardless of n, dk, scale.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Non-batched 2-D attention -- uniform inputs"

  (let* ((n  4)
         (dk 6)
         (dv 6)
         (ones-Q  (morph-from-list (make-list (* n dk) 1.0) (vector n dk)  'f64))
         (ones-K  (morph-from-list (make-list (* n dk) 1.0) (vector n dk)  'f64))
         (ones-V  (morph-from-list (make-list (* n dv) 1.0) (vector n dv)  'f64))
         (out     (realize (attention-morphism ones-Q ones-K ones-V))))

    (test "output shape"
      (vector n dv)
      (get-morphism-shape out))

    (test-assert "all output elements equal 1.0"
      (every (lambda (v) (approx= v 1.0)) (flat-values out)))

    (test-assert "output is concrete"
      (concrete-array? out))
  )

  ;; Attention weights should sum to 1 per row.
  ;; Verify via the reference helper on the non-trivial (n=3, dk=4) matrices.
  (let* ((weights (ref-attention-weights Q3x4-flat K3x4-flat n3 dk4 scale-default))
         (row-sums (map (lambda (i)
                          (apply + (map (lambda (k) (list-ref weights (+ (* i n3) k)))
                                        (iota n3))))
                        (iota n3))))
    (test-assert "attention weight rows sum to 1.0"
      (every (lambda (s) (approx= s 1.0 1e-14)) row-sums))
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4: Non-batched 2-D attention -- cross-validation vs reference
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Non-batched 2-D attention -- cross-validation"

  (let* ((expected (ref-attention Q3x4-flat K3x4-flat V3x4-flat
                                  n3 dk4 dv4 scale-default))
         (m   (attention-morphism Q3-morph K3-morph V3-morph))
         (out (realize m)))

    (test "output shape (n=3, dv=4)"
      (vector n3 dv4)
      (get-morphism-shape out))

    (test-assert "output values match reference to 1e-10"
      (arrays-approx? out expected 1e-10))
  )

  ;; Second instance: Q=K=identity-prefix, V=identity-prefix
  (let* ((n  3) (dk 3) (dv 3)
         (Q-flat '(1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 1.0))
         (K-flat '(1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 1.0))
         (V-flat '(1.0 2.0 3.0  4.0 5.0 6.0  7.0 8.0 9.0))
         (sc     (/ 1.0 (sqrt 3.0)))
         (Q-m    (morph-from-list Q-flat (vector n dk) 'f64))
         (K-m    (morph-from-list K-flat (vector n dk) 'f64))
         (V-m    (morph-from-list V-flat (vector n dv) 'f64))
         (expected (ref-attention Q-flat K-flat V-flat n dk dv sc))
         (out (realize (attention-morphism Q-m K-m V-m))))
    (test-assert "identity Q/K instance matches reference"
      (arrays-approx? out expected 1e-10))
  )

  ;; Verify that attention weights for n=3 principal instance sum to 1 after realization.
  (let* ((out    (realize (attention-morphism Q3-morph K3-morph V3-morph)))
         (out-flat (flat-values out))
         ;; The n=3 cross-check: output is a weighted combination of V rows.
         ;; If V = I_3x4, each output row is a weight vector padded with 0.
         ;; Weights for row i sum to 1 iff out[i,0]+out[i,1]+out[i,2] = 1
         ;; (since V = [e1, e2, e3] and 4th col is 0).
         (row-sums (map (lambda (i)
                          (apply + (map (lambda (p)
                                          (list-ref out-flat (+ (* i dv4) p)))
                                        (iota (- dv4 1)))))   ; columns 0-2 only
                        (iota n3))))
    (test-assert "output rows (col 0-2) sum to 1.0 (V=I3x4)"
      (every (lambda (s) (approx= s 1.0 1e-10)) row-sums))
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5: Batched attention (rank 3, B=2)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Batched attention (rank 3)"

  ;; Stack two copies of the 3x4 instance into a (2, 3, 4) batch.
  (let* ((B 2)
         (Q-flat2 (append Q3x4-flat Q3x4-flat))
         (K-flat2 (append K3x4-flat K3x4-flat))
         (V-flat2 (append V3x4-flat V3x4-flat))
         (Q-b  (morph-from-list Q-flat2 (vector B n3 dk4) 'f64))
         (K-b  (morph-from-list K-flat2 (vector B n3 dk4) 'f64))
         (V-b  (morph-from-list V3x4-flat (vector n3 dv4) 'f64)) ; non-batched V OK? no -- must match
         ;; For rank-3 attention, V must also be rank 3
         (V-b3 (morph-from-list V-flat2 (vector B n3 dv4) 'f64))
         (out  (realize (attention-morphism Q-b K-b V-b3)))
         ;; Reference: apply 2-D attention twice
         (ref  (append
                (ref-attention Q3x4-flat K3x4-flat V3x4-flat n3 dk4 dv4 scale-default)
                (ref-attention Q3x4-flat K3x4-flat V3x4-flat n3 dk4 dv4 scale-default))))

    (test "output shape (B=2, n=3, dv=4)"
      (vector B n3 dv4)
      (get-morphism-shape out))

    (test-assert "batched output matches 2x reference to 1e-10"
      (arrays-approx? out ref 1e-10))

    (test-assert "batch-element 0 matches non-batched result"
      (let* ((out-2d (realize (attention-morphism Q3-morph K3-morph V3-morph)))
             (out-flat (flat-values out))
             (out-2d-flat (flat-values out-2d))
             (n-elem (* n3 dv4)))
        (every (lambda (i)
                 (approx= (list-ref out-flat i)
                          (list-ref out-2d-flat i) 1e-10))
               (iota n-elem))))
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6: Transpose-into-reduction verification (Item 2)
;;;
;;; matmul(A, transpose(B)) must equal matmul(A, B_explicit) where B_explicit
;;; is a freshly-materialized transpose of B.  This confirms the zero-copy
;;; strided-GEMM path handles transposed views without materializing a copy.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Transpose-into-reduction (Item 2)"

  ;; 3x4 * (3x4)^T = 3x3
  (let* ((A (morph-from-list Q3x4-flat (vector n3 dk4) 'f64))
         (B (morph-from-list K3x4-flat (vector n3 dk4) 'f64))
         ;; Path 1: morphism with transposed B (zero-copy view)
         (Bt       (morph-transpose B (vector 1 0)))
         (out-view (realize (morph-matmul A Bt)))
         ;; Path 2: explicit contiguous transposition, then matmul
         (B-data   (flat-values (realize B)))
         (Bt-data  (let loop ((i 0) (j 0) (acc '()))
                     (if (= i dk4) (reverse acc)
                         (if (= j n3)
                             (loop (+ i 1) 0 acc)
                             (loop i (+ j 1)
                                   (cons (list-ref B-data (+ (* j dk4) i)) acc))))))
         (Bt-explicit (morph-from-list Bt-data (vector dk4 n3) 'f64))
         (out-explicit (realize (morph-matmul A Bt-explicit))))

    (test-assert "matmul(A, transpose(B)) shape is (n,n)"
      (equal? (vector n3 n3) (get-morphism-shape out-view)))

    (test-assert "zero-copy transpose path matches explicit transpose"
      (arrays-approx? out-view (flat-values out-explicit) 1e-12))
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 7: Scale default 1/sqrt(dk)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Scale default"

  (let* ((explicit-scale (/ 1.0 (sqrt (exact->inexact dk4))))
         (out-default  (realize (attention-morphism Q3-morph K3-morph V3-morph)))
         (out-explicit (realize (attention-morphism Q3-morph K3-morph V3-morph
                                                   explicit-scale))))
    (test-assert "attention-morphism with default scale matches explicit 1/sqrt(dk)"
      (arrays-approx? out-default (flat-values out-explicit) 1e-14))
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 8: Larger cross-validation (n=8, dk=4, dv=6)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Larger cross-validation (n=8, dk=4, dv=6)"

  (let* ((n  8) (dk 4) (dv 6)
         (sc (/ 1.0 (sqrt (exact->inexact dk))))
         ;; Manually specified flat values (avoids random number dependency)
         (Q-flat (list  0.5 -0.3  1.2  0.8
                        0.1  0.9 -0.4  0.6
                       -0.7  0.2  0.5 -0.9
                        1.1 -0.5  0.3  0.4
                       -0.2  0.8 -0.6  1.0
                        0.6  0.1  0.9 -0.3
                       -0.8  0.4 -0.1  0.7
                        0.3 -0.9  0.6  0.2))
         (K-flat (list  0.4 -0.1  0.8  0.3
                        0.7  0.5 -0.2  0.9
                       -0.3  1.0  0.4 -0.6
                        0.2  0.6  0.1  0.5
                       -0.9  0.3  0.7  0.1
                        0.5 -0.7  0.2  0.8
                        0.1  0.4 -0.5  0.6
                       -0.4  0.8  0.3 -0.2))
         (V-flat (list  0.1  0.2  0.3  0.4  0.5  0.6
                        0.7  0.8  0.9  1.0  0.1  0.2
                        0.3  0.4  0.5  0.6  0.7  0.8
                        0.9  1.0  0.1  0.2  0.3  0.4
                        0.5  0.6  0.7  0.8  0.9  1.0
                        0.1  0.3  0.5  0.7  0.9  0.2
                        0.4  0.6  0.8  1.0  0.2  0.4
                        0.6  0.8  1.0  0.2  0.4  0.6))
         (Q-m (morph-from-list Q-flat (vector n dk) 'f64))
         (K-m (morph-from-list K-flat (vector n dk) 'f64))
         (V-m (morph-from-list V-flat (vector n dv) 'f64))
         (expected (ref-attention Q-flat K-flat V-flat n dk dv sc))
         (out (realize (attention-morphism Q-m K-m V-m))))

    (test "output shape (n=8, dv=6)"
      (vector n dv)
      (get-morphism-shape out))

    (test-assert "output matches reference to 1e-10"
      (arrays-approx? out expected 1e-10))

    ;; Verify attention weight rows still sum to 1 for a larger instance
    (let* ((weights (ref-attention-weights Q-flat K-flat n dk sc))
           (row-sums (map (lambda (i)
                            (apply + (map (lambda (k) (list-ref weights (+ (* i n) k)))
                                          (iota n))))
                          (iota n))))
      (test-assert "attention weight rows sum to 1 for n=8 instance"
        (every (lambda (s) (approx= s 1.0 1e-13)) row-sums)))
  )
)

(test-exit)
