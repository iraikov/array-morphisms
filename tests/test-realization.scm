;;; test-array-morphisms-realization.scm
;;; Test Suite for MoA Realization Engine
;;;
;;; Tests materialization of abstract morphisms:
;;; - Concrete array pass-through
;;; - Affine operations (reshape, transpose, slice)
;;; - Computational operations (arithmetic, transcendental)
;;; - Window operations (im2col, padding)
;;; - Reduction operations (sum, mean, max, min)
;;; - Composed operations
;;; - Integration with previous phases

(import scheme chicken.base)
(import test)
(import (only srfi-1 make-list iota every fold))
(import datatype matchable)
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (approx= a b #!optional (tol 1e-6))
  "Test if two numbers are approximately equal"
  (< (abs (- a b)) tol))

(define (vectors-approx-equal? v1 v2 tol)
  "Test if two vectors are approximately equal element-wise"
  (let ((len1 (cond
                ((f32vector? v1) (f32vector-length v1))
                ((f64vector? v1) (f64vector-length v1))
                ((s32vector? v1) (s32vector-length v1))
                ((s64vector? v1) (s64vector-length v1))
                (else 0)))
        (len2 (cond
                ((f32vector? v2) (f32vector-length v2))
                ((f64vector? v2) (f64vector-length v2))
                ((s32vector? v2) (s32vector-length v2))
                ((s64vector? v2) (s64vector-length v2))
                (else 0))))
    
    (and (= len1 len2)
         (let ((dtype (cond
                        ((f32vector? v1) 'f32)
                        ((f64vector? v1) 'f64)
                        ((s32vector? v1) 's32)
                        ((s64vector? v1) 's64)
                        (else 'unknown))))
           (every (lambda (i)
                    (approx= (typed-vector-ref v1 dtype i)
                            (typed-vector-ref v2 dtype i)
                            tol))
                  (iota len1))))))

(define (morphism-values-equal? m expected-list #!optional (tol 1e-6))
  "Test if realized morphism has expected values"
  (let* ((realized (realize m))
         (actual-list (morph->list realized))
         (flat-actual (flatten-nested-list actual-list))
         (flat-expected (flatten-nested-list expected-list)))
    
    (and (= (length flat-actual) (length flat-expected))
         (every (lambda (a e)
                  (approx= a e tol))
                flat-actual flat-expected))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Concrete Array Pass-Through Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Concrete Arrays"
  
  (test-assert "realize concrete array returns same array"
    (let* ((m (morph-from-list '(1 2 3 4) #(4) 'f64))
           (realized (realize m)))
      (and (concrete-array? realized)
           (equal? (get-morphism-shape realized) #(4))
           (equal? (get-morphism-dtype realized) 'f64))))
  
  (test-assert "realize concrete array preserves values"
    (let ((m (morph-from-list '(1.0 2.0 3.0) #(3) 'f64)))
      (morphism-values-equal? m '(1.0 2.0 3.0))))
  
  (test-assert "realize concrete 2D array"
    (let ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64)))
      (morphism-values-equal? m '((1 2 3) (4 5 6)))))
  
  (test-assert "realize preserves dtype"
    (let* ((m-f32 (morph-from-list '(1.0 2.0) #(2) 'f32))
           (m-s32 (morph-from-list '(1 2 3) #(3) 's32))
           (r-f32 (realize m-f32))
           (r-s32 (realize m-s32)))
      (and (equal? 'f32 (get-morphism-dtype r-f32))
           (equal? 's32 (get-morphism-dtype r-s32))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Reshape Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Reshape"
  
  (test-assert "realize reshape preserves values"
    (let* ((m (morph-from-list '(1 2 3 4 5 6) #(6) 'f64))
           (reshaped (morph-reshape m #(2 3))))
      (morphism-values-equal? reshaped '((1 2 3) (4 5 6)))))
  
  (test-assert "realize reshape flatten"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (flat (morph-reshape m #(4))))
      (morphism-values-equal? flat '(1 2 3 4))))
  
  (test-assert "realize reshape with inference"
    (let* ((m (morph-from-list '(1 2 3 4 5 6) #(6) 'f64))
           (reshaped (morph-reshape m '(2 -1))))
      (morphism-values-equal? reshaped '((1 2 3) (4 5 6)))))
  
  (test-assert "realize reshape 3D"
    (let* ((m (morph-from-list (make-list 24 1.0) #(24) 'f64))
           (reshaped (morph-reshape m #(2 3 4)))
           (realized (realize reshaped)))
      (and (concrete-array? realized)
           (equal? (get-morphism-shape realized) #(2 3 4))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Transpose Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Transpose"
  
  (test-assert "realize 2D transpose"
    (let* ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (transposed (morph-transpose m)))
      (morphism-values-equal? transposed '((1 4) (2 5) (3 6)))))
  
  (test-assert "realize transpose preserves total size"
    (let* ((m (morph-from-list '((1 2) (3 4) (5 6)) #(3 2) 'f64))
           (transposed (morph-transpose m))
           (realized (realize transposed)))
      (= 6 (morph-size realized))))
  
  (test-assert "realize custom permutation"
    (let* ((m (morph-from-list '(((1 2) (3 4)) ((5 6) (7 8))) #(2 2 2) 'f64))
           ;; Swap first and last axes: (2,2,2) -> (2,2,2) with [2,1,0]
           (transposed (morph-transpose m '(2 1 0)))
           (realized (realize transposed)))
      (and (concrete-array? realized)
           (equal? (get-morphism-shape realized) #(2 2 2)))))
  
  (test-assert "realize transpose identity permutation"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (transposed (morph-transpose m '(0 1))))
      (morphism-values-equal? transposed '((1 2) (3 4))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Slice Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Slice"
  
  (test-assert "realize 1D slice"
    (let* ((m (morph-from-list '(0 1 2 3 4 5) #(6) 'f64))
           (sliced (morph-slice m '(1) '(4))))
      (morphism-values-equal? sliced '(1 2 3))))
  
  (test-assert "realize 1D slice with step"
    (let* ((m (morph-from-list '(0 1 2 3 4 5 6 7 8 9) #(10) 'f64))
           (sliced (morph-slice m '(0) '(10) 2)))
      (morphism-values-equal? sliced '(0 2 4 6 8))))
  
  (test-assert "realize 2D slice rows and columns"
    (let* ((m (morph-from-list '((1 2 3) (4 5 6) (7 8 9)) #(3 3) 'f64))
           (sliced (morph-slice m '(0 1) '(2 3))))
      (morphism-values-equal? sliced '((2 3) (5 6)))))
  
  (test-assert "realize 2D slice single row"
    (let* ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (sliced (morph-slice m '(1 0) '(2 3))))
      (morphism-values-equal? sliced '((4 5 6)))))
  
  (test-assert "realize slice with negative indices"
    (let* ((m (morph-from-list '(0 1 2 3 4) #(5) 'f64))
           (sliced (morph-slice m '(-2) '(-1))))
      (morphism-values-equal? sliced '(3)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Binary Arithmetic Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Binary Arithmetic"
  
  (test-assert "realize addition"
    (let* ((m1 (morph-from-list '(1 2 3) #(3) 'f64))
           (m2 (morph-from-list '(4 5 6) #(3) 'f64))
           (sum (morph+ m1 m2)))
      (morphism-values-equal? sum '(5 7 9))))
  
  (test-assert "realize subtraction"
    (let* ((m1 (morph-from-list '(10 20 30) #(3) 'f64))
           (m2 (morph-from-list '(1 2 3) #(3) 'f64))
           (diff (morph- m1 m2)))
      (morphism-values-equal? diff '(9 18 27))))
  
  (test-assert "realize multiplication"
    (let* ((m1 (morph-from-list '(2 3 4) #(3) 'f64))
           (m2 (morph-from-list '(5 6 7) #(3) 'f64))
           (prod (morph* m1 m2)))
      (morphism-values-equal? prod '(10 18 28))))
  
  (test-assert "realize division"
    (let* ((m1 (morph-from-list '(10.0 20.0 30.0) #(3) 'f64))
           (m2 (morph-from-list '(2.0 4.0 5.0) #(3) 'f64))
           (quot (morph/ m1 m2)))
      (morphism-values-equal? quot '(5.0 5.0 6.0))))
  
  (test-assert "realize power"
    (let* ((m1 (morph-from-list '(2.0 3.0 4.0) #(3) 'f64))
           (m2 (morph-from-list '(2.0 2.0 2.0) #(3) 'f64))
           (pow (morph-pow m1 m2)))
      (morphism-values-equal? pow '(4.0 9.0 16.0))))
  
  (test-assert "realize with broadcasting"
    (let* ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (m2 (morph-from-list '(10) #(1) 'f64))
           (sum (morph+ m1 m2)))
      (morphism-values-equal? sum '((11 12) (13 14))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Unary Operations Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Unary Operations"
  
  (test-assert "realize sqrt"
    (let* ((m (morph-from-list '(1.0 4.0 9.0 16.0) #(4) 'f64))
           (sqrt-m (morph-sqrt m)))
      (morphism-values-equal? sqrt-m '(1.0 2.0 3.0 4.0))))
  
  (test-assert "realize exp"
    (let* ((m (morph-from-list '(0.0 1.0) #(2) 'f64))
           (exp-m (morph-exp m)))
      (morphism-values-equal? exp-m (list 1.0 (exp 1.0)))))
  
  (test-assert "realize log"
    (let* ((m (morph-from-list '(1.0 2.71828) #(2) 'f64))
           (log-m (morph-log m)))
      (morphism-values-equal? log-m (list 0.0 1.0) 1e-4)))
  
  (test-assert "realize negate"
    (let* ((m (morph-from-list '(1 -2 3 -4) #(4) 'f64))
           (neg-m (morph-negate m)))
      (morphism-values-equal? neg-m '(-1 2 -3 4))))
  
  (test-assert "realize abs"
    (let* ((m (morph-from-list '(-1 2 -3 4) #(4) 'f64))
           (abs-m (morph-abs m)))
      (morphism-values-equal? abs-m '(1 2 3 4))))
  
  (test-assert "realize transcendental promotes integers"
    (let* ((m (morph-from-list '(1 4 9) #(3) 's32))
           (sqrt-m (morph-sqrt m))
           (realized (realize sqrt-m)))
      (equal? 'f64 (get-morphism-dtype realized)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Comparison Operations Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Comparisons"
  
  (test-assert "realize greater-than"
    (let* ((m1 (morph-from-list '(1 2 3) #(3) 'f64))
           (m2 (morph-from-list '(2 2 2) #(3) 'f64))
           (gt (morph> m1 m2)))
      (morphism-values-equal? gt '(0.0 0.0 1.0))))
  
  (test-assert "realize less-than"
    (let* ((m1 (morph-from-list '(1 2 3) #(3) 'f64))
           (m2 (morph-from-list '(2 2 2) #(3) 'f64))
           (lt (morph< m1 m2)))
      (morphism-values-equal? lt '(1.0 0.0 0.0))))
  
  (test-assert "realize equality"
    (let* ((m1 (morph-from-list '(1 2 3) #(3) 'f64))
           (m2 (morph-from-list '(1 2 4) #(3) 'f64))
           (eq (morph= m1 m2)))
      (morphism-values-equal? eq '(1.0 1.0 0.0)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Padding Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Padding"
  
  (test-assert "realize constant padding"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (padded (morph-pad m '((2 1)) 'constant 0.0)))
      (morphism-values-equal? padded '(0.0 0.0 1.0 2.0 3.0 0.0))))
  
  (test-assert "realize edge padding"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (padded (morph-pad m '((2 1)) 'edge)))
      (morphism-values-equal? padded '(1.0 1.0 1.0 2.0 3.0 3.0))))
  
  (test-assert "realize reflect padding"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (padded (morph-pad m '((2 2)) 'reflect)))
      (morphism-values-equal? padded '(2.0 1.0 1.0 2.0 3.0 3.0 2.0))))
  
  (test-assert "realize 2D constant padding"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (padded (morph-pad m '((1 1) (1 1)) 'constant 0.0)))
      (morphism-values-equal? padded 
        '((0.0 0.0 0.0 0.0)
          (0.0 1.0 2.0 0.0)
          (0.0 3.0 4.0 0.0)
          (0.0 0.0 0.0 0.0))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Reduction Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Reductions"
  
  (test-assert "realize sum all elements"
    (let* ((m (morph-from-list '(1 2 3 4) #(4) 'f64))
           (sum (morph-reduce 'sum m)))
      (morphism-values-equal? sum '(10.0))))
  
  (test-assert "realize sum along axis"
    (let* ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (sum (morph-reduce 'sum m '(0))))
      (morphism-values-equal? sum '(5.0 7.0 9.0))))
  
  (test-assert "realize sum keepdims"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (sum (morph-reduce 'sum m '(0) #t))
           (realized (realize sum)))
      (and (equal? (get-morphism-shape realized) #(1 2))
           (morphism-values-equal? sum '((4.0 6.0))))))
  
  (test-assert "realize mean"
    (let* ((m (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64))
           (mean (morph-reduce 'mean m)))
      (morphism-values-equal? mean '(2.5))))
  
  (test-assert "realize max"
    (let* ((m (morph-from-list '(1 5 3 2 4) #(5) 'f64))
           (max-m (morph-reduce 'max m)))
      (morphism-values-equal? max-m '(5.0))))
  
  (test-assert "realize min along axis"
    (let* ((m (morph-from-list '((3 1 4) (1 5 9)) #(2 3) 'f64))
           (min-m (morph-reduce 'min m '(1))))
      (morphism-values-equal? min-m '(1.0 1.0))))
  
  (test-assert "realize product"
    (let* ((m (morph-from-list '(2 3 4) #(3) 'f64))
           (prod (morph-reduce 'prod m)))
      (morphism-values-equal? prod '(24.0)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; im2col Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - im2col"
  
  (test-assert "realize im2col basic"
    (let* ((m (morph-from-list 
                '(((1 2 3) (4 5 6) (7 8 9)))  ; (1,3,3)
                #(1 3 3) 'f64))
           (col (im2col-morph m '(2 2) 1 0))
           (realized (realize col)))
      (and (concrete-array? realized)
           (equal? (get-morphism-shape realized) #(4 4)))))
  
  (test-assert "realize im2col with padding"
    (let* ((m (morph-from-list 
                '(((1 2) (3 4)))  ; (1,2,2)
                #(1 2 2) 'f64))
           (col (im2col-morph m '(3 3) 1 1))
           (realized (realize col)))
      (and (concrete-array? realized)
           ;; With padding=1, output is (2,2)
           ;; Shape: (C*KH*KW, OH*OW) = (1*3*3, 2*2) = (9, 4)
           (equal? (get-morphism-shape realized) #(9 4)))))
  
  (test-assert "realize im2col batched"
    (let* ((m (morph-from-list 
                (make-list 72 1.0)  ; (2,2,6,3)
                #(2 2 6 3) 'f64 batch-axis: 0))
           (col (im2col-morph m '(2 2) 1 0))
           (realized (realize col)))
      (and (concrete-array? realized)
           (equal? (get-morphism-shape realized) #(2 8 10))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Helper Operations Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Helper Operations"
  
  (test-assert "realize squeeze"
    (let* ((m (morph-from-list '(((1 2))) #(1 1 2) 'f64))
           (squeezed (morph-squeeze m)))
      (morphism-values-equal? squeezed '(1 2))))
  
  (test-assert "realize unsqueeze"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (unsqueezed (morph-unsqueeze m 0))
           (realized (realize unsqueezed)))
      (and (equal? (get-morphism-shape realized) #(1 3))
           (morphism-values-equal? unsqueezed '((1 2 3))))))
  
  (test-assert "realize morph-map"
    (let* ((m (morph-from-list '(1 2 3 4) #(4) 'f64))
           (squared (morph-map (lambda (x) (* x x)) m)))
      (morphism-values-equal? squared '(1 4 9 16)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Stack/Concat/Split Realization Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Batch Operations"
  
  (test-assert "realize stack"
    (let* ((m1 (morph-from-list '(1 2) #(2) 'f64))
           (m2 (morph-from-list '(3 4) #(2) 'f64))
           (m3 (morph-from-list '(5 6) #(2) 'f64))
           (stacked (morph-stack (list m1 m2 m3))))
      (morphism-values-equal? stacked '((1 2) (3 4) (5 6)))))
  
  (test-assert "realize concat"
    (let* ((m1 (morph-from-list '(1 2) #(2) 'f64))
           (m2 (morph-from-list '(3 4 5) #(3) 'f64))
           (concatenated (morph-concat (list m1 m2) 0)))
      (morphism-values-equal? concatenated '(1 2 3 4 5))))
  
  (test-assert "realize split then stack"
    (let* ((m (morph-from-list '(1 2 3 4 5 6) #(6) 'f64))
           (splits (morph-split m 3))
           (stacked (morph-stack splits)))
      (morphism-values-equal? stacked '((1 2) (3 4) (5 6))))))


(let* ((a1 (morph-from-list '(1 2 3) #(3) 'f64))
       (a2 (morph-from-list '(4 5 6) #(3) 'f64))
       (stacked (realize (morph-stack (list a1 a2) 0))))
  (print "concrete-array? stacked = " (concrete-array? stacked))
  (print "(equal? (get-morphism-shape stacked) #(2 3)) = "
         (equal? (get-morphism-shape stacked) #(2 3)))
  (print (morph->list stacked))
  )

(test-group "Stack Kernel - Basic Functionality"
  
  (test-assert "stack two 1D arrays"
    (let* ((a1 (morph-from-list '(1 2 3) #(3) 'f64))
           (a2 (morph-from-list '(4 5 6) #(3) 'f64))
           (stacked (realize (morph-stack (list a1 a2) 0))))
      
      (and (concrete-array? stacked)
           (equal? (get-morphism-shape stacked) #(2 3))
           (equal? (morph->list stacked) '((1.0 2.0 3.0)
                                           (4.0 5.0 6.0))))))
  
  (test-assert "stack two 2D arrays along axis 0"
    (let* ((a1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (a2 (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (stacked (realize (morph-stack (list a1 a2) 0))))
      
      (and (concrete-array? stacked)
           (equal? (get-morphism-shape stacked) #(2 2 2))
           (equal? (morph->list stacked) '(((1.0 2.0) (3.0 4.0)) ((5.0 6.0) (7.0 8.0)))))))
  
  (test-assert "stack three arrays"
    (let* ((a1 (morph-from-list '(1 2) #(2) 'f64))
           (a2 (morph-from-list '(3 4) #(2) 'f64))
           (a3 (morph-from-list '(5 6) #(2) 'f64))
           (stacked (realize (morph-stack (list a1 a2 a3) 0))))
      
      (and (concrete-array? stacked)
           (equal? (get-morphism-shape stacked) #(3 2))
           (equal? (morph->list stacked) '((1.0 2.0) (3.0 4.0) (5.0 6.0))))))
  
  (test-assert "stack 3D arrays (image batching case)"
    (let* ((img1 (morph-from-list '(((1 2 3) (4 5 6) (7 8 9))) #(1 3 3) 'f64))
           (img2 (morph-from-list '(((9 8 7) (6 5 4) (3 2 1))) #(1 3 3) 'f64))
           (batched (realize (morph-stack (list img1 img2) 0))))
      
      (and (concrete-array? batched)
           (equal? (get-morphism-shape batched) #(2 1 3 3))
           (= (get-morphism-batch-axis batched) 0)))))

(test-group "Stack Kernel - im2col Integration"
  
  (test-assert "im2col on stacked arrays"
    (let* ((img1 (morph-from-list '(((1 2 3) (4 5 6) (7 8 9))) #(1 3 3) 'f64))
           (img2 (morph-from-list '(((9 8 7) (6 5 4) (3 2 1))) #(1 3 3) 'f64))
           (batched (realize (morph-stack (list img1 img2) 0)))
           (col (realize (im2col-morph batched '(2 2) 1 0))))
      
      (and (concrete-array? col)
           (equal? (get-morphism-shape col) #(2 4 4)))))
  
  (test-assert "col2im on stacked then im2col'd arrays"
    (let* ((img1 (morph-from-list '(((1 2 3) (4 5 6) (7 8 9))) #(1 3 3) 'f64))
           (img2 (morph-from-list '(((9 8 7) (6 5 4) (3 2 1))) #(1 3 3) 'f64))
           (batched (realize (morph-stack (list img1 img2) 0)))
           (col (realize (im2col-morph batched '(2 2) 1 0)))
           (reconstructed (realize (col2im-morph col #(2 1 3 3) '(2 2) 1 0))))
      
      (and (concrete-array? reconstructed)
           (equal? (get-morphism-shape reconstructed) #(2 1 3 3)))))
  
  (test-assert "stacked matches direct batched array"
    ;; Verify stack produces same result as directly created batched array
    (let* ((img1 (morph-from-list '(((1 2) (3 4))) #(1 2 2) 'f64))
           (img2 (morph-from-list '(((5 6) (7 8))) #(1 2 2) 'f64))
           
           ;; Via stack
           (batched-stack (realize (morph-stack (list img1 img2) 0)))
           
           ;; Direct
           (batched-direct (morph-from-list 
                             '((((1 2) (3 4))) (((5 6) (7 8))))
                             #(2 1 2 2) 'f64)))
      
      (and (equal? (get-morphism-shape batched-stack) 
                   (get-morphism-shape batched-direct))
           (equal? (morph->list batched-stack)
                   (morph->list batched-direct))))))

(test-group "Stack Kernel - Edge Cases"
  
  (test-assert "stack single array"
    (let* ((a (morph-from-list '(1 2 3) #(3) 'f64))
           (stacked (realize (morph-stack (list a) 0))))
      
      (and (concrete-array? stacked)
           (equal? (get-morphism-shape stacked) #(1 3)))))
  
  (test-assert "stack with different dtypes promotes correctly"
    (let* ((a1 (morph-from-list '(1 2) #(2) 'f32))
           (a2 (morph-from-list '(3 4) #(2) 'f64))
           (stacked (realize (morph-stack (list a1 a2) 0))))
      
      ;; Should promote to f64
      (and (concrete-array? stacked)
           (eq? (get-morphism-dtype stacked) 'f64)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Integration Tests - Chained Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Integration"
  
  (test-assert "realize reshape then transpose"
    (let* ((m (morph-from-list '(1 2 3 4 5 6) #(6) 'f64))
           (reshaped (morph-reshape m #(2 3)))
           (transposed (morph-transpose reshaped)))
      (morphism-values-equal? transposed '((1 4) (2 5) (3 6)))))
  
  (test-assert "realize arithmetic then reduction"
    (let* ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (m2 (morph-from-list '((2 2) (2 2)) #(2 2) 'f64))
           (prod (morph* m1 m2))
           (sum (morph-reduce 'sum prod)))
      (morphism-values-equal? sum '(20.0))))
  
  (test-assert "realize slice then arithmetic"
    (let* ((m (morph-from-list '(1 2 3 4 5) #(5) 'f64))
           (sliced (morph-slice m '(1) '(4)))
           (doubled (morph* sliced (morph-from-list '(2 2 2) #(3) 'f64))))
      (morphism-values-equal? doubled '(4 6 8))))
  
  (test-assert "realize complex pipeline"
    (let* ((m (morph-from-list '(1 2 3 4) #(4) 'f64))
           (reshaped (morph-reshape m #(2 2)))
           (plus-ten (morph+ reshaped (morph-from-list '(10) #(1) 'f64)))
           (transposed (morph-transpose plus-ten))
           (sum (morph-reduce 'sum transposed '(0))))
      (morphism-values-equal? sum '(23.0 27.0))))
  
  (test-assert "realize nested operations"
    (let* ((m (morph-from-list '(1 2 3 4) #(4) 'f64))
           (sqrt-m (morph-sqrt m))
           (doubled (morph* sqrt-m (morph-from-list '(2) #(1) 'f64)))
           (sum (morph-reduce 'sum doubled)))
      (morphism-values-equal? sum '(12.2925) 1e-4)))
  
  (test-assert "realize with multiple reshapes"
    (let* ((m (morph-from-list (make-list 24 1.0) #(24) 'f64))
           (r1 (morph-reshape m #(2 3 4)))
           (r2 (morph-reshape r1 #(6 4)))
           (r3 (morph-reshape r2 #(24)))
           (realized (realize r3)))
      (and (equal? (get-morphism-shape realized) #(24))
           (morphism-values-equal? r3 (make-list 24 1.0))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Edge Cases and Error Handling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Edge Cases"
  
  (test-assert "realize empty reduction produces scalar"
    (let* ((m (morph-from-list '(42.0) #(1) 'f64))
           (sum (morph-reduce 'sum m))
           (realized (realize sum)))
      (and (equal? (get-morphism-shape realized) #())
           (morphism-values-equal? sum '(42.0)))))
  
  (test-assert "realize single element operations"
    (let* ((m (morph-from-list '(5.0) #(1) 'f64))
           (sqrt-m (morph-sqrt m)))
      (morphism-values-equal? sqrt-m (list (sqrt 5.0)) 1e-6)))
  
  (test-assert "realize with different dtypes promotes"
    (let* ((m1 (morph-from-list '(1 2 3) #(3) 's32))
           (m2 (morph-from-list '(1.5 2.5 3.5) #(3) 'f64))
           (sum (morph+ m1 m2))
           (realized (realize sum)))
      (equal? 'f64 (get-morphism-dtype realized))))
  
  (test-assert "realize preserves batch axis through pipeline"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64 batch-axis: 0))
           (doubled (morph* m (morph-from-list '(2) #(1) 'f64)))
           (realized (realize doubled)))
      (= 0 (get-morphism-batch-axis realized)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Performance Characteristics
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "realize - Performance"
  
  (test-assert "realize large array"
    (let* ((m (morph-from-list (make-list 10000 1.0) #(10000) 'f64))
           (doubled (morph* m (morph-from-list '(2.0) #(1) 'f64)))
           (realized (realize doubled)))
      (and (concrete-array? realized)
           (= 10000 (morph-size realized)))))
  
  (test-assert "realize deep operation chain"
    (let* ((m (morph-from-list (make-list 100 1.0) #(100) 'f64))
           (result (fold (lambda (i acc)
                          (morph+ acc (morph-from-list '(0.1) #(1) 'f64)))
                        m
                        (iota 10)))
           (realized (realize result)))
      (concrete-array? realized)))
  
  (test-assert "realize doesn't modify original"
    (let* ((orig (morph-from-list '(1 2 3) #(3) 'f64))
           (doubled (morph* orig (morph-from-list '(2) #(1) 'f64)))
           (realized-doubled (realize doubled))
           (realized-orig (realize orig)))
      (morphism-values-equal? orig '(1 2 3)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Run All Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-exit)
