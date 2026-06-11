;;; test-array-morphisms-phase4.scm
;;; Test Suite for MoA Structural Morphisms
;;;
;;; Tests zero-copy structural operations:
;;; - reshape (with inference)
;;; - transpose (dimension permutation)
;;; - slice (subarray extraction)
;;; - im2col/col2im (convolution helpers)
;;; - batch operations (stack, split, concat)

(import scheme chicken.base)
(import test)
(import (only srfi-1 make-list iota every fold))
(import datatype matchable)
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (approx= a b #!optional (tol 1e-6))
  "Test if two numbers are approximately equal"
  (< (abs (- a b)) tol))

(define (shapes-equal? s1 s2)
  "Test if two shapes (vectors) are equal"
  (and (= (vector-length s1) (vector-length s2))
       (every (lambda (i) (= (vector-ref s1 i) (vector-ref s2 i)))
              (iota (vector-length s1)))))

(define (morphisms-equal-structure? m1 m2)
  "Test if two morphisms have same structure (shape, dtype, batch-axis)"
  (and (equal? (get-morphism-shape m1) (get-morphism-shape m2))
       (eq? (get-morphism-dtype m1) (get-morphism-dtype m2))
       (= (get-morphism-batch-axis m1) (get-morphism-batch-axis m2))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1: morph-reshape Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "morph-reshape - Basic Structure"
  
  (test-assert "reshape creates morphism-expr"
    (let* ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (reshaped (morph-reshape m #(6))))
      (and (abstract-morphism? reshaped)
           (morphism-expr? reshaped))))
  
  (test-assert "reshape operation is 'reshape"
    (let* ((m (morph-from-list '(1 2 3 4) #(4) 'f32))
           (reshaped (morph-reshape m #(2 2))))
      (cases array-morphism reshaped
        (morphism-expr (op operands index-fn shape dtype metadata batch-axis)
          (equal? 'reshape op))
        (else #f))))
  
  (test-assert "reshape has correct output shape"
    (let* ((m (morph-from-list '(1 2 3 4 5 6) #(6) 'f64))
           (reshaped (morph-reshape m #(2 3))))
      (shapes-equal? #(2 3) (get-morphism-shape reshaped))))
  
  (test-assert "reshape preserves dtype"
    (let* ((m (morph-from-list '(1 2 3 4) #(4) 's32))
           (reshaped (morph-reshape m #(2 2))))
      (equal? 's32 (get-morphism-dtype reshaped))))
  
  (test-assert "reshape has single operand"
    (let* ((m (morph-from-list '(1 2 3 4) #(4) 'f64))
           (reshaped (morph-reshape m #(2 2))))
      (= 1 (length (get-operands reshaped))))))

(test-group "morph-reshape - Shape Inference"
  
  (test-assert "reshape infers -1 dimension"
    (let* ((m (morph-from-list '(1 2 3 4 5 6) #(6) 'f64))
           (reshaped (morph-reshape m '(2 -1))))
      (shapes-equal? #(2 3) (get-morphism-shape reshaped))))
  
  (test-assert "reshape infers -1 in flatten"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (reshaped (morph-reshape m '(-1))))
      (shapes-equal? #(4) (get-morphism-shape reshaped))))
  
  (test-assert "reshape infers -1 in middle dimension"
    (let* ((m (morph-from-list (make-list 24 1) #(2 3 4) 'f64))
           (reshaped (morph-reshape m '(2 -1 4))))
      (shapes-equal? #(2 3 4) (get-morphism-shape reshaped))))
  
  (test-error "reshape multiple -1 raises error"
    (let ((m (morph-from-list '(1 2 3 4) #(4) 'f64)))
      (morph-reshape m '(-1 -1))))
  
  (test-error "reshape incompatible -1 raises error"
    (let ((m (morph-from-list '(1 2 3 4 5) #(5) 'f64)))
      (morph-reshape m '(2 -1)))))

(test-group "morph-reshape - Size Validation"
  
  (test-error "reshape validates size match"
    (let ((m (morph-from-list '(1 2 3 4) #(4) 'f64)))
      (morph-reshape m #(5))))
  
  (test-assert "reshape same size different shape OK"
    (let* ((m (morph-from-list '(1 2 3 4 5 6) #(6) 'f64))
           (r1 (morph-reshape m #(2 3)))
           (r2 (morph-reshape m #(3 2)))
           (r3 (morph-reshape m #(1 6))))
      (and (morphism-expr? r1)
           (morphism-expr? r2)
           (morphism-expr? r3)))))

(test-group "morph-reshape - Batch Axis Tracking"
  
  (test-assert "reshape preserves batch axis when dimension unchanged"
    (let* ((m (morph-from-list '((1 2) (3 4) (5 6)) #(3 2) 'f64 batch-axis: 0))
           (reshaped (morph-reshape m #(3 2))))
      (= 0 (get-morphism-batch-axis reshaped))))
  
  (test-assert "reshape loses batch axis when dimension absorbed"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64 batch-axis: 0))
           (reshaped (morph-reshape m #(4))))
       (= -1 (get-morphism-batch-axis reshaped))))
  
  (test-assert "reshape finds new batch axis position"
    (let* ((m (morph-from-list (make-list 12 1) #(3 4) 'f64 batch-axis: 0))
           (reshaped (morph-reshape m #(3 2 2))))
      ;; Batch size 3 should be found at position 0
      (= 0 (get-morphism-batch-axis reshaped)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2: morph-transpose Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "morph-transpose - Basic Structure"
  
  (test-assert "transpose creates morphism-expr"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (transposed (morph-transpose m)))
      (and (abstract-morphism? transposed)
           (morphism-expr? transposed))))
  
  (test-assert "transpose operation is 'transpose"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (transposed (morph-transpose m)))
      (cases array-morphism transposed
        (morphism-expr (op operands index-fn shape dtype metadata batch-axis)
          (equal? 'transpose op))
        (else (test-assert #f)))))
  
  (test-assert "transpose preserves dtype"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 's32))
           (transposed (morph-transpose m)))
      (equal? 's32 (get-morphism-dtype transposed)))))

(test-group "morph-transpose - Default Permutation"
  
  (test-assert "transpose 2D reverses axes"
    (let* ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (transposed (morph-transpose m)))
      (shapes-equal? #(3 2) (get-morphism-shape transposed))))
  
  (test-assert "transpose 3D reverses all axes"
    (let* ((m (morph-from-list (make-list 24 1) #(2 3 4) 'f64))
           (transposed (morph-transpose m)))
      (shapes-equal? #(4 3 2) (get-morphism-shape transposed))))
  
  (test-assert "transpose 1D is identity shape"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (transposed (morph-transpose m)))
      (shapes-equal? #(3) (get-morphism-shape transposed)))))

(test-group "morph-transpose - Custom Permutations"
  
  (test-assert "transpose with explicit 2D permutation"
    (let* ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (transposed (morph-transpose m '(1 0))))
      (shapes-equal? #(3 2) (get-morphism-shape transposed))))
  
  (test-assert "transpose swap last two axes"
    (let* ((m (morph-from-list (make-list 24 1) #(2 3 4) 'f64))
           (transposed (morph-transpose m '(0 2 1))))
      (shapes-equal? #(2 4 3) (get-morphism-shape transposed))))
  
  (test-assert "transpose rotate axes right"
    (let* ((m (morph-from-list (make-list 24 1) #(2 3 4) 'f64))
           (transposed (morph-transpose m '(2 0 1))))
      (shapes-equal? #(4 2 3) (get-morphism-shape transposed))))
  
  (test-assert "transpose identity permutation preserves shape"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (transposed (morph-transpose m '(0 1))))
      (shapes-equal? #(2 2) (get-morphism-shape transposed)))))

(test-group "morph-transpose - Validation"
  
  (test-error "transpose validates permutation length"
    (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64)))
      (morph-transpose m '(0))))
  
  (test-error "transpose validates permutation range"
    (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64)))
      (morph-transpose m '(0 2))))
  
  (test-error "transpose validates duplicate indices"
    (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64)))
      (morph-transpose m '(0 0)))))

(test-group "morph-transpose - Batch Axis Tracking"
  
  (test-assert "transpose tracks batch axis movement"
    (let* ((m (morph-from-list (make-list 12 1) #(3 2 2) 'f64 batch-axis: 0))
           (transposed (morph-transpose m '(1 0 2))))
      ;; Batch axis 0 moves to position 1
      (= 1 (get-morphism-batch-axis transposed))))
  
  (test-assert "transpose batch axis to end"
    (let* ((m (morph-from-list (make-list 12 1) #(3 2 2) 'f64 batch-axis: 0))
           (transposed (morph-transpose m '(1 2 0))))
      (= 2 (get-morphism-batch-axis transposed))))
  
  (test-assert "transpose non-batched preserves -1"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (transposed (morph-transpose m)))
      (= -1 (get-morphism-batch-axis transposed)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3: morph-slice Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "morph-slice - Basic Structure"
  
  (test-assert "slice creates morphism-expr"
    (let* ((m (morph-from-list '(1 2 3 4 5) #(5) 'f64))
           (sliced (morph-slice m '(1) '(4))))
      (and (abstract-morphism? sliced)
           (morphism-expr? sliced))))
  
  (test-assert "slice operation is 'slice"
    (let* ((m (morph-from-list '(1 2 3 4 5) #(5) 'f64))
           (sliced (morph-slice m '(0) '(3))))
      (cases array-morphism sliced
        (morphism-expr (op operands index-fn shape dtype metadata batch-axis)
          (equal? 'slice op))
        (else #f))))
  
  (test-assert "slice preserves dtype"
    (let* ((m (morph-from-list '(1 2 3 4 5) #(5) 's32))
           (sliced (morph-slice m '(1) '(4))))
      (equal? 's32 (get-morphism-dtype sliced)))))

(test-group "morph-slice - 1D Slicing"
  
  (test-assert "slice 1D basic range"
    (let* ((m (morph-from-list '(0 1 2 3 4 5) #(6) 'f64))
           (sliced (morph-slice m '(1) '(4))))
      (shapes-equal? #(3) (get-morphism-shape sliced))))
  
  (test-assert "slice 1D with step"
    (let* ((m (morph-from-list '(0 1 2 3 4 5 6 7 8 9) #(10) 'f64))
           (sliced (morph-slice m '(0) '(10) 2)))
      (shapes-equal? #(5) (get-morphism-shape sliced))))
  
  (test-assert "slice 1D negative indices"
    (let* ((m (morph-from-list '(0 1 2 3 4) #(5) 'f64))
           (sliced (morph-slice m '(-2) '(-1))))
      (shapes-equal? #(1) (get-morphism-shape sliced))))
  
  (test-assert "slice 1D full range"
    (let* ((m (morph-from-list '(1 2 3 4) #(4) 'f64))
           (sliced (morph-slice m '(0) '(4))))
      (shapes-equal? #(4) (get-morphism-shape sliced)))))

(test-group "morph-slice - 2D Slicing"
  
  (test-assert "slice 2D rows and columns"
    (let* ((m (morph-from-list '((1 2 3) (4 5 6) (7 8 9)) #(3 3) 'f64))
           (sliced (morph-slice m '(0 1) '(2 3))))
      (shapes-equal? #(2 2) (get-morphism-shape sliced))))
  
  (test-assert "slice 2D single row"
    (let* ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (sliced (morph-slice m '(1 0) '(2 3))))
      (shapes-equal? #(1 3) (get-morphism-shape sliced))))
  
  (test-assert "slice 2D with steps"
    (let* ((m (morph-from-list (make-list 20 1) #(4 5) 'f64))
           (sliced (morph-slice m '(0 0) '(4 5) '(2 2))))
      (shapes-equal? #(2 3) (get-morphism-shape sliced)))))

(test-group "morph-slice - Validation"
  
  (test-error "slice validates start/end lengths"
    (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64)))
      (morph-slice m '(0) '(2 2))))
  
  (test-error "slice validates step positive"
    (let ((m (morph-from-list '(1 2 3) #(3) 'f64)))
      (morph-slice m '(0) '(3) -1)))
  
  (test-error "slice validates range"
    (let ((m (morph-from-list '(1 2 3) #(3) 'f64)))
      (morph-slice m '(0) '(5))))
  
  (test-error "slice validates start < end"
    (let ((m (morph-from-list '(1 2 3) #(3) 'f64)))
      (morph-slice m '(2) '(1)))))

(test-group "morph-slice - Batch Axis Tracking"
  
  (test-assert "slice preserves batch axis when full range"
    (let* ((m (morph-from-list (make-list 12 1) #(3 4) 'f64 batch-axis: 0))
           (sliced (morph-slice m '(0 1) '(3 3))))
      (= 0 (get-morphism-batch-axis sliced))))
  
  (test-assert "slice loses batch axis when sliced"
    (let* ((m (morph-from-list (make-list 12 1) #(3 4) 'f64 batch-axis: 0))
           (sliced (morph-slice m '(1 0) '(2 4))))
      (= -1 (get-morphism-batch-axis sliced))))
  
  (test-assert "slice batch axis with step loses batching"
    (let* ((m (morph-from-list (make-list 12 1) #(4 3) 'f64 batch-axis: 0))
           (sliced (morph-slice m '(0 0) '(4 3) '(2 1))))
      (= -1 (get-morphism-batch-axis sliced)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4: Helper Morphisms (squeeze, unsqueeze, pad)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "morph-squeeze"
  
  (test-assert "squeeze removes all singleton dimensions"
    (let* ((m (morph-from-list '(((1 2))) #(1 1 2) 'f64))
           (squeezed (morph-squeeze m)))
      (shapes-equal? #(2) (get-morphism-shape squeezed))))
  
  (test-assert "squeeze specific axes"
    (let* ((m (morph-from-list (make-list 6 1) #(1 2 1 3) 'f64))
           (squeezed (morph-squeeze m '(0 2))))
      (shapes-equal? #(2 3) (get-morphism-shape squeezed))))
  
  (test-error "squeeze validates dimension is 1"
    (let ((m (morph-from-list '((1 2)) #(1 2) 'f64)))
       (morph-squeeze m '(1))))
  
  (test-assert "squeeze no-op when no singleton dims"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (squeezed (morph-squeeze m)))
      (shapes-equal? #(2 2) (get-morphism-shape squeezed)))))

(test-group "morph-unsqueeze"
  
  (test-assert "unsqueeze adds dimension at front"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (unsqueezed (morph-unsqueeze m 0)))
      (shapes-equal? #(1 3) (get-morphism-shape unsqueezed))))
  
  (test-assert "unsqueeze adds dimension at end"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (unsqueezed (morph-unsqueeze m 1)))
      (shapes-equal? #(3 1) (get-morphism-shape unsqueezed))))
  
  (test-assert "unsqueeze in middle"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (unsqueezed (morph-unsqueeze m 1)))
      (shapes-equal? #(2 1 2) (get-morphism-shape unsqueezed))))
  
  (test-assert "unsqueeze negative axis"
    (let* ((m (morph-from-list '(1 2) #(2) 'f64))
           (unsqueezed (morph-unsqueeze m -1)))
      (shapes-equal? #(2 1) (get-morphism-shape unsqueezed))))
  
  (test-error "unsqueeze validates axis range"
    (let ((m (morph-from-list '(1 2) #(2) 'f64)))
      (morph-unsqueeze m 3))))

(test-group "morph-pad"
  
  (test-assert "pad creates morphism-expr"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (padded (morph-pad m '((1 1)))))
      (abstract-morphism? padded)))
  
  (test-assert "pad computes correct output shape"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (padded (morph-pad m '((1 1) (2 2)))))
      (shapes-equal? #(4 6) (get-morphism-shape padded))))
  
  (test-error "pad validates padding length"
    (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64)))
      (morph-pad m '((1 1)))))

  (test-assert "pad edge mode structure"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (padded (morph-pad m '((1 1)) 'edge)))
      (and (morphism-expr? padded)
           (shapes-equal? #(5) (get-morphism-shape padded)))))

  (test-assert "pad reflect mode structure"
    (let* ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (padded (morph-pad m '((1 1) (1 1)) 'reflect)))
      (and (morphism-expr? padded)
           (shapes-equal? #(4 4) (get-morphism-shape padded)))))

  (test-assert "pad index function - constant mode"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (padded (morph-pad m '((2 1)) 'constant 0.0))
           (idx-fn (get-index-fn padded)))
      ;; Test out-of-bounds indices return constant marker
      (and (equal? '(constant 0.0) (idx-fn '(0)))
           (equal? '(constant 0.0) (idx-fn '(1)))
           ;; Test in-bounds indices
           (equal? '(0) (idx-fn '(2)))
           (equal? '(1) (idx-fn '(3)))
           (equal? '(2) (idx-fn '(4)))
           ;; Test right padding
           (equal? '(constant 0.0) (idx-fn '(5))))))
  
  (test-assert "pad index function - edge mode"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (padded (morph-pad m '((2 1)) 'edge))
           (idx-fn (get-index-fn padded)))
      ;; Left padding: clamp to 0
      (and (equal? '(0) (idx-fn '(0)))
           (equal? '(0) (idx-fn '(1)))
           ;; Regular indices
           (equal? '(0) (idx-fn '(2)))
           (equal? '(1) (idx-fn '(3)))
           (equal? '(2) (idx-fn '(4)))
           ;; Right padding: clamp to 2 (last index)
           (equal? '(2) (idx-fn '(5))))))

  (test-assert "pad index function - reflect mode"
    (let* ((m (morph-from-list '(1 2 3) #(3) 'f64))
           (padded (morph-pad m '((2 2)) 'reflect))
           (idx-fn (get-index-fn padded)))
      ;; Array: [1 2 3]
      ;; Reflected: [2 1 | 1 2 3 | 3 2]
      ;;            -2 -1  0 1 2   3 4
      
      ;; Left reflection
      (and (equal? '(1) (idx-fn '(0)))  ; reflects to index 1
           (equal? '(0) (idx-fn '(1)))  ; reflects to index 0
      
           ;; Regular indices
           (equal? '(0) (idx-fn '(2)))
           (equal? '(1) (idx-fn '(3)))
           (equal? '(2) (idx-fn '(4)))
      
           ;; Right reflection
           (equal? '(2) (idx-fn '(5)))  ; reflects to index 2
           (equal? '(1) (idx-fn '(6))))))  ; reflects to index 1
  )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5: im2col-morph Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "im2col-morph - Basic Structure"
  
  (test-assert "im2col creates morphism-expr"
    (let* ((m (morph-from-list (make-list 18 1) #(2 3 3) 'f64))
           (col (im2col-morph m '(2 2))))
      (and (abstract-morphism? col)
           (morphism-expr? col))))
  
  (test-assert "im2col operation is 'im2col"
    (let* ((m (morph-from-list (make-list 18 1) #(2 3 3) 'f64))
           (col (im2col-morph m '(2 2))))
      (cases array-morphism col
        (morphism-expr (op operands index-fn shape dtype metadata batch-axis)
          (equal? 'im2col op))
        (else #f))))
  
  (test-assert "im2col preserves dtype"
    (let* ((m (morph-from-list (make-list 18 1) #(2 3 3) 'f32))
           (col (im2col-morph m '(2 2))))
      (equal? 'f32 (get-morphism-dtype col)))))

(test-group "im2col-morph - Shape Computation"
  
  (test-assert "im2col non-batched shape"
    (let* ((m (morph-from-list (make-list 18 1) #(2 3 3) 'f64))
           (col (im2col-morph m '(2 2) 1 0)))
      ;; C=2, H=3, W=3, KH=2, KW=2, S=1, P=0
      ;; Output: (C*KH*KW, OH*OW) = (2*2*2, 2*2) = (8, 4)
      (shapes-equal? #(8 4) (get-morphism-shape col))))
  
  (test-assert "im2col batched shape"
    (let* ((m (morph-from-list (make-list 72 1) #(4 2 3 3) 'f64 batch-axis: 0))
           (col (im2col-morph m '(2 2) 1 0)))
      ;; N=4, C=2, Output: (N, C*KH*KW, OH*OW) = (4, 8, 4)
      (shapes-equal? #(4 8 4) (get-morphism-shape col))))
  
  (test-assert "im2col with stride"
    (let* ((m (morph-from-list (make-list 50 1) #(2 5 5) 'f64))
           (col (im2col-morph m '(3 3) 2 0)))
      ;; OH = (5-3)/2 + 1 = 2, OW = 2
      ;; Output: (2*3*3, 2*2) = (18, 4)
      (shapes-equal? #(18 4) (get-morphism-shape col))))
  
  (test-assert "im2col with padding"
    (let* ((m (morph-from-list (make-list 18 1) #(2 3 3) 'f64))
           (col (im2col-morph m '(3 3) 1 1)))
      ;; OH = (3+2*1-3)/1 + 1 = 3, OW = 3
      ;; Output: (2*3*3, 3*3) = (18, 9)
      (shapes-equal? #(18 9) (get-morphism-shape col)))))

(test-group "im2col-morph - Parameter Parsing"
  
  (test-assert "im2col integer kernel size"
    (let* ((m (morph-from-list (make-list 18 1) #(2 3 3) 'f64))
           (col (im2col-morph m 2)))
      ;; Kernel (2,2)
      (shapes-equal? #(8 4) (get-morphism-shape col))))
  
  (test-assert "im2col tuple parameters"
    (let* ((m (morph-from-list (make-list 32 1) #(2 4 4) 'f64))
           (col (im2col-morph m '(2 3) '(1 2) '(0 1))))
      ;; KH=2, KW=3, SH=1, SW=2, PH=0, PW=1
      ;; OH = (4+0-2)/1 + 1 = 3, OW = (4+2-3)/2 + 1 = 2
      (shapes-equal? #(12 6) (get-morphism-shape col)))))

(test-group "im2col-morph - Batch Tracking"
  
  (test-assert "im2col non-batched has batch-axis -1"
               (let* ((m (morph-from-list (make-list 18 1) #(2 3 3) 'f64))
                      (col (im2col-morph m '(2 2))))
                 (= -1 (get-morphism-batch-axis col))))
  
  (test-assert "im2col batched has batch-axis 0"
               (let* ((m (morph-from-list (make-list 72 1) #(4 2 3 3) 'f64 batch-axis: 0))
                      (col (im2col-morph m '(2 2))))
                 (= 0 (get-morphism-batch-axis col)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6: col2im-morph Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "col2im-morph - Basic Structure"
  
  (test-assert "col2im creates morphism-expr"
    (let* ((col (morph-from-list (make-list 32 1) #(8 4) 'f64))
           (im (col2im-morph col #(2 3 3) '(2 2))))
      (and (abstract-morphism? im)
           (morphism-expr? im))))
  
  (test-assert "col2im operation is 'col2im"
    (let* ((col (morph-from-list (make-list 32 1) #(8 4) 'f64))
           (im (col2im-morph col #(2 3 3) '(2 2))))
      (cases array-morphism im
        (morphism-expr (op operands index-fn shape dtype metadata batch-axis)
          (equal? 'col2im op))
        (else #f))))
  
  (test-assert "col2im has correct output shape"
    (let* ((col (morph-from-list (make-list 32 1) #(8 4) 'f64))
           (im (col2im-morph col #(2 3 3) '(2 2))))
      (shapes-equal? #(2 3 3) (get-morphism-shape im))))
  
  (test-assert "col2im preserves dtype"
    (let* ((col (morph-from-list (make-list 32 1) #(8 4) 'f32))
           (im (col2im-morph col #(2 3 3) '(2 2))))
      (equal? 'f32 (get-morphism-dtype im)))))

(test-group "col2im-morph - Batch Handling"
  
  (test-assert "col2im non-batched"
               (let* ((col (morph-from-list (make-list 8 (make-list 4 1.0)) #(8 4) 'f64))
                      (im (col2im-morph col #(2 3 3) '(2 2))))
                 (= -1 (get-morphism-batch-axis im))))
  
  (test-assert "col2im batched"
    (let* ((col (morph-from-list (make-list 128 1) #(4 8 4) 'f64 batch-axis: 0))
           (im (col2im-morph col #(4 2 3 3) '(2 2))))
      (= 0 (get-morphism-batch-axis im))))
  
  (test-error "col2im validates batch consistency"
    (let ((col (morph-from-list (make-list 32 1) #(8 4) 'f64)))
      (col2im-morph col #(4 2 3 3) '(2 2)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 7: morph-stack Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "morph-stack"
  
  (test-assert "stack creates morphism-expr"
    (let* ((m1 (morph-from-list '(1 2) #(2) 'f64))
           (m2 (morph-from-list '(3 4) #(2) 'f64))
           (stacked (morph-stack (list m1 m2))))
      (abstract-morphism? stacked)))
  
  (test-assert "stack along axis 0 (default)"
    (let* ((m1 (morph-from-list '(1 2) #(2) 'f64))
           (m2 (morph-from-list '(3 4) #(2) 'f64))
           (m3 (morph-from-list '(5 6) #(2) 'f64))
           (stacked (morph-stack (list m1 m2 m3))))
      (shapes-equal? #(3 2) (get-morphism-shape stacked))))
  
  (test-assert "stack along axis 1"
    (let* ((m1 (morph-from-list '(1 2) #(2) 'f64))
           (m2 (morph-from-list '(3 4) #(2) 'f64))
           (stacked (morph-stack (list m1 m2) 1)))
      (shapes-equal? #(2 2) (get-morphism-shape stacked))))
  
  (test-assert "stack 2D matrices"
    (let* ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (m2 (morph-from-list '((5 6) (7 8)) #(2 2) 'f64))
           (stacked (morph-stack (list m1 m2) 0)))
      (shapes-equal? #(2 2 2) (get-morphism-shape stacked))))
  
  (test-error "stack validates shape match"
    (let ((m1 (morph-from-list '(1 2) #(2) 'f64))
          (m2 (morph-from-list '(3 4 5) #(3) 'f64)))
      (morph-stack (list m1 m2))))
  
  (test-assert "stack promotes types"
    (let* ((m1 (morph-from-list '(1 2) #(2) 's32))
           (m2 (morph-from-list '(3.0 4.0) #(2) 'f64))
           (stacked (morph-stack (list m1 m2))))
      (equal? 'f64 (get-morphism-dtype stacked))))
  
  (test-assert "stack sets batch axis to stacked dimension"
    (let* ((m1 (morph-from-list '(1 2) #(2) 'f64))
           (m2 (morph-from-list '(3 4) #(2) 'f64))
           (stacked (morph-stack (list m1 m2) 0)))
      (= 0 (get-morphism-batch-axis stacked)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 8: morph-concat Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "morph-concat"
  
  (test-assert "concat creates morphism-expr"
    (let* ((m1 (morph-from-list '(1 2) #(2) 'f64))
           (m2 (morph-from-list '(3 4 5) #(3) 'f64))
           (concatenated (morph-concat (list m1 m2))))
      (abstract-morphism? concatenated)))
  
  (test-assert "concat along axis 0"
    (let* ((m1 (morph-from-list '(1 2) #(2) 'f64))
           (m2 (morph-from-list '(3 4 5) #(3) 'f64))
           (concatenated (morph-concat (list m1 m2) 0)))
      (shapes-equal? #(5) (get-morphism-shape concatenated))))
  
  (test-assert "concat 2D along rows"
    (let* ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (m2 (morph-from-list '((5 6)) #(1 2) 'f64))
           (concatenated (morph-concat (list m1 m2) 0)))
      (shapes-equal? #(3 2) (get-morphism-shape concatenated))))
  
  (test-assert "concat 2D along columns"
    (let* ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
           (m2 (morph-from-list '((5) (6)) #(2 1) 'f64))
           (concatenated (morph-concat (list m1 m2) 1)))
      (shapes-equal? #(2 3) (get-morphism-shape concatenated))))
  
  (test-error "concat validates shape match"
    (let ((m1 (morph-from-list '((1 2)) #(1 2) 'f64))
          (m2 (morph-from-list '((3 4 5)) #(1 3) 'f64)))
       (morph-concat (list m1 m2) 0)))
  
  (test-assert "concat promotes types"
    (let* ((m1 (morph-from-list '(1 2) #(2) 's32))
           (m2 (morph-from-list '(3.0 4.0) #(2) 'f64))
           (concatenated (morph-concat (list m1 m2))))
      (equal? 'f64 (get-morphism-dtype concatenated)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 9: morph-split Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "morph-split"
  
  (test-assert "split into equal sections"
    (let* ((m (morph-from-list '(1 2 3 4 5 6) #(6) 'f64))
           (splits (morph-split m 3)))
      (and (= 3 (length splits))
           (every (lambda (s) (shapes-equal? #(2) (get-morphism-shape s)))
                         splits))))
  
  (test-assert "split with custom sizes"
    (let* ((m (morph-from-list '(1 2 3 4 5 6) #(6) 'f64))
           (splits (morph-split m '(1 2 3))))
      (and (= 3 (length splits))
           (shapes-equal? #(1) (get-morphism-shape (car splits)))
           (shapes-equal? #(2) (get-morphism-shape (cadr splits)))
           (shapes-equal? #(3) (get-morphism-shape (caddr splits))))))
  
  (test-assert "split 2D along rows"
    (let* ((m (morph-from-list '((1 2) (3 4) (5 6) (7 8)) #(4 2) 'f64))
           (splits (morph-split m 2 0)))
      (and (= 2 (length splits))
           (every (lambda (s) (shapes-equal? #(2 2) (get-morphism-shape s)))
                  splits))))
  
  (test-error "split validates divisibility"
    (let ((m (morph-from-list '(1 2 3 4 5) #(5) 'f64)))
      (morph-split m 2)))
  
  (test-error "split validates size sum"
    (let ((m (morph-from-list '(1 2 3 4) #(4) 'f64)))
      (morph-split m '(2 3)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 10: Integration Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Integration - Chained Operations"
  
  (test-assert "reshape then transpose"
    (let* ((m (morph-from-list '(1 2 3 4 5 6) #(6) 'f64))
           (reshaped (morph-reshape m #(2 3)))
           (transposed (morph-transpose reshaped)))
      (shapes-equal? #(3 2) (get-morphism-shape transposed))))
  
  (test-assert "transpose then slice"
    (let* ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
           (transposed (morph-transpose m))
           (sliced (morph-slice transposed '(0 0) '(2 1))))
      (shapes-equal? #(2 1) (get-morphism-shape sliced))))
  
  (test-assert "squeeze then unsqueeze"
    (let* ((m (morph-from-list '(((1 2))) #(1 1 2) 'f64))
           (squeezed (morph-squeeze m))
           (unsqueezed (morph-unsqueeze squeezed 0)))
      (shapes-equal? #(1 2) (get-morphism-shape unsqueezed))))
  
  (test-assert "stack then split"
    (let* ((m1 (morph-from-list '(1 2) #(2) 'f64))
           (m2 (morph-from-list '(3 4) #(2) 'f64))
           (m3 (morph-from-list '(5 6) #(2) 'f64))
           (stacked (morph-stack (list m1 m2 m3)))
           (splits (morph-split stacked 3)))
      (and (= 3 (length splits))
           (every (lambda (s) (shapes-equal? #(1 2) (get-morphism-shape s)))
                  splits))))
  
  (test-assert "im2col shape matches expected"
    (let* ((m (morph-from-list (make-list 32 1) #(2 4 4) 'f64))
           (col (im2col-morph m '(3 3) 1 1)))
      ;; OH = (4+2-3)/1+1 = 4, OW = 4
      ;; Output: (2*3*3, 4*4) = (18, 16)
      (shapes-equal? #(18 16) (get-morphism-shape col)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Run All Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-exit)
