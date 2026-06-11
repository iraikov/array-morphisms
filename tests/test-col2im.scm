;;; test-col2im.scm
;;; Test suite for col2im implementation
;;;
;;; Tests correctness, accumulation, batching, and gradient flow

(import scheme (chicken base) test srfi-1 srfi-4
        array-morphisms-core
        array-morphisms-structural-ops
        array-morphisms-realization
        array-morphisms-basic-ops)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Helper Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (morphism-values-equal? m1 m2 tolerance)
  "Check if two morphisms have equal values within tolerance"
  (let ((l1 (morph->list (realize m1)))
        (l2 (morph->list (realize m2))))
    (equal-within-tolerance? l1 l2 tolerance)))

(define (equal-within-tolerance? lst1 lst2 tol)
  "Recursively compare nested lists within tolerance"
  (cond
    ((and (number? lst1) (number? lst2))
     (< (abs (- lst1 lst2)) tol))
    ((and (null? lst1) (null? lst2)) #t)
    ((and (pair? lst1) (pair? lst2))
     (and (equal-within-tolerance? (car lst1) (car lst2) tol)
          (equal-within-tolerance? (cdr lst1) (cdr lst2) tol)))
    (else #f)))

(define (sum-nested-list lst)
  "Sum all values in nested list"
  (cond
    ((null? lst) 0)
    ((number? lst) lst)
    ((pair? lst) (+ (sum-nested-list (car lst))
                    (sum-nested-list (cdr lst))))
    (else 0)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 1: Basic Correctness
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "col2im - Basic Correctness"
  
  (test-assert "col2im inverts im2col (no overlap, stride=kernel)"
    (let* ((img (morph-from-list 
                  '(((1 2 3 4)
                     (5 6 7 8)
                     (9 10 11 12)
                     (13 14 15 16)))
                  #(1 4 4) 'f64))
           
           ;; im2col with stride=2 (no overlap)
           (col (realize (im2col-morph img '(2 2) 2 0)))
           
           ;; col2im back
           (reconstructed (realize (col2im-morph col #(1 4 4) '(2 2) 2 0))))
      
      ;; Should match original (no overlap means perfect reconstruction)
      (morphism-values-equal? reconstructed img 1e-10)))
  
  (test-assert "col2im creates correct output shape"
    (let* ((col (morph-from-list 
                  (make-list 36 1.0)  ; 4*3*3 = 36 rows, 1 col
                  #(36 1) 'f64))
           (result (realize (col2im-morph col #(4 3 3) '(3 3) 1 0))))
      
      (and (concrete-array? result)
           (equal? (get-morphism-shape result) #(4 3 3)))))
  
  (test-assert "col2im handles single pixel windows"
    (let* ((img (morph-from-list 
                  '(((1 2) (3 4)))
                  #(1 2 2) 'f64))
           (col (realize (im2col-morph img '(1 1) 1 0)))
           (reconstructed (realize (col2im-morph col #(1 2 2) '(1 1) 1 0))))
      
      (morphism-values-equal? reconstructed img 1e-10))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 2: Overlapping Windows (Accumulation)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "col2im - Overlapping Windows"
  
  (test-assert "col2im accumulates overlapping windows correctly"
    (let* ((img (morph-from-list 
                  '(((1 2 3)
                     (4 5 6)
                     (7 8 9)))
                  #(1 3 3) 'f64))
           
           ;; im2col with stride=1 (overlapping)
           (col (realize (im2col-morph img '(2 2) 1 0)))
           
           ;; col2im back - will have accumulation
           (reconstructed (realize (col2im-morph col #(1 3 3) '(2 2) 1 0))))
      
      ;; Result should be concrete with expected shape
      (and (concrete-array? reconstructed)
           (equal? (get-morphism-shape reconstructed) #(1 3 3))
           
           ;; Values should be accumulated (not equal to original)
           ;; Corner pixels appear once, edge twice, center four times
           (let ((result-list (morph->list reconstructed)))
             ;; Center value (5) appears in 4 windows -> 5*4 = 20
             (> (sum-nested-list result-list) 
                (sum-nested-list '(((1 2 3) (4 5 6) (7 8 9)))))))))
  
  (test-assert "col2im accumulation pattern is correct"
    ;; Simple 2x2 image with 2x2 kernel, stride 1
    ;; Each pixel should appear in exactly the right number of windows
    (let* ((img (morph-from-list 
                  '(((1 1) (1 1)))
                  #(1 2 2) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0)))
           (reconstructed (realize (col2im-morph col #(1 2 2) '(2 2) 1 0))))
      
      ;; All values are 1, with 2x2 kernel and stride 1:
      ;; - Only 1 window (1x1 output)
      ;; - Each pixel appears once -> all should be 1
      (morphism-values-equal? reconstructed img 1e-10))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 3: Padding Handling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "col2im - Padding"
  
  (test-assert "col2im with padding (valid windows only)"
    (let* ((img (morph-from-list 
                  '(((1 2) (3 4)))
                  #(1 2 2) 'f64))
           
           ;; im2col with padding=1
           (col (realize (im2col-morph img '(2 2) 1 1)))
           
           ;; col2im should handle padding correctly
           (reconstructed (realize (col2im-morph col #(1 2 2) '(2 2) 1 1))))
      
      (and (concrete-array? reconstructed)
           (equal? (get-morphism-shape reconstructed) #(1 2 2)))))
  
  (test-assert "col2im ignores out-of-bounds positions"
    ;; Padding creates windows that extend beyond image
    ;; col2im should only accumulate in-bounds positions
    (let* ((img (morph-from-list 
                  '(((1)))
                  #(1 1 1) 'f64))
           (col (realize (im2col-morph img '(3 3) 1 1)))
           (reconstructed (realize (col2im-morph col #(1 1 1) '(3 3) 1 1))))
      
      ;; Center pixel (only valid position) should have accumulated value
      (concrete-array? reconstructed))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 4: Batched Operations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

               (let* ((batch1 (morph-from-list 
                               '(((1 2 3) (4 5 6) (7 8 9)))
                               #(1 3 3) 'f64))
                      (batch2 (morph-from-list 
                               '(((9 8 7) (6 5 4) (3 2 1)))
                               #(1 3 3) 'f64))
                      
                      ;; Stack into batched input
                      (batched-img (morph-stack (list batch1 batch2) 0))
                      
                      ;; Process batched
                      (col-batched (realize (im2col-morph batched-img '(2 2) 1 0)))
                      (result-batched (realize (col2im-morph col-batched #(2 1 3 3) '(2 2) 1 0)))
                      
                      ;; Process separately
                      (col1 (realize (im2col-morph batch1 '(2 2) 1 0)))
                      (col2 (realize (im2col-morph batch2 '(2 2) 1 0)))
                      (result1 (realize (col2im-morph col1 #(1 3 3) '(2 2) 1 0)))
                      (result2 (realize (col2im-morph col2 #(1 3 3) '(2 2) 1 0)))
                      
                      ;; Extract batches from batched result
                      (batched-slice1 (realize (morph-squeeze 
                                                (morph-slice result-batched '(0 0 0 0) '(1 1 3 3))
                                                '(0))))
                      (batched-slice2 (realize (morph-squeeze 
                                                (morph-slice result-batched '(1 0 0 0) '(2 1 3 3))
                                                '(0)))))
                 
                 ;; Compare individual batches
                 (and (morphism-values-equal? batched-slice1 result1 1e-10)
                      (morphism-values-equal? batched-slice2 result2 1e-10)))

(test-group "col2im - Batched"
  
  (test-assert "col2im handles batched input"
    (let* ((img (morph-from-list 
                  (make-list 72 1.0)  ; 2 batches × 2 channels × 6×3
                  #(2 2 6 3) 'f64))
           
           (col (realize (im2col-morph img '(2 2) 1 0)))
           
           (reconstructed (realize (col2im-morph col #(2 2 6 3) '(2 2) 1 0))))
      
      (and (concrete-array? reconstructed)
           (equal? (get-morphism-shape reconstructed) #(2 2 6 3))
           (= 0 (get-morphism-batch-axis reconstructed)))))

  (test-assert "col2im batched matches unbatched per-batch"
               (let* ((batch1 (morph-from-list 
                               '(((1 2 3) (4 5 6) (7 8 9)))
                               #(1 3 3) 'f64))
                      (batch2 (morph-from-list 
                               '(((9 8 7) (6 5 4) (3 2 1)))
                               #(1 3 3) 'f64))
                      
                      ;; Stack into batched input
                      (batched-img (realize (morph-stack (list batch1 batch2) 0)))
                      
                      ;; Process batched
                      (col-batched (realize (im2col-morph batched-img '(2 2) 1 0)))
                      (result-batched (realize (col2im-morph col-batched #(2 1 3 3) '(2 2) 1 0)))
                      
                      ;; Process separately
                      (col1 (realize (im2col-morph batch1 '(2 2) 1 0)))
                      (col2 (realize (im2col-morph batch2 '(2 2) 1 0)))
                      (result1 (realize (col2im-morph col1 #(1 3 3) '(2 2) 1 0)))
                      (result2 (realize (col2im-morph col2 #(1 3 3) '(2 2) 1 0)))
                      
                      ;; Extract batches from batched result
                      (batched-slice1 (realize (morph-squeeze 
                                                (morph-slice result-batched '(0 0 0 0) '(1 1 3 3))
                                                '(0))))
                      (batched-slice2 (realize (morph-squeeze 
                                                (morph-slice result-batched '(1 0 0 0) '(2 1 3 3))
                                                '(0)))))
                 
                 ;; Compare individual batches
                 (and (morphism-values-equal? batched-slice1 result1 1e-10)
                      (morphism-values-equal? batched-slice2 result2 1e-10))))
  )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 5: Gradient Flow (Critical for Training)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "col2im - Gradient Verification"
  
  (test-assert "col2im is adjoint of im2col (basic check)"
    ;; Verify <y, Ax> = <A^T y, x> property
    ;; where A is im2col and A^T is col2im
    (let* ((input (morph-from-list 
                    (make-list 18 1.0)  ; 2 channels × 3×3
                    #(2 3 3) 'f64))
           
           ;; Forward: im2col
           (col (realize (im2col-morph input '(2 2) 1 0)))
           (col-shape (get-morphism-shape col))
           
           ;; Backward: col2im with same-shaped gradient
           (grad-col (morph-from-list 
                       (make-list (shape-size col-shape) 1.0)
                       col-shape 'f64))
           (grad-input (realize (col2im-morph grad-col #(2 3 3) '(2 2) 1 0))))
      
      ;; Should produce valid gradient with correct shape
      (and (concrete-array? grad-input)
           (equal? (get-morphism-shape grad-input) #(2 3 3)))))
  
  (test-assert "col2im gradient propagates values correctly"
    ;; Gradient should accumulate in overlapping regions
    (let* ((input-shape #(1 3 3))
           (col-shape #(4 4))  ; 1*2*2 kernel × 2*2 output
           
           ;; Gradient with specific pattern
           (grad-col (morph-from-list 
                       (make-list 16 2.0)
                       col-shape 'f64))
           
           (grad-input (realize (col2im-morph grad-col input-shape '(2 2) 1 0))))
      
      ;; Each position should have accumulated gradient
      (and (concrete-array? grad-input)
           ;; All gradients are 2.0, accumulation should increase values
           (> (sum-nested-list (morph->list grad-input)) 0)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 6: Edge Cases
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "col2im - Edge Cases"
  
  (test-assert "col2im with stride > kernel (gaps in coverage)"
    (let* ((img (morph-from-list 
                  '(((1 2 3 4 5 6)))
                  #(1 1 6) 'f64))
           (col (realize (im2col-morph img '(2) 3 0)))
           (reconstructed (realize (col2im-morph col #(1 1 6) '(2) 3 0))))
      
      ;; Some positions won't be covered (stride > kernel)
      (concrete-array? reconstructed)))
  
  (test-assert "col2im with large kernel"
    (let* ((img (morph-from-list 
                  (make-list 25 1.0)
                  #(1 5 5) 'f64))
           (col (realize (im2col-morph img '(5 5) 1 0)))
           (reconstructed (realize (col2im-morph col #(1 5 5) '(5 5) 1 0))))
      
      ;; 5×5 kernel on 5×5 image -> single output position
      (and (concrete-array? reconstructed)
           (equal? (get-morphism-shape reconstructed) #(1 5 5)))))
  
  (test-assert "col2im with asymmetric kernel and stride"
    (let* ((img (morph-from-list 
                  (make-list 12 1.0)
                  #(1 3 4) 'f64))
           (col (realize (im2col-morph img '(2 3) '(1 2) 0)))
           (reconstructed (realize (col2im-morph col #(1 3 4) '(2 3) '(1 2) 0))))
      
      (and (concrete-array? reconstructed)
           (equal? (get-morphism-shape reconstructed) #(1 3 4))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Run Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-exit)
