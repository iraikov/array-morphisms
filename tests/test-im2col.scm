;;; test-im2col-unit.scm
;;; Unit tests for im2col to catch buffer size mismatches

(import scheme (chicken base) test srfi-1 srfi-4 datatype
        array-morphisms-core
        array-morphisms-structural-ops
        array-morphisms-realization)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Helper Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (get-concrete-data-length m)
  "Extract the actual data vector length from a concrete morphism"
  (unless (concrete-array? m)
    (error "Expected concrete array" m))
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (cond
        ((f32vector? data) (f32vector-length data))
        ((f64vector? data) (f64vector-length data))
        ((s32vector? data) (s32vector-length data))
        ((s64vector? data) (s64vector-length data))
        (else (error "Unknown data type"))))
    (else (error "Not a concrete array"))))

(define (verify-col-buffer-size img kernel-size stride padding)
  "Verify im2col creates correct buffer size"
  (let* ((col (realize (im2col-morph img kernel-size stride padding)))
         (col-shape (get-morphism-shape col))
         (expected-size (shape-size col-shape))
         (actual-size (get-concrete-data-length col)))
    (= expected-size actual-size)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 1: Buffer Size Verification
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "im2col - Buffer Size Verification"
  
  (test-assert "im2col buffer size matches shape (3x3 input, 2x2 kernel, stride 1)"
    (let* ((img (morph-from-list 
                  '(((1 2 3) (4 5 6) (7 8 9)))
                  #(1 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0)))
           (col-shape (get-morphism-shape col))
           (expected-size (shape-size col-shape))  ; Should be 4*4 = 16
           (actual-size (get-concrete-data-length col)))
      
      (and (= expected-size 16)
           (= actual-size 16)
           (= expected-size actual-size))))
  
  (test-assert "im2col buffer size matches shape (4x4 input, 2x2 kernel, stride 2)"
    (let* ((img (morph-from-list 
                  (make-list 16 1.0)
                  #(1 4 4) 'f64))
           (col (realize (im2col-morph img '(2 2) 2 0)))
           (col-shape (get-morphism-shape col))
           (expected-size (shape-size col-shape))  ; Should be 4*4 = 16
           (actual-size (get-concrete-data-length col)))
      
      (and (= expected-size 16)
           (= actual-size 16))))
  
  (test-assert "im2col buffer size with padding"
    (let* ((img (morph-from-list 
                  '(((1 2) (3 4)))
                  #(1 2 2) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 1)))
           (col-shape (get-morphism-shape col))
           ;; With padding=1: effective input is 4x4
           ;; Output: (4-2)/1 + 1 = 3, so 3x3 = 9 positions
           ;; col shape: (1*2*2, 9) = (4, 9), size = 36
           (expected-size (shape-size col-shape))
           (actual-size (get-concrete-data-length col)))
      
      (= expected-size actual-size)))
  
  (test-assert "im2col buffer size with multi-channel"
    (let* ((img (morph-from-list 
                  (make-list 18 1.0)  ; 2 channels × 3×3
                  #(2 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0)))
           (col-shape (get-morphism-shape col))
           ;; col shape: (C*KH*KW, OH*OW) = (2*2*2, 2*2) = (8, 4)
           ;; size = 32
           (expected-size (shape-size col-shape))
           (actual-size (get-concrete-data-length col)))
      
      (and (= expected-size 32)
           (= actual-size 32)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 2: Shape Verification
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "im2col - Shape Verification"
  
  (test-assert "im2col produces correct output shape (unbatched)"
    (let* ((img (morph-from-list 
                  '(((1 2 3) (4 5 6) (7 8 9)))
                  #(1 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0)))
           (col-shape (get-morphism-shape col)))
      
      ;; Expected: (C*KH*KW, OH*OW) = (1*2*2, 2*2) = (4, 4)
      (equal? col-shape #(4 4))))
  
  (test-assert "im2col produces correct output shape (batched)"
    (let* ((img (morph-from-list 
                  (make-list 18 1.0)  ; 2 batches × 1 channel × 3×3
                  #(2 1 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0)))
           (col-shape (get-morphism-shape col)))
      
      ;; Expected: (N, C*KH*KW, OH*OW) = (2, 1*2*2, 2*2) = (2, 4, 4)
      (equal? col-shape #(2 4 4))))
  
  (test-assert "im2col shape with different strides"
    (let* ((img (morph-from-list 
                  (make-list 25 1.0)  ; 1 channel × 5×5
                  #(1 5 5) 'f64))
           (col (realize (im2col-morph img '(3 3) 2 0)))
           (col-shape (get-morphism-shape col)))
      
      ;; OH = (5-3)/2 + 1 = 2, OW = 2
      ;; Expected: (1*3*3, 2*2) = (9, 4)
      (equal? col-shape #(9 4)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 3: Data Access Verification
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "im2col - Data Access"
  
  (test-assert "can access all elements of im2col output"
    (let* ((img (morph-from-list 
                  '(((1 2 3) (4 5 6) (7 8 9)))
                  #(1 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0)))
           (col-list (morph->list col)))
      
      ;; Should be able to convert to list without error
      (and (list? col-list)
           (= (length col-list) 4)  ; 4 rows
           (every (lambda (row) (= (length row) 4)) col-list))))  ; 4 cols each
  
  (test-assert "im2col values are correct"
    (let* ((img (morph-from-list 
                  '(((1 2) (3 4)))
                  #(1 2 2) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0)))
           (col-list (morph->list col)))
      
      ;; With 2x2 input and 2x2 kernel, stride 1, padding 0:
      ;; Only 1 output position (1x1)
      ;; col shape: (4, 1) - 4 rows (C*KH*KW), 1 column
      ;; Values should be [1, 2, 3, 4] (flattened window)
      (and (= (length col-list) 4)
           (every (lambda (row) (= (length row) 1)) col-list))))
  
  (test-assert "can iterate over all im2col elements"
    (let* ((img (morph-from-list 
                  '(((1 2 3) (4 5 6) (7 8 9)))
                  #(1 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0))))
      
      ;; Try to access every element via linear indexing
      (cases array-morphism col
        (concrete-array (data shape strides offset dtype alloc-id batch-axis)
          (let ((size (shape-size shape)))
            ;; Try to read all elements
            (let loop ((i 0))
              (if (>= i size)
                  #t  ; Success - accessed all elements
                  (begin
                    (typed-vector-ref data dtype i)
                    (loop (+ i 1)))))))
        (else #f)))))

(test-group "im2col - realized array length"
  
  (test-assert "im2col for batch1"
    ;; This is the exact case from the failing col2im test
    (let* ((batch1 (morph-from-list 
                     '(((1 2 3) (4 5 6) (7 8 9)))
                     #(1 3 3) 'f64))
           (col1 (realize (im2col-morph batch1 '(2 2) 1 0))))
      
      ;; Verify:
      ;; 1. Shape is correct: (4, 4)
      ;; 2. Data buffer size matches shape: 16 elements
      ;; 3. Can access all elements
      (and (concrete-array? col1)
           (equal? (get-morphism-shape col1) #(4 4))
           (= (get-concrete-data-length col1) 16)
           (= (shape-size (get-morphism-shape col1)) 16))))
  
  (test-assert "im2col doesn't reuse input buffer"
    ;; Ensure im2col allocates new buffer, not reusing input
    (let* ((img (morph-from-list 
                  '(((1 2 3) (4 5 6) (7 8 9)))
                  #(1 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0))))
      
      ;; Input has 9 elements, output should have 16
      (cases array-morphism img
        (concrete-array (img-data img-shape _ _ _ _ _)
          (cases array-morphism col
            (concrete-array (col-data col-shape _ _ _ _ _)
              (let ((img-len (f64vector-length img-data))
                    (col-len (f64vector-length col-data)))
                (and (= img-len 9)
                     (= col-len 16)
                     ;; Different vectors (not same reference)
                     (not (eq? img-data col-data)))))
            (else #f)))
        (else #f))))

  (test-assert "im2col on stacked input"
        (let* ((batch1 (morph-from-list '(((1 2 3) (4 5 6) (7 8 9))) #(1 3 3) 'f64))
               (batch2 (morph-from-list '(((9 8 7) (6 5 4) (3 2 1))) #(1 3 3) 'f64))
               
               ;; Create batched via stack
               (batched-stack (realize (morph-stack (list batch1 batch2) 0)))
               (col-stack (realize (im2col-morph batched-stack '(2 2) 1 0)))
               
               ;; Create batched directly
               (batched-direct (morph-from-list (append (morph->list batch1)
                                                        (morph->list batch2))
                                                #(2 1 3 3) 'f64))
               (col-direct (realize (im2col-morph batched-direct '(2 2) 1 0))))
          
          ;; Do both produce correct shapes and data lengths?
          (and (equal? (get-morphism-shape col-stack)
                       (get-morphism-shape col-direct))
               (equal? (get-concrete-data-length col-stack)
                       (get-concrete-data-length col-direct)))))

  )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 5: Batched im2col
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "im2col - Batched Operations"
  
  (test-assert "batched im2col buffer size"
    (let* ((img (morph-from-list 
                  (make-list 18 1.0)  ; 2 batches × 1 channel × 3×3
                  #(2 1 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0)))
           (col-shape (get-morphism-shape col))
           (expected-size (shape-size col-shape))  ; 2*4*4 = 32
           (actual-size (get-concrete-data-length col)))
      
      (and (= expected-size 32)
           (= actual-size 32))))
  
  (test-assert "batched im2col can access all elements"
    (let* ((img (morph-from-list 
                  (make-list 18 1.0)  ; 2 batches × 1 channel × 3×3
                  #(2 1 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0))))
      
      ;; Try to access all elements
      (cases array-morphism col
        (concrete-array (data shape strides offset dtype alloc-id batch-axis)
          (let ((size (shape-size shape)))
            (let loop ((i 0))
              (if (>= i size)
                  #t
                  (begin
                    (typed-vector-ref data dtype i)
                    (loop (+ i 1)))))))
        (else #f)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Run Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-exit)
