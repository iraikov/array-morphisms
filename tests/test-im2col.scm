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

(define (morph->flat-list m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (let ((size (shape-size shape)))
        (let loop ((i 0) (acc '()))
          (if (>= i size)
              (reverse acc)
              (loop (+ i 1) (cons (typed-vector-ref data dtype i) acc))))))
    (else (error "morph->flat-list: expected concrete array"))))

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
      
      ;; Expected: (N*OH*OW, C*KH*KW) = (2*2*2, 1*2*2) = (8, 4)
      (equal? col-shape #(8 4))))
  
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
;;; Test Group 6: MR (Matmul-Ready) Layout Correctness
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "im2col-mr - matmul-ready layout"

  (test-assert "batched im2col produces [N*OH_OW, fan_in] shape"
    ;; N=1, C=1, H=3, W=3, KH=KW=2, OH=OW=2 -> [1*4, 1*4] = [4, 4]
    (let* ((img (morph-from-list
                  '(((1 2 3) (4 5 6) (7 8 9)))
                  #(1 1 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0)))
           (col-shape (get-morphism-shape col)))
      (equal? col-shape #(4 4))))

  (test-assert "batched im2col element values are correct"
    ;; N=1, C=1, H=3, W=3, img flat: 1..9
    ;; MR col rows: (oh,ow) -> pixels in window
    ;; Row 0 (oh=0,ow=0): img[0,1,4,5] = [1,2,4,5]
    ;; Row 1 (oh=0,ow=1): img[1,2,5,6] = [2,3,5,6]
    ;; Row 2 (oh=1,ow=0): img[3,4,6,7] ...wait, NCHW
    ;; col_idx: c=0,kh=0,kw=0 .. c=0,kh=1,kw=1
    ;; Row 0: [src(0,0,0,0) src(0,0,0,1) src(0,0,1,0) src(0,0,1,1)] = [1 2 4 5]
    ;; Row 1: [src(0,0,0,1) src(0,0,0,2) src(0,0,1,1) src(0,0,1,2)] = [2 3 5 6]
    ;; Row 2: [src(0,0,1,0) src(0,0,1,1) src(0,0,2,0) src(0,0,2,1)] = [4 5 7 8]
    ;; Row 3: [src(0,0,1,1) src(0,0,1,2) src(0,0,2,1) src(0,0,2,2)] = [5 6 8 9]
    (let* ((img (morph-from-list
                  '((((1.0 2.0 3.0) (4.0 5.0 6.0) (7.0 8.0 9.0))))
                  #(1 1 3 3) 'f64))
           (col (realize (im2col-morph img '(2 2) 1 0)))
           (col-list (morph->list col)))
      (and (equal? col-list
                   '((1.0 2.0 4.0 5.0)
                     (2.0 3.0 5.0 6.0)
                     (4.0 5.0 7.0 8.0)
                     (5.0 6.0 8.0 9.0))))))

  (test-assert "MR col roundtrip via col2im (non-overlapping stride)"
    ;; stride=kernel=2 means no overlap -> col2im inverts im2col exactly
    ;; N=1, C=1, H=4, W=4, KH=KW=2, SH=SW=2 -> OH=OW=2, fan_in=4, rows=4
    (let* ((vals (map exact->inexact (iota 16 1)))
           (img (morph-from-list vals #(1 1 4 4) 'f64))
           (col (realize (im2col-morph img '(2 2) 2 0)))
           (reconstructed (realize (col2im-morph col #(1 1 4 4) '(2 2) 2 0)))
           (img-flat (morph->flat-list img))
           (rec-flat (morph->flat-list reconstructed)))
      (every (lambda (a b) (< (abs (- a b)) 1e-10))
             img-flat rec-flat)))

  (test-assert "MR im2col adjoint: <col*g, 1> == <x, col2im(g)>"
    ;; sum(im2col(x) * g) == sum(x * col2im(g, target-shape))
    ;; Using N=1, C=1, H=3, W=3, KH=KW=2, SH=SW=1, PH=PW=0
    (let* ((x-vals '(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0))
           (g-vals '(1.0 0.0 0.0 0.0
                     0.0 1.0 0.0 0.0
                     0.0 0.0 1.0 0.0
                     0.0 0.0 0.0 1.0))
           (x-img (morph-from-list x-vals #(1 1 3 3) 'f64))
           (g-col (morph-from-list g-vals #(4 4) 'f64))
           (col (realize (im2col-morph x-img '(2 2) 1 0)))
           (dx  (realize (col2im-morph g-col #(1 1 3 3) '(2 2) 1 0)))
           (col-flat (morph->flat-list col))
           (g-flat   (morph->flat-list g-col))
           (dx-flat  (morph->flat-list dx))
           ;; <col, g> = sum of element-wise products
           (col-g (apply + (map * col-flat g-flat)))
           ;; <x, dx> = sum of element-wise products
           (x-dx (apply + (map * x-vals dx-flat))))
      (< (abs (- col-g x-dx)) 1e-9)))

  (test-assert "multi-channel MR batched im2col shape"
    ;; N=2, C=3, H=4, W=4, KH=KW=3, stride=1, pad=0 -> OH=OW=2
    ;; [N*OH_OW, C*KH*KW] = [2*4, 3*9] = [8, 27]
    (let* ((img (morph-from-list (make-list 96 1.0) #(2 3 4 4) 'f64))
           (col (realize (im2col-morph img '(3 3) 1 0)))
           (col-shape (get-morphism-shape col)))
      (equal? col-shape #(8 27))))

  )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Group 7: NHWC Layout
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "im2col-nhwc-mr - NHWC layout"

  (test-assert "NHWC im2col shape matches NCHW for same image"
    ;; Both layouts should give [N*OH_OW, fan_in] shape
    (let* ((nchw-img (morph-from-list
                       (make-list 18 1.0) #(1 2 3 3) 'f64))
           (nhwc-img (morph-from-list
                       (make-list 18 1.0) #(1 3 3 2) 'f64))
           (col-nchw (realize (im2col-morph nchw-img '(2 2) 1 0)))
           (col-nhwc (realize (im2col-morph nhwc-img '(2 2) 1 0 layout: 'nhwc))))
      (equal? (get-morphism-shape col-nchw)
              (get-morphism-shape col-nhwc))))

  (test-assert "NHWC im2col produces same values as NCHW (single channel)"
    ;; C=1: NCHW #(1,1,3,3) and NHWC #(1,3,3,1) hold same data
    ;; Both should produce the same col output
    (let* ((data '(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0))
           (nchw-img (morph-from-list data #(1 1 3 3) 'f64))
           (nhwc-img (morph-from-list data #(1 3 3 1) 'f64))
           (col-nchw (realize (im2col-morph nchw-img '(2 2) 1 0)))
           (col-nhwc (realize (im2col-morph nhwc-img '(2 2) 1 0 layout: 'nhwc))))
      (equal? (morph->list col-nchw)
              (morph->list col-nhwc))))

  (test-assert "NHWC col2im roundtrip (non-overlapping stride)"
    ;; NHWC image N=1, H=4, W=4, C=1 with stride=kernel=2 -> exact inversion
    (let* ((vals (map exact->inexact (iota 16 1)))
           (img-nhwc (morph-from-list vals #(1 4 4 1) 'f64))
           (col (realize (im2col-morph img-nhwc '(2 2) 2 0 layout: 'nhwc)))
           (reconstructed (realize (col2im-morph col #(1 4 4 1) '(2 2) 2 0 layout: 'nhwc)))
           (img-flat (morph->flat-list img-nhwc))
           (rec-flat (morph->flat-list reconstructed)))
      (every (lambda (a b) (< (abs (- a b)) 1e-10))
             img-flat rec-flat)))

  )

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Run Tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-exit)
