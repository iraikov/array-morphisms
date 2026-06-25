;;; test-array-morphisms-phase1.scm
;;; Test suite for Array Morphisms Phase 1

(import scheme (chicken base) test srfi-4
        array-morphisms-core)

;;;; ============================================================
;;;; Test Utilities
;;;; ============================================================

(define (approx= a b #!optional (tol 1e-6))
  "Check if two numbers are approximately equal"
  (< (abs (- a b)) tol))

(define (vectors-equal? v1 v2 #!optional (tol 1e-6))
  "Check if two vectors are equal (with tolerance for floats)"
  (and (= (vector-length v1) (vector-length v2))
       (let loop ((i 0))
         (cond
           ((= i (vector-length v1)) #t)
           ((approx= (vector-ref v1 i) (vector-ref v2 i) tol)
            (loop (+ i 1)))
           (else #f)))))

(define (lists-equal? l1 l2 #!optional (tol 1e-6))
  "Check if two nested lists are equal"
  (cond
    ((and (null? l1) (null? l2)) #t)
    ((or (null? l1) (null? l2)) #f)
    ((and (not (pair? l1)) (not (pair? l2)))
     (approx= l1 l2 tol))
    ((or (not (pair? l1)) (not (pair? l2))) #f)
    (else
     (and (lists-equal? (car l1) (car l2) tol)
          (lists-equal? (cdr l1) (cdr l2) tol)))))

;;;; ============================================================
;;;; Shape Utilities Tests
;;;; ============================================================

(test-group "Shape Utilities"
  
  (test "shape->list converts vector to list"
    '(2 3 4)
    (shape->list #(2 3 4)))
  
  (test "list->shape converts list to vector"
    #(2 3 4)
    (list->shape '(2 3 4)))
  
  (test "shape-rank returns number of dimensions"
    3
    (shape-rank #(2 3 4)))
  
  (test "shape-size returns total elements"
    24
    (shape-size #(2 3 4)))
  
  (test "shape-dim returns dimension at axis"
    3
    (shape-dim #(2 3 4) 1))
  
  (test "compute-strides for 1D"
    #(1)
    (compute-strides #(5)))
  
  (test "compute-strides for 2D"
    #(4 1)
    (compute-strides #(3 4)))
  
  (test "compute-strides for 3D"
    #(20 5 1)
    (compute-strides #(3 4 5)))
  
  (test "compute-strides for 4D"
    #(120 40 10 1)
    (compute-strides #(2 3 4 10)))
  
  (test "normalize-axis handles positive axis"
    1
    (normalize-axis 1 3))
  
  (test "normalize-axis handles negative axis"
    2
    (normalize-axis -1 3))
  
  (test "normalize-axis handles -2"
    1
    (normalize-axis -2 3))
  
  (test "validate-shape accepts valid shape"
    #(2 3 4)
    (validate-shape #(2 3 4)))
  
  (test-error "validate-shape rejects zero dimension"
    (validate-shape #(2 0 4)))
  
  (test-error "validate-shape rejects negative dimension"
    (validate-shape #(2 -3 4)))
)

;;;; ============================================================
;;;; Index Conversion Tests
;;;; ============================================================

(test-group "Index Conversion"
  
  (test "linear-to-multi-index for 1D"
    #(3)
    (linear-to-multi-index 3 #(5)))
  
  (test "linear-to-multi-index for 2D first element"
    #(0 0)
    (linear-to-multi-index 0 #(3 4)))
  
  (test "linear-to-multi-index for 2D middle element"
    #(1 2)
    (linear-to-multi-index 6 #(3 4)))
  
  (test "linear-to-multi-index for 2D last element"
    #(2 3)
    (linear-to-multi-index 11 #(3 4)))
  
  (test "linear-to-multi-index for 3D"
    #(1 0 3)
    (linear-to-multi-index 23 #(3 4 5)))
  
  (test "linear-to-multi-index for 4D"
    #(0 2 1 3)
    (linear-to-multi-index 23 #(2 3 2 4)))
  
  (test "multi-to-linear-index for 1D"
    3
    (multi-to-linear-index #(3) #(1)))
  
  (test "multi-to-linear-index for 2D"
    6
    (multi-to-linear-index #(1 2) #(4 1)))
  
  (test "multi-to-linear-index for 3D"
    23
    (multi-to-linear-index #(1 0 3) #(20 5 1)))
  
  (test "multi-to-linear-index with offset"
    33
    (multi-to-linear-index #(1 0 3) #(20 5 1) 10))
  
  ;; Round-trip tests
  (test "round-trip linear->multi->linear for 2D"
    15
    (let* ((shape #(4 5))
           (strides (compute-strides shape))
           (multi (linear-to-multi-index 15 shape)))
      (multi-to-linear-index multi strides)))
  
  (test "round-trip linear->multi->linear for 3D"
    42
    (let* ((shape #(3 4 5))
           (strides (compute-strides shape))
           (multi (linear-to-multi-index 42 shape)))
      (multi-to-linear-index multi strides)))
)

;;;; ============================================================
;;;; Broadcasting Tests
;;;; ============================================================

(test-group "Broadcasting"
  
  (test "shapes-compatible? same shapes"
    #t
    (shapes-compatible? #(3 4) #(3 4)))
  
  (test "shapes-compatible? with broadcasting"
    #t
    (shapes-compatible? #(3 1 5) #(4 5)))
  
  (test "shapes-compatible? single element broadcast"
    #t
    (shapes-compatible? #(3 4 5) #(1)))
  
  (test "shapes-compatible? incompatible shapes"
    #f
    (shapes-compatible? #(3 4) #(5 6)))
  
  (test "broadcast-shapes same shapes"
    #(3 4)
    (broadcast-shapes #(3 4) #(3 4)))
  
  (test "broadcast-shapes with trailing dims"
    #(3 4 5)
    (broadcast-shapes #(3 1 5) #(4 5)))
  
  (test "broadcast-shapes scalar broadcast"
    #(3 4 5)
    (broadcast-shapes #(3 4 5) #(1)))
  
  (test "broadcast-shapes both have 1s"
    #(3 4 5)
    (broadcast-shapes #(3 1 5) #(1 4 1)))
  
  (test-error "broadcast-shapes incompatible"
    (broadcast-shapes #(3 4) #(5 6)))
  
  (test "reshape-compatible? same size"
    #t
    (reshape-compatible? #(2 3 4) #(6 4)))
  
  (test "reshape-compatible? to 1D"
    #t
    (reshape-compatible? #(2 3 4) #(24)))
  
  (test "reshape-compatible? incompatible"
    #f
    (reshape-compatible? #(2 3 4) #(5 5)))
)

;;;; ============================================================
;;;; Type System Tests
;;;; ============================================================

(test-group "Type System"
  
  (test-assert "valid-dtype? f32"
    (valid-dtype? 'f32))
  
  (test-assert "valid-dtype? f64"
    (valid-dtype? 'f64))
  
  (test-assert "valid-dtype? s32"
               (valid-dtype? 's32))
  
  (test-assert "valid-dtype? invalid"
    (valid-dtype? 'invalid))
  
  (test "dtype-size f32"
    4
    (dtype-size 'f32))
  
  (test "dtype-size f64"
    8
    (dtype-size 'f64))
  
  (test-assert "dtype-floating? f64"
    (dtype-floating? 'f64))
  
  (test "dtype-floating? s32"
    #f
    (dtype-floating? 's32))
  
  (test-assert "dtype-signed? s32"
    (dtype-signed? 's32))
  
  (test "dtype-signed? u32"
    #f
    (dtype-signed? 'u32))
  
  (test "promote-types f64 dominant"
    'f64
    (promote-types 'f32 'f64))
  
  (test "promote-types f32 over int"
    'f32
    (promote-types 'f32 's32))
  
  (test "promote-types same type"
    'f32
    (promote-types 'f32 'f32))
  
  (test "promote-types s64 over s32"
    's64
    (promote-types 's32 's64))
  
  (test "infer-reduction-dtype sum preserves"
    'f32
    (infer-reduction-dtype 'sum 'f32))
  
  (test "infer-reduction-dtype mean promotes int to f64"
    'f64
    (infer-reduction-dtype 'mean 's32))
  
  (test "infer-reduction-dtype mean preserves float"
    'f32
    (infer-reduction-dtype 'mean 'f32))
  
  (test "infer-reduction-dtype argmax returns s32"
    's32
    (infer-reduction-dtype 'argmax 'f64))
)

;;;; ============================================================
;;;; Typed Vector Tests
;;;; ============================================================

(test-group "Typed Vectors"
  
  (test "allocate-typed-vector f32"
    10
    (f32vector-length (allocate-typed-vector 'f32 10)))
  
  (test "allocate-typed-vector f64"
    10
    (f64vector-length (allocate-typed-vector 'f64 10)))
  
  (test "allocate-typed-vector s32"
    10
    (s32vector-length (allocate-typed-vector 's32 10)))
  
  (test "typed-vector-ref/set! f32"
    3.14
    (let ((v (allocate-typed-vector 'f32 5)))
      (typed-vector-set! v 'f32 2 3.14)
      (typed-vector-ref v 'f32 2)))
  
  (test "typed-vector-ref/set! f64"
    2.718
    (let ((v (allocate-typed-vector 'f64 5)))
      (typed-vector-set! v 'f64 3 2.718)
      (typed-vector-ref v 'f64 3)))
  
  (test "typed-vector-ref/set! s32"
    42
    (let ((v (allocate-typed-vector 's32 5)))
      (typed-vector-set! v 's32 1 42)
      (typed-vector-ref v 's32 1)))
)

;;;; ============================================================
;;;; Morphism Construction Tests
;;;; ============================================================

(test-group "Morphism Construction"
  
  (test "make-morphism creates concrete array"
    #t
    (let* ((data (f32vector 1.0 2.0 3.0 4.0))
           (m (make-morphism data #(2 2) 'f32)))
      (concrete-array? m)))
  
  (test "make-morphism preserves shape"
    #(2 3)
    (let* ((data (f64vector 1.0 2.0 3.0 4.0 5.0 6.0))
           (m (make-morphism data #(2 3) 'f64)))
      (get-morphism-shape m)))
  
  (test "make-morphism preserves dtype"
    'f32
    (let* ((data (f32vector 1.0 2.0 3.0 4.0))
           (m (make-morphism data #(2 2) 'f32)))
      (get-morphism-dtype m)))
  
  (test-error "make-morphism size mismatch"
    (let ((data (f32vector 1.0 2.0 3.0)))
      (make-morphism data #(2 2) 'f32)))
  
  (test "make-morphism with batch-axis"
    0
    (let* ((data (f32vector 1.0 2.0 3.0 4.0))
           (m (make-morphism data #(2 2) 'f32 batch-axis: 0)))
      (get-morphism-batch-axis m)))
  
  (test "morph-from-list 1D"
    '(1.0 2.0 3.0)
    (let ((m (morph-from-list '(1 2 3) #(3) 'f32)))
      (morph->list m)))
  
  (test "morph-from-list 2D"
    #t
    (let ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64)))
      (lists-equal? (morph->list m)
                   '((1.0 2.0 3.0) (4.0 5.0 6.0)))))
  
  (test "morph-from-list 3D"
    #t
    (let ((m (morph-from-list '(((1 2) (3 4)) ((5 6) (7 8))) 
                             #(2 2 2) 'f32)))
      (lists-equal? (morph->list m)
                   '(((1.0 2.0) (3.0 4.0)) 
                     ((5.0 6.0) (7.0 8.0))))))
  
  (test "morph-from-list preserves dtype"
    's32
    (let ((m (morph-from-list '(1 2 3 4) #(2 2) 's32)))
      (get-morphism-dtype m)))
  
  (test-error "morph-from-list size mismatch"
    (morph-from-list '(1 2 3) #(2 2) 'f32))
)

;;;; ============================================================
;;;; Morphism Accessor Tests
;;;; ============================================================

(test-group "Morphism Accessors"
  
  (let* ((data (f32vector 1.0 2.0 3.0 4.0 5.0 6.0))
         (m (make-morphism data #(2 3) 'f32)))
    
    (test "get-morphism-shape"
      #(2 3)
      (get-morphism-shape m))
    
    (test "get-morphism-dtype"
      'f32
      (get-morphism-dtype m))
    
    (test "get-morphism-batch-axis default"
      -1
      (get-morphism-batch-axis m))
    
    (test "get-operands empty for concrete"
      '()
      (get-operands m))
    
    (test "get-allocation-id default"
      #t
      (< (get-allocation-id m) 0))
    
    (test "concrete-array? true"
      #t
      (concrete-array? m))
    
    (test "abstract-morphism? false"
      #f
      (abstract-morphism? m)))
  
  (let* ((data (f64vector 1.0 2.0 3.0 4.0))
         (m (make-morphism data #(2 2) 'f64 
                          batch-axis: 0 
                          allocation-id: 42)))
    
    (test "get-morphism-batch-axis custom"
      0
      (get-morphism-batch-axis m))
    
    (test "get-allocation-id custom"
      42
      (get-allocation-id m)))
)

;;;; ============================================================
;;;; Morphism Information Tests
;;;; ============================================================

(test-group "Morphism Information"
  
  (let ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f32)))
    
    (test "morph-shape"
      #(2 3)
      (morph-shape m))
    
    (test "morph-dtype"
      'f32
      (morph-dtype m))
    
    (test "morph-size"
      6
      (morph-size m))
    
    (test "morph-rank"
      2
      (morph-rank m))
    
    (test "batched? false by default"
      #f
      (batched? m)))
  
  (let ((m (morph-from-list '((1 2) (3 4) (5 6)) #(3 2) 'f64 
                           batch-axis: 0)))
    
    (test "batched? true when specified"
      #t
      (batched? m))
    
    (test "batch-size"
      3
      (batch-size m)))
)

;;;; ============================================================
;;;; Conversion Tests
;;;; ============================================================

(test-group "Conversion Utilities"
  
  (test "morph->list 1D"
    '(1.0 2.0 3.0 4.0 5.0)
    (let ((m (morph-from-list '(1 2 3 4 5) #(5) 'f32)))
      (morph->list m)))
  
  (test "morph->list 2D"
    #t
    (let ((m (morph-from-list '((1 2) (3 4) (5 6)) #(3 2) 'f64)))
      (lists-equal? (morph->list m)
                   '((1.0 2.0) (3.0 4.0) (5.0 6.0)))))
  
  (test "morph->list 3D"
    #t
    (let ((m (morph-from-list '(((1 2) (3 4)) ((5 6) (7 8))) 
                             #(2 2 2) 'f32)))
      (lists-equal? (morph->list m)
                   '(((1.0 2.0) (3.0 4.0)) 
                     ((5.0 6.0) (7.0 8.0))))))
  
  (test "round-trip list->morph->list"
    #t
    (let* ((original '((1 2 3) (4 5 6)))
           (m (morph-from-list original #(2 3) 'f64))
           (recovered (morph->list m)))
      (lists-equal? original recovered)))
  
  (test "round-trip preserves integers in s32"
    '(1 2 3 4)
    (let* ((original '(1 2 3 4))
           (m (morph-from-list original #(4) 's32))
           (recovered (morph->list m)))
      recovered))
)

;;;; ============================================================
;;;; Edge Cases and Error Handling
;;;; ============================================================

(test-group "Edge Cases"
  
  (test "scalar shape (0D tensor)"
    1
    (shape-size #()))
  
  (test "empty batch axis handling"
    #t
    (let ((m (morph-from-list '(1 2 3) #(3) 'f32)))
      (not (batched? m))))
  
  (test-error "batch-size on non-batched morphism"
    (let ((m (morph-from-list '(1 2 3) #(3) 'f32)))
      (batch-size m)))
  
  (test "large shape computation"
    1000000
    (shape-size #(100 100 100)))
  
  (test "single element array"
    '(42.0)
    (let ((m (morph-from-list '(42) #(1) 'f32)))
      (morph->list m)))
)

;;;; ============================================================
;;;; Run All Tests
;;;; ============================================================

(test-exit)
