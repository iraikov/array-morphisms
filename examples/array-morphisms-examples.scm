;;; array-morphisms-examples.scm
;;; Usage Examples for Array Morphisms

(import scheme (chicken base) (chicken time) (chicken gc) srfi-1 srfi-4)
(import array-morphisms-core)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)

;;;=============================================================================
;;; Basic Functional Operations
;;;=============================================================================

(define (example-basic-operations)
  "Array morphism basic operations examples"
  
  (print "=== Map, Reduce, Slice, Concat Operations ===")
  (let* ((a (morph-from-list '(1.0 2.0 3.0 4.0 5.0) '(5) 'f64))
         (b (morph-from-list '(2.0 4.0 6.0 8.0 10.0) '(5) 'f64))
         
         ;; Map operation - all lazy!
         (squared (morph-map (lambda (x) (* x x)) a))
         (doubled (morph-map (lambda (x) (* x 2)) b))
         
         ;; Complex expression with map
         (complex-expr (morph+ squared doubled))
         
         ;; Reduction operations
         (sum-a (morph-reduce 'sum a))
         (mean-b (morph-reduce 'mean b))
         
         ;; Slice operations
         (slice-a (morph-slice a '(1) '(4)))
         (every-other (morph-slice b '(0) '(5) 2))
         
         ;; Concatenation
         (concat-ab (morph-concat (list a b) 0)))
    
    (print "  Arrays: a=" (morph->list (realize a)) 
           " b=" (morph->list (realize b)))
    (print "  Squared a: " (morph->list (realize squared)))
    (print "  Doubled b: " (morph->list (realize doubled)))
    (print "  Complex (a^2 + 2b): " (morph->list (realize complex-expr)))
    (print "  Sum of a: " (morph->list (realize sum-a)))
    (print "  Mean of b: " (morph->list (realize mean-b)))
    (print "  Slice a[1:4]: " (morph->list (realize slice-a)))
    (print "  Every other b: " (morph->list (realize every-other)))
    (print "  Concatenated: " (morph->list (realize concat-ab)))
    (print "  Estimated materialization bytes: " 
           (estimated-materialization-bytes complex-expr))
    (print)
    )
  
  ;; Performance demonstration with large datasets
  (print "=== Large-Scale Performance ===")
  (let* ((n 100000)
         (large-a (morph-from-list 
                    (map (lambda (i) (+ 0.0 i)) (iota n))
                    (list n) 'f64))
         (large-b (morph-from-list 
                    (map (lambda (i) (+ 0.0 (* 2 i))) (iota n))
                    (list n) 'f64))
         (large-c (morph-from-list 
                    (map (lambda (i) (+ 0.0 (* 3 i))) (iota n))
                    (list n) 'f64))
         
         ;; Mega-complex operation using all features
         (mega-expr 
          (morph-reduce 'mean
                        (morph-map (lambda (x) (* x x))
                                   (morph+ (morph-slice large-a '(0) (list n))
                                           (morph* (morph-map (lambda (x) (* x 2.0)) large-b)
                                                   (morph-sqrt (morph-map abs large-c)))))))
         )
    
    (print "  Large arrays: " n " elements each")
    (print "  Operation: mean(map(x^2, slice(a) + (2b * sqrt(|c|))))")
    (print "  Estimated materialization bytes: " 
           (estimated-materialization-bytes mega-expr))
    
    (let ((start (current-process-milliseconds))
          (mem-before (memory-statistics)))
      (let ((result (realize mega-expr)))
        (let ((end (current-process-milliseconds))
              (mem-after (memory-statistics)))
          (print "  Result: " (morph->list result))
          (print "  Morphism time: " (- end start) " ms")
          (print "  Memory delta: " 
                 (- (vector-ref mem-after 1) (vector-ref mem-before 1)) " bytes"))))
    (print)
    )
  
  ;; Functional programming patterns
  (print "=== Functional Patterns ===")
  (let* ((data (morph-from-list 
                (map exact->inexact (iota 100 1)) 
                '(100) 'f64))  ; 1 to 100
         
         ;; Map-reduce pattern: sum of squares of even numbers
         (even-squares-sum
          (morph-reduce 'sum
                        (morph-map (lambda (x) 
                                    (if (= (modulo (inexact->exact x) 2) 0)
                                        (* x x)
                                        0.0))
                                  data)))
         
         ;; Complex transformation pipeline
         (sliced-data (morph-slice data '(10) '(90)))
         (transformed (morph-map (lambda (x) (sin (* x 0.1))) sliced-data))
         (scaled (morph-map (lambda (x) (* x 100.0)) transformed))
         
         ;; Multi-array operations
         (combined (morph+ (morph-slice data '(0) '(50))
                           (morph-slice data '(50) '(100)))))
    
    (print "  Data: 1 to 100")
    (print "  Even squares sum: " (morph->list (realize even-squares-sum)))
    (print "  Transformed pipeline size: " (morph-size transformed))
    (print "  Combined array size: " (morph-size combined))
    (print)
    )
  )

;;;=============================================================================
;;; Structural Operations (Zero-Copy Views)
;;;=============================================================================

(define (example-structural-operations)
  "Demonstrate zero-copy structural morphisms"
  
  (print "=== Zero-Copy Structural Operations ===")
  (let* ((x (morph-from-list '(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0) '(8) 'f64))
         
         ;; Stride-2 downsample (non-contiguous)
         (downsampled (morph-slice x '(0) '(8) 2))
         
         ;; Reshape after non-contiguous slice
         (as-2x2 (morph-reshape downsampled #(2 2)))
         (as-1x4 (morph-reshape downsampled #(1 4)))
         
         ;; Transpose
         (transposed (morph-transpose as-2x2 '(1 0)))
         
         ;; Stack and unstack
         (row0 (morph-slice as-2x2 '(0 0) '(1 2)))
         (row1 (morph-slice as-2x2 '(1 0) '(2 2)))
         (stacked (morph-stack (list row0 row1) 0)))
    
    (print "  Original x:          " (morph->list (realize x)))
    (print "  Stride-2 slice:      " (morph->list (realize downsampled)))
    (print "  Reshaped to [2,2]:   " (morph->list (realize as-2x2)))
    (print "  Reshaped to [1,4]:   " (morph->list (realize as-1x4)))
    (print "  Transposed [2,2]:    " (morph->list (realize transposed)))
    (print "  Row 0:               " (morph->list (realize row0)))
    (print "  Row 1:               " (morph->list (realize row1)))
    (print "  Stacked rows:        shape " (get-morphism-shape stacked))
    (print))
  )

;;;=============================================================================
;;; Logical Operations
;;;=============================================================================

(define (example-logical-operations)
  "Demonstrate logical/comparison operations"
  
  (print "=== Logical Operations ===")
  (let* ((a (morph-from-list '(0.0 1.0 0.0 1.0 0.0) '(5) 'f64))
         (b (morph-from-list '(0.0 0.0 1.0 1.0 1.0) '(5) 'f64))
         
         ;; Comparison operations return 0.0 (false) or 1.0 (true)
         (gt-result (morph> a b))
         (lt-result (morph< a b))
         (eq-result (morph= a b)))
    
    (print "  a = " (morph->list (realize a)))
    (print "  b = " (morph->list (realize b)))
    (print "  a > b = " (morph->list (realize gt-result)))
    (print "  a < b = " (morph->list (realize lt-result)))
    (print "  a = b = " (morph->list (realize eq-result)))
    (print))
  
  ;; Complex conditions
  (print "=== Complex Conditions ===")
  (let* ((data (morph-from-list '(-2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0) '(8) 'f64))
         
         ;; Create conditions
         (positive (morph> data (morph-from-list (make-list 8 0.0) '(8) 'f64)))
         (less-than-3 (morph< data (morph-from-list (make-list 8 3.0) '(8) 'f64)))
         
         ;; Combine conditions using arithmetic (AND = *, OR = max)
         (in-range (morph* positive less-than-3)))
    
    (print "  Data: " (morph->list (realize data)))
    (print "  Positive (>0): " (morph->list (realize positive)))
    (print "  Less than 3: " (morph->list (realize less-than-3)))
    (print "  In range (0,3): " (morph->list (realize in-range)))
    (print))
  )

;;;=============================================================================
;;; Performance Benchmarks
;;;=============================================================================

(define (benchmark-performance)
  "Performance benchmarks comparing simple vs complex expressions"
  
  (print "=== Performance Benchmarks ===")
  
  ;; Expression complexity comparison
  (print "\nExpression Complexity Impact:")
  (let ((n 1000000))
    (let* ((a (morph-from-list (map exact->inexact (iota n)) (list n) 'f64))
           (b (morph-from-list (map exact->inexact (iota n 1)) (list n) 'f64))
           (c (morph-from-list (map exact->inexact (iota n 2)) (list n) 'f64))
           
           ;; Simple expression
           (simple (morph+ a b))
           
           ;; Complex expression  
           (complex (morph+ (morph* a b)
                            (morph/ (morph-sqrt c) 
                                    (morph+ a (morph-from-list (make-list n 1.0) (list n) 'f64))))))
      
      (print "  Array size: " n " elements")
      (print "  Simple expression materialization bytes: " 
             (estimated-materialization-bytes simple))
      (print "  Complex expression materialization bytes: " 
             (estimated-materialization-bytes complex))
      
      ;; Time simple expression
      (let ((start (current-process-milliseconds)))
        (let ((result1 (realize simple)))
          (let ((mid (current-process-milliseconds)))
            ;; Time complex expression  
            (let ((result2 (realize complex)))
              (let ((end (current-process-milliseconds)))
                (print "  Simple time: " (- mid start) " ms")
                (print "  Complex time: " (- end mid) " ms"))))))
      (print))
    
    ;; Memory comparison
    (print "Memory Usage Analysis:")
    (let* ((a (morph-from-list (map exact->inexact (iota n)) (list n) 'f64))
           (b (morph-from-list (map exact->inexact (iota n 1)) (list n) 'f64))
           (c (morph-from-list (map exact->inexact (iota n 2)) (list n) 'f64))
           
           ;; Deeply nested expression
           (nested-expr (morph/ (morph+ (morph* a b) 
                                       (morph- c a))
                               (morph+ (morph-sqrt a) 
                                      (morph-exp (morph-map (lambda (x) (* x 0.1)) b))))))
      
      (print "  Expression: ((a*b) + (c-a)) / (sqrt(a) + exp(b*0.1))")
      (print "  Array size: " n " elements")
      (print "  Estimated bytes: " (estimated-materialization-bytes nested-expr))
      
      ;; Actual computation with timing
      (let ((start-time (current-process-milliseconds))
            (start-mem (vector-ref (memory-statistics) 1)))
        (let ((result (realize nested-expr)))
          (gc)
          (let ((end-time (current-process-milliseconds))
                (end-mem (vector-ref (memory-statistics) 1)))
            (print "  Actual computation:")
            (print "    Time: " (- end-time start-time) " ms")
            (print "    Memory delta: " (- end-mem start-mem) " bytes")
            (print "    Result mean: " 
                   (/ (fold + 0 (morph->list result)) n))))))))

;;;=============================================================================
;;; Signal Processing Pipeline
;;;=============================================================================

(define (example-signal-processing)
  "Signal processing with morphism pipelines"
  
  (print "=== Signal Processing Pipeline ===")
  (let* (;; Generate synthetic signal: sine wave + noise
         (n 1000)
         (signal-data (map (lambda (i)
                             (+ (sin (* i 0.1)) (* 0.1 (sin (* i 0.5)))))
                           (iota n)))
         (signal (morph-from-list (map exact->inexact signal-data) (list n) 'f64))
         
         ;; Simple moving average (box filter)
         (kernel-size 5)
         (half-k (quotient kernel-size 2))
         
         ;; Create smoothing via slice and reduce pattern
         ;; (simplified - would use proper convolution in production)
         (smoothed (morph-map 
                    (lambda (x) x)  ; Identity as placeholder
                    signal)))
    
    (print "  Signal length: " n)
    (print "  First 10 samples: " 
           (take (morph->list (realize signal)) 10))
    (print "  First 10 smoothed: " 
           (take (morph->list (realize smoothed)) 10))
    
    ;; Frequency domain analysis placeholder
    (let* ((power (morph-map (lambda (x) (* x x)) signal))
           (mean-power (morph-reduce 'mean power)))
      (print "  Mean power: " (morph->list (realize mean-power)))))
  (print))

;;;=============================================================================
;;; Run all examples
;;;=============================================================================

(define (run-all-examples)
  "Execute all example functions"
  
  (print "\n" (make-string 70 #\=))
  (print "  Array Morphisms - Usage Examples")
  (print (make-string 70 #\=) "\n")
  
  (example-basic-operations)
  (example-structural-operations)
  (example-logical-operations)
  (benchmark-performance)
  (example-signal-processing)
  
  (print (make-string 70 #\=))
  (print "  All examples completed")
  (print (make-string 70 #\=) "\n"))

;; Execute examples
(run-all-examples)
