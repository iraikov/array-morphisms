;;; Basic Morphisms
;;
;;; Implements arithmetic, transcendental, and comparison morphisms
;;; with broadcasting, type promotion, and compute-index-fn generation


(module array-morphisms-basic-ops
        
        (;; Binary arithmetic
         morph+ morph- morph* morph/ morph-pow
                ;; Unary transcendental
                morph-sqrt morph-exp morph-log morph-sin morph-cos morph-tan
                ;; Unary arithmetic
                morph-negate morph-abs morph-floor morph-ceiling
                ;; Activation functions
                morph-relu morph-sigmoid morph-tanh-am
                ;; Comparisons
                morph> morph< morph= morph>= morph<=
                ;; Higher-order
                morph-map morph-reduce
                ;; Helpers
                create-binary-morphism
                create-unary-morphism
                create-comparison-morphism
                infer-binary-batch-axis
                infer-unary-dtype
                get-reduction-function)

        
        (import scheme (chicken base)
                srfi-4 datatype matchable
                srfi-1 srfi-69 array-morphisms-core
                array-morphisms-index-fn)

;;; ============================================================================
;;; Helper Functions for Binary Operations
;;; ============================================================================

        
(define (create-binary-morphism operation m1 m2)
  "Create binary morphism with broadcasting and type promotion
   
   Args:
     operation: Symbol naming the operation (add, sub, mul, div, pow)
     m1, m2: Morphisms to combine
   
   Returns:
     morphism-expr with compute-index-fn"
  
  ;; Validate inputs are morphisms
  (unless (array-morphism? m1)
    (error "First operand must be an array morphism" m1))
  (unless (array-morphism? m2)
    (error "Second operand must be an array morphism" m2))
  
  (let* ((shape1 (get-morphism-shape m1))
         (shape2 (get-morphism-shape m2))
         (dtype1 (get-morphism-dtype m1))
         (dtype2 (get-morphism-dtype m2))
         
         ;; Compute result shape and dtype
         (result-shape (broadcast-shapes shape1 shape2))
         (result-dtype (promote-types dtype1 dtype2))
         
         ;; Get combiner function for operation
         (combiner (get-binary-combiner operation))
         
         ;; Create compute-index-fn
         (index-fn (make-binary-compute-index-fn 
                    shape1 shape2 result-shape combiner))
         
         ;; Determine batch axis (if both batched, must match)
         (batch-axis (infer-binary-batch-axis m1 m2)))
    
    (morphism-expr operation 
                   (list m1 m2)
                   index-fn
                   result-shape
                   result-dtype
                   '()
                   batch-axis)))

(define (make-binary-compute-index-fn shape1 shape2 result-shape combiner)
  "Create compute-index-fn for binary operation with broadcasting
   
   Args:
     shape1, shape2: Input shapes
     result-shape: Broadcasted result shape
     combiner: Binary function to apply
   
   Returns:
     compute-index-fn record"
  
  (let ((rank1 (vector-length shape1))
        (rank2 (vector-length shape2))
        (result-rank (vector-length result-shape)))
    
    ;; Create index functions for broadcasting
    (define (broadcast-index-fn shape)
      (lambda (result-idx)
        (let* ((result-rank (length result-idx))
               (shape-rank (vector-length shape))
               (rank-diff (- result-rank shape-rank)))
          
          ;; Drop leading dimensions if result has more dimensions
          (let ((offset-idx (if (> rank-diff 0)
                               (drop result-idx rank-diff)
                               result-idx)))
            
            ;; Broadcast: if dimension is 1, use 0; otherwise use index
            (map (lambda (idx dim)
                   (if (= dim 1) 0 idx))
                 offset-idx
                 (vector->list shape))))))
    
    (let ((idx-fn-1 (broadcast-index-fn shape1))
          (idx-fn-2 (broadcast-index-fn shape2)))
      
      (make-compute-index-fn 
       (list idx-fn-1 idx-fn-2)
       combiner
       (list shape1 shape2)))))

(define (get-binary-combiner operation)
  "Return combiner function for binary operation"
  (case operation
    ((add) +)
    ((sub) -)
    ((mul) *)
    ((div) /)
    ((pow) expt)
    (else (error "Unknown binary operation" operation))))

(define (infer-binary-batch-axis m1 m2)
  "Infer batch axis for binary operation
   
   Rules:
   - If both not batched: result not batched (-1)
   - If one batched, one not: result batched (inherit batch axis)
   - If both batched with same axis: result batched (same axis)
   - If both batched with different axes: error"
  
  (let ((batch1 (get-morphism-batch-axis m1))
        (batch2 (get-morphism-batch-axis m2)))
    
    (cond
      ;; Both not batched
      ((and (= batch1 -1) (= batch2 -1)) -1)
      
      ;; m1 batched, m2 not
      ((and (>= batch1 0) (= batch2 -1)) batch1)
      
      ;; m1 not, m2 batched
      ((and (= batch1 -1) (>= batch2 0)) batch2)
      
      ;; Both batched, same axis
      ((and (>= batch1 0) (>= batch2 0) (= batch1 batch2)) batch1)
      
      ;; Both batched, different axes
      (else (error "Cannot combine morphisms with different batch axes"
                   batch1 batch2)))))

;;; ============================================================================
;;; Binary Arithmetic Morphisms
;;; ============================================================================

(define (morph+ m1 m2)
  "Element-wise addition with broadcasting
   
   Example:
     (morph+ (morph-from-list '(1 2 3)) 
             (morph-from-list '(10)))"
  (create-binary-morphism 'add m1 m2))

(define (morph- m1 m2)
  "Element-wise subtraction with broadcasting"
  (create-binary-morphism 'sub m1 m2))

(define (morph* m1 m2)
  "Element-wise multiplication with broadcasting"
  (create-binary-morphism 'mul m1 m2))

(define (morph/ m1 m2)
  "Element-wise division with broadcasting
   
   Note: No automatic check for division by zero
         User should validate inputs"
  (create-binary-morphism 'div m1 m2))

(define (morph-pow m1 m2)
  "Element-wise exponentiation with broadcasting
   
   Example:
     (morph-pow base-morph exponent-morph)"
  (create-binary-morphism 'pow m1 m2))

;;; ============================================================================
;;; Helper Functions for Unary Operations
;;; ============================================================================

(define (floating-point-type? dtype)
  (case dtype
    ((f64 f32) #t)
    (else #f)))

(define (create-unary-morphism operation m)
  "Create unary morphism
   
   Args:
     operation: Symbol naming the operation (sqrt, exp, log, etc.)
     m: Morphism to transform
   
   Returns:
     morphism-expr with compute-index-fn"
  
  (unless (array-morphism? m)
    (error "Operand must be an array morphism" m))
  
  (let* ((shape (get-morphism-shape m))
         (dtype (get-morphism-dtype m))
         
         ;; Some operations promote to float
         (result-dtype (infer-unary-dtype operation dtype))
         
         ;; Get transformer function
         (transformer (get-unary-transformer operation))
         
         ;; Create compute-index-fn
         (index-fn (make-unary-compute-index-fn transformer))
         
         ;; Preserve batch axis
         (batch-axis (get-morphism-batch-axis m)))
    
    (morphism-expr operation
                   (list m)
                   index-fn
                   shape
                   result-dtype
                   '()
                   batch-axis)))

(define (make-unary-compute-index-fn transformer)
  "Create compute-index-fn for unary operation
   
   Args:
     transformer: Unary function to apply
   
   Returns:
     compute-index-fn record"
  
  (make-compute-index-fn 
   (list (lambda (idx) idx))  ; Identity index function
   (lambda (x) (transformer x))
   '()))

(define (get-unary-transformer operation)
  "Return transformer function for unary operation"
  (case operation
    ;; Transcendental
    ((sqrt) sqrt)
    ((exp) exp)
    ((log) log)
    ((sin) sin)
    ((cos) cos)
    ((tan) tan)
    
    ;; Arithmetic
    ((negate) -)
    ((abs) abs)
    ((floor) floor)
    ((ceiling) ceiling)

    ;; Activations
    ((relu)    (lambda (x) (max 0.0 (exact->inexact x))))
    ((sigmoid) (lambda (x) (let ((xf (exact->inexact x)))
                              (/ 1.0 (+ 1.0 (exp (- xf)))))))
    ((tanh)    (lambda (x)
                 (let* ((xf (exact->inexact x))
                        (e2 (exp (* 2.0 xf))))
                   (/ (- e2 1.0) (+ e2 1.0)))))

    (else (error "Unknown unary operation" operation))))

(define (infer-unary-dtype operation dtype)
  "Infer result dtype for unary operation
   
   Transcendental functions promote integers to f64"
  
  (case operation
    ;; Transcendental always returns float
    ((sqrt exp log sin cos tan relu sigmoid tanh)
     (if (floating-point-type? dtype)
         dtype
         'f64))
    
    ;; Arithmetic preserves type
    ((negate abs floor ceiling)
     dtype)
    
    (else dtype)))

;;; ============================================================================
;;; Unary Transcendental Morphisms
;;; ============================================================================

(define (morph-sqrt m)
  "Element-wise square root
   
   Promotes integers to f64"
  (create-unary-morphism 'sqrt m))

(define (morph-exp m)
  "Element-wise exponential (e^x)
   
   Promotes integers to f64"
  (create-unary-morphism 'exp m))

(define (morph-log m)
  "Element-wise natural logarithm
   
   Promotes integers to f64
   Note: No automatic check for log of negative numbers"
  (create-unary-morphism 'log m))

(define (morph-sin m)
  "Element-wise sine
   
   Promotes integers to f64"
  (create-unary-morphism 'sin m))

(define (morph-cos m)
  "Element-wise cosine
   
   Promotes integers to f64"
  (create-unary-morphism 'cos m))

(define (morph-tan m)
  "Element-wise tangent
   
   Promotes integers to f64"
  (create-unary-morphism 'tan m))

;;; ============================================================================
;;; Unary Arithmetic Morphisms
;;; ============================================================================

(define (morph-negate m)
  "Element-wise negation
   
   Preserves dtype"
  (create-unary-morphism 'negate m))

(define (morph-abs m)
  "Element-wise absolute value
   
   Preserves dtype"
  (create-unary-morphism 'abs m))

(define (morph-floor m)
  "Element-wise floor
   
   Preserves dtype"
  (create-unary-morphism 'floor m))

(define (morph-ceiling m)
  "Element-wise ceiling

   Preserves dtype"
  (create-unary-morphism 'ceiling m))

(define (morph-relu m)    (create-unary-morphism 'relu    m))
(define (morph-sigmoid m) (create-unary-morphism 'sigmoid m))
(define (morph-tanh-am m) (create-unary-morphism 'tanh    m))

;;; ============================================================================
;;; Helper Functions for Comparison Operations
;;; ============================================================================

(define (create-comparison-morphism operation m1 m2)
  "Create comparison morphism with broadcasting
   
   Args:
     operation: Symbol naming the comparison (gt, lt, eq, etc.)
     m1, m2: Morphisms to compare
   
   Returns:
     morphism-expr with compute-index-fn
     Result dtype is f64 (returns 0.0 or 1.0)"
  
  (unless (array-morphism? m1)
    (error "First operand must be an array morphism" m1))
  (unless (array-morphism? m2)
    (error "Second operand must be an array morphism" m2))
  
  (let* ((shape1 (get-morphism-shape m1))
         (shape2 (get-morphism-shape m2))
         
         ;; Compute result shape
         (result-shape (broadcast-shapes shape1 shape2))
         
         ;; Result is always f64 (boolean as 0.0/1.0)
         (result-dtype 'f64)
         
         ;; Get comparator function
         (comparator (get-comparison-function operation))
         
         ;; Create compute-index-fn
         (index-fn (make-binary-compute-index-fn 
                    shape1 shape2 result-shape
                    (lambda (x y) 
                      (if (comparator x y) 1.0 0.0))))
         
         ;; Determine batch axis
         (batch-axis (infer-binary-batch-axis m1 m2)))
    
    (morphism-expr operation
                   (list m1 m2)
                   index-fn
                   result-shape
                   result-dtype
                   '()
                   batch-axis)))

(define (get-comparison-function operation)
  "Return comparison function for operation"
  (case operation
    ((gt) >)
    ((lt) <)
    ((eq) =)
    ((ge) >=)
    ((le) <=)
    (else (error "Unknown comparison operation" operation))))

;;; ============================================================================
;;; Comparison Morphisms
;;; ============================================================================

(define (morph> m1 m2)
  "Element-wise greater-than comparison with broadcasting
   
   Returns morphism with values 0.0 (false) or 1.0 (true)"
  (create-comparison-morphism 'gt m1 m2))

(define (morph< m1 m2)
  "Element-wise less-than comparison with broadcasting
   
   Returns morphism with values 0.0 (false) or 1.0 (true)"
  (create-comparison-morphism 'lt m1 m2))

(define (morph= m1 m2)
  "Element-wise equality comparison with broadcasting
   
   Returns morphism with values 0.0 (false) or 1.0 (true)"
  (create-comparison-morphism 'eq m1 m2))

(define (morph>= m1 m2)
  "Element-wise greater-than-or-equal comparison with broadcasting
   
   Returns morphism with values 0.0 (false) or 1.0 (true)"
  (create-comparison-morphism 'ge m1 m2))

(define (morph<= m1 m2)
  "Element-wise less-than-or-equal comparison with broadcasting
   
   Returns morphism with values 0.0 (false) or 1.0 (true)"
  (create-comparison-morphism 'le m1 m2))

;;; ============================================================================
;;; Higher-Order Morphisms
;;; ============================================================================

(define (morph-map fn m)
  "Apply function element-wise to morphism
   
   Args:
     fn: Unary function to apply
     m: Input morphism
   
   Returns:
     morphism-expr with mapped values
   
   Example:
     (morph-map (lambda (x) (* x x)) m)  ; Square each element"
  
  (unless (array-morphism? m)
    (error "Operand must be an array morphism" m))
  
  (let* ((shape (get-morphism-shape m))
         (dtype (get-morphism-dtype m))
         
         ;; Create compute-index-fn
         (index-fn (make-unary-compute-index-fn fn))
         
         ;; Preserve batch axis
         (batch-axis (get-morphism-batch-axis m)))
    
    (morphism-expr 'map
                   (list m)
                   index-fn
                   shape
                   dtype
                   `((function . ,fn))
                   batch-axis)))


(define (get-reduction-function op)
  "Return the binary accumulation function for a reduction op.

  For 'mean, the accumulator is + (same as 'sum); the caller is
  responsible for the final division by reduce-size."
  (case op
    ((sum mean) +)
    ((prod)     *)
    ((max)      max)
    ((min)      min)
    (else (error "Unknown reduction operation" op))))

(define (morph-reduce op m #!optional (axes '()) (keepdims? #f))
  "Reduce morphism along specified axes
   
   Args:
     op: Reduction operation (sum, mean, max, min, prod)
     m: Input morphism
     axes: List of axes to reduce (empty = all axes)
     keepdims?: Keep reduced dimensions with size 1?
   
   Returns:
     reduction-morphism
   
   Examples:
     (morph-reduce 'sum m)              ; Sum all elements
     (morph-reduce 'mean m '(0))        ; Mean along axis 0
     (morph-reduce 'max m '(1 2) #t)    ; Max along axes 1,2, keep dims"
  
  (unless (array-morphism? m)
    (error "Operand must be an array morphism" m))
  
  (let* ((shape (get-morphism-shape m))
         (rank (vector-length shape))
         
         ;; Normalize axes (handle negative indices)
         (normalized-axes 
          (if (null? axes)
              (iota rank)  ; Reduce all axes
              (map (lambda (ax) (normalize-axis ax rank)) axes)))
         
         ;; Compute result shape
         (result-shape 
          (if keepdims?
              ;; Keep dimensions with size 1
              (list->vector
               (map (lambda (i)
                      (if (member i normalized-axes)
                          1
                          (vector-ref shape i)))
                    (iota rank)))
              ;; Remove reduced dimensions
              (list->vector
               (fold-right
                (lambda (i acc)
                  (if (member i normalized-axes)
                      acc
                      (cons (vector-ref shape i) acc)))
                '()
                (iota rank)))))
         
         ;; Infer result dtype
         (result-dtype (infer-reduction-dtype op (get-morphism-dtype m)))

         ; Get source index function from operand
         (source-fn (get-index-fn m))

         ;; Get reducer function for operation
         (reducer (get-reduction-function op))
         
         ;; Create reduction index function
         (index-fn (make-reduction-index-fn source-fn normalized-axes
                                            reducer keepdims?))
         
         ;; Compute result batch axis
         (batch-axis (get-morphism-batch-axis m))
         (result-batch-axis 
          (cond
            ((= batch-axis -1) -1)  ; Not batched
            ((member batch-axis normalized-axes)
             (if keepdims?
                 batch-axis  ; Keep same axis
                 -1))        ; Reduced away
            (else 
             ;; Adjust for removed dimensions
             (if keepdims?
                 batch-axis
                 (- batch-axis 
                    (length (filter (lambda (ax) (< ax batch-axis))
                                   normalized-axes))))))))
    
    (reduction-morphism op
                        m
                        normalized-axes
                        index-fn
                        result-shape
                        result-dtype
                        result-batch-axis)))

)
