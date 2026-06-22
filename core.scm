;;; array-morphisms-core.scm
;;; Core Infrastructure for Array Morphisms
;;; Provides foundational data structures and utilities


;;;; ============================================================
;;;; Core Data Types
;;;; ============================================================


(module array-morphisms-core

  (;; ============================================================
   ;; Core Type Constructors and Predicates
   ;; ============================================================
   
   ;; Array morphism algebraic datatype
   array-morphism              ; Type
   array-morphism?             ; Type predicate
   concrete-array              ; Constructor
   morphism-expr               ; Constructor
   reduction-morphism          ; Constructor
   
   ;; Morphism classification
   concrete-array?
   abstract-morphism?
   morphism-expr?
   
   ;; ============================================================
   ;; Index Function Types
   ;; ============================================================
   
   ;; ;; Affine index functions (pure transformations)
   affine-index-fn?
   affine-index-fn
   identity-fn
   permutation-fn
   diagonal-fn
   general-fn
   
   ;; Computational index functions (arithmetic, transcendental)
   compute-index-fn?
   make-compute-index-fn
   compute-index-fn-input-fns
   compute-index-fn-combiner
   compute-index-fn-input-shapes
   
   ;; Composed index functions (f o g)
   composed-index-fn?
   make-composed-index-fn
   composed-index-fn-outer
   composed-index-fn-inner
   
   ;; Window index functions (im2col, convolution, pooling)
   window-index-fn?
   make-window-index-fn
   window-index-fn-source-fn
   window-index-fn-window-shape
   window-index-fn-stride-shape
   window-index-fn-pad-shape
   window-index-fn-mode
   
   ;; Reduction index functions (aggregate over axes)
   reduction-index-fn?
   make-reduction-index-fn
   reduction-index-fn-source-fn
   reduction-index-fn-reduce-axes
   reduction-index-fn-reducer
   reduction-index-fn-keepdims?
   
   ;; Batch map index functions (per-batch-element operations)
   batch-map-index-fn?
   make-batch-map-index-fn
   batch-map-index-fn-inner-fn
   batch-map-index-fn-batch-axis
   batch-map-index-fn-elem-shape
   
   ;; Batch reduce index functions (cross-batch reduction)
   batch-reduce-index-fn?
   make-batch-reduce-index-fn
   batch-reduce-index-fn-reducer
   batch-reduce-index-fn-batch-axis
   batch-reduce-index-fn-keepdims?

   ;; col2im index functions (accumulation)
   col2im-index-fn?
   make-col2im-index-fn-record
   col2im-index-fn-kernel-h
   col2im-index-fn-kernel-w
   col2im-index-fn-stride-h
   col2im-index-fn-stride-w
   col2im-index-fn-pad-h
   col2im-index-fn-pad-w
   col2im-index-fn-batched?

   ;; Routing index functions (multi-source: stack, concat)
   stack-index-fn
   make-stack-index-fn
   stack-index-fn?
   stack-index-fn-axis
   concat-index-fn
   make-concat-index-fn
   concat-index-fn?
   concat-index-fn-axis
   concat-index-fn-offsets

   ;; Classification predicates
   index-fn?
   single-source-index-fn?
   routing-index-fn?
   
   ;; ============================================================
   ;; Shape and Stride Utilities
   ;; ============================================================
   
   ;; Shape conversion
   shape->list
   list->shape
   
   ;; Shape queries
   shape-rank
   shape-size
   shape-dim
   
   ;; Shape validation
   validate-shape
   
   ;; Stride computation
   compute-strides
   
   ;; Index transformation
   linear-to-multi-index
   multi-to-linear-index
   
   ;; Axis handling
   normalize-axis
   
   ;; Broadcasting
   shapes-compatible?
   broadcast-shapes
   
   ;; Reshape validation
   reshape-compatible?
   
   ;; ============================================================
   ;; Type System
   ;; ============================================================
   
   ;; Type validation and queries
   valid-dtype?
   dtype-size
   dtype-floating?
   dtype-signed?
   
   ;; Type promotion and inference
   promote-types
   infer-reduction-dtype
   
   ;; ============================================================
   ;; Typed Vector Operations
   ;; ============================================================
   
   ;; Vector allocation
   allocate-typed-vector
   
   ;; Vector access
   typed-vector-ref
   typed-vector-set!
   
   ;; ============================================================
   ;; Morphism Accessors
   ;; ============================================================
   
   ;; Core properties
   get-morphism-shape
   get-morphism-dtype
   get-morphism-batch-axis
   
   ;; Structural information
   get-index-fn
   get-operands
   get-allocation-id
   
   ;; ============================================================
   ;; Morphism Construction
   ;; ============================================================
   
   ;; Construction from data
   make-morphism
   morph-from-list
   
   ;; ============================================================
   ;; Morphism Information
   ;; ============================================================

   ;; Static cost estimation (Item 1 of FUSED_ATTENTION_IMPLEMENTATION_PROPOSAL)
   estimated-materialization-bytes

   ;; Shape and type queries (convenience aliases)
   morph-shape
   morph-dtype
   morph-size
   morph-rank
   
   ;; Batch information
   batched?
   batch-size
   
   ;; ============================================================
   ;; Conversion Utilities
   ;; ============================================================
   
   ;; Morphism to list conversion
   morph->list
   
   ;; Helper for nested list construction
   nest-list
   
   ;; Helper for flattening
   flatten-nested-list
   )

  (import scheme (chicken base) srfi-4 datatype srfi-1 srfi-69)

  (define (typed-vector? x)
    (or (f32vector? x)
        (f64vector? x)
        (s32vector? x)
        (s64vector? x)))
        

  
;;; Array Morphism - Main algebraic datatype
(define-datatype array-morphism array-morphism?
  ;; Terminal morphism: materialized concrete array
  (concrete-array
   (data typed-vector?)     ; Typed vector (f32/f64/s32/s64)
   (shape vector?)          ; Shape [d0, d1, ..., dn]
   (strides vector?)        ; Row-major strides
   (offset exact-integer?)  ; Base offset (for views)
   (dtype symbol?)          ; Element type
   (allocation-id exact-integer?) ; For memory reuse tracking
   (batch-axis exact-integer?))   ; Batch dimension (-1 if none)
  
  ;; Abstract morphism: deferred computation
  (morphism-expr
   (operation symbol?)      ; Operation name (add, mul, im2col, etc.)
   (operands list?)         ; List of morphism operands
   (index-fn index-fn?)     ; MoA index function (output -> input indices)
   (shape vector?)          ; Result shape
   (dtype symbol?)          ; Result type
   (metadata list?)         ; Optional metadata (alist)
   (batch-axis exact-integer?))  ; Batch dimension tracking
  
  ;; Reduction morphism: reduces rank/dimensions
  (reduction-morphism
   (operation symbol?)      ; Reduction op (sum, mean, max, etc.)
   (operand array-morphism?) ; Single operand
   (reduce-axes list?)      ; Axes to reduce
   (index-fn reduction-index-fn?)    ; Accumulation function
   (shape vector?)          ; Result shape
   (dtype symbol?)          ; Result type
   (batch-axis exact-integer?))) ; Batch axis after reduction

(define (morphism-expr? x)
  (cases array-morphism x
         (morphism-expr (op operands idx-fn shape dtype metadata batch-axis)
                        #t)
         (else #f)))
  
;;; Index Function Types

;; Pure affine transformations (reshape, transpose, slice, pad)
(define-datatype affine-index-fn affine-index-fn?
  ;; f(i) = i
  ;; Used by: reshape - reinterpret flat buffer with new shape.
  (identity-fn)

  ;; f(i)[k] = i[perm[k]]
  ;; Used by: transpose.
  ;; perm: list of length rank, each element in [0, rank).
  (permutation-fn (perm list?))

  ;; f(i)[k] = diag[k] * i[k] + bias[k]
  ;; Used by: slice - diag holds step sizes, bias holds start offsets.
  ;; diag, bias: lists of length rank.
  (diagonal-fn (diag list?) (bias list?))

  ;; f(i) = A*i + b
  ;; Used by: composed affine transforms that do not reduce to the above.
  ;; matrix: nested list (row-major), bias: list or #f.
  (general-fn (matrix list?) (bias list?)))

;; Computational transformations (arithmetic, transcendental)
(define-record compute-index-fn
  input-fns     ; List of index functions for operands
  combiner      ; Function combining retrieved values
  input-shapes) ; Operand shapes for validation

;; Morphism composition: f ∘ g
(define-record composed-index-fn
  outer         ; Outer morphism index function
  inner)        ; Inner morphism index function

;; Window operations (im2col, convolution, pooling)
(define-record window-index-fn
  source-fn     ; Source array index function
  window-shape  ; Window dimensions (list)
  stride-shape  ; Stride values (list)
  pad-shape     ; Padding values (list of pairs: ((pad-before . pad-after) ...))
  mode)         ; 'valid, 'same, or 'full

;; Reduction (aggregate over axes)
(define-record reduction-index-fn
  source-fn     ; Input index function
  reduce-axes   ; Axes to reduce (list)
  reducer       ; Reduction operation (procedure)
  keepdims?)    ; Keep reduced dimensions?

;; Batch operations
(define-record batch-map-index-fn
  inner-fn      ; Function per batch element
  batch-axis    ; Batch dimension
  elem-shape)   ; Element shape (without batch dimension)

(define-record batch-reduce-index-fn
  reducer       ; Reduction across batch
  batch-axis    ; Batch dimension
  keepdims?)    ; Keep batch dimension?

(define-record-type col2im-index-fn
  (make-col2im-index-fn-record kernel-h kernel-w stride-h stride-w pad-h pad-w batched?)
  col2im-index-fn?
  (kernel-h col2im-index-fn-kernel-h)
  (kernel-w col2im-index-fn-kernel-w)
  (stride-h col2im-index-fn-stride-h)
  (stride-w col2im-index-fn-stride-w)
  (pad-h col2im-index-fn-pad-h)
  (pad-w col2im-index-fn-pad-w)
  (batched? col2im-index-fn-batched?))

;; Stack: inserts a new axis at position `axis`.
;; apply-stack-index-fn returns (source-id . source-idx).
;; source-id  = out-idx[axis]   (which operand)
;; source-idx = out-idx with axis dimension removed
(define-record stack-index-fn axis)

;; Concat: concatenates along an existing axis.
;; apply-concat-index-fn returns (source-id . source-idx).
;; `offsets` is a list of cumulative start offsets, length = n-operands.
(define-record concat-index-fn axis offsets)


(define (index-fn? obj)
  "Predicate for any index function type"
  (or (affine-index-fn? obj)
      (compute-index-fn? obj)
      (composed-index-fn? obj)
      (window-index-fn? obj)
      (reduction-index-fn? obj)
      (batch-map-index-fn? obj)
      (batch-reduce-index-fn? obj)
      (col2im-index-fn? obj)
      (stack-index-fn? obj)    
      (concat-index-fn? obj)   
      (procedure? obj)
      ))

(define (single-source-index-fn? fn)
  "True for all index-fn variants that map one output index to one source index."
  (or (affine-index-fn? fn)
      (compute-index-fn? fn)
      (composed-index-fn? fn)
      (window-index-fn? fn)
      (reduction-index-fn? fn)
      (batch-map-index-fn? fn)
      (batch-reduce-index-fn? fn)
      (col2im-index-fn? fn)))

(define (routing-index-fn? fn)
  "True for index-fn variants that return (source-id . source-idx)."
  (or (stack-index-fn? fn)
      (concat-index-fn? fn)))

;;;; ============================================================
;;;; Shape and Stride Utilities
;;;; ============================================================

(define (shape->list shape)
  "Convert shape vector to list"
  (if (vector? shape)
      (vector->list shape)
      shape))

(define (list->shape lst)
  "Convert list to shape vector"
  (if (list? lst)
      (list->vector lst)
      lst))

(define (shape-rank shape)
  "Return number of dimensions in shape"
  (if (vector? shape)
      (vector-length shape)
      (length shape)))

(define (shape-size shape)
  "Return total number of elements in shape"
  (if (vector? shape)
      (apply * (vector->list shape))
      (apply * shape)))

(define (shape-dim shape axis)
  "Return dimension size at given axis"
  (if (vector? shape)
      (vector-ref shape axis)
      (list-ref shape axis)))

(define (compute-strides shape)
  "Compute row-major strides for given shape
   
   Example: shape [3, 4, 5] -> strides [20, 5, 1]"
  (let* ((dims (shape->list shape))
         (rank (length dims))
         (strides (make-vector rank)))
    
    ;; Compute strides from right to left
    (let loop ((i (- rank 1))
               (stride 1))
      (when (>= i 0)
        (vector-set! strides i stride)
        (loop (- i 1) (* stride (list-ref dims i)))))
    
    strides))

(define (linear-to-multi-index linear-idx shape)
  "Convert linear index to multi-dimensional index
   
   Example: linear-idx=23, shape=[3,4,5] -> [1,0,3]"
  (let* ((dims (shape->list shape))
         (rank (length dims))
         (strides (vector->list (compute-strides (list->shape dims))))
         (multi-idx (make-vector rank)))
    
    (let loop ((i 0)
               (remaining linear-idx))
      (when (< i rank)
        (let ((stride (list-ref strides i)))
          (vector-set! multi-idx i (quotient remaining stride))
          (loop (+ i 1) (modulo remaining stride)))))
    
    multi-idx))

(define (multi-to-linear-index multi-idx strides #!optional (offset 0))
  "Convert multi-dimensional index to linear index
   
   Example: multi-idx=[1,0,3], strides=[20,5,1], offset=0 -> 23"
  (+ offset
     (apply + (map * 
                   (vector->list multi-idx)
                   (vector->list strides)))))

(define (normalize-axis axis rank)
  "Normalize axis to positive value, handling negative indexing
   
   Example: axis=-1, rank=3 -> 2
            axis=1, rank=3 -> 1"
  (if (negative? axis)
      (+ rank axis)
      axis))

(define (validate-shape shape)
  "Validate that shape contains only positive integers"
  (let ((dims (shape->list shape)))
    (unless (every (lambda (d) (and (exact-integer? d) (positive? d))) dims)
      (error "Invalid shape: dimensions must be positive integers" shape))
    shape))

(define (shapes-compatible? shape1 shape2)
  "Check if two shapes are compatible for broadcasting"
  (let ((dims1 (reverse (shape->list shape1)))
        (dims2 (reverse (shape->list shape2))))
    
    (let loop ((d1 dims1) (d2 dims2))
      (cond
        ((and (null? d1) (null? d2)) #t)
        ((null? d1) #t)
        ((null? d2) #t)
        (else
         (let ((size1 (car d1))
               (size2 (car d2)))
           (if (or (= size1 size2) (= size1 1) (= size2 1))
               (loop (cdr d1) (cdr d2))
               #f)))))))

(define (broadcast-shapes shape1 shape2)
  "Compute broadcasted shape from two compatible shapes"
  (unless (shapes-compatible? shape1 shape2)
    (error "Shapes not compatible for broadcasting" shape1 shape2))
  
  (let* ((dims1 (reverse (shape->list shape1)))
         (dims2 (reverse (shape->list shape2)))
         (max-rank (max (length dims1) (length dims2)))
         (result (make-vector max-rank)))
    
    ;; Pad shorter shape with 1s
    (let ((padded1 (append dims1 (make-list (- max-rank (length dims1)) 1)))
          (padded2 (append dims2 (make-list (- max-rank (length dims2)) 1))))
      
      ;; Compute broadcasted dimensions
      (let ((bc-dims
             (let loop ((i 0) (p1 padded1) (p2 padded2) (result '()))
               (if (= i max-rank)
                   result
                   (let ((d1 (car p1))
                         (d2 (car p2)))
                     (loop (+ i 1) (cdr p1) (cdr p2)
                           (cons (max d1 d2) result)))))))
        
        ;; Result is already in correct order (built from reversed dims)
        (list->vector bc-dims))
      ))
  )

(define (reshape-compatible? old-shape new-shape)
  "Check if reshape is valid (same total size)"
  (= (shape-size old-shape) (shape-size new-shape)))

;;;; ============================================================
;;;; Type System
;;;; ============================================================

;;; Supported dtypes
(define *supported-dtypes*
  '(f32 f64 s32 s64 u32 u64))

(define (valid-dtype? dtype)
  "Check if dtype is supported"
  (memq dtype *supported-dtypes*))

(define (dtype-size dtype)
  "Return size in bytes for dtype"
  (case dtype
    ((f32 s32 u32) 4)
    ((f64 s64 u64) 8)
    (else (error "Unknown dtype" dtype))))

(define (dtype-floating? dtype)
  "Check if dtype is floating-point"
  (memq dtype '(f32 f64)))

(define (dtype-signed? dtype)
  "Check if dtype is signed integer"
  (memq dtype '(s32 s64)))

(define (estimated-materialization-bytes m)
  "Bytes that would be written if morphism m were materialized now.
   Computed from shape and dtype alone; no realization occurs.
   Use as a static cost signal for fuse-vs-materialize decisions."
  (* (shape-size (get-morphism-shape m))
     (dtype-size  (get-morphism-dtype m))))

(define (promote-types dtype1 dtype2)
  "Promote two dtypes to common type
   
   Promotion rules:
   - f64 > f32 > s64 > s32 > u64 > u32
   - mixed signed/unsigned promotes to signed
   - mixed int/float promotes to float"
  
  (cond
    ((eq? dtype1 dtype2) dtype1)
    ((eq? dtype1 'f64) 'f64)
    ((eq? dtype2 'f64) 'f64)
    ((eq? dtype1 'f32) 'f32)
    ((eq? dtype2 'f32) 'f32)
    ((eq? dtype1 's64) 's64)
    ((eq? dtype2 's64) 's64)
    ((eq? dtype1 's32) 's32)
    ((eq? dtype2 's32) 's32)
    ((eq? dtype1 'u64) 'u64)
    ((eq? dtype2 'u64) 'u64)
    (else 'u32)))

(define (infer-reduction-dtype op dtype)
  "Infer result dtype for reduction operation
   
   Most reductions preserve dtype, except:
   - mean: always promotes to floating point
   - argmax/argmin: return s32 indices"
  
  (case op
    ((mean)
     (if (dtype-floating? dtype)
         dtype
         'f64))
    ((argmax argmin)
     's32)
    (else dtype)))

;;;; ============================================================
;;;; Typed Vector Allocation
;;;; ============================================================

(define (allocate-typed-vector dtype size)
  "Allocate typed vector of given size and dtype"
  (case dtype
    ((f32) (make-f32vector size 0.0))
    ((f64) (make-f64vector size 0.0))
    ((s32) (make-s32vector size 0))
    ((s64) (make-s64vector size 0))
    ((u32) (make-u32vector size 0))
    ((u64) (make-u64vector size 0))
    (else (error "Unsupported dtype" dtype))))

(define (typed-vector-ref vec dtype idx)
  "Get element from typed vector"
  (case dtype
    ((f32) (f32vector-ref vec idx))
    ((f64) (f64vector-ref vec idx))
    ((s32) (s32vector-ref vec idx))
    ((s64) (s64vector-ref vec idx))
    ((u32) (u32vector-ref vec idx))
    ((u64) (u64vector-ref vec idx))
    (else (error "Unsupported dtype" dtype))))

(define (typed-vector-set! vec dtype idx val)
  "Set element in typed vector"
  (case dtype
    ((f32) (f32vector-set! vec idx (exact->inexact val)))
    ((f64) (f64vector-set! vec idx (exact->inexact val)))
    ((s32) (s32vector-set! vec idx (inexact->exact (truncate val))))
    ((s64) (s64vector-set! vec idx (inexact->exact (truncate val))))
    ((u32) (u32vector-set! vec idx (inexact->exact (abs (truncate val)))))
    ((u64) (u64vector-set! vec idx (inexact->exact (abs (truncate val)))))
    (else (error "Unsupported dtype" dtype))))

;;;; ============================================================
;;;; Morphism Accessors
;;;; ============================================================

(define (get-morphism-shape m)
  "Extract shape from morphism"
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
                    shape)
    (morphism-expr (op operands idx-fn shape dtype metadata batch-axis)
                   shape)
    (reduction-morphism (op operand axes idx-fn shape dtype batch-axis)
                        shape)))

(define (get-morphism-dtype m)
  "Extract dtype from morphism"
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
                    dtype)
    (morphism-expr (op operands idx-fn shape dtype metadata batch-axis)
                   dtype)
    (reduction-morphism (op operand axes idx-fn shape dtype batch-axis)
                        dtype)))

(define (get-morphism-batch-axis m)
  "Extract batch axis from morphism (-1 if not batched)"
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
                    batch-axis)
    (morphism-expr (op operands idx-fn shape dtype metadata batch-axis)
                   batch-axis)
    (reduction-morphism (op operand axes idx-fn shape dtype batch-axis)
                        batch-axis)))

(define (get-index-fn m)
  "Extract index function from morphism"
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
                    ;; Identity index function for concrete arrays
                    (identity-fn))
    (morphism-expr (op operands idx-fn shape dtype metadata batch-axis)
                   idx-fn)
    (reduction-morphism (op operand axes idx-fn shape dtype batch-axis)
                        idx-fn)))

(define (get-operands m)
  "Extract operands from morphism"
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
                    '())
    (morphism-expr (op operands idx-fn shape dtype metadata batch-axis)
                   operands)
    (reduction-morphism (op operand axes idx-fn shape dtype batch-axis)
                        (list operand))))

(define (get-allocation-id m)
  "Extract allocation ID from morphism (-1 if not concrete)"
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
                    alloc-id)
    (else -1)))

(define (concrete-array? m)
  "Check if morphism is concrete (materialized)"
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis) #t)
    (else #f)))

(define (abstract-morphism? m)
  "Check if morphism is abstract (deferred)"
  (not (concrete-array? m)))

;;;; ============================================================
;;;; Basic Morphism Construction
;;;; ============================================================

(define (make-morphism data shape dtype #!key 
                      (batch-axis -1)
                      (allocation-id -1))
  "Create concrete morphism from data
   
   Args:
     data: typed vector containing array data
     shape: shape vector or list
     dtype: element type symbol
     batch-axis: batch dimension (default: -1, not batched)
     allocation-id: allocation ID for memory reuse (default: -1)
   
   Returns:
     Concrete array morphism"
  
  (let ((shape-vec (list->shape shape)))
    (validate-shape shape-vec)
    (unless (valid-dtype? dtype)
      (error "Invalid dtype" dtype))
    
    ;; Verify data size matches shape
    (let ((expected-size (shape-size shape-vec))
          (actual-size (cond
                         ((f32vector? data) (f32vector-length data))
                         ((f64vector? data) (f64vector-length data))
                         ((s32vector? data) (s32vector-length data))
                         ((s64vector? data) (s64vector-length data))
                         ((u32vector? data) (u32vector-length data))
                         ((u64vector? data) (u64vector-length data))
                         (else (error "Invalid data vector" data)))))
      
      (unless (= expected-size actual-size)
        (error "Data size mismatch" 
               `((expected ,expected-size) (actual ,actual-size)))))
    
    (concrete-array data 
                   shape-vec
                   (compute-strides shape-vec)
                   0  ; offset
                   dtype
                   allocation-id
                   batch-axis)))

(define (morph-from-list lst shape dtype #!key (batch-axis -1))
  "Create morphism from nested list
   
   Args:
     lst: nested list of numbers
     shape: desired shape
     dtype: element type
     batch-axis: batch dimension (default: -1)
   
   Returns:
     Concrete array morphism"
  
  (let* ((shape-vec (list->shape shape))
         (size (shape-size shape-vec))
         (data (allocate-typed-vector dtype size)))
    
    (validate-shape shape-vec)
    
    ;; Flatten nested list and fill typed vector
    (let ((flat-lst (flatten-nested-list lst)))
      (unless (= (length flat-lst) size)
        (error "List size doesn't match shape"
               `((list-size ,(length flat-lst)) (shape-size ,size))))
      
      (do ((i 0 (+ i 1))
           (vals flat-lst (cdr vals)))
          ((= i size))
        (typed-vector-set! data dtype i (car vals))))
    
    (make-morphism data shape-vec dtype 
                   batch-axis: batch-axis)))

(define (flatten-nested-list lst)
  "Recursively flatten nested list"
  (cond
    ((null? lst) '())
    ((not (pair? lst)) (list lst))
    (else
     (append (flatten-nested-list (car lst))
             (flatten-nested-list (cdr lst))))))

;;;; ============================================================
;;;; Morphism Information
;;;; ============================================================

(define (morph-shape m)
  "Get morphism shape (alias for get-morphism-shape)"
  (get-morphism-shape m))

(define (morph-dtype m)
  "Get morphism dtype (alias for get-morphism-dtype)"
  (get-morphism-dtype m))

(define (morph-size m)
  "Get total number of elements in morphism"
  (shape-size (get-morphism-shape m)))

(define (morph-rank m)
  "Get number of dimensions in morphism"
  (shape-rank (get-morphism-shape m)))

(define (batched? m)
  "Check if morphism is batched"
  (not (= (get-morphism-batch-axis m) -1)))

(define (batch-size m)
  "Get batch size (error if not batched)"
  (let ((batch-axis (get-morphism-batch-axis m)))
    (when (= batch-axis -1)
      (error "Morphism is not batched" m))
    (shape-dim (get-morphism-shape m) batch-axis)))

;;;; ============================================================
;;;; Conversion Utilities
;;;; ============================================================

(define (morph->list m)
  "Convert concrete morphism to nested list"
  (unless (concrete-array? m)
    (error "Can only convert concrete morphism to list" m))
  
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (let* ((dims (shape->list shape))
             (size (shape-size shape))
             (flat-list
              (let loop ((i 0) (result '()))
                (if (= i size)
                    (reverse result)
                    ;; Convert logical linear index -> multi-index -> physical index
                    ;; This correctly handles views with non-standard strides/offset.
                    (let* ((multi-idx (linear-to-multi-index i shape))
                           (physical  (multi-to-linear-index multi-idx strides offset))
                           (val       (typed-vector-ref data dtype physical)))
                      (loop (+ i 1) (cons val result)))))))
        (nest-list flat-list dims)))
    (else (error "Not a concrete array" m))))
  

(define (nest-list flat-lst dims)
  "Reshape flat list into nested structure.

   Examples:
     '(42.0) '()      -> 42.0          ; scalar
     '(1 2 3) '(3)    -> '(1 2 3)      ; 1D
     '(1..6)  '(2 3)  -> '((1 2 3) (4 5 6))"
  (cond
    ;; Scalar: no dimensions left, return the single element unwrapped.
    ((null? dims)
     (car flat-lst))
    ;; 1D: return the flat list as-is.
    ((= (length dims) 1)
     flat-lst)
    ;; N-D: partition into outer-size chunks and recurse.
    (else
     (let* ((outer-size  (car dims))
            (inner-dims  (cdr dims))
            (chunk-size  (apply * inner-dims)))
       (let loop ((i         0)
                  (remaining flat-lst)
                  (result    '()))
         (if (= i outer-size)
             (reverse result)
             (loop (+ i 1)
                   (drop remaining chunk-size)
                   (cons (nest-list (take remaining chunk-size) inner-dims)
                         result))))))))

)
