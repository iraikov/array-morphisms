;;; MoA Structural Morphisms
;;;
;;; Zero-copy structural operations using affine index functions.
;;; These operations manipulate array views without copying data.

(module array-morphisms-structural-ops
        
  (;; Structural morphisms
   morph-reshape
   morph-transpose
   morph-slice
   morph-concat
   morph-pad
   morph-squeeze
   morph-unsqueeze
   
   ;; Convolution helpers
   im2col-morph
   col2im-morph
   make-col2im-index-fn
   
   ;; Batch operations
   morph-stack
   morph-unstack
   morph-split
   
   ;; Utilities
   validate-reshape
   infer-reshape-dimension
   normalize-slice-indices
   compute-output-shape-conv
   )
  
  (import scheme chicken.base chicken.module chicken.sort)
  (import (only srfi-1 make-list fold iota every zip drop-right take last drop append-map filter-map filter count))
  (import (only srfi-4 f32vector f64vector s32vector s64vector u32vector u64vector
                       f32vector-length f64vector-length s32vector-length
                       s64vector-length u32vector-length u64vector-length))
  (import datatype matchable)
  

  (import array-morphisms-core)
  (import array-morphisms-index-fn)
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Reshape Morphism
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (morph-reshape m new-shape)
    "Reshape morphism - zero-copy view with new shape
    
    Args:
      m: Input morphism
      new-shape: Target shape (vector or list), may contain -1 for inference
    
    Returns:
      Morphism with new shape (identity index function)
    
    Examples:
      (morph-reshape m #(2 3))      ; Explicit shape
      (morph-reshape m '(2 -1))     ; Infer second dimension
      (morph-reshape m #(-1))       ; Flatten
    
    Constraints:
      - Total size must match (unless -1 used for inference)
      - At most one -1 allowed
      - Result is zero-copy (identity index function)"
    
    (let* ((current-shape (get-morphism-shape m))
           (current-size (shape-size current-shape))
           (target-shape-list (if (vector? new-shape)
                                 (vector->list new-shape)
                                 new-shape))
           
           ;; Count -1's and infer if needed
           (neg-count (count (lambda (x) (= x -1)) target-shape-list))
           
           (inferred-shape
            (cond
              ((= neg-count 0)
               ;; No inference needed
               (list->vector target-shape-list))
              
              ((= neg-count 1)
               ;; Infer single dimension
               (let* ((known-product (fold * 1 (filter (lambda (x) (> x 0)) 
                                                       target-shape-list)))
                      (inferred-dim (quotient current-size known-product)))
                 
                 (unless (= (* known-product inferred-dim) current-size)
                   (error "Cannot infer reshape dimension - sizes incompatible"
                          current-shape target-shape-list current-size))
                 
                 (list->vector
                  (map (lambda (x) (if (= x -1) inferred-dim x))
                       target-shape-list))))
              
              (else
               (error "At most one -1 allowed in reshape" target-shape-list)))))
      
      ;; Validate total size matches
      (validate-reshape current-size inferred-shape)
      
      ;; Adjust batch axis for reshape
      (let ((batch-axis (get-morphism-batch-axis m))
            (new-rank (vector-length inferred-shape)))
        
        (let ((new-batch-axis 
               (cond
                 ;; No batch axis
                 ((< batch-axis 0) -1)
                 
                 ;; Batch dimension preserved at same position
                 ((and (< batch-axis new-rank)
                       (= (vector-ref current-shape batch-axis)
                          (vector-ref inferred-shape batch-axis)))
                  batch-axis)
                 
                 ;; Batch dimension moved/removed - scan for it
                 (else
                  (let ((batch-size (vector-ref current-shape batch-axis)))
                    (let loop ((i 0))
                      (cond
                        ((>= i new-rank) -1) ; Batch dimension absorbed
                        ((= (vector-ref inferred-shape i) batch-size) i)
                        (else (loop (+ i 1))))))))))
          
          ;; Create morphism-expr with identity index function
          (morphism-expr
           (gensym 'morph-)
           'reshape
           (list m)
           (make-reshape-index-fn current-shape inferred-shape)
           inferred-shape
           (get-morphism-dtype m)
           `((original-shape . ,current-shape))
           new-batch-axis)))))
  
  (define (validate-reshape current-size new-shape)
    "Validate reshape is size-preserving"
    (let ((new-size (shape-size new-shape)))
      (unless (= current-size new-size)
        (error "Reshape size mismatch" current-size new-size))))
  
  (define (infer-reshape-dimension shape size)
    "Infer the -1 dimension in reshape
    
    Args:
      shape: Shape with possible -1
      size: Total number of elements
    
    Returns:
      Inferred dimension value"
    
    (let* ((known-dims (filter (lambda (x) (> x 0)) (vector->list shape)))
           (known-product (fold * 1 known-dims)))
      
      (let ((inferred (quotient size known-product)))
        (unless (= (* known-product inferred) size)
          (error "Cannot infer dimension - size mismatch" shape size))
        inferred)))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Transpose Morphism
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (morph-transpose m #!optional permutation)
    "Transpose morphism - permute dimensions via permutation matrix
    
    Args:
      m: Input morphism
      permutation: Dimension permutation (list or vector), optional
                  If omitted, reverses all axes (for 2D: standard transpose)
    
    Returns:
      Morphism with permuted dimensions (zero-copy via permutation matrix)
    
    Examples:
      (morph-transpose m)           ; Reverse all axes
      (morph-transpose m '(1 0))    ; 2D transpose
      (morph-transpose m '(0 2 1))  ; Swap last two axes
      (morph-transpose m #(2 0 1))  ; Rotate axes right
    
    Constraints:
      - Permutation must be valid (each index appears exactly once)
      - Permutation length must match rank"
    
    (let* ((shape (get-morphism-shape m))
           (rank (vector-length shape))
           
           ;; Default permutation: reverse all axes
           (perm (if permutation
                    (if (vector? permutation)
                        (vector->list permutation)
                        permutation)
                    (reverse (iota rank))))
           
           ;; Validate permutation
           (_ (validate-permutation perm rank))
           
           ;; Compute new shape by permuting dimensions
           (new-shape (list->vector
                       (map (lambda (i) (vector-ref shape i)) perm)))
           
           ;; Adjust batch axis
           (batch-axis (get-morphism-batch-axis m))
           (new-batch-axis
            (if (< batch-axis 0)
                -1
                ;; Find where batch-axis moved to in permutation
                (let loop ((i 0))
                  (cond
                    ((>= i rank) -1)
                    ((= (list-ref perm i) batch-axis) i)
                    (else (loop (+ i 1))))))))
      
      ;; Create morphism-expr with transpose index function
      (morphism-expr
       (gensym 'morph-)
       'transpose
       (list m)
       (make-transpose-index-fn perm)
       new-shape
       (get-morphism-dtype m)
       `((permutation . ,perm))
       new-batch-axis)))
  
  (define (validate-permutation perm rank)
    "Validate permutation is valid"
    (unless (= (length perm) rank)
      (error "Permutation length mismatch" perm rank))
    
    (unless (every (lambda (i) (and (>= i 0) (< i rank))) perm)
      (error "Permutation indices out of range" perm rank))
    
    ;; Check each index appears exactly once
    (let ((sorted (sort perm <)))
      (unless (equal? sorted (iota rank))
        (error "Invalid permutation - duplicate or missing indices" perm))))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Slice Morphism
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (morph-slice m start end #!optional (step #f))
    "Slice morphism - extract subarray via affine transformation
    
    Args:
      m: Input morphism
      start: Start indices (list or vector)
      end: End indices (list or vector), exclusive
      step: Step sizes (list/vector/integer), optional (default 1)
    
    Returns:
      Morphism with sliced shape (zero-copy via affine index function)
    
    Examples:
      (morph-slice m '(0 0) '(2 3))           ; Rows 0-1, cols 0-2
      (morph-slice m '(1) '(5) 2)             ; Elements 1,3 (step=2)
      (morph-slice m #(0 1 0) #(2 3 4))       ; 3D slice
      (morph-slice m '(-2) '(-1))             ; Last element (negative indices)
    
    Constraints:
      - start, end, step must have same length as rank
      - 0 <= start < end <= dim (after normalization)
      - step > 0"
    
    (let* ((shape (get-morphism-shape m))
           (rank (vector-length shape))
           
           ;; Normalize to lists
           (start-list (if (vector? start) (vector->list start) start))
           (end-list (if (vector? end) (vector->list end) end))
           (step-list (cond
                        ((not step) (make-list rank 1))
                        ((number? step) (make-list rank step))
                        ((vector? step) (vector->list step))
                        (else step)))
           
           ;; Validate lengths
           (_ (begin
                (unless (= (length start-list) rank)
                  (error "Start length mismatch" start-list rank))
                (unless (= (length end-list) rank)
                  (error "End length mismatch" end-list rank))
                (unless (= (length step-list) rank)
                  (error "Step length mismatch" step-list rank))))

           ;; Normalize negative indices
           (norm-start (normalize-slice-indices start-list shape))
           (norm-end (normalize-slice-indices end-list shape))
           
           ;; Validate ranges
           (_ (for-each
               (lambda (s e dim step-val)
                 (unless (> step-val 0)
                   (error "Step must be positive" step-val))
                 (unless (and (>= s 0) (< s dim))
                   (error "Start index out of range" s dim))
                 (unless (and (> e s) (<= e dim))
                   (error "End index out of range or not > start" e dim s)))
               norm-start norm-end (vector->list shape) step-list))
           
           ;; Compute output shape
           (output-shape
            (list->vector
             (map (lambda (s e step-val)
                    (quotient (+ (- e s) step-val -1) step-val))
                  norm-start norm-end step-list)))
           
           ;; Batch axis handling
           (batch-axis (get-morphism-batch-axis m))
           (new-batch-axis
            (if (or (< batch-axis 0) (>= batch-axis rank))
                -1
                ;; Check if batch dimension is preserved
                (if (and (= (list-ref norm-start batch-axis) 0)
                        (= (list-ref norm-end batch-axis)
                           (vector-ref shape batch-axis))
                        (= (list-ref step-list batch-axis) 1))
                    batch-axis
                    -1)))) ;; Batch dimension sliced - no longer batched
      
      ;; Create morphism-expr with slice index function
      (morphism-expr
       (gensym 'morph-)
       'slice
       (list m)
       (make-slice-index-fn norm-start norm-end step-list)
       output-shape
       (get-morphism-dtype m)
       `((start . ,norm-start)
         (end . ,norm-end)
         (step . ,step-list))
       new-batch-axis)))
  
  (define (normalize-slice-indices indices shape)
    "Normalize negative indices relative to shape"
    (map (lambda (idx dim)
           (if (< idx 0)
               (+ dim idx)
               idx))
         indices
         (vector->list shape)))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Helper Morphisms
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (morph-squeeze m #!optional axes)
    "Remove dimensions of size 1
    
    Args:
      m: Input morphism
      axes: Axes to squeeze (list), or #f to squeeze all size-1 dims
    
    Returns:
      Morphism with size-1 dimensions removed"
    
    (let* ((shape (get-morphism-shape m))
           (rank (vector-length shape))
           (axes-to-remove
            (if axes
                ;; Normalize and validate specified axes
                (map (lambda (ax) (normalize-axis ax rank)) axes)
                ;; Find all size-1 dimensions
                (filter-map (lambda (i)
                             (if (= (vector-ref shape i) 1) i #f))
                           (iota rank)))))
      
      ;; Validate axes are size 1
      (for-each
       (lambda (ax)
         (unless (= (vector-ref shape ax) 1)
           (error "Cannot squeeze non-singular dimension" ax 
                  (vector-ref shape ax))))
       axes-to-remove)
      
      ;; Compute new shape
      (let ((new-shape
             (list->vector
              (filter-map (lambda (i)
                           (if (member i axes-to-remove) #f (vector-ref shape i)))
                         (iota rank)))))
        
        (morph-reshape m new-shape))))
  
  (define (morph-unsqueeze m axis)
    "Add dimension of size 1 at specified axis
    
    Args:
      m: Input morphism  
      axis: Position to insert new dimension (can be negative)
    
    Returns:
      Morphism with added dimension"
    
    (let* ((shape (get-morphism-shape m))
           (rank (vector-length shape))
           (norm-axis (if (< axis 0) (+ rank 1 axis) axis)))
      
      (unless (and (>= norm-axis 0) (<= norm-axis rank))
        (error "Unsqueeze axis out of range" axis rank))
      
      (let* ((shape-list (vector->list shape))
             (new-shape (append (take shape-list norm-axis)
                               '(1)
                               (drop shape-list norm-axis))))
        (morph-reshape m (list->vector new-shape)))))
  
  (define (morph-pad m padding #!optional (mode 'constant) (value 0.0))
    "Pad morphism with values
    
    Args:
      m: Input morphism
      padding: Padding specification - list of (before, after) pairs per dimension
      mode: 'constant, 'edge, or 'reflect
      value: Constant value for constant mode
    
    Returns:
      Morphism with padding applied
    
    Note: This creates a morphism-expr, actual implementation deferred to Phase 5"
    
    (let* ((shape (get-morphism-shape m))
           (rank (vector-length shape))
           (_ (unless (= (length padding) rank)
                (error "Padding length must match rank" padding rank)))
           
           (new-shape
            (list->vector
             (map (lambda (dim pad-pair)
                    (+ dim (car pad-pair) (cadr pad-pair)))
                  (vector->list shape)
                  padding))))
      
      (morphism-expr
       (gensym 'morph-)
       'pad
       (list m)
       (make-pad-index-fn padding mode value shape)
       new-shape
       (get-morphism-dtype m)
       `((padding . ,padding)
         (mode . ,mode)
         (value . ,value))
       (get-morphism-batch-axis m))))
  
  (define (make-pad-index-fn padding mode value orig-shape)
    "Create index function for padding operation
    
    Returns function: output-idx -> (input-idx | 'pad-value)"

    (lambda (out-idx)
      (case mode
        ;; Constant padding
        ((constant)
         (let* ((in-idx-and-valid
                 (map (lambda (out-i pad-pair dim)
                        (let* ((before (car pad-pair))
                               (in-i (- out-i before)))
                          (cons in-i (and (>= in-i 0) (< in-i dim)))))
                      out-idx padding (vector->list orig-shape))))
           
           (if (every cdr in-idx-and-valid)
               ;; All indices in bounds - return input indices
               (map car in-idx-and-valid)
               ;; Out of bounds - return constant padding marker
               `(constant ,value))))
        
        ;; Edge padding (clamp to boundaries)
        ((edge)
         (map (lambda (out-i pad-pair dim)
                (let* ((before (car pad-pair))
                       (in-i (- out-i before)))
                  ;; Clamp to [0, dim-1]
                  (cond
                   ((< in-i 0) 0)
                   ((>= in-i dim) (- dim 1))
                   (else in-i))))
              out-idx padding (vector->list orig-shape)))
        
        ;; Reflect padding (mirror at boundaries)
        ((reflect)
         (map (lambda (out-i pad-pair dim)
                (let* ((before (car pad-pair))
                       (in-i (- out-i before)))
                  ;; Reflect formula:
                  ;; - If in-i < 0: reflect as (-in-i - 1)
                  ;; - If in-i >= dim: reflect as (2*dim - in-i - 1)
                  ;; - Repeat reflection if still out of bounds
                  (let loop ((idx in-i))
                    (cond
                     ;; In bounds
                     ((and (>= idx 0) (< idx dim)) idx)
                     
                     ;; Below lower bound - reflect up
                     ((< idx 0)
                      (loop (- -1 idx)))
                     
                     ;; Above upper bound - reflect down
                     (else
                      (loop (- (* 2 dim) idx 1)))))))
              out-idx padding (vector->list orig-shape)))
        
        (else
         (error "Unknown padding mode" mode)))))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; im2col and col2im Morphisms
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (im2col-morph input kernel-size #!optional (stride 1) (padding 0))
    "Image-to-column morphism for convolution
    
    Transforms image into column matrix where each column is a flattened
    receptive field (kernel window).
    
    Args:
      input: Input morphism with shape (C,H,W) or (N,C,H,W)
      kernel-size: (KH, KW) tuple
      stride: Stride value(s) - integer or (SH, SW) tuple
      padding: Padding value(s) - integer or (PH, PW) tuple
    
    Returns:
      Morphism with shape (C*KH*KW, OH*OW) or (N, C*KH*KW, OH*OW)
      where OH, OW are output spatial dimensions
    
    Zero-padded positions return special 'pad-zero marker."
    
    (let* ((shape (get-morphism-shape input))
           (rank (vector-length shape))
           (batched? (= rank 4))
           
           ;; Parse input shape
           (N (if batched? (vector-ref shape 0) 1))
           (C (vector-ref shape (if batched? 1 0)))
           (H (vector-ref shape (if batched? 2 1)))
           (W (vector-ref shape (if batched? 3 2)))
           
           ;; Parse kernel size
           (KH (if (pair? kernel-size) (car kernel-size) kernel-size))
           (KW (if (pair? kernel-size) 
                   (if (null? (cdr kernel-size)) (car kernel-size) (cadr kernel-size))
                   kernel-size))
           
           ;; Parse stride
           (SH (if (pair? stride) (car stride) stride))
           (SW (if (pair? stride)
                   (if (null? (cdr stride)) (car stride) (cadr stride))
                   stride))
           
           ;; Parse padding
           (PH (if (pair? padding) (car padding) padding))
           (PW (if (pair? padding)
                   (if (null? (cdr padding)) (car padding) (cadr padding))
                   padding))
           
           ;; Compute output spatial dimensions
           (OH (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH)))
           (OW (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW)))
           
           ;; Output shape
           (output-shape (if batched?
                            (vector N (* C KH KW) (* OH OW))
                            (vector (* C KH KW) (* OH OW)))))
      
      (morphism-expr
       (gensym 'morph-)
       'im2col
       (list input)
       (make-im2col-index-fn C H W KH KW SH SW PH PW OH OW batched?)
       output-shape
       (get-morphism-dtype input)
       `((kernel-size . (,KH ,KW))
         (stride . (,SH ,SW))
         (padding . (,PH ,PW))
         (spatial-output . (,OH ,OW)))
       (if batched? 0 -1))))
  
  (define (make-im2col-index-fn C H W KH KW SH SW PH PW OH OW batched?)
    "Create im2col index function
    
    Maps column indices to image indices with padding handling"
    
    (if batched?
        ;; Batched: (n, col_row, col_col) -> (n, c, h, w) or 'pad-zero
        (lambda (indices)
          (let* ((n (car indices))
                 (col-row (cadr indices))
                 (col-col (caddr indices))
                 
                 ;; Decompose column indices
                 (c (quotient col-row (* KH KW)))
                 (kh (modulo (quotient col-row KW) KH))
                 (kw (modulo col-row KW))
                 (oh (quotient col-col OW))
                 (ow (modulo col-col OW))
                 
                 ;; Map to input coordinates
                 (in-h (+ (* oh SH) kh (- PH)))
                 (in-w (+ (* ow SW) kw (- PW))))
            
            ;; Bounds check (padding)
            (if (and (>= in-h 0) (< in-h H)
                     (>= in-w 0) (< in-w W))
                (list n c in-h in-w)
                'pad-zero)))
        
        ;; Non-batched: (col_row, col_col) -> (c, h, w) or 'pad-zero
        (lambda (indices)
          (let* ((col-row (car indices))
                 (col-col (cadr indices))
                 
                 (c (quotient col-row (* KH KW)))
                 (kh (modulo (quotient col-row KW) KH))
                 (kw (modulo col-row KW))
                 (oh (quotient col-col OW))
                 (ow (modulo col-col OW))
                 
                 (in-h (+ (* oh SH) kh (- PH)))
                 (in-w (+ (* ow SW) kw (- PW))))
            
            (if (and (>= in-h 0) (< in-h H)
                     (>= in-w 0) (< in-w W))
                (list c in-h in-w)
                'pad-zero)))))
  
  (define (col2im-morph col output-shape kernel-size #!optional (stride 1) (padding 0))
    "Column-to-image morphism (adjoint of im2col)
    
    Inverse operation of im2col - scatter columns back to image.
    For overlapping windows, values are accumulated (summed).
    
    Args:
      col: Column morphism with shape (C*KH*KW, OH*OW) or (N, C*KH*KW, OH*OW)
      output-shape: Target image shape (C,H,W) or (N,C,H,W)
      kernel-size: (KH, KW) tuple
      stride: Stride value(s)
      padding: Padding value(s)
    
    Returns:
      Morphism with specified output shape
    
    Note: Actual accumulation deferred to Phase 5 realization"
    
    (let* ((col-shape (get-morphism-shape col))
           (col-rank (vector-length col-shape))
           (col-batched? (= col-rank 3))

           (target-shape (if (vector? output-shape) output-shape 
                            (list->vector output-shape)))
           (target-rank (vector-length target-shape))
           (target-batched? (= target-rank 4))
           
           (valid-dims? (or (and (= col-rank 2) (= target-rank 3))
                            (and (= col-rank 3) (= target-rank 4))))
           (_ (unless valid-dims?
                (error "Batch mismatch between col and output shape"
                       col-shape target-shape)))
           
           ;; Parse parameters
           (KH (if (pair? kernel-size) (car kernel-size) kernel-size))
           (KW (if (pair? kernel-size)
                   (if (null? (cdr kernel-size)) (car kernel-size) (cadr kernel-size))
                   kernel-size))
           (SH (if (pair? stride) (car stride) stride))
           (SW (if (pair? stride)
                   (if (null? (cdr stride)) (car stride) (cadr stride))
                   stride))
           (PH (if (pair? padding) (car padding) padding))
           (PW (if (pair? padding)
                   (if (null? (cdr padding)) (car padding) (cadr padding))
                   padding)))
      
      (morphism-expr
       (gensym 'morph-)
       'col2im
       (list col)
       (make-col2im-index-fn kernel-size stride padding col-batched?)
       target-shape
       (get-morphism-dtype col)
       `((kernel-size . (,KH ,KW))
         (stride . (,SH ,SW))
         (padding . (,PH ,PW)))
       (if target-batched? 0 -1))))
  

  (define (make-col2im-index-fn kernel-size stride padding batched?)
    "Create col2im index function with accumulation metadata
  
    Returns a special record type that the realization engine
    can detect and route to specialized accumulation kernel
  
    Args:
      kernel-size: (KH, KW) or integer
      stride: (SH, SW) or integer  
      padding: (PH, PW) or integer
      batched?: Boolean indicating if input is batched
    
    Returns:
      col2im-index-fn record with all parameters"
  
    ;; Parse kernel size
    (let ((KH (if (pair? kernel-size) (car kernel-size) kernel-size))
          (KW (if (pair? kernel-size)
                  (if (null? (cdr kernel-size)) (car kernel-size) (cadr kernel-size))
                  kernel-size))
          
          ;; Parse stride
          (SH (if (pair? stride) (car stride) stride))
          (SW (if (pair? stride)
                  (if (null? (cdr stride)) (car stride) (cadr stride))
                  stride))
          
          ;; Parse padding
          (PH (if (pair? padding) (car padding) padding))
          (PW (if (pair? padding)
                  (if (null? (cdr padding)) (car padding) (cadr padding))
                  padding)))
      
      ;; Return col2im index function record
      (make-col2im-index-fn-record KH KW SH SW PH PW batched?)))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Batch Stack/Split Operations
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (morph-stack morphisms #!optional (axis 0))
    "Stack morphisms along new dimension
    
    Args:
      morphisms: List of morphisms with identical shapes
      axis: Axis to stack along (new dimension inserted here)
    
    Returns:
      Morphism with stacked shape
    
    Example:
      (morph-stack (list m1 m2 m3) 0)  ; Stack 3 (2,3) -> (3,2,3)"
    
    (when (null? morphisms)
      (error "morph-stack requires at least one morphism"))
    
    (let* ((first-shape (get-morphism-shape (car morphisms)))
           (rank (vector-length first-shape))
           (n (length morphisms))
           
           ;; Validate all shapes match
           (_ (for-each
               (lambda (m)
                 (unless (equal? (get-morphism-shape m) first-shape)
                   (error "All morphisms must have same shape for stack"
                          (map get-morphism-shape morphisms))))
               morphisms))
           
           ;; Normalize axis
           (norm-axis (if (< axis 0) (+ rank 1 axis) axis))
           
           (_ (unless (and (>= norm-axis 0) (<= norm-axis rank))
                (error "Stack axis out of range" axis rank)))
           
           ;; Compute output shape
           (shape-list (vector->list first-shape))
           (output-shape (list->vector
                         (append (take shape-list norm-axis)
                                (list n)
                                (drop shape-list norm-axis))))
           
           ;; Common dtype (promote if needed)
           (dtypes (map get-morphism-dtype morphisms))
           (result-dtype (fold promote-types (car dtypes) (cdr dtypes))))
      
      (morphism-expr
       (gensym 'morph-)
       'stack
       morphisms
       (%make-stack-index-fn norm-axis (length morphisms))
       output-shape
       result-dtype
       `((axis . ,norm-axis)
         (count . ,(length morphisms)))
       norm-axis))) ;; Stacked dimension is the batch axis
  
  (define (%make-stack-index-fn axis count)
      (make-stack-index-fn axis))
  
  (define (morph-unstack m #!optional (axis 0))
    "Unstack morphism along axis into list of morphisms
    
    Args:
      m: Input morphism
      axis: Axis to unstack along
    
    Returns:
      List of morphisms (one per element along axis)
    
    Example:
      (morph-unstack m 0)  ; (3,2,3) -> list of 3 (2,3) morphisms"
    
    (let* ((shape (get-morphism-shape m))
           (rank (vector-length shape))
           (norm-axis (normalize-axis axis rank))
           (n (vector-ref shape norm-axis)))
      
      (map (lambda (i)
             (morph-squeeze
              (morph-slice m
                           (map (lambda (ax) (if (= ax norm-axis) i 0))
                                (iota rank))
                           (map (lambda (ax) (if (= ax norm-axis) (+ i 1)
                                                 (vector-ref shape ax)))
                                (iota rank)))
              (list norm-axis)))
           (iota n))))
  
  (define (morph-split m sizes-or-sections #!optional (axis 0))
    "Split morphism into multiple morphisms along axis
    
    Args:
      m: Input morphism
      sizes-or-sections: Either integer (equal splits) or list of sizes
      axis: Axis to split along
    
    Returns:
      List of morphisms"
    
    (let* ((shape (get-morphism-shape m))
           (rank (vector-length shape))
           (norm-axis (normalize-axis axis rank))
           (dim-size (vector-ref shape norm-axis)))
      
      (let ((sections
             (if (integer? sizes-or-sections)
                 ;; Equal sections
                 (let ((section-size (quotient dim-size sizes-or-sections)))
                   (unless (= (* section-size sizes-or-sections) dim-size)
                     (error "Dimension not evenly divisible" dim-size sizes-or-sections))
                   (make-list sizes-or-sections section-size))
                 ;; Custom sizes
                 sizes-or-sections)))
        
        ;; Validate total size
        (unless (= (fold + 0 sections) dim-size)
          (error "Split sizes don't sum to dimension size" sections dim-size))
        
        ;; Generate slices
        (let loop ((offset 0)
                  (remaining sections)
                  (result '()))
          (if (null? remaining)
              (reverse result)
              (let* ((size (car remaining))
                     (start (map (lambda (ax) (if (= ax norm-axis) offset 0))
                                 (iota rank)))
                     (end (map (lambda (ax) (if (= ax norm-axis) (+ offset size)
                                                (vector-ref shape ax)))
                               (iota rank)))
                     (slice (morph-slice m start end)))
                (loop (+ offset size) (cdr remaining) (cons slice result))))))))
  
  (define (morph-concat morphisms #!optional (axis 0))
    "Concatenate morphisms along existing axis
    
    Args:
      morphisms: List of morphisms
      axis: Axis to concatenate along
    
    Returns:
      Morphism with concatenated shape
    
    Note: Actual implementation deferred to Phase 5"
    
    (when (null? morphisms)
      (error "morph-concat requires at least one morphism"))
    
    (let* ((first-shape (get-morphism-shape (car morphisms)))
           (rank (vector-length first-shape))
           (norm-axis (normalize-axis axis rank))
           
           ;; Validate shapes match except along concat axis
           (_ (for-each
               (lambda (m)
                 (let ((s (get-morphism-shape m)))
                   (do ((i 0 (+ i 1)))
                       ((= i rank))
                     (when (and (not (= i norm-axis))
                               (not (= (vector-ref s i) (vector-ref first-shape i))))
                       (error "Shape mismatch in concat" first-shape s)))))
               morphisms))
           
           ;; Compute concatenated size along axis
           (concat-size (fold + 0 (map (lambda (m)
                                        (vector-ref (get-morphism-shape m) norm-axis))
                                      morphisms)))
           
           ;; Output shape
           (output-shape (make-vector (vector-length first-shape)))
           (_ (vector-copy! first-shape output-shape))
           (_ (vector-set! output-shape norm-axis concat-size))
           
           ;; Common dtype
           (dtypes (map get-morphism-dtype morphisms))
           (result-dtype (fold promote-types (car dtypes) (cdr dtypes))))
      
      (morphism-expr
       (gensym 'morph-)
       'concat
       morphisms
       (%make-concat-index-fn norm-axis (map get-morphism-shape morphisms))
       output-shape
       result-dtype
       `((axis . ,norm-axis))
       (get-morphism-batch-axis (car morphisms)))))

  (define (%make-concat-index-fn axis shapes)
    "Create concat index function"
    ;; Pre-compute cumulative start offsets once at construction time.
    ;; Example: axis sizes (2 3 4) -> offsets = (0 2 5)
    (let* ((sizes   (map (lambda (s) (vector-ref s axis)) shapes))
           (offsets (let loop ((remaining sizes) (acc '(0)))
                      (if (null? remaining)
                          ;; Drop the trailing sentinel (total size) and reverse.
                          (reverse (cdr acc))
                          (loop (cdr remaining)
                                (cons (+ (car acc) (car remaining))
                                      acc))))))
      (make-concat-index-fn axis offsets)))
  
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Utility Functions
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (compute-output-shape-conv input-shape kernel-size stride padding)
    "Compute output shape for convolution
    
    Args:
      input-shape: (C,H,W) or (N,C,H,W)
      kernel-size: (KH, KW) or integer
      stride: (SH, SW) or integer
      padding: (PH, PW) or integer
    
    Returns:
      Output spatial dimensions (OH, OW)"
    
    (let* ((batched? (= (vector-length input-shape) 4))
           (H (vector-ref input-shape (if batched? 2 1)))
           (W (vector-ref input-shape (if batched? 3 2)))
           
           (KH (if (pair? kernel-size) (car kernel-size) kernel-size))
           (KW (if (pair? kernel-size)
                   (if (null? (cdr kernel-size)) (car kernel-size) (cadr kernel-size))
                   kernel-size))
           (SH (if (pair? stride) (car stride) stride))
           (SW (if (pair? stride)
                   (if (null? (cdr stride)) (car stride) (cadr stride))
                   stride))
           (PH (if (pair? padding) (car padding) padding))
           (PW (if (pair? padding)
                   (if (null? (cdr padding)) (car padding) (cadr padding))
                   padding))
           
           (OH (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH)))
           (OW (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW))))
      
      (list OH OW)))
  
) ;; end module
