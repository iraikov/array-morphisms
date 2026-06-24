;;; Index Function Algebra
;;; Composition, optimization, and simplification of index functions


(module array-morphisms-index-fn
        
        (;; Matrix operations
         index-fn-make-identity-matrix
         index-fn-matrix-multiply
         index-fn-vector-add
         index-fn-vector-scale
         index-fn-vector-zero?
         index-fn-matrix-identity?
         
         ;; Identity and constant functions
         make-identity-index-fn
         identity-index-fn?
         make-constant-index-fn
         constant-index-fn?
         
         ;; Composition
         compose-affine-index-fns
         compose-index-fns
         
         ;; Application
         apply-index-fn
         apply-affine-index-fn
         apply-compute-index-fn
         apply-stack-index-fn
         apply-concat-index-fn
         
         ;; Simplification
         simplify-index-fn
         
         ;; Permutations
         identity-permutation?
         compose-permutations
         invert-permutation
         permutation-to-matrix
         
         ;; Specialized constructors
         make-reshape-index-fn
         make-transpose-index-fn
         make-slice-index-fn
         
         ;; Information
         index-fn-invertible?
         index-fn-rank

         ;; MoA psi-composition for element-wise combiners
         compose-flat-combiners
         )

        (import scheme (chicken base)
                srfi-4 datatype matchable
                srfi-1 srfi-69 array-morphisms-core)

;;;; ============================================================
;;;; Matrix Operations
;;;; ============================================================

(define (index-fn-make-identity-matrix n)
  "Create n x n identity matrix as nested list"
  (let loop ((i 0) (result '()))
    (if (= i n) (reverse result)
        (loop (+ i 1)
              (cons (let inner ((j 0) (row '()))
                      (if (= j n) (reverse row)
                          (inner (+ j 1)
                                 (cons (if (= i j) 1 0) row))))
                    result)
              ))
    ))


(define (index-fn-matrix-multiply A B)
  "Multiply two matrices or matrix × vector
   
   A: m×n matrix (list of m rows) or #f (identity)
   B: n×p matrix (list of n rows), vector of length n, or #f (identity)
   Returns: m×p matrix or vector of length m"
  
  (cond
    ;; Special case: identity matrices
    ((not A) B)
    ((not B) A)
    
    ;; B is a vector - convert to column matrix, multiply, extract column
    ((and (pair? B) (not (pair? (car B))))
     (let* ((col-matrix (map list B))  ; [a b c] -> [[a] [b] [c]]
            (result-matrix (index-fn-matrix-multiply A col-matrix)))
       (map car result-matrix)))  ; Extract first (only) column
    
    (else
     (let* ((m (length A))
            (n (length (car A)))
            (p (length (car B))))
       
       ;; Validate dimensions
       (unless (= n (length B))
         (error "Matrix dimension mismatch for multiplication"
                `((A-shape ,(length A) ,n) (B-shape ,(length B) ,p))))
       
       ;; Compute C = A * B
       (define (dot-product v1 v2) (fold + 0 (map * v1 v2)))
       (map (lambda (arow)
              (apply map
                     (lambda bcol (dot-product arow bcol))
                     B))
            A)))))

(define (index-fn-vector-add v1 v2)
  "Add two vectors (represented as lists)"
  (cond
    ((not v1) v2)
    ((not v2) v1)
    (else
     (unless (= (length v1) (length v2))
       (error "Vector length mismatch for addition"
              `((v1-length ,(length v1)) (v2-length ,(length v2)))))
     (map + v1 v2))))

(define (index-fn-vector-scale c v)
  "Multiply vector v by scalar c"
  (if (not v)
      #f (map (lambda (x) (* c x)) v)))

(define (index-fn-vector-zero? v)
  "Check if vector is all zeros"
  (or (not v) (every zero? v)))

(define (index-fn-matrix-identity? A)
  "Check if matrix is identity"
  (or (not A)
      (let ((n (length A)))
        (and (= n (length (car A)))
             (every (lambda (i)
                      (every (lambda (j)
                               (if (= i j)
                                   (= (list-ref (list-ref A i) j) 1)
                                   (= (list-ref (list-ref A i) j) 0)))
                             (iota n)))
                    (iota n))))))

;;;; ============================================================
;;;; Identity and Constant Index Functions
;;;; ============================================================

(define (make-identity-index-fn rank)
  "Create identity index function for given rank
   
   Identity: f(i) = i (no transformation)"
  (identity-fn))

(define (identity-index-fn? fn)
  "Check if index function is identity"
  (and (affine-index-fn? fn)
       (cases affine-index-fn fn
         (identity-fn () #t)
         (else         #f))))

(define (make-constant-index-fn value)
  "Create constant index function
   
   Constant: f(i) = value (ignores input)"
  (lambda (indices) value))

(define (constant-index-fn? fn)
  "Check if index function always returns same value"
  ;; Simple heuristic: not a record type we recognize
  (and (procedure? fn)
       (not (affine-index-fn? fn))
       (not (compute-index-fn? fn))
       (not (composed-index-fn? fn))
       (not (window-index-fn? fn))
       (not (reduction-index-fn? fn))
       (not (batch-map-index-fn? fn))
       (not (batch-reduce-index-fn? fn))))

;;;; ============================================================
;;;; Affine Index Function Composition
;;;; ============================================================

(define (compose-affine-index-fns f g)
  "Compose two affine index functions
   
   f: Af·i + bf
   g: Ag·i + bg
   
   (f ∘ g)(i) = f(g(i)) = Af·(Ag·i + bg) + bf
                        = (Af·Ag)·i + (Af·bg + bf)
   
   Returns: Composed affine index function"
  
  (cases affine-index-fn f

    ;; identity o g  ->  g
    (identity-fn ()
      g)

    ;; permutation o g
    (permutation-fn (pf)
      (cases affine-index-fn g
        (identity-fn ()
          f)
        (permutation-fn (pg)
          ;; perm o perm: O(n) composition
          (permutation-fn (compose-permutations pf pg)))
        (diagonal-fn (dg bg)
          ;; perm o diag: permute the diagonal entries and biases
          (diagonal-fn (map (lambda (k) (list-ref dg k)) pf)
                       (map (lambda (k) (list-ref bg k)) pf)))
        (general-fn (Ag bg)
          ;; perm o general: permute rows of Ag and entries of bg
          (general-fn (index-fn-matrix-multiply (permutation-to-matrix pf) Ag)
                      (if bg (map (lambda (k) (list-ref bg k)) pf) #f)))))

    ;; diagonal o g
    (diagonal-fn (df bf)
      (cases affine-index-fn g
        (identity-fn ()
          f)
        (permutation-fn (pg)
          ;; diag o perm: permute diagonal entries and biases
          (diagonal-fn (map (lambda (k) (list-ref df k)) pg)
                       (map (lambda (k) (list-ref bf k)) pg)))
        (diagonal-fn (dg bg)
          ;; diag o diag: element-wise scale composition
          ;; (df * (dg * i + bg) + bf)[k] = (df[k]*dg[k])*i[k] + (df[k]*bg[k]+bf[k])
          (diagonal-fn (map * df dg)
                       (map (lambda (dfi bfi bgi) (+ (* dfi bgi) bfi)) df bf bg)))
        (general-fn (Ag bg)
          ;; diag o general: scale rows of Ag
          (let ((Df (diagonal-to-matrix df)))
            (general-fn (index-fn-matrix-multiply Df Ag)
                        (if bg
                            (index-fn-vector-add
                             (index-fn-matrix-multiply Df bg)
                             bf)
                            bf))))))

    (general-fn (Af bf)
      (let* ((Ag     (affine-fn->matrix g))
             (bg     (affine-fn->bias   g))
             (A-comp (index-fn-matrix-multiply Af Ag))
             (b-comp (index-fn-vector-add
                      (if (and Af bg)
                          (index-fn-matrix-multiply Af bg)
                          bg)
                      bf)))
        (general-fn A-comp b-comp)))))


;;;; ============================================================
;;;; General Index Function Composition
;;;; ============================================================

(define (compose-index-fns f g)
  "Compose index functions: (f ∘ g)(i) = f(g(i))
   
   Applies optimizations for common patterns:
   - Affine ∘ Affine -> Affine (matrix multiplication)
   - Identity elimination
   - Constant propagation"
  
  (cond
    ;; Identity elimination
    ((identity-index-fn? f) g)
    ((identity-index-fn? g) f)
    
    ;; Affine ∘ Affine -> optimized Affine
    ((and (affine-index-fn? f) (affine-index-fn? g))
     (compose-affine-index-fns f g))
    
    ;; Affine ∘ Identity -> Affine
    ((and (affine-index-fn? f) (identity-index-fn? g))
     f)
    
    ;; Identity ∘ Affine -> Affine
    ((and (identity-index-fn? f) (affine-index-fn? g))
     g)
    
    ;; General composition (no optimization)
    (else
     (make-composed-index-fn f g))))


;;;; ============================================================
;;;; MoA psi-composition for element-wise combiners
;;;;
;;;; Implements Psi(f, Psi(g, A)) = Psi(f o g, A) at the combiner level.
;;;; Used by the SSA element-wise fusion pass.
;;;; ============================================================

(define (compose-flat-combiners p-combiner p-nargs c-combiner c-extra-nargs)
  "Compose two element-wise combiners: c-combiner(p-combiner(p-args...), c-rest...).
   p-nargs:       number of inputs the producer combiner takes.
   c-extra-nargs: number of consumer inputs beyond the producer's output.
   Uses fixed-arity lambdas to avoid rest-list allocation on every element call."
  (cond
    ((and (= p-nargs 1) (= c-extra-nargs 0))
     (lambda (x) (c-combiner (p-combiner x))))
    ((and (= p-nargs 2) (= c-extra-nargs 0))
     (lambda (x y) (c-combiner (p-combiner x y))))
    ((and (= p-nargs 1) (= c-extra-nargs 1))
     (lambda (x y) (c-combiner (p-combiner x) y)))
    ((and (= p-nargs 2) (= c-extra-nargs 1))
     (lambda (x y z) (c-combiner (p-combiner x y) z)))
    (else (error "compose-flat-combiners: unsupported arity"
                 `(p-nargs ,p-nargs) `(c-extra-nargs ,c-extra-nargs)))))


(define (apply-index-fn fn indices)
  "Apply index function to indices.

  Single-source variants return a source index (list).
  Routing variants return (source-id . source-idx)."

  (cond
    ;; Affine transformation
    ((affine-index-fn? fn)
     (apply-affine-index-fn fn indices))

    ;; Composed function: (f o g)(i) = f(g(i))
    ((composed-index-fn? fn)
     (let ((outer (composed-index-fn-outer fn))
           (inner (composed-index-fn-inner fn)))
       (apply-index-fn outer (apply-index-fn inner indices))))

    ;; Computational function
    ((compute-index-fn? fn)
     (apply-compute-index-fn fn indices))

    ;; Routing: stack -- returns (source-id . source-idx)
    ((stack-index-fn? fn)
     (apply-stack-index-fn fn indices))

    ;; Routing: concat -- returns (source-id . source-idx)
    ((concat-index-fn? fn)
     (apply-concat-index-fn fn indices))

    ;; Procedure fallback (padding lambdas, etc.)
    ((procedure? fn)
     (fn indices))

    (else
     (error "Unknown index function type" fn))))


(define (apply-affine-index-fn fn indices)
  "Apply affine index function to a multi-index (list or vector)."
  (let ((idx (if (vector? indices) (vector->list indices) indices)))
    (cases affine-index-fn fn
      (identity-fn ()
        idx)
      (permutation-fn (p)
        (map (lambda (k) (list-ref idx k)) p))
      (diagonal-fn (d b)
        (map (lambda (di bi i) (+ (* di i) bi)) d b idx))
      (general-fn (A b)
        (let ((Ai (index-fn-matrix-multiply A idx)))
          (if b (index-fn-vector-add Ai b) Ai))))))

(define (apply-compute-index-fn fn indices)
  "Apply computational index function"
  
  (let ((input-fns (compute-index-fn-input-fns fn))
        (combiner (compute-index-fn-combiner fn)))
    
    ;; Apply each input function and combine
    (let ((input-values (map (lambda (f) (apply-index-fn f indices))
                            input-fns)))
      (apply combiner input-values))))

(define (apply-stack-index-fn fn indices)
  "Apply stack routing index function.

  Returns (source-id . source-idx):
    source-id  = indices[axis]
    source-idx = indices with the axis position removed."
  (let ((axis (stack-index-fn-axis fn)))
    (cons (list-ref indices axis)
          (append (take indices axis)
                  (drop indices (+ axis 1))))))

(define (apply-concat-index-fn fn indices)
  "Apply concat routing index function.

  Returns (source-id . source-idx):
    source-id determined by binary-searching offsets for indices[axis],
    source-idx is indices with the axis coordinate remapped to its
    local offset within the chosen operand."
  (let* ((axis    (concat-index-fn-axis    fn))
         (offsets (concat-index-fn-offsets fn))
         (ax-idx  (list-ref indices axis))
         (n       (length offsets)))
    (let loop ((i 0))
      (cond
        ((>= i n)
         (error "concat-index-fn: index out of range" ax-idx offsets))
        ;; Last operand, or ax-idx falls before the next operand's start.
        ((or (= i (- n 1))
             (< ax-idx (list-ref offsets (+ i 1))))
         (let* ((local-idx (- ax-idx (list-ref offsets i)))
                (src-idx   (append (take indices axis)
                                   (list local-idx)
                                   (drop indices (+ axis 1)))))
           (cons i src-idx)))
        (else
         (loop (+ i 1)))))))

(define (affine-fn->matrix fn)
  "Extract the equivalent dense matrix from any affine-index-fn variant."
  (cases affine-index-fn fn
         (identity-fn   ()     #f)
         (permutation-fn (p)   (permutation-to-matrix p))
         (diagonal-fn   (d b)  (diagonal-to-matrix d))
         (general-fn    (A b)  A)))

(define (affine-fn->bias fn)
  "Extract the bias list from any affine-index-fn variant, or #f."
  (cases affine-index-fn fn
         (identity-fn   ()     #f)
         (permutation-fn (p)   #f)
         (diagonal-fn   (d b)  b)
         (general-fn    (A b)  b)))

;;;; ============================================================
;;;; Index Function Simplification
;;;; ============================================================

(define (simplify-index-fn fn)
  "Apply algebraic simplification rules to index function
   
   Simplification rules:
   - (f ∘ id) -> f
   - (id ∘ f) -> f
   - (f ∘ f^{-1}) -> id (for invertible f)
   - Nested compositions flattened
   - Identity matrices eliminated"
  
  (cond
    ;; identity-fn is already in simplest form
    ((and (affine-index-fn? fn)
          (cases affine-index-fn fn
            (identity-fn () #t)
            (else         #f)))
     fn)

    ;; Composed with identity -> simplify
    ((composed-index-fn? fn)
     (let ((outer (composed-index-fn-outer fn))
           (inner (composed-index-fn-inner fn)))
       (cond
         ((identity-index-fn? outer) (simplify-index-fn inner))
         ((identity-index-fn? inner) (simplify-index-fn outer))
         (else
          (let ((outer-simp (simplify-index-fn outer))
                (inner-simp (simplify-index-fn inner)))
            (if (and (eq? outer outer-simp)
                     (eq? inner inner-simp))
                fn
                (make-composed-index-fn outer-simp inner-simp)))))))

    (else fn)))

  
;;;; ============================================================
;;;; Permutation Utilities (for transpose)
;;;; ============================================================

(define (identity-permutation? perm)
  "Check if permutation is identity [0,1,2,...,n-1]"
  (let ((n (length perm)))
    (every (lambda (i) (= (list-ref perm i) i))
           (iota n))))

(define (compose-permutations p1 p2)
  "Compose two permutations: (p1 ∘ p2)(i) = p1[p2[i]]
   
   Example: p1=[2,0,1], p2=[1,2,0]
            result[0] = p1[p2[0]] = p1[1] = 0
            result[1] = p1[p2[1]] = p1[2] = 1
            result[2] = p1[p2[2]] = p1[0] = 2
            -> [0,1,2] (identity)"
  
  (unless (= (length p1) (length p2))
    (error "Permutation length mismatch" 
           `((p1-length ,(length p1)) (p2-length ,(length p2)))))
  
  (map (lambda (i) (list-ref p1 (list-ref p2 i)))
       (iota (length p1))))

(define (invert-permutation perm)
  "Compute inverse of permutation
   
   If perm[i] = j, then inv[j] = i"
  
  (let ((n (length perm))
        (inv (make-vector (length perm))))
    
    (do ((i 0 (+ i 1)))
        ((= i n))
      (vector-set! inv (list-ref perm i) i))
    
    (vector->list inv)))

(define (permutation-to-matrix perm)
  "Convert permutation to permutation matrix
   
   Example: [1,2,0] -> [[0,1,0],
                       [0,0,1],
                       [1,0,0]]"
  
  (let ((n (length perm)))
    (let loop ((i 0))
      (if (= i n)
          '()
          (cons (let inner ((j 0))
                  (if (= j n)
                      '()
                      (cons (if (= j (list-ref perm i)) 1 0)
                            (inner (+ j 1)))))
                (loop (+ i 1)))))))

(define (diagonal-to-matrix d)
  "Convert a diagonal vector (list) to a square diagonal matrix."
  (let ((n (length d)))
    (map (lambda (i)
           (map (lambda (j) (if (= i j) (list-ref d i) 0))
                (iota n)))
         (iota n))))

;;;; ============================================================
;;;; Specialized Index Function Constructors
;;;; ============================================================

(define (make-reshape-index-fn old-shape new-shape)
  "Create index function for reshape operation
   
   Reshape is identity on linear indices (just reinterpret shape)"
  
  (unless (reshape-compatible? old-shape new-shape)
    (error "Incompatible shapes for reshape" 
           `((old ,old-shape) (new ,new-shape))))
  
  ;; Reshape is identity transformation
  (identity-fn))

(define (make-transpose-index-fn permutation)
  "Create index function for transpose operation
   
   Transpose permutes dimensions: output[i,j,k] = input[perm[i],perm[j],perm[k]]"
  
  (permutation-fn permutation))

(define (make-slice-index-fn start end step)
  "Create index function for slice operation
   
   Maps output indices to input indices with offset and stride
   output_idx -> start + output_idx * step"
  
  (diagonal-fn step start))

;;;; ============================================================
;;;; Index Function Information
;;;; ============================================================

(define (index-fn-invertible? fn)
  "Check if index function is invertible.

   identity-fn:    trivially invertible.
   permutation-fn: permutations are always bijections.
   diagonal-fn:    invertible iff every scale factor is non-zero.
   general-fn:     conservative #f (full rank check not implemented)."
  (cond
    ((affine-index-fn? fn)
     (cases affine-index-fn fn
       (identity-fn   ()    #t)
       (permutation-fn (p)  #t)
       (diagonal-fn   (d b) (every (lambda (x) (not (zero? x))) d))
       (general-fn    (A b) #f)))
    (else #f)))

(define (index-fn-rank fn)
  "Estimate output rank of index function."
  (cond
    ((affine-index-fn? fn)
     (cases affine-index-fn fn
       (identity-fn   ()    0)   ; rank not knowable without shape context
       (permutation-fn (p)  (length p))
       (diagonal-fn   (d b) (length d))
       (general-fn    (A b) (length A))))
    ((composed-index-fn? fn)
     (index-fn-rank (composed-index-fn-outer fn)))
    (else 0)))

)
