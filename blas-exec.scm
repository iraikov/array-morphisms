;;; array-morphisms-blas-exec.scm
;;; BLAS Execution Kernels
;;;
;;; Provides:
;;;   - Backend record type with normalized kernel interface
;;;   - Backend registration mechanism (no hard BLAS library dependency)
;;;   - Pure Scheme fallback kernels for GEMM, GEMV, DOT, AXPY
;;;   - Execute-or-fallback dispatch functions
;;;   - Morphism constructors for matmul/matvec/dot/axpy
;;;
;;; The backend registration pattern decouples this module from any specific
;;; BLAS implementation.  Backends for the Chicken 'blas' egg, CUDA, or
;;; chicken-crunch kernels are registered at application startup by loading
;;; the corresponding adapter module and calling register-blas-backend!.
;;;
;;; Normalized Kernel Interface
;;; ===========================
;;; Every blas-backend record stores eight function slots, two per operation
;;; (f64 and f32 variants).  Each slot must hold a procedure obeying the
;;; following Scheme calling convention, or #f if the backend does not
;;; provide that variant.
;;;
;;; All kernels assume row-major, contiguous (offset=0, standard strides)
;;; storage.  Non-contiguous operands must be densified before calling.
;;;
;;;   gemm-fn : M N K ALPHA DATA-A DATA-B BETA DATA-C -> unspecified
;;;     C := alpha * A * B + beta * C   (in-place on DATA-C)
;;;     DATA-A : f32/f64vector  -- M x K matrix, row-major
;;;     DATA-B : f32/f64vector  -- K x N matrix, row-major
;;;     DATA-C : f32/f64vector  -- M x N matrix, row-major (mutated)
;;;
;;;   gemv-fn : M N ALPHA DATA-A DATA-X BETA DATA-Y -> unspecified
;;;     y := alpha * A * x + beta * y   (in-place on DATA-Y)
;;;     DATA-A : f32/f64vector  -- M x N matrix, row-major
;;;     DATA-X : f32/f64vector  -- length N
;;;     DATA-Y : f32/f64vector  -- length M (mutated)
;;;
;;;   dot-fn  : N DATA-X DATA-Y -> number
;;;     result := x . y        (returns a Scheme number, no mutation)
;;;     DATA-X : f32/f64vector  -- length N
;;;     DATA-Y : f32/f64vector  -- length N
;;;
;;;   axpy-fn : N ALPHA DATA-X DATA-Y -> unspecified
;;;     y := alpha * x + y     (in-place on DATA-Y)
;;;     DATA-X : f32/f64vector  -- length N
;;;     DATA-Y : f32/f64vector  -- length N (mutated)

(module array-morphisms-blas-exec

  (;; Backend record type
   make-blas-backend
   blas-backend?
   blas-backend-name
   blas-backend-gemm-f64
   blas-backend-gemm-f32
   blas-backend-gemm-strided-f64
   blas-backend-gemm-strided-f32
   blas-backend-gemv-f64
   blas-backend-gemv-f32
   blas-backend-dot-f64
   blas-backend-dot-f32
   blas-backend-axpy-f64
   blas-backend-axpy-f32

   ;; Backend registration and inspection
   register-blas-backend!
   active-blas-backend
   blas-available?
   *active-backend*

   ;; Global BLAS configuration
   blas-enabled?
   enable-blas!
   disable-blas!
   *blas-size-threshold*

   ;; Morphism constructors
   morph-matmul
   morph-matvec
   morph-dot
   morph-axpy
   attention-morphism

   ;; Pure Scheme fallback kernels
   execute-scheme-gemm
   execute-scheme-gemv
   execute-scheme-dot
   execute-scheme-axpy

   ;; Execute-or-fallback dispatch
   execute-blas-gemm
   execute-blas-gemv
   execute-blas-dot
   execute-blas-axpy

   ;; Re-exported from blas-compat for use in realization and grad modules
   matmul-concrete?

   ;; High-level morphism-expr dispatch (used by realization engine)
   execute-blas-operation

   ;; In-place variants: write result into caller-supplied typed-vector.
   ;; Used by realize-morphism-expr/ctx
   execute-blas-gemm/into!
   execute-blas-gemv/into!

   ;; Strided GEMM: handles transposed zero-copy views via Trans flag
   execute-blas-gemm-strided/into!
   execute-blas-gemm-strided

   ;; Re-exported from blas-compat for convenience
   array->gemm-blas-params)

  (import scheme (chicken base))
  (import (only srfi-1 every))
  (import datatype)
  (import array-morphisms-core)
  (import array-morphisms-blas-compat)

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Backend Record
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; A blas-backend bundles all kernel function pointers for one BLAS
  ;; implementation.  The name field is a symbol used for logging and
  ;; future routing logic (e.g. 'openblas, 'mkl, 'cuda, 'crunch).
  ;;
  ;; Each kernel field holds a procedure implementing the normalized interface
  ;; described in the module header, or #f when that variant is not provided
  ;; by this backend.
  ;;
  ;; The normalized interface is intentionally decoupled from the Chicken
  ;; 'blas' egg's keyword-argument API.  Adapter modules (e.g.
  ;; array-morphisms-blas-egg-backend.scm) bridge any concrete library into
  ;; this interface.  This means blas-exec.scm carries no direct dependency
  ;; on the 'blas' egg, CUDA bindings, or chicken-crunch.
  (define-record blas-backend
    name              ; symbol: backend identity
    gemm-f64          ; (M N K alpha data-A data-B beta data-C) -> void, or #f
    gemm-f32          ; (M N K alpha data-A data-B beta data-C) -> void, or #f
    gemm-strided-f64  ; (M N K alpha data-A lda-A transa data-B ldb-B transb beta data-C) -> void, or #f
    gemm-strided-f32  ; (M N K alpha data-A lda-A transa data-B ldb-B transb beta data-C) -> void, or #f
    gemv-f64          ; (M N alpha data-A data-x beta data-y) -> void, or #f
    gemv-f32          ; (M N alpha data-A data-x beta data-y) -> void, or #f
    dot-f64           ; (N data-x data-y) -> number, or #f
    dot-f32           ; (N data-x data-y) -> number, or #f
    axpy-f64          ; (N alpha data-x data-y) -> void, or #f
    axpy-f32)         ; (N alpha data-x data-y) -> void, or #f

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Configuration
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define *active-backend*      #f)
  (define *blas-enabled*        #t)
  (define *blas-size-threshold* 64)

  (define (blas-enabled?)
    "True when BLAS optimisation is globally enabled."
    *blas-enabled*)

  (define (blas-available?)
    "True when a BLAS backend has been registered via register-blas-backend!."
    (and *active-backend* #t))

  (define (active-blas-backend)
    "Return the currently registered blas-backend record, or #f."
    *active-backend*)

  (define (enable-blas!)
    "Enable BLAS optimisation (default)."
    (set! *blas-enabled* #t))

  (define (disable-blas!)
    "Disable BLAS optimisation; all operations fall back to pure Scheme."
    (set! *blas-enabled* #f))

  (define (register-blas-backend! backend)
    "Register a blas-backend record as the active BLAS implementation.

    The backend's kernel procedures must implement the normalized interface
    described in the module header.  Call this function once at startup after
    loading an adapter module such as array-morphisms-blas-egg-backend.

    Args:
      backend: a blas-backend record created with make-blas-backend.

    Example using the Chicken 'blas' egg adapter:
      (import array-morphisms-blas-egg-backend)
      (register-blas-backend! (make-blas-egg-backend))

    Example using a custom CUDA backend:
      (register-blas-backend!
        (make-blas-backend 'cuda
          cuda-dgemm cuda-sgemm
          cuda-dgemv cuda-sgemv
          cuda-ddot  cuda-sdot
          cuda-daxpy cuda-saxpy))

    where each cuda-* procedure obeys the normalized kernel interface."
    (unless (blas-backend? backend)
      (error "register-blas-backend!: expected a blas-backend record" backend))
    (set! *active-backend* backend))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Internal Helpers
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (use-blas-for-shape? shape)
    "True when the total number of output elements meets the size threshold.
    Checks product of all dimensions rather than per-dimension size, so that
    non-square matrices like (10 x 8) with 80 total elements correctly
    trigger BLAS rather than being rejected because 10 < threshold."
    (>= (apply * (vector->list shape)) *blas-size-threshold*))

  (define (%select-kernel backend dtype f64-accessor f32-accessor)
    "Extract the dtype-specialised kernel from backend.
    Returns the procedure or #f when the backend does not supply it."
    (if (eq? dtype 'f64)
        (f64-accessor backend)
        (f32-accessor backend)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Morphism Constructors
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (morph-matmul A B)
    "Matrix-matrix multiply morphism: (M,K) x (K,N) -> (M,N).

    Creates a morphism-expr with op 'matmul.  The identity-fn placeholder
    is replaced by a specialised kernel in Phase 3; until then use
    execute-scheme-gemm or execute-blas-gemm directly on concrete arrays.

    Type-promotes operand dtypes."
    (unless (array-morphism? A) (error "morph-matmul: A must be a morphism" A))
    (unless (array-morphism? B) (error "morph-matmul: B must be a morphism" B))
    (let* ((sa (get-morphism-shape A))
           (sb (get-morphism-shape B)))
      (unless (= 2 (vector-length sa))
        (error "morph-matmul: A must be 2-D" sa))
      (unless (= 2 (vector-length sb))
        (error "morph-matmul: B must be 2-D" sb))
      (let ((K-a (vector-ref sa 1))
            (K-b (vector-ref sb 0)))
        (unless (= K-a K-b)
          (error "morph-matmul: inner dimensions must match" K-a K-b)))
      (let* ((M     (vector-ref sa 0))
             (N     (vector-ref sb 1))
             (K     (vector-ref sa 1))
             (dtype (promote-types (get-morphism-dtype A) (get-morphism-dtype B))))
        (morphism-expr
         (gensym 'morph-) 'matmul (list A B)
         (identity-fn)
         (vector M N)
         dtype
         `((k-dim . ,K))
         -1))))

  (define (morph-matvec A v)
    "Matrix-vector multiply morphism: (M,N) x (N,) -> (M,).

    Creates a morphism-expr with op 'matvec."
    (unless (array-morphism? A) (error "morph-matvec: A must be a morphism" A))
    (unless (array-morphism? v) (error "morph-matvec: v must be a morphism" v))
    (let* ((sa (get-morphism-shape A))
           (sv (get-morphism-shape v)))
      (unless (= 2 (vector-length sa))
        (error "morph-matvec: A must be 2-D" sa))
      (unless (= 1 (vector-length sv))
        (error "morph-matvec: v must be 1-D" sv))
      (let ((N-a (vector-ref sa 1))
            (N-v (vector-ref sv 0)))
        (unless (= N-a N-v)
          (error "morph-matvec: inner dimensions must match" N-a N-v)))
      (let* ((M     (vector-ref sa 0))
             (N     (vector-ref sa 1))
             (dtype (promote-types (get-morphism-dtype A) (get-morphism-dtype v))))
        (morphism-expr
         (gensym 'morph-) 'matvec (list A v)
         (identity-fn)
         (vector M)
         dtype
         `((n-dim . ,N))
         -1))))

  (define (morph-dot v1 v2)
    "Vector dot product morphism: (N,) . (N,) -> scalar.

    Creates a morphism-expr with op 'dot and scalar result shape #()."
    (unless (array-morphism? v1) (error "morph-dot: v1 must be a morphism" v1))
    (unless (array-morphism? v2) (error "morph-dot: v2 must be a morphism" v2))
    (let* ((s1 (get-morphism-shape v1))
           (s2 (get-morphism-shape v2)))
      (unless (= 1 (vector-length s1))
        (error "morph-dot: v1 must be 1-D" s1))
      (unless (= 1 (vector-length s2))
        (error "morph-dot: v2 must be 1-D" s2))
      (let ((N1 (vector-ref s1 0))
            (N2 (vector-ref s2 0)))
        (unless (= N1 N2)
          (error "morph-dot: vectors must have same length" N1 N2)))
      (let* ((N     (vector-ref s1 0))
             (dtype (promote-types (get-morphism-dtype v1) (get-morphism-dtype v2))))
        (morphism-expr
         (gensym 'morph-) 'dot (list v1 v2)
         (identity-fn)
         #()                  ; scalar output
         dtype
         `((n-dim . ,N))
         -1))))

  (define (morph-axpy alpha x y)
    "AXPY morphism: result = alpha * x + y (non-destructive, returns new array).

    Creates a morphism-expr with op 'axpy.
    alpha: scalar multiplier (exact or inexact number)
    x, y:  morphisms with identical shapes."
    (unless (number? alpha)       (error "morph-axpy: alpha must be a number" alpha))
    (unless (array-morphism? x)   (error "morph-axpy: x must be a morphism"    x))
    (unless (array-morphism? y)   (error "morph-axpy: y must be a morphism"    y))
    (let ((sx (get-morphism-shape x))
          (sy (get-morphism-shape y)))
      (unless (equal? sx sy)
        (error "morph-axpy: x and y must have the same shape" sx sy)))
    (let ((dtype (promote-types (get-morphism-dtype x) (get-morphism-dtype y))))
      (morphism-expr
       (gensym 'morph-) 'axpy (list x y)
       (identity-fn)
       (get-morphism-shape x)
       dtype
       `((alpha . ,alpha))
       -1)))

  (define (attention-morphism Q K V #!optional (scale #f))
    "Fused scaled dot-product attention morphism: softmax(scale*Q*K^T)*V.

    Supports non-batched (rank 2) and batched (rank 3 or 4) inputs:
      Q, K : (n, dk)             V : (n, dv)          -> (n, dv)
      Q, K : (B, n, dk)          V : (B, n, dv)        -> (B, n, dv)
      Q, K : (B, h, n, dk)       V : (B, h, n, dv)     -> (B, h, n, dv)

    scale: real scalar; defaults to 1/sqrt(dk) when #f.

    Returns a morphism-expr with op 'attention.  The realization engine
    dispatches it to execute-fused-attention, which uses a streaming
    numerically-stable softmax (O(n) scratch, no n x n score matrix)."
    (unless (array-morphism? Q) (error "attention-morphism: Q must be a morphism" Q))
    (unless (array-morphism? K) (error "attention-morphism: K must be a morphism" K))
    (unless (array-morphism? V) (error "attention-morphism: V must be a morphism" V))
    (let* ((sq  (get-morphism-shape Q))
           (sk  (get-morphism-shape K))
           (sv  (get-morphism-shape V))
           (rq  (vector-length sq))
           (rk  (vector-length sk))
           (rv  (vector-length sv)))
      (unless (= rq rk rv)
        (error "attention-morphism: Q, K, V must have the same rank" rq rk rv))
      (unless (memv rq '(2 3 4))
        (error "attention-morphism: rank must be 2, 3, or 4" rq))
      (let* ((n    (vector-ref sq (- rq 2)))
             (dk   (vector-ref sq (- rq 1)))
             (dv   (vector-ref sv (- rv 1)))
             (n-v  (vector-ref sv (- rv 2))))
        (unless (equal? sq sk)
          (error "attention-morphism: Q and K must have the same shape" sq sk))
        (unless (= n n-v)
          (error "attention-morphism: V sequence length must match Q/K" n-v n))
        (when (> rq 2)
          (let loop ((i 0))
            (when (< i (- rq 2))
              (unless (= (vector-ref sq i) (vector-ref sv i))
                (error "attention-morphism: leading batch dimensions must match" sq sv))
              (loop (+ i 1)))))
        (let* ((resolved-scale
                (if scale
                    (exact->inexact scale)
                    (/ 1.0 (sqrt (exact->inexact dk)))))
               (out-shape
                (let ((v (make-vector rq)))
                  (do ((i 0 (+ i 1))) ((= i rq))
                    (vector-set! v i (vector-ref sq i)))
                  (vector-set! v (- rq 1) dv)
                  v))
               (batch-axis (if (= rq 2) -1 0))
               (dtype (promote-types
                       (promote-types (get-morphism-dtype Q) (get-morphism-dtype K))
                       (get-morphism-dtype V))))
          (morphism-expr
           (gensym 'morph-) 'attention (list Q K V)
           (identity-fn)
           out-shape
           dtype
           `((scale     . ,resolved-scale)
             (n         . ,n)
             (dk        . ,dk)
             (dv        . ,dv)
             (n-leading . ,(- rq 2)))
           batch-axis)))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Pure Scheme Fallback Kernels
  ;;; All kernels are stride-aware and handle non-contiguous operands.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (execute-scheme-gemm A B)
    "Pure Scheme matrix-matrix multiply: C = A * B.

    Uses a stride-aware triple loop; handles transposed / sliced operands
    correctly by honouring each array's strides and offset.

    Args:
      A: concrete-array with shape (M, K)
      B: concrete-array with shape (K, N)
    Returns:
      Fresh contiguous concrete-array with shape (M, N) and dtype of A."
    (unless (concrete-array? A)
      (error "execute-scheme-gemm: A must be a concrete array" A))
    (unless (concrete-array? B)
      (error "execute-scheme-gemm: B must be a concrete array" B))
    (unless (= 2 (morph-rank A))
      (error "execute-scheme-gemm: A must be 2-D" (get-morphism-shape A)))
    (unless (= 2 (morph-rank B))
      (error "execute-scheme-gemm: B must be 2-D" (get-morphism-shape B)))
    (cases array-morphism A
      (concrete-array (data-A shape-A strides-A offset-A dtype-A _ _)
        (cases array-morphism B
          (concrete-array (data-B shape-B strides-B offset-B dtype-B _ _)
            (let* ((M   (vector-ref shape-A 0))
                   (K   (vector-ref shape-A 1))
                   (N   (vector-ref shape-B 1))
                   (_ (unless (= K (vector-ref shape-B 0))
                        (error "execute-scheme-gemm: incompatible shapes"
                               shape-A shape-B)))
                   (result-shape (vector M N))
                   (result-data  (allocate-typed-vector dtype-A (* M N)))
                   (sa0 (vector-ref strides-A 0))
                   (sa1 (vector-ref strides-A 1))
                   (sb0 (vector-ref strides-B 0))
                   (sb1 (vector-ref strides-B 1)))
              (do ((i 0 (+ i 1)))
                  ((= i M))
                (do ((j 0 (+ j 1)))
                    ((= j N))
                  (let loop ((k 0) (acc 0.0))
                    (if (= k K)
                        (typed-vector-set! result-data dtype-A
                                           (+ (* i N) j) acc)
                        (loop (+ k 1)
                              (+ acc
                                 (* (typed-vector-ref data-A dtype-A
                                      (+ offset-A (* i sa0) (* k sa1)))
                                    (typed-vector-ref data-B dtype-B
                                      (+ offset-B (* k sb0) (* j sb1))))))))))
              (concrete-array result-data result-shape
                              (compute-strides result-shape)
                              0 dtype-A -1 -1)))
          (else (error "execute-scheme-gemm: B must be a concrete array" B))))
      (else (error "execute-scheme-gemm: A must be a concrete array" A))))

  (define (execute-scheme-gemv A v)
    "Pure Scheme matrix-vector multiply: y = A * x.

    Stride-aware double loop; handles non-contiguous operands.

    Args:
      A: concrete-array with shape (M, N)
      v: concrete-array with shape (N,)
    Returns:
      Fresh contiguous concrete-array with shape (M,)."
    (unless (concrete-array? A)
      (error "execute-scheme-gemv: A must be a concrete array" A))
    (unless (concrete-array? v)
      (error "execute-scheme-gemv: v must be a concrete array" v))
    (cases array-morphism A
      (concrete-array (data-A shape-A strides-A offset-A dtype-A _ _)
        (cases array-morphism v
          (concrete-array (data-v shape-v strides-v offset-v dtype-v _ _)
            (let* ((M   (vector-ref shape-A 0))
                   (N   (vector-ref shape-A 1))
                   (_ (unless (= N (vector-ref shape-v 0))
                        (error "execute-scheme-gemv: shape mismatch"
                               shape-A shape-v)))
                   (result-shape (vector M))
                   (result-data  (allocate-typed-vector dtype-A M))
                   (sa0 (vector-ref strides-A 0))
                   (sa1 (vector-ref strides-A 1))
                   (sv0 (vector-ref strides-v 0)))
              (do ((i 0 (+ i 1)))
                  ((= i M))
                (let loop ((j 0) (acc 0.0))
                  (if (= j N)
                      (typed-vector-set! result-data dtype-A i acc)
                      (loop (+ j 1)
                            (+ acc
                               (* (typed-vector-ref data-A dtype-A
                                    (+ offset-A (* i sa0) (* j sa1)))
                                  (typed-vector-ref data-v dtype-v
                                    (+ offset-v (* j sv0)))))))))
              (concrete-array result-data result-shape
                              (compute-strides result-shape)
                              0 dtype-A -1 -1)))
          (else (error "execute-scheme-gemv: v must be a concrete array" v))))
      (else (error "execute-scheme-gemv: A must be a concrete array" A))))

  (define (execute-scheme-dot v1 v2)
    "Pure Scheme vector dot product: result = v1 . v2.

    Stride-aware single loop.

    Args:
      v1, v2: concrete-arrays with shape (N,)
    Returns:
      Fresh concrete-array with scalar shape #() holding the dot product."
    (unless (concrete-array? v1)
      (error "execute-scheme-dot: v1 must be a concrete array" v1))
    (unless (concrete-array? v2)
      (error "execute-scheme-dot: v2 must be a concrete array" v2))
    (cases array-morphism v1
      (concrete-array (data1 shape1 strides1 offset1 dtype1 _ _)
        (cases array-morphism v2
          (concrete-array (data2 shape2 strides2 offset2 dtype2 _ _)
            (let* ((N  (vector-ref shape1 0))
                   (_ (unless (= N (vector-ref shape2 0))
                        (error "execute-scheme-dot: vectors must have same length"
                               N (vector-ref shape2 0))))
                   (s1  (vector-ref strides1 0))
                   (s2  (vector-ref strides2 0))
                   (result-data (allocate-typed-vector dtype1 1)))
              (let loop ((i 0) (acc 0.0))
                (if (= i N)
                    (begin
                      (typed-vector-set! result-data dtype1 0 acc)
                      (concrete-array result-data #() #() 0 dtype1 -1 -1))
                    (loop (+ i 1)
                          (+ acc
                             (* (typed-vector-ref data1 dtype1
                                  (+ offset1 (* i s1)))
                                (typed-vector-ref data2 dtype2
                                  (+ offset2 (* i s2))))))))))
          (else (error "execute-scheme-dot: v2 must be a concrete array" v2))))
      (else (error "execute-scheme-dot: v1 must be a concrete array" v1))))

  (define (execute-scheme-axpy alpha x y)
    "Pure Scheme AXPY: result = alpha * x + y.

    Non-destructive (does not modify y); returns a fresh array.
    Stride-aware; handles non-contiguous operands.

    Args:
      alpha: scalar
      x, y:  concrete-arrays with the same shape
    Returns:
      Fresh contiguous concrete-array with shape matching x."
    (unless (concrete-array? x)
      (error "execute-scheme-axpy: x must be a concrete array" x))
    (unless (concrete-array? y)
      (error "execute-scheme-axpy: y must be a concrete array" y))
    (cases array-morphism x
      (concrete-array (data-x shape-x strides-x offset-x dtype-x _ _)
        (cases array-morphism y
          (concrete-array (data-y shape-y strides-y offset-y dtype-y _ _)
            (unless (equal? shape-x shape-y)
              (error "execute-scheme-axpy: shapes must match" shape-x shape-y))
            (let* ((N  (vector-ref shape-x 0))
                   (sx (vector-ref strides-x 0))
                   (sy (vector-ref strides-y 0))
                   (result-data (allocate-typed-vector dtype-x N)))
              (do ((i 0 (+ i 1)))
                  ((= i N)
                   (concrete-array result-data shape-x
                                   (compute-strides shape-x)
                                   0 dtype-x -1 -1))
                (typed-vector-set! result-data dtype-x i
                  (+ (* alpha
                        (typed-vector-ref data-x dtype-x
                          (+ offset-x (* i sx))))
                     (typed-vector-ref data-y dtype-y
                       (+ offset-y (* i sy))))))))
          (else (error "execute-scheme-axpy: y must be a concrete array" y))))
      (else (error "execute-scheme-axpy: x must be a concrete array" x))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; In-Place Execution Variants
  ;;;
  ;;; These write results into a caller-supplied typed-vector instead of
  ;;; allocating.  Used by realize-morphism-expr/ctx so that both trace and
  ;;; replay modes can fill the same pool buffer with BLAS kernels.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; -- GEMM/into! -----------------------------------------------------------

  (define (%scheme-gemm/into! A B result-data)
    "Pure Scheme GEMM writing into result-data (no allocation)."
    (cases array-morphism A
      (concrete-array (data-A shape-A strides-A offset-A dtype-A _ _)
        (cases array-morphism B
          (concrete-array (data-B shape-B strides-B offset-B _ _ _)
            (let* ((M   (vector-ref shape-A 0))
                   (K   (vector-ref shape-A 1))
                   (N   (vector-ref shape-B 1))
                   (sa0 (vector-ref strides-A 0))
                   (sa1 (vector-ref strides-A 1))
                   (sb0 (vector-ref strides-B 0))
                   (sb1 (vector-ref strides-B 1)))
              (do ((i 0 (+ i 1)))
                  ((= i M))
                (do ((j 0 (+ j 1)))
                    ((= j N))
                  (let loop ((k 0) (acc 0.0))
                    (if (= k K)
                        (typed-vector-set! result-data dtype-A
                                           (+ (* i N) j) acc)
                        (loop (+ k 1)
                              (+ acc
                                 (* (typed-vector-ref data-A dtype-A
                                      (+ offset-A (* i sa0) (* k sa1)))
                                    (typed-vector-ref data-B dtype-A
                                      (+ offset-B (* k sb0) (* j sb1))))))))))))
          (else (error "%scheme-gemm/into!: B must be a concrete array" B))))
      (else (error "%scheme-gemm/into!: A must be a concrete array" A))))

  (define (%blas-gemm/into! A B result-data)
    "Invoke active backend GEMM kernel into result-data (no allocation)."
    (cases array-morphism A
      (concrete-array (data-A shape-A _ _ dtype-A _ _)
        (cases array-morphism B
          (concrete-array (data-B shape-B _ _ _ _ _)
            (let* ((M  (vector-ref shape-A 0))
                   (K  (vector-ref shape-A 1))
                   (N  (vector-ref shape-B 1))
                   (fn (%select-kernel *active-backend* dtype-A
                                       blas-backend-gemm-f64
                                       blas-backend-gemm-f32)))
              (unless fn
                (error "%blas-gemm/into!: active backend has no GEMM kernel"
                       dtype-A (blas-backend-name *active-backend*)))
              (fn M N K 1.0 data-A data-B 0.0 result-data)))
          (else (error "%blas-gemm/into!: B must be a concrete array" B))))
      (else (error "%blas-gemm/into!: A must be a concrete array" A))))

  (define (execute-blas-gemm/into! A B result-data)
    "Execute matrix-matrix multiply into caller-supplied result-data.

    Uses BLAS when available and beneficial; falls back to pure Scheme.
    Writes M*N values into result-data (row-major); returns unspecified.

    Args:
      A, B:        concrete-arrays with compatible shapes
      result-data: typed-vector of length M*N (pre-allocated by caller)"
    (if (and *blas-enabled*
             *active-backend*
             (blas-compatible-matmul? A B)
             (>= (* (vector-ref (get-morphism-shape A) 0)
                    (vector-ref (get-morphism-shape B) 1))
                 *blas-size-threshold*))
        (%blas-gemm/into! A B result-data)
        (%scheme-gemm/into! A B result-data)))

  ;; -- Strided GEMM/into! ---------------------------------------------------
  ;; Handles transposed zero-copy views by extracting Trans flags and physical
  ;; leading dimensions from the array strides, then calling the backend's
  ;; gemm-strided kernel (e.g. dgemm! with Trans).  Falls back to the
  ;; stride-aware %scheme-gemm/into! when no strided kernel is registered or
  ;; when the stride pattern cannot be expressed as a BLAS Trans flag.

  (define (%blas-gemm-strided/into! A B result-data)
    "Invoke backend strided GEMM into result-data.
    Falls back to %scheme-gemm/into! when the backend has no strided kernel
    or when array->gemm-blas-params cannot classify the operand layout."
    (let ((params-A (array->gemm-blas-params A))
          (params-B (array->gemm-blas-params B)))
      (if (and params-A params-B)
          (cases array-morphism A
            (concrete-array (_ shape-A _ _ dtype-A _ _)
              (cases array-morphism B
                (concrete-array (_ shape-B _ _ _ _ _)
                  (let* ((M      (vector-ref shape-A 0))
                         (K      (vector-ref shape-A 1))
                         (N      (vector-ref shape-B 1))
                         (data-A (car params-A))
                         (transA (cadr params-A))
                         (ldA    (caddr params-A))
                         (data-B (car params-B))
                         (transB (cadr params-B))
                         (ldB    (caddr params-B))
                         (fn     (%select-kernel *active-backend* dtype-A
                                                 blas-backend-gemm-strided-f64
                                                 blas-backend-gemm-strided-f32)))
                    (if fn
                        (fn M N K 1.0 data-A ldA transA data-B ldB transB 0.0 result-data)
                        (%scheme-gemm/into! A B result-data))))
                (else (%scheme-gemm/into! A B result-data))))
            (else (%scheme-gemm/into! A B result-data)))
          (%scheme-gemm/into! A B result-data))))

  (define (execute-blas-gemm-strided/into! A B result-data)
    "Execute strided matrix-matrix multiply into caller-supplied result-data.

    Uses the backend's strided GEMM kernel (with Trans flags) when available
    and the operand layouts are recognisable; falls back to the stride-aware
    pure Scheme triple loop otherwise.  Unlike execute-blas-gemm/into! this
    does NOT require row-major operands.

    Args:
      A, B:        concrete-arrays with compatible shapes (may be transposed views)
      result-data: typed-vector of length M*N (pre-allocated, row-major)"
    (if (and *blas-enabled*
             *active-backend*
             (>= (* (vector-ref (get-morphism-shape A) 0)
                    (vector-ref (get-morphism-shape B) 1))
                 *blas-size-threshold*))
        (%blas-gemm-strided/into! A B result-data)
        (%scheme-gemm/into! A B result-data)))

  (define (execute-blas-gemm-strided A B)
    "Execute strided matrix-matrix multiply; returns a fresh concrete-array.

    Used by execute-blas-operation for the gemm-strided case."
    (cases array-morphism A
      (concrete-array (_ shape-A _ _ dtype-A _ _)
        (cases array-morphism B
          (concrete-array (_ shape-B _ _ _ _ _)
            (let* ((M            (vector-ref shape-A 0))
                   (N            (vector-ref shape-B 1))
                   (result-shape (vector M N))
                   (result-data  (allocate-typed-vector dtype-A (* M N))))
              (execute-blas-gemm-strided/into! A B result-data)
              (concrete-array result-data result-shape
                              (compute-strides result-shape) 0 dtype-A -1 -1)))
          (else (error "execute-blas-gemm-strided: B must be a concrete array" B))))
      (else (error "execute-blas-gemm-strided: A must be a concrete array" A))))

  ;; -- GEMV/into! -----------------------------------------------------------

  (define (%scheme-gemv/into! A v result-data)
    "Pure Scheme GEMV writing into result-data (no allocation)."
    (cases array-morphism A
      (concrete-array (data-A shape-A strides-A offset-A dtype-A _ _)
        (cases array-morphism v
          (concrete-array (data-v shape-v strides-v offset-v _ _ _)
            (let* ((M   (vector-ref shape-A 0))
                   (N   (vector-ref shape-A 1))
                   (sa0 (vector-ref strides-A 0))
                   (sa1 (vector-ref strides-A 1))
                   (sv0 (vector-ref strides-v 0)))
              (do ((i 0 (+ i 1)))
                  ((= i M))
                (let loop ((j 0) (acc 0.0))
                  (if (= j N)
                      (typed-vector-set! result-data dtype-A i acc)
                      (loop (+ j 1)
                            (+ acc
                               (* (typed-vector-ref data-A dtype-A
                                    (+ offset-A (* i sa0) (* j sa1)))
                                  (typed-vector-ref data-v dtype-A
                                    (+ offset-v (* j sv0)))))))))))
          (else (error "%scheme-gemv/into!: v must be a concrete array" v))))
      (else (error "%scheme-gemv/into!: A must be a concrete array" A))))

  (define (%blas-gemv/into! A v result-data)
    "Invoke active backend GEMV kernel into result-data (no allocation)."
    (cases array-morphism A
      (concrete-array (data-A shape-A _ _ dtype-A _ _)
        (cases array-morphism v
          (concrete-array (data-v _ _ _ _ _ _)
            (let* ((M  (vector-ref shape-A 0))
                   (N  (vector-ref shape-A 1))
                   (fn (%select-kernel *active-backend* dtype-A
                                       blas-backend-gemv-f64
                                       blas-backend-gemv-f32)))
              (unless fn
                (error "%blas-gemv/into!: active backend has no GEMV kernel"
                       dtype-A (blas-backend-name *active-backend*)))
              (fn M N 1.0 data-A data-v 0.0 result-data)))
          (else (error "%blas-gemv/into!: v must be a concrete array" v))))
      (else (error "%blas-gemv/into!: A must be a concrete array" A))))

  (define (execute-blas-gemv/into! A v result-data)
    "Execute matrix-vector multiply into caller-supplied result-data.

    Uses BLAS when available and beneficial; falls back to pure Scheme.
    Writes M values into result-data; returns unspecified.

    Args:
      A:           concrete-array with shape (M, N)
      v:           concrete-array with shape (N,)
      result-data: typed-vector of length M (pre-allocated by caller)"
    (if (and *blas-enabled*
             *active-backend*
             (blas-compatible-matvec? A v)
             (use-blas-for-shape? (get-morphism-shape A)))
        (%blas-gemv/into! A v result-data)
        (%scheme-gemv/into! A v result-data)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; BLAS Dispatch Helpers
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; BLAS kernels require contiguous row-major arrays.  When an operand is
  ;; non-contiguous (sliced or transposed), it must be densified first.
  ;; densify-if-needed returns a contiguous concrete-array, which may be the
  ;; same object when it is already contiguous.
  (define (densify-if-needed m)
    "Return m if already contiguous row-major; otherwise error.
    Non-contiguous operands are filtered out by blas-compatible-*? predicates
    before this function is reached, so this branch indicates a logic error."
    (if (contiguous-row-major? m)
        m
        (error "densify-if-needed: non-contiguous operand reached BLAS path; \
this is a bug (compat predicates should have rejected it)" m)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Execute-or-Fallback Dispatch
  ;;; Each function checks: (1) globally enabled, (2) backend registered,
  ;;; (3) shape large enough, then calls the normalized kernel.
  ;;; Falls back to the pure Scheme kernel on any failure condition.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; ---- GEMM ---------------------------------------------------------------

  (define (execute-blas-gemm A B)
    "Execute matrix-matrix multiply, using a registered BLAS backend when
    available and beneficial, falling back to pure Scheme otherwise.

    Args:
      A, B: concrete-arrays (shapes are validated by the kernel or fallback)
    Returns:
      concrete-array with shape (M, N)."
    (if (and *blas-enabled*
             *active-backend*
             (blas-compatible-matmul? A B)
             (>= (* (vector-ref (get-morphism-shape A) 0)
                    (vector-ref (get-morphism-shape B) 1))
                 *blas-size-threshold*))
        (%blas-gemm A B)
        (execute-scheme-gemm A B)))

  (define (%blas-gemm A B)
    "Invoke the active backend's GEMM kernel (normalized interface).
    Assumes compatibility has already been verified."
    (cases array-morphism A
      (concrete-array (data-A shape-A _ _ dtype-A _ _)
        (cases array-morphism B
          (concrete-array (data-B shape-B _ _ _ _ _)
            (let* ((M   (vector-ref shape-A 0))
                   (K   (vector-ref shape-A 1))
                   (N   (vector-ref shape-B 1))
                   (result-shape (vector M N))
                   (result-data  (allocate-typed-vector dtype-A (* M N)))
                   (fn  (%select-kernel *active-backend* dtype-A
                                        blas-backend-gemm-f64
                                        blas-backend-gemm-f32)))
              (unless fn
                (error "%blas-gemm: active backend has no GEMM kernel for dtype"
                       dtype-A
                       (blas-backend-name *active-backend*)))
              ;; Normalized call: (M N K alpha data-A data-B beta data-C)
              (fn M N K 1.0 data-A data-B 0.0 result-data)
              (concrete-array result-data result-shape
                              (compute-strides result-shape)
                              0 dtype-A -1 -1)))
          (else (error "%blas-gemm: B must be a concrete array"))))
      (else (error "%blas-gemm: A must be a concrete array"))))

  ;; ---- GEMV ---------------------------------------------------------------

  (define (execute-blas-gemv A v)
    "Execute matrix-vector multiply, using a registered BLAS backend when
    available, falling back to pure Scheme otherwise."
    (if (and *blas-enabled*
             *active-backend*
             (blas-compatible-matvec? A v)
             (use-blas-for-shape? (get-morphism-shape A)))
        (%blas-gemv A v)
        (execute-scheme-gemv A v)))

  (define (%blas-gemv A v)
    "Invoke the active backend's GEMV kernel (normalized interface)."
    (cases array-morphism A
      (concrete-array (data-A shape-A _ _ dtype-A _ _)
        (cases array-morphism v
          (concrete-array (data-v _ _ _ _ _ _)
            (let* ((M   (vector-ref shape-A 0))
                   (N   (vector-ref shape-A 1))
                   (result-data (allocate-typed-vector dtype-A M))
                   (fn  (%select-kernel *active-backend* dtype-A
                                        blas-backend-gemv-f64
                                        blas-backend-gemv-f32)))
              (unless fn
                (error "%blas-gemv: active backend has no GEMV kernel for dtype"
                       dtype-A
                       (blas-backend-name *active-backend*)))
              ;; Normalized call: (M N alpha data-A data-x beta data-y)
              (fn M N 1.0 data-A data-v 0.0 result-data)
              (concrete-array result-data (vector M)
                              (compute-strides (vector M))
                              0 dtype-A -1 -1)))
          (else (error "%blas-gemv: v must be a concrete array"))))
      (else (error "%blas-gemv: A must be a concrete array"))))

  ;; ---- DOT ----------------------------------------------------------------

  (define (execute-blas-dot v1 v2)
    "Execute vector dot product, using a registered BLAS backend when
    available, falling back to pure Scheme otherwise."
    (if (and *blas-enabled*
             *active-backend*
             (blas-compatible-dot? v1 v2))
        (%blas-dot v1 v2)
        (execute-scheme-dot v1 v2)))

  (define (%blas-dot v1 v2)
    "Invoke the active backend's DOT kernel (normalized interface).
    Returns a concrete scalar array wrapping the dot product value."
    (cases array-morphism v1
      (concrete-array (data1 shape1 _ _ dtype1 _ _)
        (cases array-morphism v2
          (concrete-array (data2 _ _ _ _ _ _)
            (let* ((N   (vector-ref shape1 0))
                   (fn  (%select-kernel *active-backend* dtype1
                                         blas-backend-dot-f64
                                         blas-backend-dot-f32)))
              (unless fn
                (error "%blas-dot: active backend has no DOT kernel for dtype"
                       dtype1
                       (blas-backend-name *active-backend*)))
              ;; Normalized call: (N data-x data-y) -> number
              (let* ((val         (fn N data1 data2))
                     (result-data (allocate-typed-vector dtype1 1)))
                (typed-vector-set! result-data dtype1 0 val)
                (concrete-array result-data #() #() 0 dtype1 -1 -1))))
          (else (error "%blas-dot: v2 must be a concrete array"))))
      (else (error "%blas-dot: v1 must be a concrete array"))))

  ;; ---- AXPY ---------------------------------------------------------------

  (define (execute-blas-axpy alpha x y)
    "Execute AXPY, using a registered BLAS backend when available,
    falling back to pure Scheme otherwise.

    Non-destructive: the input y is copied before calling the backend
    so that the morphism semantics (y unchanged, fresh result returned)
    are preserved regardless of which backend is active."
    (if (and *blas-enabled*
             *active-backend*
             (blas-compatible-dot? x y))   ; same shape/dtype/contiguous check
        (%blas-axpy alpha x y)
        (execute-scheme-axpy alpha x y)))

  (define (%blas-axpy alpha x y)
    "Invoke the active backend's AXPY kernel (normalized interface).
    Copies y into result-data first; AXPY mutates result-data in-place."
    (cases array-morphism x
      (concrete-array (data-x shape-x strides-x offset-x dtype-x _ _)
        (cases array-morphism y
          (concrete-array (data-y _ strides-y offset-y _ _ _)
            (let* ((N    (vector-ref shape-x 0))
                   (sy   (vector-ref strides-y 0))
                   ;; Pre-allocate output and copy y into it.
                   ;; This gives us the non-destructive semantics required by
                   ;; the morphism model while still calling an in-place kernel.
                   (result-data (allocate-typed-vector dtype-x N))
                   (fn   (%select-kernel *active-backend* dtype-x
                                          blas-backend-axpy-f64
                                          blas-backend-axpy-f32)))
              (unless fn
                (error "%blas-axpy: active backend has no AXPY kernel for dtype"
                       dtype-x
                       (blas-backend-name *active-backend*)))
              ;; Copy y into result (stride-aware; y may not be contiguous)
              (do ((i 0 (+ i 1)))
                  ((= i N))
                (typed-vector-set! result-data dtype-x i
                  (typed-vector-ref data-y dtype-x (+ offset-y (* i sy)))))
              ;; Normalized call: (N alpha data-x data-result) -> void
              (fn N alpha data-x result-data)
              (concrete-array result-data shape-x
                              (compute-strides shape-x)
                              0 dtype-x -1 -1)))
          (else (error "%blas-axpy: y must be a concrete array"))))
      (else (error "%blas-axpy: x must be a concrete array"))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; High-Level Morphism-Expr Dispatch
  ;;; Called by the realization engine after operand realization.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (execute-blas-operation morphism)
    "Attempt to execute morphism via a registered BLAS backend.

    Returns a concrete-array when successful; #f when the morphism is not
    BLAS-compatible (e.g. abstract operands, wrong op, unsupported dtype,
    no backend registered, or BLAS globally disabled).

    Intended calling convention from realize-morphism-expr:
      1. Realize all operands.
      2. Construct a fresh morphism-expr with concrete operands.
      3. Call (execute-blas-operation updated-expr).
      4. If #f, fall back to standard execute-index-fn path."
    (let ((info (blas-compatible-operation? morphism)))
      (if info
          (let* ((blas-op  (car info))
                 (ops      (cdr info)))
            (case blas-op
              ((gemm)        (execute-blas-gemm         (car ops) (cadr ops)))
              ((gemm-strided)(execute-blas-gemm-strided (car ops) (cadr ops)))
              ((gemv)        (execute-blas-gemv    (car ops) (cadr ops)))
              ((dot)         (execute-blas-dot     (car ops) (cadr ops)))
              (else          #f)))
          #f)))

) ;; end module array-morphisms-blas-exec
