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
   blas-backend-conv-fwd-im2col-f32
   blas-backend-conv-bwd-data-im2col-f32
   blas-backend-conv-bwd-weights-im2col-f32
   blas-backend-conv-fwd-nhwc-im2col-f32
   blas-backend-conv-bwd-data-nhwc-im2col-f32
   blas-backend-conv-bwd-weights-nhwc-im2col-f32
   blas-backend-gemm-offset-f64
   blas-backend-gemm-offset-f32
   blas-backend-dot-offset-f64
   blas-backend-dot-offset-f32

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
   morph-batch-matmul
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
   promote-blas-operands

   ;; In-place variants: write result into caller-supplied typed-vector.
   ;; Used by realize-morphism-expr/ctx
   execute-blas-gemm/into!
   execute-blas-gemv/into!

   ;; Strided GEMM: handles transposed zero-copy views via Trans flag
   execute-blas-gemm-strided/into!
   execute-blas-gemm-strided

   ;; Prepared GEMM: operand checks done once, for repeated products
   gemm-plan?
   gemm-plan-uses-kernel?
   prepare-blas-gemm/into
   prepare-blas-gemm-strided/into
   execute-gemm-plan/into!

   ;; Batched GEMM: one prepared plan for all cells of a batched product
   batch-matmul-concrete?
   batched-gemm-plan?
   batched-gemm-plan-cell
   prepare-batched-gemm
   execute-batched-gemm-plan/into!
   execute-blas-batch-matmul/into!

   ;; Re-exported from blas-compat for convenience
   array->gemm-blas-params)

  (import scheme (chicken base))
  (import (only srfi-1 every))
  (import (only (chicken memory) move-memory!))
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
  ;;
  ;; Four optional fields hold GEMM and DOT kernels that read and write
  ;; their operands at element offsets into the vectors:
  ;;   gemm-offset-f64/f32 :
  ;;     (M N K alpha data-A off-A lda transa data-B off-B ldb transb beta
  ;;      data-C off-C) -> void
  ;;   dot-offset-f64/f32 : (N data-x off-x data-y off-y) -> number
  ;; They compute what the strided GEMM kernel and the DOT kernel compute
  ;; for operands stored from those offsets; C is row-major with leading
  ;; dimension N.  A backend without them leaves the fields #f.
  (define-record-type blas-backend
    (%make-blas-backend name
                        gemm-f64 gemm-f32 gemm-strided-f64 gemm-strided-f32
                        gemv-f64 gemv-f32 dot-f64 dot-f32 axpy-f64 axpy-f32
                        conv-fwd-im2col-f32 conv-bwd-data-im2col-f32
                        conv-bwd-weights-im2col-f32
                        conv-fwd-nhwc-im2col-f32 conv-bwd-data-nhwc-im2col-f32
                        conv-bwd-weights-nhwc-im2col-f32
                        gemm-offset-f64 gemm-offset-f32
                        dot-offset-f64 dot-offset-f32)
    blas-backend?
    (name              blas-backend-name)     ; symbol: backend identity
    (gemm-f64          blas-backend-gemm-f64) ; (M N K alpha data-A data-B beta data-C) -> void, or #f
    (gemm-f32          blas-backend-gemm-f32) ; (M N K alpha data-A data-B beta data-C) -> void, or #f
    ;; (M N K alpha data-A lda-A transa data-B ldb-B transb beta data-C) -> void, or #f
    (gemm-strided-f64  blas-backend-gemm-strided-f64)
    (gemm-strided-f32  blas-backend-gemm-strided-f32)
    (gemv-f64          blas-backend-gemv-f64) ; (M N alpha data-A data-x beta data-y) -> void, or #f
    (gemv-f32          blas-backend-gemv-f32)
    (dot-f64           blas-backend-dot-f64)  ; (N data-x data-y) -> number, or #f
    (dot-f32           blas-backend-dot-f32)
    (axpy-f64          blas-backend-axpy-f64) ; (N alpha data-x data-y) -> void, or #f
    (axpy-f32          blas-backend-axpy-f32)
    ;; (out bias col src weight M N K out-ch Nbatch C H W KH KW SH SW PH PW OH OW) -> void, or #f
    (conv-fwd-im2col-f32         blas-backend-conv-fwd-im2col-f32)
    ;; (dx col g wt M K N out-ch Nbatch C H W KH KW SH SW PH PW OH OW) -> void, or #f
    (conv-bwd-data-im2col-f32    blas-backend-conv-bwd-data-im2col-f32)
    ;; (dwt col src g fan-in out-ch M Nbatch C H W KH KW SH SW PH PW OH OW) -> void, or #f
    (conv-bwd-weights-im2col-f32 blas-backend-conv-bwd-weights-im2col-f32)
    ;; col is a scratch buffer of M * C*KH*KW elements that other calls also
    ;; use, so a hook writes all of it before reading it, zeros included.
    ;; NHWC counterparts: identical signatures to the slots above, but the
    ;; im2col/col2im step reads/writes src/dx in NHWC layout [N,H,W,C]
    ;; instead of NCHW. The col/weight/output layouts are unchanged (always
    ;; the layout-independent matmul-ready [.., C*KH*KW] / [.., out_ch] form).
    (conv-fwd-nhwc-im2col-f32         blas-backend-conv-fwd-nhwc-im2col-f32)
    (conv-bwd-data-nhwc-im2col-f32    blas-backend-conv-bwd-data-nhwc-im2col-f32)
    (conv-bwd-weights-nhwc-im2col-f32 blas-backend-conv-bwd-weights-nhwc-im2col-f32)
    (gemm-offset-f64   blas-backend-gemm-offset-f64)
    (gemm-offset-f32   blas-backend-gemm-offset-f32)
    (dot-offset-f64    blas-backend-dot-offset-f64)
    (dot-offset-f32    blas-backend-dot-offset-f32))

  (define (make-blas-backend name
                             gemm-f64 gemm-f32 gemm-strided-f64 gemm-strided-f32
                             gemv-f64 gemv-f32 dot-f64 dot-f32 axpy-f64 axpy-f32
                             conv-fwd-im2col-f32 conv-bwd-data-im2col-f32
                             conv-bwd-weights-im2col-f32
                             conv-fwd-nhwc-im2col-f32 conv-bwd-data-nhwc-im2col-f32
                             conv-bwd-weights-nhwc-im2col-f32
                             #!optional (gemm-offset-f64 #f) (gemm-offset-f32 #f)
                             (dot-offset-f64 #f) (dot-offset-f32 #f))
    "Make a blas-backend record from its kernels, in the order of the fields
    above; each is a procedure or #f.  The offset GEMM and DOT kernels are
    optional."
    (%make-blas-backend name
                        gemm-f64 gemm-f32 gemm-strided-f64 gemm-strided-f32
                        gemv-f64 gemv-f32 dot-f64 dot-f32 axpy-f64 axpy-f32
                        conv-fwd-im2col-f32 conv-bwd-data-im2col-f32
                        conv-bwd-weights-im2col-f32
                        conv-fwd-nhwc-im2col-f32 conv-bwd-data-nhwc-im2col-f32
                        conv-bwd-weights-nhwc-im2col-f32
                        gemm-offset-f64 gemm-offset-f32
                        dot-offset-f64 dot-offset-f32))

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
          cuda-dgemm cuda-sgemm cuda-conv-fwd-im2col-f32
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
    (let loop ((i 0) (n 1))
      (if (= i (vector-length shape))
          (>= n *blas-size-threshold*)
          (loop (+ i 1) (* n (vector-ref shape i))))))

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
     (if (batched-matmul-ranks? (vector-length sa) (vector-length sb))
      (morph-batch-matmul A B)
      (begin
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
         -1))))))

  ;; True when operands of ranks ra and rb make a batched product: both of
  ;; rank 2 or more, and one of rank 3 or more.
  (define (batched-matmul-ranks? ra rb)
    (and (>= ra 2) (>= rb 2) (or (> ra 2) (> rb 2))))

  (define (morph-batch-matmul A B)
    "Batched matrix multiply morphism: F ++ (M,K) x F ++ (K,N) -> F ++ (M,N).

    The leading dimensions F of an operand of rank 3 or more are its frame;
    each cell of the frame is an M x K (or K x N) matrix, and the result
    holds the product of the corresponding cells.  One operand may be a
    plain matrix, which is then shared by every cell (B shared: frame form
    a; A shared: frame form b; both framed: frame form both).  The frames of
    two framed operands must be equal.  Two matrices give the plain matmul
    morphism of morph-matmul.

    Creates a morphism-expr with op 'batch-matmul, an identity-fn
    placeholder, the promoted dtype of the operands, batch-axis 0, and the
    metadata k-dim, frame (a vector) and frame-form."
    (unless (array-morphism? A) (error "morph-batch-matmul: A must be a morphism" A))
    (unless (array-morphism? B) (error "morph-batch-matmul: B must be a morphism" B))
    (let* ((sa (get-morphism-shape A))
           (sb (get-morphism-shape B))
           (ra (vector-length sa))
           (rb (vector-length sb)))
      (unless (and (>= ra 2) (>= rb 2))
        (error "morph-batch-matmul: operands must have rank 2 or more" sa sb))
      (if (and (= ra 2) (= rb 2))
          (morph-matmul A B)
          (let* ((fa    (subvector sa 0 (- ra 2)))
                 (fb    (subvector sb 0 (- rb 2)))
                 (form  (cond ((= rb 2) 'a) ((= ra 2) 'b) (else 'both)))
                 (frame (if (eq? form 'b) fb fa))
                 (M     (vector-ref sa (- ra 2)))
                 (K     (vector-ref sa (- ra 1)))
                 (K-b   (vector-ref sb (- rb 2)))
                 (N     (vector-ref sb (- rb 1))))
            (when (and (eq? form 'both) (not (equal? fa fb)))
              (error "morph-batch-matmul: frames must be equal" fa fb))
            (unless (= K K-b)
              (error "morph-batch-matmul: inner dimensions must match" K K-b))
            (morphism-expr
             (gensym 'morph-) 'batch-matmul (list A B)
             (identity-fn)
             (vector-append frame (vector M N))
             (promote-types (get-morphism-dtype A) (get-morphism-dtype B))
             `((k-dim . ,K) (frame . ,frame) (frame-form . ,form))
             0)))))

  (define (vector-append u v)
    (list->vector (append (vector->list u) (vector->list v))))

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

;; -- Prepared GEMM ---------------------------------------------------------
  ;; execute-blas-gemm/into! and execute-blas-gemm-strided/into! check their
  ;; operands on every call.  A caller that multiplies arrays of the same
  ;; shape, strides and dtype many times, such as a compiled replay plan,
  ;; can instead check them once with prepare-blas-gemm/into or
  ;; prepare-blas-gemm-strided/into and then run each product with
  ;; execute-gemm-plan/into!.
  ;;
  ;; A plan records the decision the corresponding execute-blas-* procedure
  ;; would make for operands of that layout: the backend and kernel to use,
  ;; with the dimensions and, for the strided kernel, the Trans flags and
  ;; leading dimensions.  When that decision was not to use a backend
  ;; kernel, the kernel field is #f.  A plan is used only while BLAS is
  ;; enabled and the backend it was made for is still the active one;
  ;; otherwise execute-gemm-plan/into! falls back to the execute-blas-*
  ;; procedure, which checks the operands again.  Either way the result is
  ;; the one that procedure would compute.

  (define-record gemm-plan
    backend   ; the blas-backend the plan was made for, or #f
    kernel    ; gemm or gemm-strided kernel of that backend, or #f
    strided?  ; #t when made by prepare-blas-gemm-strided/into
    M N K
    lda transa ldb transb)   ; strided kernel arguments; unused otherwise

  (define (gemm-plan-uses-kernel? plan)
    "True when plan calls a backend GEMM kernel; false when its products
    are computed by the execute-blas-* procedure it falls back to."
    (and (gemm-plan-kernel plan) #t))

  (define (prepare-blas-gemm/into A B)
    "Check A and B once for execute-gemm-plan/into!.  Returns a gemm-plan
    that computes what (execute-blas-gemm/into! A B result-data) computes,
    for A and B and for any later arrays with the same shapes, strides and
    dtype."
    (let ((kernel
           (and *blas-enabled*
                *active-backend*
                (blas-compatible-matmul? A B)
                (>= (* (vector-ref (get-morphism-shape A) 0)
                       (vector-ref (get-morphism-shape B) 1))
                    *blas-size-threshold*)
                (%select-kernel *active-backend* (get-morphism-dtype A)
                                blas-backend-gemm-f64
                                blas-backend-gemm-f32))))
      (if kernel
          (let ((sA (get-morphism-shape A))
                (sB (get-morphism-shape B)))
            (make-gemm-plan *active-backend* kernel #f
                            (vector-ref sA 0) (vector-ref sB 1) (vector-ref sA 1)
                            0 'no-trans 0 'no-trans))
          (make-gemm-plan *active-backend* #f #f 0 0 0 0 'no-trans 0 'no-trans))))

  (define (prepare-blas-gemm-strided/into A B)
    "Check A and B once for execute-gemm-plan/into!.  Returns a gemm-plan
    that computes what (execute-blas-gemm-strided/into! A B result-data)
    computes, for A and B and for any later arrays with the same shapes,
    strides and dtype."
    (let* ((params-A (array->gemm-blas-params A))
           (params-B (array->gemm-blas-params B))
           (kernel
            (and *blas-enabled*
                 *active-backend*
                 (>= (* (vector-ref (get-morphism-shape A) 0)
                        (vector-ref (get-morphism-shape B) 1))
                     *blas-size-threshold*)
                 params-A params-B
                 (concrete-array? A) (concrete-array? B)
                 (%select-kernel *active-backend* (get-morphism-dtype A)
                                 blas-backend-gemm-strided-f64
                                 blas-backend-gemm-strided-f32))))
      (if kernel
          (let ((sA (get-morphism-shape A))
                (sB (get-morphism-shape B)))
            (make-gemm-plan *active-backend* kernel #t
                            (vector-ref sA 0) (vector-ref sB 1) (vector-ref sA 1)
                            (caddr params-A) (cadr params-A)
                            (caddr params-B) (cadr params-B)))
          (make-gemm-plan *active-backend* #f #t 0 0 0 0 'no-trans 0 'no-trans))))

  (define (execute-gemm-plan/into! plan A B result-data)
    "Multiply A by B into result-data as prepared by plan.  A and B must
    have the shapes, strides and dtype of the arrays the plan was made for;
    they are not checked again."
    (let ((kernel (gemm-plan-kernel plan)))
      (if (and kernel
               *blas-enabled*
               (eq? *active-backend* (gemm-plan-backend plan)))
          (cases array-morphism A
            (concrete-array (data-A _ _ _ _ _ _)
              (cases array-morphism B
                (concrete-array (data-B _ _ _ _ _ _)
                  (if (gemm-plan-strided? plan)
                      (kernel (gemm-plan-M plan) (gemm-plan-N plan) (gemm-plan-K plan) 1.0
                              data-A (gemm-plan-lda plan) (gemm-plan-transa plan)
                              data-B (gemm-plan-ldb plan) (gemm-plan-transb plan)
                              0.0 result-data)
                      (kernel (gemm-plan-M plan) (gemm-plan-N plan) (gemm-plan-K plan) 1.0
                              data-A data-B 0.0 result-data)))
                (else (error "execute-gemm-plan/into!: B must be a concrete array" B))))
            (else (error "execute-gemm-plan/into!: A must be a concrete array" A)))
          (if (gemm-plan-strided? plan)
              (execute-blas-gemm-strided/into! A B result-data)
              (execute-blas-gemm/into! A B result-data)))))

  ;; -- Batched GEMM ---------------------------------------------------------
  ;; A batched product multiplies the cells of framed operands (see
  ;; morph-batch-matmul).  Each cell is computed as the 2-D product of that
  ;; cell stored on its own would be: the cell's gemm-plan is prepared from
  ;; an array with the cell's shape and strides that starts at the cell's
  ;; first element.  A cell whose rows (or columns) are contiguous but lie
  ;; further apart than its width (or height), as when frame axes come
  ;; between its axes, is a padded cell: its own array is the contiguous
  ;; one, in the same orientation, and kernels read it in place with its
  ;; actual leading dimension.  When that plan calls a kernel of a backend that has an
  ;; offset GEMM kernel, the kernel reads the cell in place.  Otherwise the
  ;; cell is copied, strides and all, into a scratch buffer, the plan runs
  ;; on the copy, and the product is copied into place.

  (define-record batched-gemm-plan
    cell        ; gemm-plan of one cell
    dtype
    M N
    a-shared?   ; #t when A is a plain matrix used by every cell
    b-shared?
    a-layout    ; layout array of an A cell copy (offset = -lo), or A itself when shared
    b-layout
    a-lo b-lo   ; offset of a cell's lowest element from its first element
    a-span b-span ; elements spanned by one cell
    a-offs b-offs ; vector: offset of each cell's first element from the array's offset
    a-pad b-pad   ; #f, or (rows . ld) or (cols . ld) for a padded cell
    scratch)    ; #f, or vector of the scratch buffers for A, B and C

  (define (dtype-bytes dtype)
    (if (eq? dtype 'f64) 8 4))

  ;; The lowest and highest offset, relative to its first element, that a
  ;; matrix of the given shape and strides reads.
  (define (matrix-extent shape strides)
    (let loop ((i 0) (lo 0) (hi 0))
      (if (= i (vector-length shape))
          (values lo hi)
          (let ((n (vector-ref shape i)) (st (vector-ref strides i)))
            (if (= n 0)
                (values 0 -1)
                (let ((d (* (- n 1) st)))
                  (loop (+ i 1) (+ lo (min 0 d)) (+ hi (max 0 d)))))))))

  ;; The offsets of the cells of frame, in row-major order, for frame
  ;; strides fstrides.
  (define (frame-offsets frame fstrides)
    (let* ((r (vector-length frame))
           (n (let loop ((i 0) (n 1))
                (if (= i r) n (loop (+ i 1) (* n (vector-ref frame i))))))
           (offs (make-vector n 0))
           (idx  (make-vector r 0)))
      (do ((c 0 (+ c 1)))
          ((= c n) offs)
        (let loop ((i 0) (o 0))
          (if (= i r)
              (vector-set! offs c o)
              (loop (+ i 1) (+ o (* (vector-ref idx i) (vector-ref fstrides i))))))
        (let carry ((i (- r 1)))
          (when (>= i 0)
            (let ((v (+ 1 (vector-ref idx i))))
              (if (< v (vector-ref frame i))
                  (vector-set! idx i v)
                  (begin (vector-set! idx i 0) (carry (- i 1))))))))))

  (define (batch-matmul-concrete? A B)
    "True when concrete arrays A and B, of rank 2 or more and one of rank 3
    or more, have one dtype, f32 or f64, and shapes that morph-batch-matmul
    accepts."
    (and (concrete-array? A) (concrete-array? B)
         (let* ((sa (get-morphism-shape A)) (sb (get-morphism-shape B))
                (ra (vector-length sa)) (rb (vector-length sb))
                (da (get-morphism-dtype A)))
           (and (batched-matmul-ranks? ra rb)
                (memq da '(f32 f64))
                (eq? da (get-morphism-dtype B))
                (= (vector-ref sa (- ra 1)) (vector-ref sb (- rb 2)))
                (or (= ra 2) (= rb 2)
                    (equal? (subvector sa 0 (- ra 2)) (subvector sb 0 (- rb 2))))))))

  (define (prepare-batched-gemm A B)
    "Check concrete arrays A and B once for execute-batched-gemm-plan/into!.
    Returns a batched-gemm-plan that multiplies A by B, and any later
    arrays with the same shapes, strides and dtype, cell by cell; or #f
    when batch-matmul-concrete? does not hold.  Only the shapes, strides,
    offsets and dtypes of A and B are read."
    (and
     (batch-matmul-concrete? A B)
     (let ()
       (define (cell-of m)
         ;; (shared? layout lo span fstrides frame pad) for operand m.
         (cases array-morphism m
           (concrete-array (_ shape strides offset dtype _ _)
             (let ((r (vector-length shape)))
               (if (= r 2)
                   (list #t m 0 0 #f #f #f)
                   (let* ((cs  (subvector shape (- r 2) r))
                          (cst (subvector strides (- r 2) r))
                          (R   (vector-ref cs 0)) (C (vector-ref cs 1))
                          (s0  (vector-ref cst 0)) (s1 (vector-ref cst 1))
                          (pad (cond ((and (= s1 1) (> R 1) (> s0 C)) (cons 'rows s0))
                                     ((and (= s0 1) (> C 1) (> s1 R)) (cons 'cols s1))
                                     (else #f)))
                          (fst (subvector strides 0 (- r 2)))
                          (frm (subvector shape 0 (- r 2))))
                     (if pad
                         (list #f
                               (concrete-array (allocate-typed-vector dtype 0) cs
                                               (if (eq? (car pad) 'rows) (vector C 1) (vector 1 R))
                                               0 dtype -1 -1)
                               0 (* R C) fst frm pad)
                         (let-values (((lo hi) (matrix-extent cs cst)))
                           (list #f
                                 (concrete-array (allocate-typed-vector dtype 0) cs cst (- lo)
                                                 dtype -1 -1)
                                 lo (max 0 (+ 1 (- hi lo)))
                                 fst frm #f)))))))
           (else (error "prepare-batched-gemm: operand not concrete" m))))
       (let* ((a  (cell-of A))
              (b  (cell-of B))
              (al (cadr a))
              (bl (cadr b))
              (frame (or (list-ref a 5) (list-ref b 5)))
              (cell (cond ((blas-compatible-matmul? al bl) (prepare-blas-gemm/into al bl))
                          (else (prepare-blas-gemm-strided/into al bl))))
              (dtype (get-morphism-dtype A)))
         (make-batched-gemm-plan
          cell dtype
          (vector-ref (get-morphism-shape al) 0) (vector-ref (get-morphism-shape bl) 1)
          (car a) (car b) al bl
          (caddr a) (caddr b) (cadddr a) (cadddr b)
          (if (car a) #f (frame-offsets frame (list-ref a 4)))
          (if (car b) #f (frame-offsets frame (list-ref b 4)))
          (list-ref a 6) (list-ref b 6)
          #f)))))

  (define (batched-gemm-plan-cells plan)
    (vector-length (or (batched-gemm-plan-a-offs plan) (batched-gemm-plan-b-offs plan))))

  ;; The offset GEMM kernel that runs a cell of plan in place, or #f.
  (define (batched-offset-kernel plan)
    (let* ((cell    (batched-gemm-plan-cell plan))
           (backend (gemm-plan-backend cell)))
      (and (gemm-plan-kernel cell)
           *blas-enabled*
           (eq? *active-backend* backend)
           (%select-kernel backend (batched-gemm-plan-dtype plan)
                           blas-backend-gemm-offset-f64
                           blas-backend-gemm-offset-f32))))

  (define (batched-scratch plan)
    (or (batched-gemm-plan-scratch plan)
        (let* ((dt (batched-gemm-plan-dtype plan))
               (s  (vector (allocate-typed-vector dt (batched-gemm-plan-a-span plan))
                           (allocate-typed-vector dt (batched-gemm-plan-b-span plan))
                           (allocate-typed-vector dt (* (batched-gemm-plan-M plan)
                                                        (batched-gemm-plan-N plan))))))
          (batched-gemm-plan-scratch-set! plan s)
          s)))

  (define (execute-batched-gemm-plan/into! plan A B result-data)
    "Multiply A by B cell by cell into result-data, row-major with the
    frame dimensions first, as prepared by plan.  A and B must have the
    shapes, strides and dtype of the arrays the plan was made for."
    (let* ((n      (batched-gemm-plan-cells plan))
           (cell   (batched-gemm-plan-cell plan))
           (M      (batched-gemm-plan-M plan))
           (N      (batched-gemm-plan-N plan))
           (MN     (* M N))
           (kernel (and (> n 0) (> MN 0) (batched-offset-kernel plan))))
      (cases array-morphism A
        (concrete-array (data-A _ _ off-A _ _ _)
          (cases array-morphism B
            (concrete-array (data-B _ _ off-B _ _ _)
              (cond
               ((or (= n 0) (= MN 0)) (void))
               (kernel
                (let ((a-offs (batched-gemm-plan-a-offs plan))
                      (b-offs (batched-gemm-plan-b-offs plan))
                      (K      (gemm-plan-K cell)))
                  (let*-values (((lda ta ldb tb)
                                 (if (gemm-plan-strided? cell)
                                     (values (gemm-plan-lda cell) (gemm-plan-transa cell)
                                             (gemm-plan-ldb cell) (gemm-plan-transb cell))
                                     (values K 'no-trans N 'no-trans)))
                                ((a-pad) (batched-gemm-plan-a-pad plan))
                                ((b-pad) (batched-gemm-plan-b-pad plan))
                                ((lda) (if a-pad (cdr a-pad) lda))
                                ((ldb) (if b-pad (cdr b-pad) ldb)))
                    (do ((c 0 (+ c 1)))
                        ((= c n))
                      (kernel M N K 1.0
                              data-A (if a-offs (+ off-A (vector-ref a-offs c)) off-A) lda ta
                              data-B (if b-offs (+ off-B (vector-ref b-offs c)) off-B) ldb tb
                              0.0 result-data (* c MN))))))
               (else
                (let* ((dt      (batched-gemm-plan-dtype plan))
                       (bytes   (dtype-bytes dt))
                       (scratch (batched-scratch plan))
                       (c-buf   (vector-ref scratch 2)))
                  ;; The cell of operand m at cell index c, as an array with
                  ;; the plan's cell layout.  A padded cell is packed row by
                  ;; row (or column by column).
                  (define (cell-array m data off shared? layout lo span offs buf c pad)
                    (if shared?
                        m
                        (let ((first (+ off (vector-ref offs c) lo)))
                          (cases array-morphism layout
                            (concrete-array (_ cs cst loff _ _ _)
                              (cond
                               (pad
                                (let* ((rows? (eq? (car pad) 'rows))
                                       (n     (vector-ref cs (if rows? 0 1)))
                                       (len   (vector-ref cs (if rows? 1 0))))
                                  (do ((i 0 (+ i 1))) ((= i n))
                                    (move-memory! data buf (* len bytes)
                                                  (* (+ first (* i (cdr pad))) bytes)
                                                  (* i len bytes)))
                                  (concrete-array buf cs cst 0 dt -1 -1)))
                               ((= first 0)
                                (concrete-array data cs cst loff dt -1 -1))
                               (else
                                (move-memory! data buf (* span bytes) (* first bytes) 0)
                                (concrete-array buf cs cst loff dt -1 -1))))
                            (else m)))))
                  (do ((c 0 (+ c 1)))
                      ((= c n))
                    (let ((a (cell-array A data-A off-A (batched-gemm-plan-a-shared? plan)
                                         (batched-gemm-plan-a-layout plan)
                                         (batched-gemm-plan-a-lo plan)
                                         (batched-gemm-plan-a-span plan)
                                         (batched-gemm-plan-a-offs plan)
                                         (vector-ref scratch 0) c
                                         (batched-gemm-plan-a-pad plan)))
                          (b (cell-array B data-B off-B (batched-gemm-plan-b-shared? plan)
                                         (batched-gemm-plan-b-layout plan)
                                         (batched-gemm-plan-b-lo plan)
                                         (batched-gemm-plan-b-span plan)
                                         (batched-gemm-plan-b-offs plan)
                                         (vector-ref scratch 1) c
                                         (batched-gemm-plan-b-pad plan))))
                      (if (= c 0)
                          (execute-gemm-plan/into! cell a b result-data)
                          (begin
                            (execute-gemm-plan/into! cell a b c-buf)
                            (move-memory! c-buf result-data (* MN bytes) 0
                                          (* c MN bytes))))))))))
            (else (error "execute-batched-gemm-plan/into!: B must be a concrete array" B))))
        (else (error "execute-batched-gemm-plan/into!: A must be a concrete array" A)))))

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

  ;; -- Operand dtype promotion -----------------------------------------------
  ;; morph-matmul, morph-matvec and morph-dot give their expression the
  ;; promoted dtype of the two operands, but the GEMM, GEMV and DOT kernels
  ;; read both operands with a single dtype.  When the operand dtypes
  ;; differ, each operand whose dtype is not the expression dtype is copied
  ;; into a contiguous row-major array of that dtype before dispatch.

  (define (%convert-concrete-dtype m dtype)
    "Copy concrete-array m into a new contiguous array of the given dtype."
    (cases array-morphism m
      (concrete-array (data shape strides offset src-dtype alloc-id batch-axis)
        (let* ((n   (shape-size shape))
               (out (allocate-typed-vector dtype n)))
          (do ((i 0 (+ i 1)))
              ((= i n))
            (let ((phys (multi-to-linear-index
                         (linear-to-multi-index i shape) strides offset)))
              (typed-vector-set! out dtype i
                                 (exact->inexact
                                  (typed-vector-ref data src-dtype phys)))))
          (concrete-array out shape (compute-strides shape) 0 dtype -1
                          batch-axis)))
      (else (error "%convert-concrete-dtype: expected a concrete array" m))))

  (define (promote-blas-operands op operands dtype)
    "Return operands with every concrete operand converted to dtype.

    Applies only to the linear-algebra ops matmul, batch-matmul, matvec and
    dot, and only
    when dtype is f32 or f64; other operand lists are returned unchanged.

    Args:
      op:       the morphism-expr op symbol
      operands: list of realized operands
      dtype:    the dtype of the morphism-expr"
    (if (and (memq op '(matmul matvec dot batch-matmul))
             (memq dtype '(f32 f64)))
        (map (lambda (m)
               (if (and (concrete-array? m)
                        (not (eq? (get-morphism-dtype m) dtype)))
                   (%convert-concrete-dtype m dtype)
                   m))
             operands)
        operands))

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
    (let ((info (and (not (eq? (morphism-op morphism) 'batch-matmul))
                     (blas-compatible-operation? morphism))))
      (if (eq? (morphism-op morphism) 'batch-matmul)
          (let* ((ops (morphism-operands morphism))
                 (A   (car ops))
                 (B   (cadr ops))
                 (shape (get-morphism-shape morphism))
                 (dtype (get-morphism-dtype morphism))
                 (result-data (allocate-typed-vector dtype (shape-size shape))))
            (and (execute-blas-batch-matmul/into! A B result-data)
                 (concrete-array result-data shape (compute-strides shape) 0 dtype -1 0)))
      (if info
          (let* ((blas-op  (car info))
                 (ops      (cdr info)))
            (case blas-op
              ((gemm)        (execute-blas-gemm         (car ops) (cadr ops)))
              ((gemm-strided)(execute-blas-gemm-strided (car ops) (cadr ops)))
              ((gemv)        (execute-blas-gemv    (car ops) (cadr ops)))
              ((dot)         (execute-blas-dot     (car ops) (cadr ops)))
              (else          #f)))
          #f))))

  (define (morphism-op m)
    (cases array-morphism m
      (morphism-expr (_ op _ _ _ _ _ _) op)
      (else #f)))

  (define (morphism-operands m)
    (cases array-morphism m
      (morphism-expr (_ _ operands _ _ _ _ _) operands)
      (else '())))

  (define (execute-blas-batch-matmul/into! A B result-data)
    "Multiply concrete arrays A and B cell by cell, as morph-batch-matmul
    describes, into result-data (row-major, frame dimensions first).
    Returns #t, or #f without writing when batch-matmul-concrete? does not
    hold for A and B."
    (let ((plan (prepare-batched-gemm A B)))
      (and plan
           (begin (execute-batched-gemm-plan/into! plan A B result-data) #t))))

) ;; end module array-morphisms-blas-exec
