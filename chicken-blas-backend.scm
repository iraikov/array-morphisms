;;; array-morphisms-blas-egg-backend.scm
;;; Adapter: bridges the Chicken 5 'blas' egg into the normalized kernel
;;; interface expected by array-morphisms-blas-exec.
;;;
;;; This module is intentionally separate from array-morphisms-blas-exec so
;;; that the core framework carries no hard dependency on the 'blas' egg.
;;; Load this module at startup only when the egg is installed:
;;;
;;;   (import array-morphisms-blas-egg-backend)
;;;   (register-blas-backend! (make-blas-egg-backend))
;;;
;;; The Chicken 'blas' egg API used here:
;;;
;;;   Level 3:
;;;     (dgemm! ORDER TRANSA TRANSB M N K ALPHA A B BETA C #:lda K #:ldb N #:ldc N)
;;;     (sgemm! ORDER TRANSA TRANSB M N K ALPHA A B BETA C #:lda K #:ldb N #:ldc N)
;;;
;;;   Level 2:
;;;     (dgemv! ORDER TRANS M N ALPHA A X BETA Y #:lda N)
;;;     (sgemv! ORDER TRANS M N ALPHA A X BETA Y #:lda N)
;;;
;;;   Level 1:
;;;     (ddot  N X Y) -> number
;;;     (sdot  N X Y) -> number
;;;     (daxpy! N ALPHA X Y)   ; in-place on Y
;;;     (saxpy! N ALPHA X Y)   ; in-place on Y
;;;
;;; Notes on LDA defaults:
;;;   The egg's default LDA for GEMM is (if (= TRANSA NoTrans) M K).
;;;   For row-major storage this default is WRONG (it gives M instead of K).
;;;   We always supply #:lda, #:ldb, #:ldc explicitly to avoid silent
;;;   incorrect results with non-square matrices.
;;;   The same issue applies to GEMV: we always supply #:lda N.

(module array-morphisms-blas-egg-backend

  (make-blas-egg-backend)

  (import scheme (chicken base))
  (import blas)                       ; Chicken 5 'blas' egg
  (import array-morphisms-blas-exec)  ; for make-blas-backend and register-blas-backend!

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; GEMM kernels
  ;;; Normalized signature: (M N K alpha data-A data-B beta data-C) -> void
  ;;; data-C is mutated in-place.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%egg-dgemm M N K alpha data-A data-B beta data-C)
    ;; RowMajor, NoTrans A, NoTrans B.
    ;; For row-major M*K matrix A: lda = K (number of columns).
    ;; For row-major K*N matrix B: ldb = N.
    ;; For row-major M*N matrix C: ldc = N.
    (dgemm! RowMajor NoTrans NoTrans M N K
            alpha data-A data-B beta data-C
            #:lda K #:ldb N #:ldc N))

  (define (%egg-sgemm M N K alpha data-A data-B beta data-C)
    (sgemm! RowMajor NoTrans NoTrans M N K
            alpha data-A data-B beta data-C
            #:lda K #:ldb N #:ldc N))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Strided GEMM kernels
  ;;; Normalized signature:
  ;;;   (M N K alpha data-A lda-A transa data-B ldb-B transb beta data-C) -> void
  ;;; transa and transb are Scheme symbols: 'no-trans or 'trans.
  ;;; lda-A / ldb-B are the physical leading dimensions of the underlying
  ;;; row-major buffers (= number of physical columns).
  ;;; ldc is always N: the result C is freshly allocated contiguous row-major.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%egg-dgemm-strided M N K alpha data-A lda-A transa data-B ldb-B transb beta data-C)
    (dgemm! RowMajor
            (if (eq? transa 'trans) Trans NoTrans)
            (if (eq? transb 'trans) Trans NoTrans)
            M N K alpha data-A data-B beta data-C
            #:lda lda-A #:ldb ldb-B #:ldc N))

  (define (%egg-sgemm-strided M N K alpha data-A lda-A transa data-B ldb-B transb beta data-C)
    (sgemm! RowMajor
            (if (eq? transa 'trans) Trans NoTrans)
            (if (eq? transb 'trans) Trans NoTrans)
            M N K alpha data-A data-B beta data-C
            #:lda lda-A #:ldb ldb-B #:ldc N))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; GEMV kernels
  ;;; Normalized signature: (M N alpha data-A data-x beta data-y) -> void
  ;;; data-y is mutated in-place.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%egg-dgemv M N alpha data-A data-x beta data-y)
    ;; RowMajor, NoTrans.
    ;; For row-major M*N matrix A: lda = N.
    ;; incx and incy default to 1 (contiguous vectors).
    (dgemv! RowMajor NoTrans M N
            alpha data-A data-x beta data-y
            #:lda N))

  (define (%egg-sgemv M N alpha data-A data-x beta data-y)
    (sgemv! RowMajor NoTrans M N
            alpha data-A data-x beta data-y
            #:lda N))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; DOT kernels
  ;;; Normalized signature: (N data-x data-y) -> number
  ;;; No mutation; returns a Scheme number.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%egg-ddot N data-x data-y)
    ;; incx and incy default to 1.
    (ddot N data-x data-y))

  (define (%egg-sdot N data-x data-y)
    (sdot N data-x data-y))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; AXPY kernels
  ;;; Normalized signature: (N alpha data-x data-y) -> void
  ;;; data-y is mutated in-place (caller is responsible for pre-copying y).
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%egg-daxpy N alpha data-x data-y)
    ;; incx and incy default to 1.
    (daxpy! N alpha data-x data-y))

  (define (%egg-saxpy N alpha data-x data-y)
    (saxpy! N alpha data-x data-y))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Public Constructor
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (make-blas-egg-backend)
    "Construct a blas-backend record that wraps the Chicken 'blas' egg.

    All eight kernel slots are populated; both f64 and f32 variants are
    provided via the egg's d* and s* routines respectively.

    Usage:
      (import array-morphisms-blas-egg-backend)
      (register-blas-backend! (make-blas-egg-backend))"
    (make-blas-backend
     'chicken-blas-egg
     %egg-dgemm          %egg-sgemm           ; gemm-f64          gemm-f32
     %egg-dgemm-strided  %egg-sgemm-strided   ; gemm-strided-f64  gemm-strided-f32
     %egg-dgemv          %egg-sgemv           ; gemv-f64          gemv-f32
     %egg-ddot           %egg-sdot            ; dot-f64           dot-f32
     %egg-daxpy          %egg-saxpy))         ; axpy-f64          axpy-f32

) ;; end module array-morphisms-blas-egg-backend
