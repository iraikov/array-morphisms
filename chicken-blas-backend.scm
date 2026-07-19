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
  (import (chicken foreign))
  (import srfi-4)
  (import blas)                       ; Chicken 5 'blas' egg
  (import array-morphisms-blas-exec)  ; for make-blas-backend and register-blas-backend!

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Optional C hot-kernel declarations
  ;;; These routines live in kernels/im2col.c and are linked into the final
  ;;; compiled program.  They are only available in compiled mode
  ;;; (foreign-lambda); the Scheme fallback path is always present.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (foreign-declare "extern void im2col_batched_mr_f32(float* col, const float* src, int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW);")
  (foreign-declare "extern void bias_add_f32(float* out, const float* b, int M, int out_ch);")
  (foreign-declare "extern void col2im_batched_mr_f32(float* dx, const float* col, int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW);")

  (define %c-im2col-batched-mr-f32
    (foreign-lambda void "im2col_batched_mr_f32"
      f32vector f32vector int int int int int int int int int int int int))

  (define %c-bias-add-f32
    (foreign-lambda void "bias_add_f32" f32vector f32vector int int))

  (define %c-col2im-batched-mr-f32
    (foreign-lambda void "col2im_batched_mr_f32"
      f32vector f32vector int int int int int int int int int int int int))

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
  ;;; Convolution hot-kernels (optional C path)
  ;;;
  ;;; conv-fwd:  out bias col src wt M N K out-ch Nbatch C H W KH KW SH SW PH PW OH OW
  ;;;   im2col(src->col), gemm(col,wt->out, beta=0), then bias-add.
  ;;;
  ;;; conv-bwd-data: dx col g wt M K N out-ch Nbatch C H W KH KW SH SW PH PW OH OW
  ;;;   col = g @ wt^T  (gemm with transposed B), then col2im(col->dx).
  ;;;
  ;;; conv-bwd-weights: dwt col src g fan-in out-ch M Nbatch C H W KH KW SH SW PH PW OH OW
  ;;;   im2col(src->col), dwt = col^T @ g.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%egg-conv-fwd-im2col-f32 out bias col src wt M N K out-ch Nbatch C H W KH KW SH SW PH PW OH OW)
    (%c-im2col-batched-mr-f32 col src Nbatch C H W KH KW SH SW PH PW OH OW)
    (%egg-sgemm M N K 1.0 col wt 0.0 out)
    (%c-bias-add-f32 out bias M out-ch))

  (define (%egg-conv-bwd-data-im2col-f32 dx col g wt M K N out-ch Nbatch C H W KH KW SH SW PH PW OH OW)
    ;; g is [M, N]=[M,out-ch], wt is [K,N]=[fan-in,out-ch].
    ;; Compute col[M,K] = g * wt^T, then col2im(col)->dx.
    (%egg-sgemm-strided M K N
                        1.0 g N 'no-trans
                        wt N 'trans
                        0.0 col)
    (%c-col2im-batched-mr-f32 dx col Nbatch C H W KH KW SH SW PH PW OH OW))

  (define (%egg-conv-bwd-weights-im2col-f32 dwt col src g fan-in out-ch M Nbatch C H W KH KW SH SW PH PW OH OW)
    ;; col[fan_in, M] = im2col(src); dwt[fan_in, out_ch] = col^T * g.
    (%c-im2col-batched-mr-f32 col src Nbatch C H W KH KW SH SW PH PW OH OW)
    (%egg-sgemm-strided fan-in out-ch M
                        1.0 col fan-in 'trans
                        g out-ch 'no-trans
                        0.0 dwt))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Public Constructor
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (make-blas-egg-backend)
    "Construct a blas-backend record that wraps the Chicken 'blas' egg.

    All eight kernel slots are populated; both f64 and f32 variants are
    provided via the egg's d* and s* routines respectively.  When the C
    hot-kernel object is linked, the conv-* hot-kernel slots are also
    populated.

    Usage:
      (import array-morphisms-blas-egg-backend)
      (register-blas-backend! (make-blas-egg-backend))"
    (make-blas-backend
     'chicken-blas-egg
     %egg-dgemm          %egg-sgemm           ; gemm-f64          gemm-f32
     %egg-dgemm-strided  %egg-sgemm-strided   ; gemm-strided-f64  gemm-strided-f32
     %egg-dgemv          %egg-sgemv           ; gemv-f64          gemv-f32
     %egg-ddot           %egg-sdot            ; dot-f64           dot-f32
     %egg-daxpy          %egg-saxpy           ; axpy-f64          axpy-f32
     %egg-conv-fwd-im2col-f32
     %egg-conv-bwd-data-im2col-f32
     %egg-conv-bwd-weights-im2col-f32))

) ;; end module array-morphisms-blas-egg-backend
