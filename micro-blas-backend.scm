;;; micro-blas-backend.scm
;;; Adapter: bridges the vendored, header-only microBLAS library
;;; (kernels/microBLAS.h, kernels/microblas_shim.c) into the normalized
;;; kernel interface expected by array-morphisms-blas-exec.
;;;
;;; Unlike array-morphisms-blas-egg-backend (which requires the Chicken
;;; 'blas' egg / OpenBLAS to be installed), this backend has no external
;;; dependency beyond libm: microBLAS.h is vendored and compiled directly
;;; into this egg.  It is intended as the dependency-free *default*
;;; backend -- see array-morphisms-realization.scm for the auto-
;;; registration bootstrap.  The system-BLAS backend remains available and,
;;; when registered, takes priority as the higher-performance opt-in tier.
;;;
;;; microBLAS's own gemm/gemv have no lda or transpose parameters at all
;;; (its Matrix type is {data, rows, cols} with an implicit contiguous row
;;; stride).  The C shim in kernels/microblas_shim.c therefore repacks any
;;; transposed or non-default-lda operand into a contiguous scratch buffer
;;; before calling into microBLAS's real cache-blocked gemm kernel; see that
;;; file for details.  Every real call site in this codebase passes
;;; alpha=1.0, so the common case never triggers a copy.
;;;
;;;   (import array-morphisms-micro-blas-backend)
;;;   (register-blas-backend! (make-micro-blas-backend))

(module array-morphisms-micro-blas-backend

  (make-micro-blas-backend)

  (import scheme (chicken base))
  (import (chicken foreign))
  (import srfi-4)
  (import array-morphisms-blas-exec)  ; for make-blas-backend and register-blas-backend!

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; C hot-kernel declarations
  ;;; microblas_shim.c is compiled twice (f32 default, f64 via
  ;;; -DREAL_TYPE_DOUBLE) producing mb_s*/mb_d* symbol pairs; im2col.c's
  ;;; kernels are f32-only and shared with the system-BLAS egg backend.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (foreign-declare "extern int mb_sgemm(int M, int N, int K, float alpha, const float *A, int lda, int transA, const float *B, int ldb, int transB, float beta, float *C);")
  (foreign-declare "extern int mb_dgemm(int M, int N, int K, double alpha, const double *A, int lda, int transA, const double *B, int ldb, int transB, double beta, double *C);")
  (foreign-declare "extern int mb_sgemv(int M, int N, float alpha, const float *A, const float *x, float beta, float *y);")
  (foreign-declare "extern int mb_dgemv(int M, int N, double alpha, const double *A, const double *x, double beta, double *y);")
  (foreign-declare "extern float mb_sdot(int N, const float *x, const float *y);")
  (foreign-declare "extern double mb_ddot(int N, const double *x, const double *y);")
  (foreign-declare "extern int mb_saxpy(int N, float alpha, const float *x, float *y);")
  (foreign-declare "extern int mb_daxpy(int N, double alpha, const double *x, double *y);")

  (foreign-declare "extern void im2col_batched_mr_f32(float* col, const float* src, int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW);")
  (foreign-declare "extern void bias_add_f32(float* out, const float* b, int M, int out_ch);")
  (foreign-declare "extern void col2im_batched_mr_f32(float* dx, const float* col, int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW);")

  (define %c-sgemm
    (foreign-lambda int "mb_sgemm" int int int float f32vector int int f32vector int int float f32vector))
  (define %c-dgemm
    (foreign-lambda int "mb_dgemm" int int int double f64vector int int f64vector int int double f64vector))
  (define %c-sgemv
    (foreign-lambda int "mb_sgemv" int int float f32vector f32vector float f32vector))
  (define %c-dgemv
    (foreign-lambda int "mb_dgemv" int int double f64vector f64vector double f64vector))
  (define %c-sdot
    (foreign-lambda float "mb_sdot" int f32vector f32vector))
  (define %c-ddot
    (foreign-lambda double "mb_ddot" int f64vector f64vector))
  (define %c-saxpy
    (foreign-lambda int "mb_saxpy" int float f32vector f32vector))
  (define %c-daxpy
    (foreign-lambda int "mb_daxpy" int double f64vector f64vector))

  (define %c-im2col-batched-mr-f32
    (foreign-lambda void "im2col_batched_mr_f32"
      f32vector f32vector int int int int int int int int int int int int))

  (define %c-bias-add-f32
    (foreign-lambda void "bias_add_f32" f32vector f32vector int int))

  (define %c-col2im-batched-mr-f32
    (foreign-lambda void "col2im_batched_mr_f32"
      f32vector f32vector int int int int int int int int int int int int))

  (define (%check-mb-error! who err)
    (unless (= err 0)
      (error (string-append who ": microBLAS kernel failed") err))
    err)

  (define (%trans->int t)
    (if (eq? t 'trans) 1 0))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; GEMM kernels
  ;;; Normalized signature: (M N K alpha data-A data-B beta data-C) -> void
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%mb-sgemm M N K alpha data-A data-B beta data-C)
    ;; Row-major, no-trans/no-trans: lda=K, ldb=N (contiguous fast path).
    (%check-mb-error! "gemm-f32"
      (%c-sgemm M N K alpha data-A K 0 data-B N 0 beta data-C)))

  (define (%mb-dgemm M N K alpha data-A data-B beta data-C)
    (%check-mb-error! "gemm-f64"
      (%c-dgemm M N K alpha data-A K 0 data-B N 0 beta data-C)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Strided GEMM kernels
  ;;; Normalized signature:
  ;;;   (M N K alpha data-A lda-A transa data-B ldb-B transb beta data-C) -> void
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%mb-sgemm-strided M N K alpha data-A lda-A transa data-B ldb-B transb beta data-C)
    (%check-mb-error! "gemm-strided-f32"
      (%c-sgemm M N K alpha data-A lda-A (%trans->int transa)
                data-B ldb-B (%trans->int transb) beta data-C)))

  (define (%mb-dgemm-strided M N K alpha data-A lda-A transa data-B ldb-B transb beta data-C)
    (%check-mb-error! "gemm-strided-f64"
      (%c-dgemm M N K alpha data-A lda-A (%trans->int transa)
                data-B ldb-B (%trans->int transb) beta data-C)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; GEMV kernels
  ;;; Normalized signature: (M N alpha data-A data-x beta data-y) -> void
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%mb-sgemv M N alpha data-A data-x beta data-y)
    (%check-mb-error! "gemv-f32" (%c-sgemv M N alpha data-A data-x beta data-y)))

  (define (%mb-dgemv M N alpha data-A data-x beta data-y)
    (%check-mb-error! "gemv-f64" (%c-dgemv M N alpha data-A data-x beta data-y)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; DOT kernels
  ;;; Normalized signature: (N data-x data-y) -> number
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%mb-sdot N data-x data-y) (%c-sdot N data-x data-y))
  (define (%mb-ddot N data-x data-y) (%c-ddot N data-x data-y))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; AXPY kernels
  ;;; Normalized signature: (N alpha data-x data-y) -> void
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%mb-saxpy N alpha data-x data-y)
    (%check-mb-error! "axpy-f32" (%c-saxpy N alpha data-x data-y)))

  (define (%mb-daxpy N alpha data-x data-y)
    (%check-mb-error! "axpy-f64" (%c-daxpy N alpha data-x data-y)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Convolution hot-kernels (f32 only, mirrors chicken-blas-backend.scm)
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

  (define (%mb-conv-fwd-im2col-f32 out bias col src wt M N K out-ch Nbatch C H W KH KW SH SW PH PW OH OW)
    (%c-im2col-batched-mr-f32 col src Nbatch C H W KH KW SH SW PH PW OH OW)
    (%mb-sgemm M N K 1.0 col wt 0.0 out)
    (%c-bias-add-f32 out bias M out-ch))

  (define (%mb-conv-bwd-data-im2col-f32 dx col g wt M K N out-ch Nbatch C H W KH KW SH SW PH PW OH OW)
    ;; g is [M, N]=[M,out-ch], wt is [K,N]=[fan-in,out-ch].
    ;; Compute col[M,K] = g * wt^T, then col2im(col)->dx.
    (%mb-sgemm-strided M K N
                       1.0 g N 'no-trans
                       wt N 'trans
                       0.0 col)
    (%c-col2im-batched-mr-f32 dx col Nbatch C H W KH KW SH SW PH PW OH OW))

  (define (%mb-conv-bwd-weights-im2col-f32 dwt col src g fan-in out-ch M Nbatch C H W KH KW SH SW PH PW OH OW)
    ;; col[fan_in, M] = im2col(src); dwt[fan_in, out_ch] = col^T * g.
    (%c-im2col-batched-mr-f32 col src Nbatch C H W KH KW SH SW PH PW OH OW)
    (%mb-sgemm-strided fan-in out-ch M
                       1.0 col fan-in 'trans
                       g out-ch 'no-trans
                       0.0 dwt))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Public Constructor
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (make-micro-blas-backend)
    "Construct a blas-backend record backed by the vendored, header-only
    microBLAS library (kernels/microBLAS.h).  Dependency-free: requires no
    external BLAS library, only libm.

    Slower than a tuned system BLAS (no SIMD/threading in microBLAS itself),
    but far faster than the pure-Scheme fallback.  Intended as the
    dependency-free default backend; see array-morphisms-realization.scm's
    auto-registration bootstrap.

    Usage:
      (import array-morphisms-micro-blas-backend)
      (register-blas-backend! (make-micro-blas-backend))"
    (make-blas-backend
     'micro-blas
     %mb-dgemm          %mb-sgemm           ; gemm-f64          gemm-f32
     %mb-dgemm-strided  %mb-sgemm-strided   ; gemm-strided-f64  gemm-strided-f32
     %mb-dgemv          %mb-sgemv           ; gemv-f64          gemv-f32
     %mb-ddot           %mb-sdot            ; dot-f64           dot-f32
     %mb-daxpy          %mb-saxpy           ; axpy-f64          axpy-f32
     %mb-conv-fwd-im2col-f32
     %mb-conv-bwd-data-im2col-f32
     %mb-conv-bwd-weights-im2col-f32))

) ;; end module array-morphisms-micro-blas-backend
