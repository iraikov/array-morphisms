/* microblas_wrapper.c -- thin C wrapper around microBLAS.h,
 * normalizing its API to the (M N K alpha A lda transA B ldb transB beta C)
 * calling convention expected by array-morphisms-blas-exec's blas-backend
 * record (see blas-exec.scm).
 *
 * microBLAS's gemm/gemv have no lda/transpose parameters at all; Matrix is
 * {data, rows, cols} with an implicit contiguous row stride equal to cols.
 * Any transposed or non-default-lda operand is therefore physically
 * repacked into a tightly-packed scratch buffer before calling into
 * microBLAS's real (cache-blocked) gemm/gemv kernel; the scratch buffers are
 * process-global and grow-on-demand, since CHICKEN's execution model is
 * single-threaded and this avoids a malloc/free pair on every call.
 *
 * Compiled twice from this one source file: once as-is (float / mb_s*) and
 * once with -DREAL_TYPE_DOUBLE (double / mb_d*), so both precisions can be
 * linked into the same backend module without symbol collisions.
 */

#include <string.h>
#include "microBLAS.h"

#ifdef REAL_TYPE_DOUBLE
#define MB_(name) mb_d##name
#else
#define MB_(name) mb_s##name
#endif

/* ---- growable scratch buffers (repacking transposed/strided operands) -- */

static RealType *MB_(scratch_a) = NULL;
static size_t MB_(scratch_a_cap) = 0;
static RealType *MB_(scratch_b) = NULL;
static size_t MB_(scratch_b_cap) = 0;
static RealType *MB_(scratch_x) = NULL;
static size_t MB_(scratch_x_cap) = 0;

static RealType *MB_(ensure_scratch)(RealType **buf, size_t *cap, size_t need) {
    if (*cap < need) {
        RealType *p = (RealType *) realloc(*buf, need * sizeof(RealType));
        if (!p) return NULL;
        *buf = p;
        *cap = need;
    }
    return *buf;
}

/* Repack src into a tightly-packed dim0 x dim1 buffer, scaling by alpha.
 * trans == 0: src is physically (dim0 x dim1) with row-stride ld.
 * trans != 0: src is physically (dim1 x dim0) with row-stride ld (so the
 *             logical (dim0 x dim1) view is its transpose).
 */
static void MB_(pack)(RealType *dst, const RealType *src,
                       size_t dim0, size_t dim1, size_t ld, int trans,
                       RealType alpha) {
    size_t i, j;
    if (trans) {
        for (i = 0; i < dim0; i++)
            for (j = 0; j < dim1; j++)
                dst[i * dim1 + j] = alpha * src[j * ld + i];
    } else {
        for (i = 0; i < dim0; i++)
            for (j = 0; j < dim1; j++)
                dst[i * dim1 + j] = alpha * src[i * ld + j];
    }
}

/* ---------------------------------- gemm --------------------------------
 * C[M,N] := alpha * op(A)[M,K] * op(B)[K,N] + beta * C[M,N]
 * op(X) is X if trans==0, X^T if trans!=0 (X physically stored as given by
 * ld, i.e. ld is the physical row-major stride of the *untransposed*
 * physical buffer).
 */
int MB_(gemm)(int Mi, int Ni, int Ki,
              RealType alpha,
              const RealType *A, int ldaI, int transA,
              const RealType *B, int ldbI, int transB,
              RealType beta,
              RealType *C) {
    size_t M = (size_t) Mi, N = (size_t) Ni, K = (size_t) Ki;
    size_t lda = (size_t) ldaI, ldb = (size_t) ldbI;
    int need_copy_a, need_copy_b;
    const RealType *Ap, *Bp;
    Matrix Cm, Am, Bm;
    MBError err;

    if (M == 0 || N == 0) return MB_SUCCESS;

    if (K == 0) {
        size_t n = M * N;
        if (beta == (RealType) 0.0) memset(C, 0, n * sizeof(RealType));
        else if (beta != (RealType) 1.0) { size_t i; for (i = 0; i < n; i++) C[i] *= beta; }
        return MB_SUCCESS;
    }

    need_copy_a = transA || (lda != K);
    need_copy_b = transB || (ldb != N);
    if (!need_copy_a && alpha != (RealType) 1.0) need_copy_a = 1;

    Ap = A;
    Bp = B;

    if (need_copy_a) {
        RealType *buf = MB_(ensure_scratch)(&MB_(scratch_a), &MB_(scratch_a_cap), M * K);
        if (!buf) return MB_ERR_ALLOC;
        MB_(pack)(buf, A, M, K, lda, transA, alpha);
        Ap = buf;
    }
    if (need_copy_b) {
        RealType *buf = MB_(ensure_scratch)(&MB_(scratch_b), &MB_(scratch_b_cap), K * N);
        if (!buf) return MB_ERR_ALLOC;
        MB_(pack)(buf, B, K, N, ldb, transB, (RealType) 1.0);
        Bp = buf;
    }

    Cm.data = C;    Cm.rows = M; Cm.cols = N;
    Am.data = (RealType *) Ap; Am.rows = M; Am.cols = K;
    Bm.data = (RealType *) Bp; Bm.rows = K; Bm.cols = N;

    err = gemm(&Cm, &Am, &Bm, beta);
    return (int) err;
}

/* ---------------------------------- gemv --------------------------------
 * y[M] := alpha * A[M,N] * x[N] + beta * y[M]   (A row-major, no transpose)
 */
int MB_(gemv)(int Mi, int Ni,
              RealType alpha,
              const RealType *A, const RealType *x,
              RealType beta,
              RealType *y) {
    size_t M = (size_t) Mi, N = (size_t) Ni;
    const RealType *xp;
    Matrix Am;
    Vector xv, yv;
    MBError err;

    if (M == 0) return MB_SUCCESS;
    if (N == 0) {
        if (beta == (RealType) 0.0) memset(y, 0, M * sizeof(RealType));
        else if (beta != (RealType) 1.0) { size_t i; for (i = 0; i < M; i++) y[i] *= beta; }
        return MB_SUCCESS;
    }

    xp = x;
    if (alpha != (RealType) 1.0) {
        size_t i;
        RealType *buf = MB_(ensure_scratch)(&MB_(scratch_x), &MB_(scratch_x_cap), N);
        if (!buf) return MB_ERR_ALLOC;
        for (i = 0; i < N; i++) buf[i] = alpha * x[i];
        xp = buf;
    }

    Am.data = (RealType *) A; Am.rows = M; Am.cols = N;
    xv.data = (RealType *) xp; xv.n = N;
    yv.data = y; yv.n = M;

    err = gemv(&yv, &Am, &xv, beta);
    return (int) err;
}

/* ---------------------------------- dot ---------------------------------
 * returns <x, y>
 */
RealType MB_(dot)(int Ni, const RealType *x, const RealType *y) {
    size_t N = (size_t) Ni;
    Vector xv, yv;
    RealType out = (RealType) 0.0;

    if (N == 0) return out;

    xv.data = (RealType *) x; xv.n = N;
    yv.data = (RealType *) y; yv.n = N;
    vdot(&out, &xv, &yv);
    return out;
}

/* ---------------------------------- axpy ---------------------------------
 * y[N] := alpha * x[N] + y[N]
 */
int MB_(axpy)(int Ni, RealType alpha, const RealType *x, RealType *y) {
    size_t N = (size_t) Ni;
    Vector xv, yv;
    MBError err;

    if (N == 0) return MB_SUCCESS;

    xv.data = (RealType *) x; xv.n = N;
    yv.data = y; yv.n = N;
    err = vaxpy(&yv, alpha, &xv);
    return (int) err;
}
