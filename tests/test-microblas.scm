;;; tests/test-microblas.scm
;;; Test suite for the vendored, header-only microBLAS backend
;;; (kernels/microBLAS.h, kernels/microblas_shim.c, micro-blas-backend.scm).
;;;
;;; Unlike test-blas.scm (which mostly exercises the dispatch/fallback
;;; machinery in blas-exec.scm without a real backend registered), this file
;;; registers make-micro-blas-backend explicitly and drives its kernel
;;; procedures directly against an independent naive-Scheme reference, so it
;;; is the first test in the repo to exercise a real backend's numerics
;;; (including the conv im2col hot-kernel hooks) end-to-end.
;;;
;;; Organisation:
;;;   Group 1 - Backend construction
;;;   Group 2 - GEMM correctness (plain, contiguous)
;;;   Group 3 - GEMM-strided correctness (transposed / non-default-lda)
;;;   Group 4 - GEMV correctness
;;;   Group 5 - DOT / AXPY correctness
;;;   Group 6 - Conv im2col hot-kernel hooks (forward / bwd-data / bwd-weights)
;;;   Group 7 - Default-registration behaviour

(import scheme (chicken base))
(import test)
(import (only srfi-1 iota every))
(import srfi-4)
(import array-morphisms-blas-exec)
(import array-morphisms-realization)
(import array-morphisms-micro-blas-backend)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (approx= a b #!optional (tol 1e-6))
  "Approximate numeric equality within tolerance."
  (< (abs (- a b)) tol))

(define (rel-close? a b tol)
  "Equality within tol relative to the larger operand's magnitude (or
  absolute tol near zero) -- used for f32 comparisons where accumulated
  rounding scales with operand magnitude."
  (<= (abs (- a b)) (* tol (max 1.0 (abs a) (abs b)))))

;; Deterministic, bounded-magnitude, sign-varying data generators so
;; accumulated sums over K up to ~130 terms stay well-conditioned.
(define (gen-a i) (sin (* (+ i 1) 0.123)))
(define (gen-b i) (cos (* (+ i 1) 0.071)))

(define (fill-f32vec! v n f)
  (do ((i 0 (+ i 1))) ((= i n) v)
    (f32vector-set! v i (exact->inexact (f i)))))

(define (fill-f64vec! v n f)
  (do ((i 0 (+ i 1))) ((= i n) v)
    (f64vector-set! v i (exact->inexact (f i)))))

;; Naive reference gemm on plain Scheme vectors: C := alpha*A*B + beta*C
;; A: M*K row-major vector, B: K*N row-major vector, C: M*N row-major vector.
(define (naive-gemm M N K alpha A B beta C)
  (let ((out (make-vector (* M N) 0.0)))
    (do ((i 0 (+ i 1))) ((= i M) out)
      (do ((j 0 (+ j 1))) ((= j N))
        (let ((sum 0.0))
          (do ((k 0 (+ k 1))) ((= k K))
            (set! sum (+ sum (* (vector-ref A (+ (* i K) k))
                                 (vector-ref B (+ (* k N) j))))))
          (vector-set! out (+ (* i N) j)
                       (+ (* alpha sum) (* beta (vector-ref C (+ (* i N) j))))))))))

;; Naive reference for A^T (physical K x M, row-stride ld) or A (physical
;; M x K, row-stride ld) read with an explicit lda/trans, matching the
;; shim's calling convention -- used to build reference operands for the
;; strided-gemm tests directly from a physical vector.
(define (logical-ref phys dim0 dim1 ld trans)
  ;; Returns a fresh dim0*dim1 vector holding the logical (no-trans) view.
  (let ((out (make-vector (* dim0 dim1) 0.0)))
    (do ((i 0 (+ i 1))) ((= i dim0) out)
      (do ((j 0 (+ j 1))) ((= j dim1))
        (vector-set! out (+ (* i dim1) j)
                     (if (= trans 1)
                         (vector-ref phys (+ (* j ld) i))
                         (vector-ref phys (+ (* i ld) j))))))))

(define (vec->f32 v)
  (let* ((n (vector-length v)) (out (make-f32vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) out)
      (f32vector-set! out i (exact->inexact (vector-ref v i))))))

(define (f32->vec v)
  (let* ((n (f32vector-length v)) (out (make-vector n 0.0)))
    (do ((i 0 (+ i 1))) ((= i n) out)
      (vector-set! out i (f32vector-ref v i)))))

(define (vec-close? a b tol)
  (and (= (vector-length a) (vector-length b))
       (let loop ((i 0))
         (or (= i (vector-length a))
             (and (rel-close? (vector-ref a i) (vector-ref b i) tol)
                  (loop (+ i 1)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1: Backend construction
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "microBLAS - Backend construction"

  (test-assert "make-micro-blas-backend returns a blas-backend record"
    (blas-backend? (make-micro-blas-backend)))

  (test-assert "make-micro-blas-backend names the backend 'micro-blas"
    (eq? 'micro-blas (blas-backend-name (make-micro-blas-backend))))

  (test-assert "all 13 kernel slots are populated (non-#f)"
    (let ((b (make-micro-blas-backend)))
      (every (lambda (x) x)
             (list (blas-backend-gemm-f64 b) (blas-backend-gemm-f32 b)
                   (blas-backend-gemm-strided-f64 b) (blas-backend-gemm-strided-f32 b)
                   (blas-backend-gemv-f64 b) (blas-backend-gemv-f32 b)
                   (blas-backend-dot-f64 b) (blas-backend-dot-f32 b)
                   (blas-backend-axpy-f64 b) (blas-backend-axpy-f32 b)
                   (blas-backend-conv-fwd-im2col-f32 b)
                   (blas-backend-conv-bwd-data-im2col-f32 b)
                   (blas-backend-conv-bwd-weights-im2col-f32 b)))))

  (test-assert "register-blas-backend! + blas-available? round-trip"
    (let ((saved *active-backend*))
      (register-blas-backend! (make-micro-blas-backend))
      (let ((r (and (blas-available?)
                    (eq? 'micro-blas (blas-backend-name (active-blas-backend))))))
        (set! *active-backend* saved)
        r)))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2: GEMM correctness (plain, contiguous)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "microBLAS - GEMM correctness"

  (let* ((be (make-micro-blas-backend))
         (gemm-f32 (blas-backend-gemm-f32 be))
         (gemm-f64 (blas-backend-gemm-f64 be)))

    (for-each
     (lambda (size)
       (let* ((M size) (N size) (K size))
         (test-assert (string-append "gemm-f32 square size=" (number->string size))
           (let* ((A (make-f32vector (* M K) 0.0)) (Av (make-vector (* M K) 0.0))
                  (B (make-f32vector (* K N) 0.0)) (Bv (make-vector (* K N) 0.0))
                  (C (make-f32vector (* M N) 0.0)))
             (do ((i 0 (+ i 1))) ((= i (* M K)))
               (f32vector-set! A i (exact->inexact (gen-a i)))
               (vector-set! Av i (exact->inexact (gen-a i))))
             (do ((i 0 (+ i 1))) ((= i (* K N)))
               (f32vector-set! B i (exact->inexact (gen-b i)))
               (vector-set! Bv i (exact->inexact (gen-b i))))
             (gemm-f32 M N K 1.0 A B 0.0 C)
             (let ((ref (naive-gemm M N K 1.0 Av Bv 0.0 (make-vector (* M N) 0.0))))
               (vec-close? (f32->vec C) ref 1e-3))))

         (test-assert (string-append "gemm-f64 square size=" (number->string size))
           (let* ((A (make-f64vector (* M K) 0.0)) (Av (make-vector (* M K) 0.0))
                  (B (make-f64vector (* K N) 0.0)) (Bv (make-vector (* K N) 0.0))
                  (C (make-f64vector (* M N) 0.0)))
             (fill-f64vec! A (* M K) gen-a)
             (fill-f64vec! B (* K N) gen-b)
             (do ((i 0 (+ i 1))) ((= i (* M K))) (vector-set! Av i (gen-a i)))
             (do ((i 0 (+ i 1))) ((= i (* K N))) (vector-set! Bv i (gen-b i)))
             (gemm-f64 M N K 1.0 A B 0.0 C)
             (let ((ref (naive-gemm M N K 1.0 Av Bv 0.0 (make-vector (* M N) 0.0))))
               (let loop ((i 0))
                 (or (= i (* M N))
                     (and (approx= (f64vector-ref C i) (vector-ref ref i) 1e-9)
                          (loop (+ i 1)))))))))
       )
     '(1 2 63 64 65 127 128))

    (test-assert "gemm-f32 non-square (M=5,N=13,K=7)"
      (let* ((M 5) (N 13) (K 7)
             (A (make-f32vector (* M K) 0.0)) (Av (make-vector (* M K) 0.0))
             (B (make-f32vector (* K N) 0.0)) (Bv (make-vector (* K N) 0.0))
             (C (make-f32vector (* M N) 0.0)))
        (do ((i 0 (+ i 1))) ((= i (* M K)))
          (f32vector-set! A i (exact->inexact (gen-a i)))
          (vector-set! Av i (exact->inexact (gen-a i))))
        (do ((i 0 (+ i 1))) ((= i (* K N)))
          (f32vector-set! B i (exact->inexact (gen-b i)))
          (vector-set! Bv i (exact->inexact (gen-b i))))
        (gemm-f32 M N K 1.0 A B 0.0 C)
        (vec-close? (f32->vec C) (naive-gemm M N K 1.0 Av Bv 0.0 (make-vector (* M N) 0.0)) 1e-3)))

    (test-assert "gemm-f32 zero M is a no-op, does not crash"
      (let ((C (make-f32vector 0 0.0)))
        (gemm-f32 0 3 3 1.0 (make-f32vector 0 0.0) (make-f32vector 9 1.0) 0.0 C)
        #t))

    (test-assert "gemm-f32 zero K with beta=0 zeroes output"
      (let ((C (make-f32vector 4 9.0)))
        (gemm-f32 2 2 0 1.0 (make-f32vector 0 0.0) (make-f32vector 0 0.0) 0.0 C)
        (every (lambda (i) (= 0.0 (f32vector-ref C i))) (iota 4))))

    (test-assert "gemm-f32 zero K with beta=1 preserves existing C"
      (let ((C (make-f32vector 4 9.0)))
        (gemm-f32 2 2 0 1.0 (make-f32vector 0 0.0) (make-f32vector 0 0.0) 1.0 C)
        (every (lambda (i) (= 9.0 (f32vector-ref C i))) (iota 4))))

    (test-assert "gemm-f32 beta=1 accumulates onto existing C"
      (let* ((A (vec->f32 #(1.0 0.0 0.0 1.0)))   ; identity 2x2
             (B (vec->f32 #(1.0 2.0 3.0 4.0)))
             (C (vec->f32 #(10.0 10.0 10.0 10.0))))
        (gemm-f32 2 2 2 1.0 A B 1.0 C)
        (vec-close? (f32->vec C) #(11.0 12.0 13.0 14.0) 1e-3)))
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3: GEMM-strided correctness (transposed / non-default-lda)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "microBLAS - GEMM-strided correctness"

  (let* ((be (make-micro-blas-backend))
         (gemm-s-f32 (blas-backend-gemm-strided-f32 be)))

    (test-assert "strided: transposed A only"
      ;; logical A (M=3,K=4); physical Aphys stored as A^T (4x3), lda=3.
      (let* ((M 3) (K 4) (N 2)
             (Aphys (make-vector (* K M) 0.0))
             (B (make-vector (* K N) 0.0)))
        (do ((i 0 (+ i 1))) ((= i (* K M))) (vector-set! Aphys i (gen-a i)))
        (do ((i 0 (+ i 1))) ((= i (* K N))) (vector-set! B i (gen-b i)))
        (let* ((A-logical (logical-ref Aphys M K M 1))   ; trans reading of Aphys
               (ref (naive-gemm M N K 1.0 A-logical B 0.0 (make-vector (* M N) 0.0)))
               (Cbuf (make-f32vector (* M N) 0.0)))
          (gemm-s-f32 M N K 1.0 (vec->f32 Aphys) M 'trans (vec->f32 B) N 'no-trans 0.0 Cbuf)
          (vec-close? (f32->vec Cbuf) ref 1e-3))))

    (test-assert "strided: transposed B only"
      ;; logical B (K=3,N=4); physical Bphys stored as B^T (4x3), ldb=3.
      (let* ((M 2) (K 3) (N 4)
             (A (make-vector (* M K) 0.0))
             (Bphys (make-vector (* N K) 0.0)))
        (do ((i 0 (+ i 1))) ((= i (* M K))) (vector-set! A i (gen-a i)))
        (do ((i 0 (+ i 1))) ((= i (* N K))) (vector-set! Bphys i (gen-b i)))
        (let* ((B-logical (logical-ref Bphys K N K 1))
               (ref (naive-gemm M N K 1.0 A B-logical 0.0 (make-vector (* M N) 0.0)))
               (Cbuf (make-f32vector (* M N) 0.0)))
          (gemm-s-f32 M N K 1.0 (vec->f32 A) K 'no-trans (vec->f32 Bphys) K 'trans 0.0 Cbuf)
          (vec-close? (f32->vec Cbuf) ref 1e-3))))

    (test-assert "strided: both A and B transposed"
      (let* ((M 3) (K 2) (N 3)
             (Aphys (make-vector (* K M) 0.0))
             (Bphys (make-vector (* N K) 0.0)))
        (do ((i 0 (+ i 1))) ((= i (* K M))) (vector-set! Aphys i (gen-a i)))
        (do ((i 0 (+ i 1))) ((= i (* N K))) (vector-set! Bphys i (gen-b i)))
        (let* ((A-logical (logical-ref Aphys M K M 1))
               (B-logical (logical-ref Bphys K N K 1))
               (ref (naive-gemm M N K 1.0 A-logical B-logical 0.0 (make-vector (* M N) 0.0)))
               (Cbuf (make-f32vector (* M N) 0.0)))
          (gemm-s-f32 M N K 1.0 (vec->f32 Aphys) M 'trans (vec->f32 Bphys) K 'trans 0.0 Cbuf)
          (vec-close? (f32->vec Cbuf) ref 1e-3))))

    (test-assert "strided: non-default lda (row-padded A slice), no transpose"
      ;; logical A is (M=2,K=3) embedded in a physical buffer with lda=5
      ;; (2 extra padding columns per row that must not be read as data).
      (let* ((M 2) (K 3) (N 2) (lda 5)
             (Aphys (make-vector (* M lda) -999.0))
             (B (make-vector (* K N) 0.0)))
        (do ((i 0 (+ i 1))) ((= i M))
          (do ((j 0 (+ j 1))) ((= j K))
            (vector-set! Aphys (+ (* i lda) j) (gen-a (+ (* i K) j)))))
        (do ((i 0 (+ i 1))) ((= i (* K N))) (vector-set! B i (gen-b i)))
        (let* ((A-logical (logical-ref Aphys M K lda 0))
               (ref (naive-gemm M N K 1.0 A-logical B 0.0 (make-vector (* M N) 0.0)))
               (Cbuf (make-f32vector (* M N) 0.0)))
          (gemm-s-f32 M N K 1.0 (vec->f32 Aphys) lda 'no-trans (vec->f32 B) N 'no-trans 0.0 Cbuf)
          (vec-close? (f32->vec Cbuf) ref 1e-3))))

    (test-assert "strided: G x B^T (var-matmul dA backward pattern)"
      ;; G [2x2] x B^T where B is [3x2] -> B^T is [2x3].
      ;; G = [[1,2],[4,5]], B = [[1,0],[0,1],[1,1]], B^T = [[1,0,1],[0,1,1]]
      ;; G x B^T = [[1,2,3],[4,5,9]]
      (let* ((G (vec->f32 #(1.0 2.0 4.0 5.0)))
             (B (vec->f32 #(1.0 0.0 0.0 1.0 1.0 1.0)))  ; physical 3x2
             (Cbuf (make-f32vector 6 0.0)))
        (gemm-s-f32 2 3 2 1.0 G 2 'no-trans B 2 'trans 0.0 Cbuf)
        (vec-close? (f32->vec Cbuf) #(1.0 2.0 3.0 4.0 5.0 9.0) 1e-3)))
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4: GEMV correctness
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "microBLAS - GEMV correctness"

  (let* ((be (make-micro-blas-backend))
         (gemv-f32 (blas-backend-gemv-f32 be)))

    (test-assert "gemv-f32 basic: A[2x3] . x[3]"
      (let ((A (vec->f32 #(1.0 2.0 3.0 4.0 5.0 6.0)))
            (x (vec->f32 #(1.0 1.0 1.0)))
            (y (make-f32vector 2 0.0)))
        (gemv-f32 2 3 1.0 A x 0.0 y)
        (vec-close? (f32->vec y) #(6.0 15.0) 1e-3)))

    (test-assert "gemv-f32 with alpha != 1"
      (let ((A (vec->f32 #(1.0 2.0 3.0 4.0 5.0 6.0)))
            (x (vec->f32 #(1.0 1.0 1.0)))
            (y (make-f32vector 2 0.0)))
        (gemv-f32 2 3 2.0 A x 0.0 y)
        (vec-close? (f32->vec y) #(12.0 30.0) 1e-3)))

    (test-assert "gemv-f32 with beta != 0 accumulates"
      (let ((A (vec->f32 #(1.0 0.0 0.0 1.0)))
            (x (vec->f32 #(3.0 4.0)))
            (y (vec->f32 #(10.0 10.0))))
        (gemv-f32 2 2 1.0 A x 1.0 y)
        (vec-close? (f32->vec y) #(13.0 14.0) 1e-3)))

    (test-assert "gemv-f32 zero N is beta-scale-only"
      (let ((A (make-f32vector 0 0.0)) (x (make-f32vector 0 0.0))
            (y (vec->f32 #(7.0 8.0))))
        (gemv-f32 2 0 1.0 A x 1.0 y)
        (vec-close? (f32->vec y) #(7.0 8.0) 1e-3)))
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5: DOT / AXPY correctness
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "microBLAS - DOT / AXPY correctness"

  (let* ((be (make-micro-blas-backend))
         (dot-f32 (blas-backend-dot-f32 be))
         (axpy-f32 (blas-backend-axpy-f32 be)))

    (test-assert "dot-f32 basic"
      (approx= (dot-f32 3 (vec->f32 #(1.0 2.0 3.0)) (vec->f32 #(4.0 5.0 6.0))) 32.0 1e-3))

    (test-assert "dot-f32 zero length returns 0"
      (approx= (dot-f32 0 (make-f32vector 0 0.0) (make-f32vector 0 0.0)) 0.0 1e-9))

    (test-assert "axpy-f32 basic: y := 2*x + y"
      (let ((x (vec->f32 #(1.0 2.0 3.0)))
            (y (vec->f32 #(10.0 10.0 10.0))))
        (axpy-f32 3 2.0 x y)
        (vec-close? (f32->vec y) #(12.0 14.0 16.0) 1e-3)))

    (test-assert "axpy-f32 alpha=0 leaves y unchanged"
      (let ((x (vec->f32 #(1.0 2.0 3.0)))
            (y (vec->f32 #(10.0 10.0 10.0))))
        (axpy-f32 3 0.0 x y)
        (vec-close? (f32->vec y) #(10.0 10.0 10.0) 1e-9)))

    (test-assert "axpy-f32 zero length is a no-op"
      (let ((y (vec->f32 #(10.0 10.0))))
        (axpy-f32 0 5.0 (make-f32vector 0 0.0) y)
        (vec-close? (f32->vec y) #(10.0 10.0) 1e-9)))
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6: Conv im2col hot-kernel hooks
;;;
;;; Registers the microBLAS backend so execute-conv-{fwd,bwd-data,
;;; bwd-weights}-blas take the conv-*-im2col-f32 hook path, and compares
;;; against the pure-Scheme execute-conv-*-nchw reference implementations.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "microBLAS - Conv im2col hot-kernel hooks"

  (let* ((saved *active-backend*))
    (register-blas-backend! (make-micro-blas-backend))

    ;; N=2,C=2,H=5,W=5,KH=KW=3,SH=SW=1,PH=PW=1 (same-padding) -> OH=OW=5, out-ch=3
    (let* ((N 2) (C 2) (H 5) (W 5) (KH 3) (KW 3) (SH 1) (SW 1) (PH 1) (PW 1)
           (OH 5) (OW 5) (out-ch 3)
           (fan-in (* C KH KW)) (M (* N OH OW))
           (x-shape (vector N C H W))
           (src (make-f32vector (* N C H W) 0.0))
           (wt  (make-f32vector (* fan-in out-ch) 0.0))
           (b   (make-f32vector out-ch 0.0))
           (g   (make-f32vector (* M out-ch) 0.0)))
      (fill-f32vec! src (* N C H W) gen-a)
      (fill-f32vec! wt (* fan-in out-ch) gen-b)
      (fill-f32vec! b out-ch (lambda (i) (* 0.1 (+ i 1))))
      (fill-f32vec! g (* M out-ch) (lambda (i) (sin (* (+ i 3) 0.211))))

      (test-assert "conv-fwd: microBLAS hook matches scalar Scheme reference"
        (let ((ref-out (make-f32vector (* M out-ch) 0.0))
              (blas-out (make-f32vector (* M out-ch) 0.0)))
          (execute-conv-fwd-nchw ref-out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-fwd-blas blas-out src wt b N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (vec-close? (f32->vec blas-out) (f32->vec ref-out) 1e-2)))

      (test-assert "conv-bwd-data: microBLAS hook matches scalar Scheme reference"
        (let ((ref-dx (make-f32vector (* N C H W) 0.0))
              (blas-dx (make-f32vector (* N C H W) 0.0)))
          (execute-conv-bwd-data-nchw ref-dx x-shape g #f wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-bwd-data-blas blas-dx x-shape g #f wt N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (vec-close? (f32->vec blas-dx) (f32->vec ref-dx) 1e-2)))

      (test-assert "conv-bwd-weights: microBLAS hook matches scalar Scheme reference"
        (let ((ref-dwt (make-f32vector (* fan-in out-ch) 0.0))
              (blas-dwt (make-f32vector (* fan-in out-ch) 0.0))
              (wt-shape (vector fan-in out-ch)))
          (execute-conv-bwd-weights-nchw ref-dwt wt-shape g #f src N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (execute-conv-bwd-weights-blas blas-dwt wt-shape g #f src N C H W KH KW SH SW PH PW OH OW out-ch 'f32)
          (vec-close? (f32->vec blas-dwt) (f32->vec ref-dwt) 1e-2))))

    (set! *active-backend* saved))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 7: Default-registration behaviour
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "microBLAS - Default-registration behaviour"

  (test-assert "microBLAS is auto-registered as the default backend in this process"
    ;; realization.scm's load-time bootstrap already ran when this file's
    ;; own (import array-morphisms-realization) above executed; since no
    ;; test file imported earlier in the run.scm chain registers a
    ;; different backend first, microBLAS should be active here.
    (and (blas-available?)
         (eq? 'micro-blas (blas-backend-name (active-blas-backend)))))

  (test-assert "manual registration after load still wins (non-clobbering)"
    (let ((saved *active-backend*)
          (dummy (lambda args 0)))
      (register-blas-backend!
       (make-blas-backend 'dummy-backend
                           dummy dummy dummy dummy dummy dummy
                           dummy dummy dummy dummy dummy dummy dummy))
      (let ((r (eq? 'dummy-backend (blas-backend-name (active-blas-backend)))))
        (set! *active-backend* saved)
        r)))
)
