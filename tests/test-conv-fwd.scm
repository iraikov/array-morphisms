;;; tests/test-conv-fwd.scm
;;;
;;; Unit tests for Phase 2 implicit GEMM convolution execute functions and
;;; the var-conv2d SSA op (forward + backward correctness via gradient check).

(import scheme (chicken base))
(import srfi-4)
(import (only srfi-1 iota fold))
(import (only test test-group test))
(import array-morphisms-core)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-ssa)
(import array-morphisms-blas-exec)
(import array-morphisms-blas-egg-backend)


;;; ============================================================
;;; Helpers
;;; ============================================================

(define (make-f64 lst) (list->f64vector lst))

(define (f64-ref v i) (f64vector-ref v i))
(define (f64-set! v i x) (f64vector-set! v i x))

(define (zeros-f64 n) (make-f64vector n 0.0))

(define (alloc-f64 . vals)
  (list->f64vector vals))

(define (shape-product shape)
  (fold * 1 (vector->list shape)))

;; Manual im2col + GEMM: col = im2col(src,NCHW), out = col @ wt + b
;; col layout: [N*OH*OW, C*KH*KW] (classic flattened rows)
;; wt layout:  [C*KH*KW, out_ch]  (fan_in x out_ch = WT)
(define (reference-conv-fwd-nchw src N C H W KH KW SH SW PH PW OH OW out-ch wt b)
  (let* ((fan-in (* C KH KW))
         (M (* N OH OW))
         (out (make-f64vector (* M out-ch) 0.0)))
    ;; bias init
    (do ((m 0 (+ m 1))) ((= m M))
      (do ((co 0 (+ co 1))) ((= co out-ch))
        (f64vector-set! out (+ (* m out-ch) co) (f64vector-ref b co))))
    ;; accumulate
    (do ((n 0 (+ n 1))) ((= n N))
      (do ((oh 0 (+ oh 1))) ((= oh OH))
        (do ((ow 0 (+ ow 1))) ((= ow OW))
          (let ((m (+ (* n OH OW) (* oh OW) ow)))
            (let ((col-idx 0))
              (do ((c 0 (+ c 1))) ((= c C))
                (do ((kh 0 (+ kh 1))) ((= kh KH))
                  (do ((kw 0 (+ kw 1))) ((= kw KW))
                    (let* ((ih (+ (* oh SH) kh (- PH)))
                           (iw (+ (* ow SW) kw (- PW)))
                           (src-val (if (and (>= ih 0) (< ih H) (>= iw 0) (< iw W))
                                        (f64vector-ref src (+ (* n C H W) (* c H W) (* ih W) iw))
                                        0.0)))
                      (do ((co 0 (+ co 1))) ((= co out-ch))
                        (f64vector-set! out (+ (* m out-ch) co)
                          (+ (f64vector-ref out (+ (* m out-ch) co))
                             (* src-val (f64vector-ref wt (+ (* col-idx out-ch) co))))))
                      (set! col-idx (+ col-idx 1)))))))))))
    out))

;; NHWC reference: same as above but reads src[(n*H*W+ih*W+iw)*C+c]
(define (reference-conv-fwd-nhwc src N C H W KH KW SH SW PH PW OH OW out-ch wt b)
  (let* ((fan-in (* C KH KW))
         (M (* N OH OW))
         (out (make-f64vector (* M out-ch) 0.0)))
    (do ((m 0 (+ m 1))) ((= m M))
      (do ((co 0 (+ co 1))) ((= co out-ch))
        (f64vector-set! out (+ (* m out-ch) co) (f64vector-ref b co))))
    (do ((n 0 (+ n 1))) ((= n N))
      (do ((oh 0 (+ oh 1))) ((= oh OH))
        (do ((ow 0 (+ ow 1))) ((= ow OW))
          (let ((m (+ (* n OH OW) (* oh OW) ow)))
            (let ((col-idx 0))
              (do ((c 0 (+ c 1))) ((= c C))
                (do ((kh 0 (+ kh 1))) ((= kh KH))
                  (do ((kw 0 (+ kw 1))) ((= kw KW))
                    (let* ((ih (+ (* oh SH) kh (- PH)))
                           (iw (+ (* ow SW) kw (- PW)))
                           (src-val (if (and (>= ih 0) (< ih H) (>= iw 0) (< iw W))
                                        (f64vector-ref src (+ (* (* (+ (* n H) ih) W) C)
                                                              (* iw C) c))
                                        0.0)))
                      (do ((co 0 (+ co 1))) ((= co out-ch))
                        (f64vector-set! out (+ (* m out-ch) co)
                          (+ (f64vector-ref out (+ (* m out-ch) co))
                             (* src-val (f64vector-ref wt (+ (* col-idx out-ch) co))))))
                      (set! col-idx (+ col-idx 1)))))))))))
    out))

(define (f64v-close? a b tol)
  (let ((n (f64vector-length a)))
    (and (= n (f64vector-length b))
         (let loop ((i 0))
           (if (= i n)
               #t
               (and (< (abs (- (f64vector-ref a i) (f64vector-ref b i))) tol)
                    (loop (+ i 1))))))))

(define (f64v-dot a b)
  (let ((n (f64vector-length a)) (s 0.0))
    (do ((i 0 (+ i 1)) (s 0.0 (+ s (* (f64vector-ref a i) (f64vector-ref b i)))))
        ((= i n) s))))

(define (f64v-sum v)
  (let ((n (f64vector-length v)))
    (do ((i 0 (+ i 1)) (s 0.0 (+ s (f64vector-ref v i))))
        ((= i n) s))))

;;; ============================================================
;;; Test data: N=2, C=2, H=4, W=4, KH=KW=3, stride=1, pad=0
;;; OH = OW = 2;  fan_in = 2*3*3 = 18;  M = 2*2*2 = 8
;;; ============================================================

(define N 2) (define C 2) (define H 4) (define W 4)
(define KH 3) (define KW 3) (define SH 1) (define SW 1)
(define PH 0) (define PW 0) (define OH 2) (define OW 2)
(define out-ch 3)
(define fan-in (* C KH KW))  ; = 18
(define M (* N OH OW))        ; = 8

;; Random-looking but deterministic data
(define (fill-vec! v f)
  (do ((i 0 (+ i 1))) ((= i (f64vector-length v)) v)
    (f64vector-set! v i (f (+ i 1)))))

(define src-nchw
  (fill-vec! (make-f64vector (* N C H W) 0.0)
             (lambda (i) (* 0.1 i))))

(define src-nhwc
  ;; Permute nchw->nhwc: src-nhwc[n,h,w,c] = src-nchw[n,c,h,w]
  (let ((v (make-f64vector (* N H W C) 0.0)))
    (do ((n 0 (+ n 1))) ((= n N))
      (do ((c 0 (+ c 1))) ((= c C))
        (do ((h 0 (+ h 1))) ((= h H))
          (do ((ww 0 (+ ww 1))) ((= ww W))
            (f64vector-set! v (+ (* n H W C) (* h W C) (* ww C) c)
              (f64vector-ref src-nchw (+ (* n C H W) (* c H W) (* h W) ww)))))))
    v))

(define wt   ; [fan_in, out_ch]
  (fill-vec! (make-f64vector (* fan-in out-ch) 0.0)
             (lambda (i) (* 0.05 (- i (* out-ch fan-in 0.5))))))

(define bias
  (fill-vec! (make-f64vector out-ch 0.0)
             (lambda (i) (* 0.01 i))))


;;; ============================================================
;;; Test group 1: execute-conv-fwd-nchw vs reference
;;; ============================================================

(test-group "execute-conv-fwd-nchw vs reference im2col+GEMM"

  (let* ((out-buf (zeros-f64 (* M out-ch)))
         (ref     (reference-conv-fwd-nchw src-nchw N C H W KH KW SH SW PH PW OH OW out-ch wt bias))
         (_ (execute-conv-fwd-nchw out-buf src-nchw wt bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64)))

    (test "output size is N*OH*OW * out_ch"
          (* M out-ch)
          (f64vector-length out-buf))

    (test "fused NCHW forward matches reference im2col+GEMM"
          #t
          (f64v-close? out-buf ref 1e-10))))


;;; ============================================================
;;; Test group 2: execute-conv-fwd-nhwc matches NCHW result
;;; ============================================================

(test-group "execute-conv-fwd-nhwc matches NCHW"

  (let* ((out-nchw (zeros-f64 (* M out-ch)))
         (out-nhwc (zeros-f64 (* M out-ch)))
         (_ (execute-conv-fwd-nchw out-nchw src-nchw wt bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (_ (execute-conv-fwd-nhwc out-nhwc src-nhwc wt bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (ref (reference-conv-fwd-nhwc src-nhwc N C H W KH KW SH SW PH PW OH OW out-ch wt bias)))

    (test "NHWC forward matches NCHW forward"
          #t
          (f64v-close? out-nhwc out-nchw 1e-10))

    (test "NHWC forward matches reference"
          #t
          (f64v-close? out-nhwc ref 1e-10))))


;;; ============================================================
;;; Test group 3: bwd-data adjoint: <dX, delta> = <g, fwd(delta)>
;;; ============================================================

(test-group "execute-conv-bwd-data-nchw adjoint"

  (let* (;; fixed g: upstream gradient [M, out_ch]
         (g-data (fill-vec! (make-f64vector (* M out-ch) 0.0) (lambda (i) (* 0.03 i))))
         (g-shape (vector M out-ch))
         ;; x-shape for dX
         (x-shape (vector N C H W))
         (dx-size (* N C H W))
         ;; compute dX
         (dx-buf (zeros-f64 dx-size))
         (_ (execute-conv-bwd-data-nchw dx-buf x-shape g-data g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         ;; random perturbation delta of same shape as x
         (delta (fill-vec! (make-f64vector dx-size 0.0) (lambda (i) (- (* 0.07 i) 0.5))))
         ;; <dX, delta>
         (lhs (f64v-dot dx-buf delta))
         ;; fwd(delta): [M, out_ch] (zero bias for linearity)
         (zero-bias (make-f64vector out-ch 0.0))
         (fwd-delta (zeros-f64 (* M out-ch)))
         (_ (execute-conv-fwd-nchw fwd-delta delta wt zero-bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         ;; <g, fwd(delta)>
         (rhs (f64v-dot g-data fwd-delta)))

    (test "bwd-data adjoint: <dX,delta> = <g,fwd(delta)>"
          #t
          (< (abs (- lhs rhs)) 1e-8))))


;;; ============================================================
;;; Test group 4: bwd-weights adjoint: <dWT, delta> = <g, fwd_with_delta_wt(x)>
;;; ============================================================

(test-group "execute-conv-bwd-weights-nchw adjoint"

  (let* (;; fixed g: [M, out_ch]
         (g-data (fill-vec! (make-f64vector (* M out-ch) 0.0) (lambda (i) (* 0.02 i))))
         (g-shape (vector M out-ch))
         (wt-shape (vector fan-in out-ch))
         ;; compute dWT
         (dwt-buf (zeros-f64 (* fan-in out-ch)))
         (_ (execute-conv-bwd-weights-nchw dwt-buf wt-shape g-data g-shape src-nchw N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         ;; random delta of same shape as wt
         (delta-wt (fill-vec! (make-f64vector (* fan-in out-ch) 0.0) (lambda (i) (- (* 0.04 i) 0.3))))
         ;; <dWT, delta_wt>
         (lhs (f64v-dot dwt-buf delta-wt))
         ;; fwd with delta_wt: [M, out_ch]
         (zero-bias (make-f64vector out-ch 0.0))
         (fwd-delta (zeros-f64 (* M out-ch)))
         (_ (execute-conv-fwd-nchw fwd-delta src-nchw delta-wt zero-bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         ;; <g, fwd_with_delta_wt(x)>
         (rhs (f64v-dot g-data fwd-delta)))

    (test "bwd-weights adjoint: <dWT,delta> = <g,fwd_delta_wt(x)>"
          #t
          (< (abs (- lhs rhs)) 1e-8))))


;;; ============================================================
;;; Test group 5: NHWC bwd-data adjoint
;;; ============================================================

(test-group "execute-conv-bwd-data-nhwc adjoint"

  (let* ((g-data (fill-vec! (make-f64vector (* M out-ch) 0.0) (lambda (i) (* 0.03 i))))
         (g-shape (vector M out-ch))
         (x-shape (vector N H W C))   ; NHWC output shape
         (dx-size (* N H W C))
         (dx-buf  (zeros-f64 dx-size))
         (_ (execute-conv-bwd-data-nhwc dx-buf x-shape g-data g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (delta   (fill-vec! (make-f64vector dx-size 0.0) (lambda (i) (- (* 0.07 i) 0.5))))
         (lhs     (f64v-dot dx-buf delta))
         (zero-bias (make-f64vector out-ch 0.0))
         (fwd-delta (zeros-f64 (* M out-ch)))
         (_ (execute-conv-fwd-nhwc fwd-delta delta wt zero-bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (rhs (f64v-dot g-data fwd-delta)))

    (test "NHWC bwd-data adjoint: <dX,delta> = <g,fwd(delta)>"
          #t
          (< (abs (- lhs rhs)) 1e-8))))


;;; ============================================================
;;; Test group 6: BLAS-path output matches scalar (NCHW)
;;; ============================================================

(register-blas-backend! (make-blas-egg-backend))

(test-group "execute-conv-fwd-blas matches scalar NCHW"
  (let* ((out-scalar (zeros-f64 (* M out-ch)))
         (out-blas   (zeros-f64 (* M out-ch)))
         (_ (execute-conv-fwd-nchw  out-scalar src-nchw wt bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (_ (execute-conv-fwd-blas  out-blas   src-nchw wt bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64)))
    (test "BLAS forward matches scalar forward (NCHW, f64)"
          #t
          (f64v-close? out-scalar out-blas 1e-8))))

(test-group "execute-conv-bwd-data-blas matches scalar NCHW"
  (let* ((g-data  (fill-vec! (make-f64vector (* M out-ch) 0.0) (lambda (i) (* 0.03 i))))
         (g-shape (vector M out-ch))
         (x-shape (vector N C H W))
         (dx-scalar (zeros-f64 (* N C H W)))
         (dx-blas   (zeros-f64 (* N C H W)))
         (_ (execute-conv-bwd-data-nchw dx-scalar x-shape g-data g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (_ (execute-conv-bwd-data-blas dx-blas   x-shape g-data g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f64)))
    (test "BLAS bwd-data matches scalar bwd-data (NCHW, f64)"
          #t
          (f64v-close? dx-scalar dx-blas 1e-8))))

(test-group "execute-conv-bwd-weights-blas matches scalar NCHW"
  (let* ((g-data   (fill-vec! (make-f64vector (* M out-ch) 0.0) (lambda (i) (* 0.04 i))))
         (g-shape  (vector M out-ch))
         (wt-shape (vector fan-in out-ch))
         (dwt-scalar (zeros-f64 (* fan-in out-ch)))
         (dwt-blas   (zeros-f64 (* fan-in out-ch)))
         (_ (execute-conv-bwd-weights-nchw dwt-scalar wt-shape g-data g-shape src-nchw N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (_ (execute-conv-bwd-weights-blas dwt-blas   wt-shape g-data g-shape src-nchw N C H W KH KW SH SW PH PW OH OW out-ch 'f64)))
    (test "BLAS bwd-weights matches scalar bwd-weights (NCHW, f64)"
          #t
          (f64v-close? dwt-scalar dwt-blas 1e-8))))

(test-group "execute-conv-fwd-nhwc-blas matches scalar NHWC"
  (let* ((out-scalar (zeros-f64 (* M out-ch)))
         (out-blas   (zeros-f64 (* M out-ch)))
         (_ (execute-conv-fwd-nhwc      out-scalar src-nhwc wt bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (_ (execute-conv-fwd-nhwc-blas out-blas   src-nhwc wt bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64)))
    (test "BLAS forward matches scalar forward (NHWC, f64)"
          #t
          (f64v-close? out-scalar out-blas 1e-8))))

(test-group "execute-conv-bwd-data-nhwc-blas matches scalar NHWC"
  (let* ((g-data  (fill-vec! (make-f64vector (* M out-ch) 0.0) (lambda (i) (* 0.03 i))))
         (g-shape (vector M out-ch))
         (x-shape (vector N H W C))
         (dx-scalar (zeros-f64 (* N H W C)))
         (dx-blas   (zeros-f64 (* N H W C)))
         (_ (execute-conv-bwd-data-nhwc      dx-scalar x-shape g-data g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (_ (execute-conv-bwd-data-nhwc-blas dx-blas   x-shape g-data g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f64)))
    (test "BLAS bwd-data matches scalar bwd-data (NHWC, f64)"
          #t
          (f64v-close? dx-scalar dx-blas 1e-8))))

(test-group "execute-conv-bwd-weights-nhwc-blas matches scalar NHWC"
  (let* ((g-data   (fill-vec! (make-f64vector (* M out-ch) 0.0) (lambda (i) (* 0.04 i))))
         (g-shape  (vector M out-ch))
         (wt-shape (vector fan-in out-ch))
         (dwt-scalar (zeros-f64 (* fan-in out-ch)))
         (dwt-blas   (zeros-f64 (* fan-in out-ch)))
         (_ (execute-conv-bwd-weights-nhwc      dwt-scalar wt-shape g-data g-shape src-nhwc N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (_ (execute-conv-bwd-weights-nhwc-blas dwt-blas   wt-shape g-data g-shape src-nhwc N C H W KH KW SH SW PH PW OH OW out-ch 'f64)))
    (test "BLAS bwd-weights matches scalar bwd-weights (NHWC, f64)"
          #t
          (f64v-close? dwt-scalar dwt-blas 1e-8))))


;;; ============================================================
;;; Test group 7: BLAS-path adjoint tests
;;; ============================================================

(test-group "execute-conv-bwd-data-blas adjoint (NCHW)"
  (let* ((g-data  (fill-vec! (make-f64vector (* M out-ch) 0.0) (lambda (i) (* 0.03 i))))
         (g-shape (vector M out-ch))
         (x-shape (vector N C H W))
         (dx-size (* N C H W))
         (dx-buf  (zeros-f64 dx-size))
         (_ (execute-conv-bwd-data-blas dx-buf x-shape g-data g-shape wt N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (delta     (fill-vec! (make-f64vector dx-size 0.0) (lambda (i) (- (* 0.07 i) 0.5))))
         (lhs       (f64v-dot dx-buf delta))
         (zero-bias (make-f64vector out-ch 0.0))
         (fwd-delta (zeros-f64 (* M out-ch)))
         (_ (execute-conv-fwd-blas fwd-delta delta wt zero-bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (rhs (f64v-dot g-data fwd-delta)))
    (test "BLAS bwd-data adjoint: <dX,delta> = <g,fwd_blas(delta)>"
          #t
          (< (abs (- lhs rhs)) 1e-8))))

(test-group "execute-conv-bwd-weights-blas adjoint (NCHW)"
  (let* ((g-data   (fill-vec! (make-f64vector (* M out-ch) 0.0) (lambda (i) (* 0.02 i))))
         (g-shape  (vector M out-ch))
         (wt-shape (vector fan-in out-ch))
         (dwt-buf  (zeros-f64 (* fan-in out-ch)))
         (_ (execute-conv-bwd-weights-blas dwt-buf wt-shape g-data g-shape src-nchw N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (delta-wt  (fill-vec! (make-f64vector (* fan-in out-ch) 0.0) (lambda (i) (- (* 0.04 i) 0.3))))
         (lhs       (f64v-dot dwt-buf delta-wt))
         (zero-bias (make-f64vector out-ch 0.0))
         (fwd-delta (zeros-f64 (* M out-ch)))
         (_ (execute-conv-fwd-blas fwd-delta src-nchw delta-wt zero-bias N C H W KH KW SH SW PH PW OH OW out-ch 'f64))
         (rhs (f64v-dot g-data fwd-delta)))
    (test "BLAS bwd-weights adjoint: <dWT,delta> = <g,fwd_blas_delta_wt(x)>"
          #t
          (< (abs (- lhs rhs)) 1e-8))))
