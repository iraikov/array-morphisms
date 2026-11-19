;;; test-batch-matmul.scm
;;; Tests for the batched matrix multiply: morph-batch-matmul, the
;;; dispatch of morph-matmul on operands of rank 3 and more, and the
;;; realization of batch-matmul morphisms with and without a context.
;;;
;;; Each batched product is compared, element by element with eqv?, with
;;; the 2-D products of its cells, each cell stored in an array of its own
;;; with the same layout and multiplied by morph-matmul.

(import scheme (chicken base) (chicken format) (chicken condition))
(import test)
(import (only srfi-1 every iota append-map))
(import datatype)
(import array-morphisms-core)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-blas-exec)
(import array-morphisms-context)

;;;; Helpers

(define (bm-vals n seed)
  (map (lambda (i) (* 0.9 (sin (+ (* 1.37 i) seed)))) (iota n)))

;; The elements of realized m in row-major order of its shape.
(define (bm-elements m)
  (cases array-morphism (realize m)
    (concrete-array (data shape strides offset dtype _ _)
      (map (lambda (i)
             (typed-vector-ref data dtype
                               (multi-to-linear-index (linear-to-multi-index i shape)
                                                      strides offset)))
           (iota (shape-size shape))))
    (else (error "bm-elements: not concrete"))))

(define (bm-same? xs ys)
  (and (= (length xs) (length ys)) (every eqv? xs ys)))

(define (bm-product lst) (apply * lst))

(define (bm-take lst k)
  (if (= k 0) '() (cons (car lst) (bm-take (cdr lst) (- k 1)))))

;; The transpose of the last two axes of an array of rank r.
(define (bm-last-two-perm r)
  (append (iota (- r 2)) (list (- r 1) (- r 2))))

;; A matrix operand of logical cell shape (rows cols) with frame (a list,
;; empty for a plain matrix).  When t? the cells are stored as (cols rows)
;; and read through a transpose.  Returns (values operand cells): cells
;; lists the cells as 2-D arrays of their own with the same layout.
(define (bm-operand dt frame rows cols t? seed)
  (let* ((n      (bm-product frame))
         (stored (if t? (list cols rows) (list rows cols)))
         (size   (bm-product stored))
         (vals   (bm-vals (* n size) seed))
         (cell-vals (map (lambda (c) (bm-take (list-tail vals (* c size)) size)) (iota n)))
         (r      (+ 2 (length frame)))
         (whole  (morph-from-list vals (list->vector (append frame stored)) dt))
         (cell   (lambda (cv)
                   (let ((m (morph-from-list cv (list->vector stored) dt)))
                     (if t? (morph-transpose m '(1 0)) m)))))
    (values (if t? (morph-transpose whole (bm-last-two-perm r)) whole)
            (map cell cell-vals))))

;; The batched product of frame form (both, a or b), as
;; (list product-elements oracle-elements A B).
(define (bm-case dt form frame M K N tA? tB?)
  (let-values (((A a-cells) (bm-operand dt (if (eq? form 'b) '() frame) M K tA? 0.3))
               ((B b-cells) (bm-operand dt (if (eq? form 'a) '() frame) K N tB? 1.1)))
    (let* ((n     (bm-product frame))
           (cellA (lambda (c) (if (eq? form 'b) (car a-cells) (list-ref a-cells c))))
           (cellB (lambda (c) (if (eq? form 'a) (car b-cells) (list-ref b-cells c))))
           (oracle (append-map (lambda (c) (bm-elements (morph-matmul (cellA c) (cellB c))))
                               (iota n))))
      (list (bm-elements (morph-batch-matmul A B)) oracle A B))))

(define (bm-expr-fields m)
  (cases array-morphism m
    (morphism-expr (_ op _ _ shape dtype metadata batch-axis)
      (list op metadata shape dtype batch-axis))
    (else #f)))

(define (bm-error? thunk)
  (condition-case (begin (thunk) #f) (e (exn) #t)))

;; A backend with the kernels of the active one and no offset GEMM
;; kernels, so that batched products copy their cells.
(define (bm-copying-backend)
  (let ((b (active-blas-backend)))
    (make-blas-backend 'copying
                       (blas-backend-gemm-f64 b) (blas-backend-gemm-f32 b)
                       (blas-backend-gemm-strided-f64 b) (blas-backend-gemm-strided-f32 b)
                       (blas-backend-gemv-f64 b) (blas-backend-gemv-f32 b)
                       (blas-backend-dot-f64 b) (blas-backend-dot-f32 b)
                       (blas-backend-axpy-f64 b) (blas-backend-axpy-f32 b)
                       #f #f #f #f #f #f)))

;;;; Tests

(test-group "batch-matmul: values equal the cell products"
  (for-each
   (lambda (dt)
     (for-each
      (lambda (form)
        (for-each
         (lambda (frame)
           (for-each
            (lambda (mkn)
              (for-each
               (lambda (ts)
                 (let ((r (bm-case dt form frame (car mkn) (cadr mkn) (caddr mkn)
                                   (car ts) (cadr ts))))
                   (test-assert (sprintf "~A ~A frame ~A ~Ax~Ax~A~A~A" dt form frame
                                         (car mkn) (cadr mkn) (caddr mkn)
                                         (if (car ts) " A^T" "") (if (cadr ts) " B^T" ""))
                     (bm-same? (car r) (cadr r)))))
               '((#f #f) (#t #f) (#f #t) (#t #t))))
            '((3 4 5) (9 8 10))))
         '((3) (2 3))))
      '(both a b)))
   '(f32 f64)))

(test-group "batch-matmul: execution paths"
  (let ((saved (active-blas-backend)))
    (test-assert "the default backend has offset GEMM kernels"
      (and (blas-backend-gemm-offset-f32 saved) (blas-backend-gemm-offset-f64 saved) #t))
    (register-blas-backend! (bm-copying-backend))
    (for-each
     (lambda (dt)
       (for-each
        (lambda (form)
          (for-each
           (lambda (ts)
             (let ((r (bm-case dt form '(4) 9 8 10 (car ts) (cadr ts))))
               (test-assert (sprintf "cells copied to scratch: ~A ~A~A~A" dt form
                                     (if (car ts) " A^T" "") (if (cadr ts) " B^T" ""))
                 (bm-same? (car r) (cadr r)))))
           '((#f #f) (#t #t))))
        '(both a b)))
     '(f32 f64))
    (register-blas-backend! saved))
  (disable-blas!)
  (for-each
   (lambda (dt)
     (let ((r (bm-case dt 'both '(2 2) 9 8 10 #t #f)))
       (test-assert (sprintf "BLAS disabled: ~A" dt) (bm-same? (car r) (cadr r)))))
   '(f32 f64))
  (enable-blas!)
  (let* ((r   (bm-case 'f32 'a '(5) 9 8 10 #f #f))
         (ctx (make-morphism-context))
         (m   (morph-batch-matmul (caddr r) (cadddr r))))
    (test-assert "realize/ctx gives the values of realize"
      (bm-same? (bm-elements (realize/ctx ctx m)) (car r))))
  (let ((A (morph-from-list (bm-vals 24 0.0) #(2 3 4) 'f32))
        (B (morph-from-list (bm-vals 20 1.0) #(4 5) 'f64)))
    (test "mixed operand dtypes are promoted" 'f64
          (get-morphism-dtype (realize (morph-batch-matmul A B))))))

(test-group "batch-matmul: construction and dispatch"
  (let ((A (morph-from-list (bm-vals 6 0.0) #(2 3) 'f32))
        (B (morph-from-list (bm-vals 12 1.0) #(3 4) 'f32)))
    (test "2-D morph-matmul builds the plain matmul node"
          (list 'matmul '((k-dim . 3)) #(2 4) 'f32 -1)
          (bm-expr-fields (morph-matmul A B)))
    (test "2-D morph-batch-matmul builds the plain matmul node"
          (list 'matmul '((k-dim . 3)) #(2 4) 'f32 -1)
          (bm-expr-fields (morph-batch-matmul A B))))
  (let ((A (morph-from-list (bm-vals 30 0.0) #(5 2 3) 'f64))
        (B (morph-from-list (bm-vals 60 1.0) #(5 3 4) 'f64))
        (W (morph-from-list (bm-vals 12 2.0) #(3 4) 'f64))
        (X (morph-from-list (bm-vals 6 3.0) #(2 3) 'f64)))
    (test "rank-3 morph-matmul builds the batch-matmul node"
          (list 'batch-matmul '((k-dim . 3) (frame . #(5)) (frame-form . both)) #(5 2 4) 'f64 0)
          (bm-expr-fields (morph-matmul A B)))
    (test-assert "morph-matmul and morph-batch-matmul agree"
      (equal? (bm-expr-fields (morph-matmul A W)) (bm-expr-fields (morph-batch-matmul A W))))
    (test "frame form a" 'a
          (cdr (assq 'frame-form (cadr (bm-expr-fields (morph-matmul A W))))))
    (test "frame form b" '(b #(5 2 4))
          (let ((f (bm-expr-fields (morph-matmul X B))))
            (list (cdr (assq 'frame-form (cadr f))) (caddr f)))))
  (let ((A (morph-from-list (bm-vals 30 0.0) #(5 2 3) 'f64))
        (B (morph-from-list (bm-vals 48 1.0) #(4 3 4) 'f64))
        (C (morph-from-list (bm-vals 40 1.0) #(5 4 2) 'f64))
        (v (morph-from-list (bm-vals 3 1.0) #(3) 'f64)))
    (test-assert "unequal frames are an error" (bm-error? (lambda () (morph-matmul A B))))
    (test-assert "unequal inner dimensions are an error"
      (bm-error? (lambda () (morph-matmul A C))))
    (test-assert "morph-matmul of rank 3 and rank 1 is an error"
      (bm-error? (lambda () (morph-matmul A v))))
    (test-assert "morph-batch-matmul of rank 1 is an error"
      (bm-error? (lambda () (morph-batch-matmul v A))))))

;; Padded cells: rows (or columns) of each cell lie further apart than its
;; width (or height), as in a [M, F, K] array viewed as [F, M, K].  Each
;; cell is compared with the product of the cell stored contiguously, in
;; the same orientation.
(define (bm-padded dt F rows cols kind seed)
  ;; (values operand cells): kind rows stores [rows, F, cols] and views it
  ;; as [F, rows, cols]; kind cols stores [cols, F, rows] and views it as
  ;; [F, rows, cols] through a transpose of the last two axes.
  (let* ((vals (bm-vals (* F rows cols) seed))
         (at   (lambda (i j k) (list-ref vals (+ (* i F (if (eq? kind 'rows) cols rows))
                                                 (* j (if (eq? kind 'rows) cols rows)) k)))))
    (if (eq? kind 'rows)
        (values (morph-transpose (morph-from-list vals (vector rows F cols) dt) '(1 0 2))
                (map (lambda (f)
                       (morph-from-list (append-map (lambda (i) (map (lambda (k) (at i f k)) (iota cols)))
                                                    (iota rows))
                                        (vector rows cols) dt))
                     (iota F)))
        (values (morph-transpose (morph-from-list vals (vector cols F rows) dt) '(1 2 0))
                (map (lambda (f)
                       (morph-transpose
                        (morph-from-list (append-map (lambda (k) (map (lambda (i) (at k f i)) (iota rows)))
                                                     (iota cols))
                                         (vector cols rows) dt)
                        '(1 0)))
                     (iota F))))))

(test-group "batch-matmul: padded cells"
  (for-each
   (lambda (backend)
     (let ((saved (active-blas-backend)))
       (when (eq? backend 'copying) (register-blas-backend! (bm-copying-backend)))
       (for-each
        (lambda (dt)
          (for-each
           (lambda (mkn)
             (for-each
              (lambda (kinds)
                (let ((M (car mkn)) (K (cadr mkn)) (N (caddr mkn)))
                  (let-values (((A a-cells) (bm-padded dt 4 M K (car kinds) 0.3))
                               ((B b-cells) (bm-padded dt 4 K N (cadr kinds) 1.1)))
                    (test-assert (sprintf "~A ~A ~Ax~Ax~A A ~A B ~A" backend dt M K N
                                          (car kinds) (cadr kinds))
                      (bm-same? (bm-elements (morph-batch-matmul A B))
                                (append-map (lambda (a b) (bm-elements (morph-matmul a b)))
                                            a-cells b-cells))))))
              '((rows rows) (cols rows) (rows cols) (cols cols))))
           '((3 4 5) (9 8 10))))
        '(f32 f64))
       (register-blas-backend! saved)))
   '(offset copying)))
