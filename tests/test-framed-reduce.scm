;;; test-framed-reduce.scm
;;; Tests for reductions with a frame (morph-reduce with a frame rank).
;;;
;;; Each cell of a framed reduction is compared, element by element, with
;;; the same reduction of that cell stored in an array of its own with the
;;; same in-cell layout.  Values are compared with eqv?, NaN equal to NaN.

(import scheme (chicken base) (chicken format) (chicken condition))
(import test)
(import (only srfi-1 every iota append-map make-list))
(import datatype)
(import array-morphisms-core)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-blas-exec)
(import array-morphisms-context)

;;;; Helpers

(define (fr-take lst k)
  (if (= k 0) '() (cons (car lst) (fr-take (cdr lst) (- k 1)))))

(define (fr-product lst) (apply * lst))

;; Values for n elements: smooth values, or with special? NaN, -0.0, 0.0
;; and large values mixed in.  Integers for integer dtypes.
(define (fr-vals n seed dt special?)
  (map (lambda (i)
         (cond ((memq dt '(s32 s64)) (- (modulo (+ (* 7 i) (* 13 seed)) 23) 11))
               ((and special? (= (modulo (+ i seed) 11) 3)) +nan.0)
               ((and special? (= (modulo (+ i seed) 7) 2)) -0.0)
               ((and special? (= (modulo (+ i seed) 7) 5)) 0.0)
               ((and special? (= (modulo (+ i seed) 13) 1)) 3.0e7)
               (else (* 0.9 (sin (+ (* 1.37 i) seed))))))
       (iota n)))

;; The elements of realized m in row-major order of its shape.
(define (fr-elements m)
  (cases array-morphism (realize m)
    (concrete-array (data shape strides offset dtype _ _)
      (map (lambda (i)
             (typed-vector-ref data dtype
                               (multi-to-linear-index (linear-to-multi-index i shape)
                                                      strides offset)))
           (iota (shape-size shape))))
    (else (error "fr-elements: not concrete"))))

(define (fr-same? xs ys)
  (and (= (length xs) (length ys))
       (every (lambda (x y)
                (if (and (flonum? x) (flonum? y) (nan? x) (nan? y)) #t (eqv? x y)))
              xs ys)))

;; A framed source with frame (a list) and cell shape (a list), and the
;; list of its cells as arrays of their own with the same in-cell layout.
;; layout: row-major; cell-transposed (rank-2 cells stored transposed);
;; frame-strided (frame rank 1: every other frame position of a source
;; twice as long).
(define (fr-source dt frame cell layout special?)
  (let* ((n     (fr-product frame))
         (size  (fr-product cell))
         (r     (+ (length frame) (length cell)))
         (stored-cell (if (eq? layout 'cell-transposed) (reverse cell) cell))
         (count (if (eq? layout 'frame-strided) (* 2 n) n))
         (vals  (fr-vals (* count size) 1 dt special?))
         (cvals (lambda (c) (fr-take (list-tail vals (* c size)) size)))
         (own   (lambda (cv)
                  (let ((m (morph-from-list cv (list->vector stored-cell) dt)))
                    (if (eq? layout 'cell-transposed) (morph-transpose m '(1 0)) m)))))
    (case layout
      ((row-major)
       (values (morph-from-list vals (list->vector (append frame cell)) dt)
               (map (lambda (c) (own (cvals c))) (iota n))))
      ((cell-transposed)
       (values (morph-transpose (morph-from-list vals (list->vector (append frame stored-cell)) dt)
                                (append (iota (length frame)) (list (- r 1) (- r 2))))
               (map (lambda (c) (own (cvals c))) (iota n))))
      ((frame-strided)
       (let ((whole (morph-from-list vals (list->vector (cons (* 2 n) cell)) dt)))
         (values (morph-slice whole (make-list r 0) (cons (* 2 n) cell)
                              (cons 2 (make-list (length cell) 1)))
                 (map (lambda (c) (own (cvals (* 2 c)))) (iota n))))))))

;; The framed reduction and its cell-by-cell oracle, as two element lists.
(define (fr-case dt op frame cell axes keepdims? layout special?)
  (let-values (((src cells) (fr-source dt frame cell layout special?)))
    (let ((f (length frame)))
      (list (fr-elements (morph-reduce op src (map (lambda (a) (+ a f)) axes) keepdims? f))
            (append-map (lambda (c) (fr-elements (morph-reduce op c axes keepdims?))) cells)))))

(define (fr-error? thunk)
  (condition-case (begin (thunk) #f) (e (exn) #t)))

;; A backend with the kernels of the active one and no offset kernels.
(define (fr-plain-backend)
  (let ((b (active-blas-backend)))
    (make-blas-backend 'plain
                       (blas-backend-gemm-f64 b) (blas-backend-gemm-f32 b)
                       (blas-backend-gemm-strided-f64 b) (blas-backend-gemm-strided-f32 b)
                       (blas-backend-gemv-f64 b) (blas-backend-gemv-f32 b)
                       (blas-backend-dot-f64 b) (blas-backend-dot-f32 b)
                       (blas-backend-axpy-f64 b) (blas-backend-axpy-f32 b)
                       #f #f #f #f #f #f)))

;; (cell axes) pairs, one or more per cell path of the cell's own array.
(define fr-cells
  '(((7) (0))                 ; 1d
    ((5 9) (0))               ; axis0
    ((5 9) (1))               ; axis1
    ((5 9) (0 1))             ; general: both axes
    ((3 4 5) (2))             ; general: rank 3
    ((3 4 5) (0 2))))

;;;; Tests

(test-group "framed reductions: values equal the cell reductions"
  (for-each
   (lambda (dt)
     (for-each
      (lambda (op)
        (for-each
         (lambda (ca)
           (for-each
            (lambda (frame)
              (for-each
               (lambda (keepdims?)
                 (for-each
                  (lambda (special?)
                    (let ((r (fr-case dt op frame (car ca) (cadr ca) keepdims? 'row-major special?)))
                      (test-assert (sprintf "~A ~A frame ~A cell ~A axes ~A~A~A" dt op frame
                                            (car ca) (cadr ca) (if keepdims? " keepdims" "")
                                            (if special? " special" ""))
                        (fr-same? (car r) (cadr r)))))
                  (if (memq op '(sum max min)) '(#f #t) '(#f))))
               '(#f #t)))
            '((4) (2 3))))
         fr-cells))
      '(sum mean max min prod)))
   '(f32 f64))
  (for-each
   (lambda (op)
     (let ((r (fr-case 's32 op '(3) '(5 9) '(1) #f 'row-major #f)))
       (test-assert (sprintf "s32 ~A axis 1" op) (fr-same? (car r) (cadr r)))))
   '(sum max prod)))

(test-group "framed reductions: layouts and execution paths"
  (for-each
   (lambda (dt)
     (for-each
      (lambda (op)
        (for-each
         (lambda (ca)
           (let ((r (fr-case dt op '(4) (car ca) (cadr ca) #f 'cell-transposed #t)))
             (test-assert (sprintf "cells stored transposed: ~A ~A axes ~A" dt op (cadr ca))
               (fr-same? (car r) (cadr r)))))
         '(((5 9) (0)) ((5 9) (1)) ((5 9) (0 1)))))
      '(sum mean max))
     (for-each
      (lambda (op)
        (for-each
         (lambda (ca)
           (let ((r (fr-case dt op '(4) (car ca) (cadr ca) #f 'frame-strided #t)))
             (test-assert (sprintf "every other frame position: ~A ~A cell ~A axes ~A" dt op
                                   (car ca) (cadr ca))
               (fr-same? (car r) (cadr r)))))
         '(((7) (0)) ((5 9) (1)) ((5 9) (0)))))
      '(sum mean max)))
   '(f32 f64))
  (let ((saved (active-blas-backend)))
    (test-assert "the default backend has offset DOT kernels"
      (and (blas-backend-dot-offset-f32 saved) (blas-backend-dot-offset-f64 saved) #t))
    (register-blas-backend! (fr-plain-backend))
    (for-each
     (lambda (dt)
       (for-each
        (lambda (op)
          (let ((r (fr-case dt op '(2 3) '(64) '(0) #f 'row-major #f)))
            (test-assert (sprintf "rows copied to scratch: ~A ~A" dt op)
              (fr-same? (car r) (cadr r)))))
        '(sum mean)))
     '(f32 f64))
    (register-blas-backend! saved))
  (disable-blas!)
  (for-each
   (lambda (dt)
     (let ((r (fr-case dt 'sum '(4) '(64) '(0) #f 'row-major #f)))
       (test-assert (sprintf "BLAS disabled: ~A" dt) (fr-same? (car r) (cadr r)))))
   '(f32 f64))
  (enable-blas!)
  (let-values (((src cells) (fr-source 'f32 '(4) '(5 9) 'row-major #f)))
    (let ((m (morph-reduce 'sum src '(2) #f 1)))
      (test-assert "realize/ctx gives the values of realize"
        (fr-same? (fr-elements (realize/ctx (make-morphism-context) m)) (fr-elements m))))))

(test-group "framed reductions: construction"
  (let* ((m  (morph-from-list (fr-vals 60 0 'f64 #f) #(3 4 5) 'f64))
         (r0 (morph-reduce 'sum m '(1) #t))
         (fields (lambda (r)
                   (cases array-morphism r
                     (reduction-morphism (_ op _ axes fn shape dtype ba)
                       (list op axes shape dtype ba (reduction-index-fn-keepdims? fn)
                             (reduction-index-fn-frame-rank fn)))
                     (else #f)))))
    (test "without a frame rank the node is the unframed reduction"
          (list 'sum '(1) #(3 1 5) 'f64 -1 #t 0) (fields r0))
    (test "with frame rank 1" (list 'sum '(1) #(3 1 5) 'f64 -1 #t 1)
          (fields (morph-reduce 'sum m '(1) #t 1)))
    (test "empty axes reduce every cell axis" '(1 2)
          (cadr (fields (morph-reduce 'max m '() #f 1))))
    (test-assert "a frame axis cannot be reduced"
      (fr-error? (lambda () (morph-reduce 'sum m '(0 1) #f 1))))
    (test "an axis-1 cell runs as rows" '(rows 20 9)
          (framed-reduction-plan 'sum #(4 5 9) #(45 9 1) 'f32 '(2) 1))
    (test "a 1d sum cell runs as 1d rows" '(rows-1d 6 7)
          (framed-reduction-plan 'sum #(2 3 7) #(21 7 1) 'f64 '(2) 2))
    (test "a 1d max cell runs as rows" '(rows 4 7)
          (framed-reduction-plan 'max #(4 7) #(7 1) 'f32 '(1) 1))
    (test "an axis-0 cell runs directly" 'direct
          (framed-reduction-plan 'sum #(4 5 9) #(45 9 1) 'f32 '(1) 1))
    (test "frame rank 0 runs directly" 'direct
          (framed-reduction-plan 'sum #(5 9) #(9 1) 'f32 '(1) 0))
    (test-assert "the frame rank cannot exceed the rank"
      (fr-error? (lambda () (morph-reduce 'sum m '() #f 4))))))
