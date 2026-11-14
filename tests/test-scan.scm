;;; test-scan.scm
;;; Tests for scans (morph-scan, morph-cumsum, morph-cumprod).
;;;
;;; Each scan is compared, element by element, with a plain loop over the
;;; lines of a list-based copy of its operand.  Values are compared with
;;; eqv?, NaN equal to NaN.

(import scheme (chicken base) (chicken format) (chicken condition) (only (chicken flonum) fp+))
(import test)
(import (only srfi-1 every iota make-list take drop))
(import srfi-4)
(import datatype)
(import array-morphisms-core)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-context)

;;;; Helpers

(define (sc-product lst) (apply * lst))

;; Values for n elements: smooth values, or with special? NaN, -0.0, 0.0,
;; infinities and large values mixed in.  Small integers for integer dtypes.
(define (sc-vals n seed dt special?)
  (map (lambda (i)
         (cond ((memq dt '(s32 s64)) (- (modulo (+ (* 7 i) (* 13 seed)) 9) 4))
               ((and special? (= (modulo (+ i seed) 11) 3)) +nan.0)
               ((and special? (= (modulo (+ i seed) 7) 2)) -0.0)
               ((and special? (= (modulo (+ i seed) 7) 5)) 0.0)
               ((and special? (= (modulo (+ i seed) 17) 4)) -inf.0)
               ((and special? (= (modulo (+ i seed) 13) 1)) 3.0e7)
               (else (* 1.3 (sin (+ (* 1.37 i) seed))))))
       (iota n)))

;; The elements of realized m in row-major order of its shape.
(define (sc-elements m)
  (cases array-morphism (realize m)
    (concrete-array (data shape strides offset dtype _ _)
      (map (lambda (i)
             (typed-vector-ref data dtype
                               (multi-to-linear-index (linear-to-multi-index i shape)
                                                      strides offset)))
           (iota (shape-size shape))))
    (else (error "sc-elements: not concrete"))))

(define (sc-same? xs ys)
  (and (= (length xs) (length ys))
       (every (lambda (x y)
                (if (and (flonum? x) (flonum? y) (nan? x) (nan? y)) #t (eqv? x y)))
              xs ys)))

;; The op and metadata of morphism-expr m.
(define (sc-op+meta m)
  (cases array-morphism m
    (morphism-expr (_ op _ _ _ _ metadata _) (cons op metadata))
    (else #f)))

;; x rounded to dtype, as storing it would round it.
(define (sc-round x dt)
  (let ((v (allocate-typed-vector dt 1)))
    (typed-vector-set! v dt 0 x)
    (typed-vector-ref v dt 0)))

(define (sc-neutral op dt)
  (let ((f? (memq dt '(f32 f64))))
    (case op
      ((sum) (if f? 0.0 0))
      ((prod) (if f? 1.0 1))
      ((max) (if f? -inf.0 (if (eq? dt 's32) -2147483648 -9223372036854775808)))
      ((min) (if f? +inf.0 (if (eq? dt 's32) 2147483647 9223372036854775807))))))

;; One step of the plain loop: the value so far acc combined with x.
(define (sc-step op swapped?)
  (let ((f (case op
             ((sum) (lambda (a x) (if (and (flonum? a) (flonum? x)) (fp+ a x) (+ a x))))
             ((prod) *)
             ((max) (lambda (a x) (if (> x a) x a)))
             ((min) (lambda (a x) (if (< x a) x a))))))
    (if swapped? (lambda (a x) (f x a)) f)))

;; The reference scan of the row-major element list vals of shape (a list),
;; along axis, as a row-major list.  init-vals: #f or the row-major list of
;; the init array (shape without axis).
(define (sc-ref op vals shape axis dt #!key (excl #f) (rev #f) (init-vals #f) (swapped #f))
  (let* ((rank (length shape))
         (n (list-ref shape axis))
         (strides (vector->list (compute-strides (list->vector shape))))
         (out (make-vector (sc-product shape) #f))
         (v (list->vector vals))
         (iv (and init-vals (list->vector init-vals)))
         (step (sc-step op swapped))
         (par-shape (append (take shape axis) (drop shape (+ axis 1))))
         (par-strides (vector->list (compute-strides (list->vector par-shape)))))
    (for-each
     (lambda (p)
       (let* ((pidx (vector->list (linear-to-multi-index p (list->vector par-shape))))
              (base (apply + (map * (append (take pidx axis) (list 0) (drop pidx axis)) strides)))
              (seed (and iv (sc-round (vector-ref iv p) dt))))
         (let loop ((r 0) (acc seed))
           (when (< r n)
             (let* ((pos (if rev (- n 1 r) r))
                    (k (+ base (* pos (list-ref strides axis))))
                    (x (vector-ref v k))
                    (next (sc-round (if acc (step acc x) x) dt)))
               (vector-set! out k (if excl (or acc (sc-neutral op dt)) next))
               (loop (+ r 1) next))))))
     (iota (sc-product par-shape)))
    (vector->list out)))

;; Compare (morph-scan op m axis ...) with the reference on row-major data.
(define (sc-case op dt shape axis #!key (excl #f) (rev #f) (init? #f) (special #f) (swapped #f))
  (let* ((vals (map (lambda (x) (sc-round x dt)) (sc-vals (sc-product shape) 3 dt special)))
         (m (morph-from-list vals (list->vector shape) dt))
         (ax (if (< axis 0) (+ axis (length shape)) axis))
         (par-shape (append (take shape ax) (drop shape (+ ax 1))))
         (ivals (and init? (map (lambda (x) (sc-round x dt)) (sc-vals (sc-product par-shape) 5 dt special))))
         (init (and init? (morph-from-list ivals (list->vector par-shape) dt)))
         (s (morph-scan op m axis exclusive?: excl reverse?: rev init: init swapped?: swapped)))
    (sc-same? (sc-elements s)
              (sc-ref op vals shape ax dt excl: excl rev: rev init-vals: ivals swapped: swapped))))

;;;; Values

(test-group "scans: values against a plain loop"
  (for-each
   (lambda (dt)
     (for-each
      (lambda (shape+axes)
        (for-each
         (lambda (axis)
           (for-each
            (lambda (op)
              (let ((ok (every (lambda (flags)
                                 (sc-case op dt (car shape+axes) axis
                                          excl: (car flags) rev: (cadr flags) init?: (caddr flags)
                                          special: (memq dt '(f32 f64))))
                               '((#f #f #f) (#t #f #f) (#f #t #f) (#t #t #f)
                                 (#f #f #t) (#t #f #t) (#f #t #t) (#t #t #t)))))
                (test-assert (sprintf "~A ~A shape ~A axis ~A" op dt (car shape+axes) axis) ok)))
            '(sum prod max min)))
         (cdr shape+axes)))
      '(((7) 0 -1) ((3 5) 0 1 -2) ((2 3 4) 0 1 2) ((2 1 3 2) 1 3))))
   '(f32 f64 s32)))

(test-group "scans: special values"
  (test-assert "a first element -0.0 stays -0.0 in a sum"
    (eqv? -0.0 (car (sc-elements (morph-cumsum (morph-from-list '(-0.0 -0.0 1.0) #(3) 'f64) 0)))))
  (test-assert "-0.0 + -0.0 stays -0.0"
    (eqv? -0.0 (cadr (sc-elements (morph-cumsum (morph-from-list '(-0.0 -0.0 1.0) #(3) 'f64) 0)))))
  (test-assert "a first NaN is kept by max"
    (every nan? (sc-elements (morph-scan 'max (morph-from-list '(+nan.0 1.0 2.0) #(3) 'f64) 0))))
  (test "a later NaN is passed over by max" '(1.0 1.0 2.0)
        (sc-elements (morph-scan 'max (morph-from-list '(1.0 +nan.0 2.0) #(3) 'f64) 0)))
  (test "f32 sums round at every step" (make-list 4 1e8)
        (sc-elements (morph-cumsum (morph-from-list '(1e8 1.0 1.0 1.0) #(4) 'f32) 0)))
  (test "f64 sums keep the ones" '(1e8 100000001.0 100000002.0)
        (sc-elements (morph-cumsum (morph-from-list '(1e8 1.0 1.0) #(3) 'f64) 0)))
  (test "an axis of length 1, exclusive, without init" '(0.0 0.0)
        (sc-elements (morph-cumsum (morph-from-list '(5.0 6.0) #(2 1) 'f64) 1 exclusive?: #t)))
  (test "an axis of length 1, exclusive, with init" '(7.0 -1.0)
        (sc-elements (morph-cumsum (morph-from-list '(5.0 6.0) #(2 1) 'f64) 1 exclusive?: #t
                                   init: (morph-from-list '(7.0 -1.0) #(2) 'f64))))
  (test "cumprod" '(2.0 6.0 24.0)
        (sc-elements (morph-cumprod (morph-from-list '(2.0 3.0 4.0) #(3) 'f64) 0)))
  (test "reverse exclusive cumsum" '(5.0 3.0 0.0)
        (sc-elements (morph-cumsum (morph-from-list '(1.0 2.0 3.0) #(3) 'f64) 0
                                   exclusive?: #t reverse?: #t))))

;;;; Layouts

(test-group "scans: layouts"
  (let* ((vals (sc-vals 20 1 'f64 #t))
         (m (morph-from-list vals #(4 5) 'f64))
         (t (morph-transpose m '(1 0))))          ; [5 4], strided
    (for-each
     (lambda (axis)
       (test-assert (sprintf "transposed source, axis ~A" axis)
         (sc-same? (sc-elements (morph-cumsum t axis))
                   (sc-ref 'sum (sc-elements t) '(5 4) axis 'f64))))
     '(0 1)))
  (let* ((vals (sc-vals 30 2 'f32 #t))
         (m (morph-from-list vals #(5 6) 'f32))
         (s (morph-slice m '(1 1) '(5 6) '(2 2))))   ; [2 3], offset and steps
    (for-each
     (lambda (axis)
       (for-each
        (lambda (op)
          (test-assert (sprintf "sliced source, ~A axis ~A" op axis)
            (sc-same? (sc-elements (morph-scan op s axis reverse?: #t))
                      (sc-ref op (sc-elements s) '(2 3) axis 'f32 rev: #t))))
        '(sum max)))
     '(0 1)))
  (let* ((vals (sc-vals 12 4 'f64 #f))
         (m (morph-from-list vals #(3 4) 'f64))
         (init (morph-transpose (morph-from-list '(1.0 2.0 3.0) #(1 3) 'f64) '(1 0)))
         (init1 (morph-reshape init #(3))))
    (test-assert "an init that is a view"
      (sc-same? (sc-elements (morph-cumsum m 1 init: init1))
                (sc-ref 'sum vals '(3 4) 1 'f64 init-vals: '(1.0 2.0 3.0)))))
  (let ((m (morph-cumsum (morph-from-list (sc-vals 24 1 'f32 #t) #(2 3 4) 'f32) 1 reverse?: #t)))
    (test-assert "realize/ctx gives the values of realize"
      (sc-same? (sc-elements (realize/ctx (make-morphism-context) m)) (sc-elements m))))
  (let ((m (morph-cumsum (morph-exp (morph-from-list (sc-vals 12 1 'f64 #f) #(3 4) 'f64)) 0)))
    (test-assert "the operand of a scan may be an expression"
      (sc-same? (sc-elements m)
                (sc-ref 'sum (sc-elements (morph-exp (morph-from-list (sc-vals 12 1 'f64 #f) #(3 4) 'f64)))
                        '(3 4) 0 'f64)))))

;;;; Construction

(test-group "scans: construction"
  (let* ((m (morph-from-list (iota 6) #(2 3) 's32))
         (s (morph-scan 'sum m -1 exclusive?: #t)))
    (test "op" 'scan (car (sc-op+meta s)))
    (test "shape" '#(2 3) (get-morphism-shape s))
    (test "dtype" 's32 (get-morphism-dtype s))
    (test "metadata" '((scan-op . sum) (axis . 1) (exclusive? . #t) (reverse? . #f) (swapped? . #f))
          (cdr (sc-op+meta s)))
    (test "morph-cumsum is morph-scan sum" (sc-elements (morph-scan 'sum m 0))
          (sc-elements (morph-cumsum m 0))))
  (let ((m (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64)))
    (test-error "unknown op" (morph-scan 'mean m 0))
    (test-error "axis out of range" (morph-scan 'sum m 2))
    (test-error "negative axis out of range" (morph-scan 'sum m -3))
    (test-error "rank 0" (morph-scan 'sum (morph-reduce 'sum m) 0))
    (test-error "init of the wrong shape"
      (morph-scan 'sum m 0 init: (morph-from-list '(1.0 2.0 3.0) #(3) 'f64)))
    (test-error "init of the wrong dtype"
      (morph-scan 'sum m 0 init: (morph-from-list '(1.0 2.0) #(2) 'f32)))))

