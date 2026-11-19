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
(import array-morphisms-ssa)
(import (prefix array-morphisms-grad am:))
(import array-morphisms-lift)
(import (only array-morphisms-batch-ops batch-scan batch-scan/slices))
(import srfi-69)

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
  (let ((f (if (procedure? op) op (case op
             ((sum) (lambda (a x) (if (and (flonum? a) (flonum? x)) (fp+ a x) (+ a x))))
             ((prod) *)
             ((max) (lambda (a x) (if (> x a) x a)))
             ((min) (lambda (a x) (if (< x a) x a)))))))
    (if swapped? (lambda (a x . c) (apply f x a c)) f)))

;; The reference scan of the row-major element list vals of shape (a list),
;; along axis, as a row-major list.  init-vals: #f or the row-major list of
;; the init array (shape without axis).
(define (sc-ref op vals shape axis dt #!key (excl #f) (rev #f) (init-vals #f) (swapped #f) (neutral #f)
                (aux-vals '()))
  (let* ((rank (length shape))
         (n (list-ref shape axis))
         (strides (vector->list (compute-strides (list->vector shape))))
         (out (make-vector (sc-product shape) #f))
         (v (list->vector vals))
         (avs (map list->vector aux-vals))
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
                    (next (sc-round (if acc (apply step acc x (map (lambda (a) (vector-ref a k)) avs)) x) dt)))
               (vector-set! out k (if excl (or acc neutral (sc-neutral op dt)) next))
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


;;;; Combine expressions and procedures

(define (sc-lift-unsupported? thunk)
  (condition-case (begin (thunk) #f) (e (lift-unsupported) #t)))

(test-group "scans with combine expressions"
  (for-each
   (lambda (dt)
     (let* ((vals (map (lambda (x) (sc-round x dt)) (sc-vals 24 3 dt #f)))
            (m (morph-from-list vals #(4 6) dt))
            (r (lambda (x) (sc-round x dt))))
       ;; f64 needs no cast; f32 rounds the product as the expression says.
       (test-assert (sprintf "~A exponential moving sum equals a plain loop" dt)
         (sc-same? (sc-elements (morph-scan `(lambda (a b) (+ (cast ,dt (* a 0.5)) b)) m 1))
                   (sc-ref (lambda (a x) (r (+ (r (* a 0.5)) x))) vals '(4 6) 1 dt)))
       (test-assert (sprintf "~A expression along axis 0, reversed, exclusive with init" dt)
         (let ((init (morph-from-list (map r '(1.0 -2.0 0.5 3.0 0.0 -0.0)) #(6) dt)))
           (sc-same? (sc-elements (morph-scan '(lambda (a b) (- a b)) m 0 reverse?: #t exclusive?: #t
                                              init: init))
                     (sc-ref (lambda (a x) (- a x)) vals '(4 6) 0 dt rev: #t excl: #t
                             init-vals: (map r '(1.0 -2.0 0.5 3.0 0.0 -0.0))))))
       (test-assert (sprintf "~A exclusive with a neutral element" dt)
         (sc-same? (sc-elements (morph-scan '(lambda (a b) (* a b)) m 1 exclusive?: #t neutral: 1.0))
                   (sc-ref (lambda (a x) (r (* a x))) vals '(4 6) 1 dt excl: #t neutral: 1.0)))))
   '(f32 f64))

  ;; morph-scan/fn: the traced procedure, against the step-by-step program
  ;; of batch-scan (slice-and-stack), bit for bit.
  (for-each
   (lambda (dt)
     (let* ((vals (sc-vals 20 1 dt #t))
            (m (morph-from-list vals #(5 4) dt))
            (k (morph-from-list '(0.9) #(1) dt))
            (fn (lambda (acc x) (morph+ (morph* acc k) (morph-exp (morph-negate x)))))
            (init (morph-from-list '(0.0 1.0 -1.0 2.0) #(4) dt)))
       (test "the traced expression"
             (if (eq? dt 'f64)
                 `(lambda (a b) (add (mul a ,(sc-round 0.9 dt)) (exp (negate b))))
                 `(lambda (a b) (add (cast f32 (mul a ,(sc-round 0.9 dt))) (cast f32 (exp (cast f32 (negate b)))))))
             (scan-combine-expression fn dt))
       (test-assert (sprintf "~A morph-scan/fn equals batch-scan's slice-and-stack values" dt)
         (sc-same? (sc-elements (morph-scan/fn fn m 0 init: init))
                   (sc-elements (batch-scan/slices fn init m))))))
   '(f32 f64))
  (test "operand order is kept" '(lambda (a b) (sub b a))
        (scan-combine-expression (lambda (acc x) (morph- x acc)) 'f64))

  (let ((c (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
        (h (morph-from-list '(1.0) #(1) 'f64)))
    (test-assert "a map step is not a combine expression"
      (sc-lift-unsupported? (lambda () (scan-combine-expression (lambda (a b) (morph-map (lambda (x) x) (morph+ a b))) 'f64))))
    (test-assert "a captured array that is not a scalar"
      (sc-lift-unsupported? (lambda () (scan-combine-expression (lambda (a b) (morph+ (morph* a c) b)) 'f64))))
    (test-assert "a reduction in the step"
      (sc-lift-unsupported? (lambda () (scan-combine-expression (lambda (a b) (morph+ (morph-reduce 'sum a) b)) 'f64))))
    (test-assert "a constant of another dtype"
      (sc-lift-unsupported? (lambda () (scan-combine-expression (lambda (a b) (morph+ (morph* a h) b)) 'f32)))))

  (let ((m (morph-from-list '(1.0 2.0) #(2) 'f64)))
    (test-error "an exclusive custom scan needs init or a neutral element"
      (morph-scan '(lambda (a b) (+ a b)) m 0 exclusive?: #t))
    (test-error "a neutral element only with a combine expression"
      (morph-scan 'sum m 0 neutral: 0.0))
    (test-error "an invalid combine expression"
      (morph-scan '(lambda (a b) (+ a c)) m 0))))

;;;; Gradients

;; The elements of concrete array m (row-major order of its shape).
(define (sc-concrete-elements m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype _ _)
      (map (lambda (i) (typed-vector-ref data dtype
                                         (multi-to-linear-index (linear-to-multi-index i shape)
                                                                strides offset)))
           (iota (shape-size shape))))
    (else (error "sc-concrete-elements: not concrete"))))

;; The elements of the row-major list vals of shape (a list) at index pos
;; along axis, in row-major order of the other axes.
(define (sc-axis-at vals shape axis pos)
  (let ((strides (vector->list (compute-strides (list->vector shape))))
        (rest (append (take shape axis) (drop shape (+ axis 1)))))
    (map (lambda (p)
           (let ((idx (vector->list (linear-to-multi-index p (list->vector rest)))))
             (list-ref vals (apply + (map * (append (take idx axis) (list pos) (drop idx axis)) strides)))))
         (iota (sc-product rest)))))

;; loss = sum(scan(X, init: I) * c) through ssa-vjp, traced then replayed:
;; (list replayed-gradients traced-gradients joint), the gradients of X and
;; (with init) of I as element lists.
(define (sc-ssa-grads dt shape axis excl rev init? c-vals)
  (let* ((X (morph-from-list (sc-vals (sc-product shape) 1 dt #f) (list->vector shape) dt))
         (ishape (append (take shape axis) (drop shape (+ axis 1))))
         (I (and init? (morph-from-list (sc-vals (sc-product ishape) 2 dt #f) (list->vector ishape) dt)))
         (c (morph-from-list c-vals (list->vector shape) dt))
         (loss (am:make-var (morph-reduce 'sum (morph* (morph-scan 'sum X axis exclusive?: excl reverse?: rev init: I) c)) #f))
         (fwd (morphism-to-ssa loss))
         (ps (if I (list X I) (list X)))
         (joint (ssa-vjp fwd (map (lambda (p) (ssa-constant-id fwd p)) ps) (ssa-loss-binding-val fwd)))
         (ctx (make-morphism-context))
         (run (lambda ()
                (let ((res (ssa-realize/ctx ctx joint)))
                  (unless (eq? (context-mode ctx) 'replay) (finalize-context! ctx))
                  (reset-context! ctx)
                  (map sc-concrete-elements (cdr res)))))
         (traced (run)))
    (list (run) traced joint)))

(test-group "scan gradients"
  (for-each
   (lambda (dt)
     (for-each
      (lambda (axis)
        (for-each
         (lambda (flags)
           (let* ((excl (car flags)) (rev (cadr flags)) (init? (caddr flags))
                  (shape '(3 4 5))
                  (c-vals (map (lambda (x) (sc-round x dt)) (sc-vals 60 7 dt #t)))
                  (r (sc-ssa-grads dt shape axis excl rev init? c-vals))
                  (name (sprintf "~A axis ~A~A~A~A" dt axis (if excl " exclusive" "")
                                 (if rev " reverse" "") (if init? " init" ""))))
             (test-assert (string-append name ": dX is the opposite scan of g, as a plain loop")
               (sc-same? (car (car r)) (sc-ref 'sum c-vals shape axis dt excl: excl rev: (not rev))))
             (test-assert (string-append name ": replay equals the trace")
               (every sc-same? (car r) (cadr r)))
             (when init?
               (test-assert (string-append name ": dinit is the end of the reverse inclusive scan of g")
                 (sc-same? (cadr (car r))
                           (sc-axis-at (sc-ref 'sum c-vals shape axis dt rev: (not rev)) shape axis
                                       (if rev (- (list-ref shape axis) 1) 0)))))))
         '((#f #f #f) (#t #f #f) (#f #t #f) (#t #t #t) (#f #f #t))))
      '(0 2)))
   '(f32 f64))

  ;; Every fusion policy gives the same gradients.
  (let* ((c-vals (sc-vals 60 7 'f32 #f))
         (run (lambda (policy)
                (parameterize ((segop-fusion policy))
                  (car (sc-ssa-grads 'f32 '(3 4 5) 1 #f #t #t c-vals)))))
         (off (run #f)))
    (test-assert "fusion on equals fusion off" (every sc-same? off (run #t)))
    (test-assert "fusion 'all equals fusion off" (every sc-same? off (run 'all))))

  ;; Eager backward! through var-scan gives the gradients of ssa-vjp.
  (for-each
   (lambda (dt)
     (let* ((shape '(3 4 5)) (axis 1)
            (c-vals (map (lambda (x) (sc-round x dt)) (sc-vals 60 7 dt #t)))
            (xv (am:make-var (morph-from-list (sc-vals 60 1 dt #f) #(3 4 5) dt) #t))
            (iv (am:make-var (morph-from-list (sc-vals 15 2 dt #f) #(3 5) dt) #t))
            (cv (am:make-var (morph-from-list c-vals #(3 4 5) dt) #f))
            (loss (am:var-sum (am:var* (am:var-scan 'sum xv axis exclusive?: #t reverse?: #t init: iv) cv))))
       (am:backward! loss)
       (let ((r (sc-ssa-grads dt shape axis #t #t #t c-vals)))
         (test-assert (sprintf "~A eager dX equals ssa-vjp's" dt)
           (sc-same? (sc-elements (am:var-grad xv)) (car (car r))))
         (test-assert (sprintf "~A eager dinit equals ssa-vjp's" dt)
           (sc-same? (sc-elements (am:var-grad iv)) (cadr (car r)))))))
   '(f32 f64))

  (test-assert "ssa-vjp: a product scan of a parameter has a gradient"
    (let* ((X (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (fwd (morphism-to-ssa (am:make-var (morph-reduce 'sum (morph-cumprod X 0)) #f))))
      (ssa-vjp fwd (list (ssa-constant-id fwd X)) (ssa-loss-binding-val fwd))
      #t))
  (test-assert "ssa-vjp: a max scan of a constant needs no gradient"
    (let* ((X (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (K (morph-from-list '(3.0 1.0 2.0) #(3) 'f64))
           (fwd (morphism-to-ssa (am:make-var (morph-reduce 'sum (morph* X (morph-scan 'max K 0))) #f))))
      (ssa-vjp fwd (list (ssa-constant-id fwd X)) (ssa-loss-binding-val fwd))
      #t))
  (test-error "ssa-vjp: a combine-expression scan of a parameter has no gradient"
    (let* ((X (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (fwd (morphism-to-ssa (am:make-var (morph-reduce 'sum (morph-scan '(lambda (a b) (+ a b)) X 0)) #f))))
      (ssa-vjp fwd (list (ssa-constant-id fwd X)) (ssa-loss-binding-val fwd))))
  (test-error "var-scan: a combine expression of a variable that requires grad"
    (am:var-scan '(lambda (a b) (+ a b)) (am:make-var (morph-from-list '(1.0 2.0) #(2) 'f64) #t) 0))
  (test-assert "var-scan: a product scan of a variable that requires grad"
    (am:morph-variable? (am:var-scan 'prod (am:make-var (morph-from-list '(1.0 2.0) #(2) 'f64) #t) 0)))
  (test-assert "var-scan: a max scan without grad"
    (am:morph-variable? (am:var-scan 'max (am:make-var (morph-from-list '(1.0 2.0) #(2) 'f64) #f) 0))))

;;;; batch-scan

;; The number of distinct nodes of morphism m.
(define (sc-node-count m)
  (let ((seen (make-hash-table equal?)) (count 0))
    (let walk ((m m))
      (cases array-morphism m
        (concrete-array (_ _ _ _ _ _ _) (set! count (+ count 1)))
        (morphism-expr (id _ operands _ _ _ _ _)
          (unless (hash-table-ref/default seen (symbol->string id) #f)
            (hash-table-set! seen (symbol->string id) #t)
            (set! count (+ count 1))
            (for-each walk operands)))
        (reduction-morphism (id _ operand _ _ _ _ _)
          (unless (hash-table-ref/default seen (symbol->string id) #f)
            (hash-table-set! seen (symbol->string id) #t)
            (set! count (+ count 1))
            (walk operand)))))
    count))

(define (sc-op-of m) (cases array-morphism m (morphism-expr (_ op _ _ _ _ _ _) op) (else #f)))

;; batch-scan and batch-scan/slices give the same morphism shape, dtype,
;; batch axis and elements.
(define (sc-batch-same? fn init m)
  (let ((a (batch-scan fn init m)) (b (batch-scan/slices fn init m)))
    (and (equal? (get-morphism-shape a) (get-morphism-shape b))
         (eq? (get-morphism-dtype a) (get-morphism-dtype b))
         (= (get-morphism-batch-axis a) (get-morphism-batch-axis b))
         (sc-same? (sc-elements a) (sc-elements b)))))

(test-group "batch-scan as one scan"
  (for-each
   (lambda (dt)
     (let* ((m (morph-from-list (sc-vals 24 1 dt #t) #(6 4) dt))
            (init (morph-from-list (sc-vals 4 2 dt #t) #(4) dt))
            (k (morph-from-list '(0.75) #(1) dt))
            (steps (list (cons "morph+" morph+)
                         (cons "morph*" morph*)
                         (cons "the element first" (lambda (acc x) (morph+ x acc)))
                         (cons "a moving sum" (lambda (acc x) (morph+ (morph* acc k) x)))
                         (cons "exp and negate" (lambda (acc x) (morph- acc (morph-exp (morph-negate x))))))))
       (for-each
        (lambda (st)
          (test-assert (sprintf "~A ~A: one scan node" dt (car st))
            (eq? 'scan (sc-op-of (batch-scan (cdr st) init m))))
          (test-assert (sprintf "~A ~A: equal to slice-and-stack" dt (car st))
            (sc-batch-same? (cdr st) init m)))
        steps)
       (test-assert (sprintf "~A: as many nodes for 4 and 64 elements" dt)
         (= (sc-node-count (batch-scan morph+ init (morph-from-list (sc-vals 16 1 dt #f) #(4 4) dt)))
            (sc-node-count (batch-scan morph+ init (morph-from-list (sc-vals 256 1 dt #f) #(64 4) dt)))))))
   '(f32 f64))

  ;; Through SSA replay.
  (let* ((make (lambda () (batch-scan (lambda (acc x) (morph+ (morph* acc (morph-from-list '(0.5) #(1) 'f32)) x))
                                      (morph-from-list '(1.0 -1.0 0.0) #(3) 'f32)
                                      (morph-from-list (sc-vals 15 3 'f32 #f) #(5 3) 'f32))))
         (fwd (morphism-to-ssa (am:make-var (make) #f)))
         (ctx (make-morphism-context))
         (_ (begin (ssa-realize/ctx ctx fwd) (finalize-context! ctx) (reset-context! ctx)))
         (out (sc-concrete-elements (car (ssa-realize/ctx ctx fwd)))))
    (reset-context! ctx)
    (test-assert "a batch-scan replays and gives the values of realize"
      (sc-same? out (sc-elements (make)))))
  (let* ((make (lambda () (batch-scan/slices morph+ (morph-from-list '(1.0 -1.0 0.0) #(3) 'f64)
                                             (morph-from-list (sc-vals 15 3 'f64 #t) #(5 3) 'f64))))
         (fwd (morphism-to-ssa (am:make-var (make) #f)))
         (ctx (make-morphism-context))
         (_ (begin (ssa-realize/ctx ctx fwd) (finalize-context! ctx) (reset-context! ctx)))
         (out (sc-concrete-elements (car (ssa-realize/ctx ctx fwd)))))
    (reset-context! ctx)
    (test-assert "batch-scan/slices replays and gives the values of realize"
      (sc-same? out (sc-elements (make)))))

  ;; Fallbacks: each gives the values of slice-and-stack.
  (let* ((m (morph-from-list (sc-vals 12 4 'f64 #f) #(3 4) 'f64))
         (init (morph-from-list '(0.0 0.0 0.0 0.0) #(4) 'f64))
         (w (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64)))
    (test-assert "a captured array of the element's shape falls back"
      (let ((fn (lambda (acc x) (morph+ (morph* acc w) x))))
        (and (not (eq? 'scan (sc-op-of (batch-scan fn init m)))) (sc-batch-same? fn init m))))
    (test-assert "a morph-map step falls back"
      (let ((fn (lambda (acc x) (morph-map (lambda (v) (* 2.0 v)) (morph+ acc x)))))
        (and (not (eq? 'scan (sc-op-of (batch-scan fn init m)))) (sc-batch-same? fn init m))))
    (test-assert "an init of another dtype falls back"
      (let ((i32 (morph-from-list '(0.0 0.0 0.0 0.0) #(4) 'f32)))
        (not (eq? 'scan (sc-op-of (batch-scan morph+ i32 m))))))
    (test-assert "a step that reads values falls back"
      (let ((fn (lambda (acc x) (if (> (car (sc-elements x)) 100.0) acc (morph+ acc x)))))
        (and (not (eq? 'scan (sc-op-of (batch-scan fn init m)))) (sc-batch-same? fn init m)))))

  ;; The gradient of a batch-scan with morph+ is the reverse cumulative sum of g.
  (let* ((X (morph-from-list (sc-vals 12 1 'f64 #f) #(4 3) 'f64))
         (init (morph-from-list '(0.0 0.0 0.0) #(3) 'f64))
         (c-vals (sc-vals 12 7 'f64 #t))
         (c (morph-from-list c-vals #(4 3) 'f64))
         (fwd (morphism-to-ssa (am:make-var (morph-reduce 'sum (morph* (batch-scan morph+ init X) c)) #f)))
         (joint (ssa-vjp fwd (list (ssa-constant-id fwd X)) (ssa-loss-binding-val fwd)))
         (ctx (make-morphism-context))
         (res (ssa-realize/ctx ctx joint)))
    (test-assert "a batch-scan's gradient is the reverse scan of g"
      (sc-same? (sc-concrete-elements (cadr res)) (sc-ref 'sum c-vals '(4 3) 0 'f64 rev: #t)))))

;;;; Scans with aux arrays

(test-group "scans with aux arrays"
  (for-each
   (lambda (dt)
     (let* ((shape '(3 4 5))
            (r (lambda (x) (sc-round x dt)))
            (xs (map r (sc-vals 60 3 dt #f)))
            (cs (map r (sc-vals 60 8 dt #f)))
            (m (morph-from-list xs #(3 4 5) dt))
            (c (morph-from-list cs #(3 4 5) dt))
            (rec '(lambda (a b c) (+ b (* c a)))))
       (for-each
        (lambda (axis)
          (for-each
           (lambda (flags)
             (let* ((excl (car flags)) (rev (cadr flags)) (init? (caddr flags))
                    (ishape (append (take shape axis) (drop shape (+ axis 1))))
                    (ivals (and init? (map r (sc-vals (sc-product ishape) 5 dt #f))))
                    (init (and init? (morph-from-list ivals (list->vector ishape) dt)))
                    (s (morph-scan rec m axis aux: (list c) exclusive?: excl reverse?: rev init: init
                                   neutral: (and excl (not init?) 0.0))))
               (test-assert (sprintf "~A axis ~A~A~A~A: a linear recurrence equals a plain loop" dt axis
                                     (if excl " exclusive" "") (if rev " reverse" "") (if init? " init" ""))
                 (sc-same? (sc-elements s)
                           (sc-ref (lambda (a x cc) (+ x (* cc a))) xs shape axis dt
                                   excl: excl rev: rev init-vals: ivals aux-vals: (list cs) neutral: 0.0)))))
           '((#f #f #f) (#t #f #t) (#f #t #f) (#t #t #f) (#f #f #t))))
        '(0 2))
       ;; SSA replay
       (let* ((make (lambda () (morph-scan rec (morph-exp m) 1 aux: (list c) reverse?: #t)))
              (fwd (morphism-to-ssa (am:make-var (make) #f)))
              (ctx (make-morphism-context))
              (_ (begin (ssa-realize/ctx ctx fwd) (finalize-context! ctx) (reset-context! ctx)))
              (out (sc-concrete-elements (car (ssa-realize/ctx ctx fwd)))))
         (reset-context! ctx)
         (test-assert (sprintf "~A: an aux scan replays with the values of realize" dt)
           (sc-same? out (sc-elements (make)))))))
   '(f32 f64))
  (let ((m (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
        (c3 (morph-from-list '(1.0 2.0 3.0) #(3) 'f64)))
    (test-error "aux arrays must have the operand's shape"
      (morph-scan '(lambda (a b c) (+ b (* c a))) m 0 aux: (list c3)))
    (test-error "one variable per aux array"
      (morph-scan '(lambda (a b) (+ a b)) m 0 aux: (list m)))
    (test-error "a standard op takes no aux arrays"
      (morph-scan 'sum m 0 aux: (list m)))))


;;;; Gradients of prod, max and min scans

;; The flat indices of each line of a row-major array of shape (a list)
;; along axis, in the scan's order.
(define (sc-lines shape axis rev)
  (let* ((strides (vector->list (compute-strides (list->vector shape))))
         (n (list-ref shape axis))
         (par (append (take shape axis) (drop shape (+ axis 1)))))
    (map (lambda (p)
           (let* ((pidx (vector->list (linear-to-multi-index p (list->vector par))))
                  (base (apply + (map * (append (take pidx axis) (list 0) (drop pidx axis)) strides)))
                  (ks (map (lambda (r) (+ base (* r (list-ref strides axis)))) (iota n))))
             (if rev (reverse ks) ks)))
         (iota (sc-product par)))))

;; The formulas of scan-gradient for one line, in the scan's order: xs the
;; elements, ys the outputs, gs the incoming gradient, s the seed or #f.
;; Returns (dx-list . dseed).
(define (sc-line-grad op dt xs ys gs s excl swapped)
  (let* ((r  (lambda (v) (sc-round v dt)))
         (n  (length xs))
         (gp (if excl (append (cdr gs) (list 0.0)) gs))
         (at (lambda (l i) (list-ref l i)))
         ;; h_k = gp_k + step(c_k, h_{k+1}), run from the end
         (recur (lambda (c step)
                  (let loop ((i (- n 1)) (h #f) (out '()))
                    (if (< i 0) out
                        (let ((v (if h (r (step (at gp i) (at c i) h)) (at gp i))))
                          (loop (- i 1) v (cons v out)))))))
         (extra (lambda (d) (if excl (r (+ d (car gs))) d))))
    (case op
      ((prod)
       (let* ((E (let loop ((i 0) (acc (and s (r s))) (out '()))
                   (if (= i n) (reverse out)
                       (loop (+ i 1) (if acc (r (* acc (at xs i))) (at xs i))
                             (cons (or acc (r 1.0)) out)))))
              (c (append (cdr xs) (list 1.0)))
              (S (recur c (lambda (b cc a) (+ b (* cc a)))))
              (dx (map (lambda (e v) (r (* e v))) E S)))
         (cons dx (and s (extra (r (* (car xs) (car S))))))))
      (else
       (let* ((rel (if (eq? op 'max) > <))
              (prev (lambda (i) (if excl (at ys i) (if (= i 0) s (at ys (- i 1))))))
              (raw (map (lambda (i)
                          (let ((p (prev i)))
                            (if (and (= i 0) (not s)) 1.0
                                (if swapped
                                    (if (rel p (at xs i)) 0.0 1.0)
                                    (if (rel (at xs i) p) 1.0 0.0)))))
                        (iota n)))
              (keep (map (lambda (m) (- 1.0 m)) raw))
              (c (append (cdr keep) (list 0.0)))
              (T (recur c (lambda (b cc a) (if (> cc 0.5) (+ b a) b))))
              (dx (map (lambda (m t) (r (* m t))) raw T)))
         (cons dx (and s (extra (r (* (car keep) (car T)))))))))))

;; The formulas over a whole array: (dx-list . dinit-list).
(define (sc-ref-grad op dt shape axis xs ys gs init-vals excl rev swapped)
  (let* ((lines (sc-lines shape axis rev))
         (dx (make-vector (sc-product shape) #f))
         (di '()))
    (for-each
     (lambda (ks p)
       (let ((res (sc-line-grad op dt (map (lambda (k) (list-ref xs k)) ks) (map (lambda (k) (list-ref ys k)) ks)
                                (map (lambda (k) (list-ref gs k)) ks) (and init-vals (list-ref init-vals p))
                                excl swapped)))
         (for-each (lambda (k v) (vector-set! dx k v)) ks (car res))
         (set! di (cons (cdr res) di))))
     lines (iota (length lines)))
    (cons (vector->list dx) (reverse di))))

;; loss = sum(scan(X, init: I) * c) through ssa-vjp, traced then replayed.
;; Returns (grads ys X-vals I-vals).
(define (sc-ssa-grads* op dt shape axis excl rev swapped init? xs c-vals #!optional (policy #t))
  (parameterize ((segop-fusion policy))
    (let* ((X (morph-from-list xs (list->vector shape) dt))
           (ishape (append (take shape axis) (drop shape (+ axis 1))))
           (ivals (and init? (map (lambda (x) (sc-round x dt)) (sc-vals (sc-product ishape) 2 dt #t))))
           (I (and init? (morph-from-list ivals (list->vector ishape) dt)))
           (sc (morph-scan op X axis exclusive?: excl reverse?: rev init: I swapped?: swapped))
           (c (morph-from-list c-vals (list->vector shape) dt))
           (loss (am:make-var (morph-reduce 'sum (morph* sc c)) #f))
           (fwd (morphism-to-ssa loss))
           (joint (ssa-vjp fwd (map (lambda (p) (ssa-constant-id fwd p)) (if I (list X I) (list X)))
                           (ssa-loss-binding-val fwd)))
           (ctx (make-morphism-context))
           (run (lambda ()
                  (let ((res (ssa-realize/ctx ctx joint)))
                    (unless (eq? (context-mode ctx) 'replay) (finalize-context! ctx))
                    (reset-context! ctx)
                    (map sc-concrete-elements (cdr res))))))
      (run)
      (list (run) (sc-elements sc) ivals))))

(test-group "prod, max and min scan gradients"
  (for-each
   (lambda (dt)
     (for-each
      (lambda (op)
        (for-each
         (lambda (shape+axis)
           (for-each
            (lambda (flags)
              (let* ((shape (car shape+axis)) (axis (cadr shape+axis))
                     (excl (car flags)) (rev (cadr flags)) (init? (caddr flags)) (swapped (cadddr flags))
                     (n (sc-product shape))
                     (xs (map (lambda (x) (sc-round x dt)) (sc-vals n 1 dt #t)))
                     (c-vals (map (lambda (x) (sc-round x dt)) (sc-vals n 7 dt #f)))
                     (r (sc-ssa-grads* op dt shape axis excl rev swapped init? xs c-vals))
                     (ref (sc-ref-grad op dt shape axis xs (cadr r) c-vals (caddr r) excl rev swapped))
                     (name (sprintf "~A ~A ~A axis ~A~A~A~A~A" dt op shape axis (if excl " exclusive" "")
                                    (if rev " reverse" "") (if init? " init" "") (if swapped " swapped" ""))))
                (test-assert (string-append name ": dX equals the formulas")
                  (sc-same? (car (car r)) (car ref)))
                (when init?
                  (test-assert (string-append name ": dinit equals the formulas")
                    (sc-same? (cadr (car r)) (cdr ref))))))
            (if (eq? op 'prod)
                '((#f #f #f #f) (#t #f #f #f) (#f #t #t #f) (#t #t #t #f) (#f #f #t #f))
                '((#f #f #f #f) (#t #f #f #f) (#f #t #t #f) (#t #t #t #f) (#f #f #t #t) (#t #f #f #t) (#f #t #f #t)))))
         '(((3 4 5) 0) ((3 4 5) 2) ((3 1 2) 1))))
      '(prod max min)))
   '(f32 f64))

  ;; Hand-computed cases, g = 1.
  (let ((grad (lambda (op xs)
                (let* ((X (morph-from-list xs (vector (length xs)) 'f64))
                       (fwd (morphism-to-ssa (am:make-var (morph-reduce 'sum (morph-scan op X 0)) #f)))
                       (joint (ssa-vjp fwd (list (ssa-constant-id fwd X)) (ssa-loss-binding-val fwd)))
                       (ctx (make-morphism-context)))
                  (sc-concrete-elements (cadr (ssa-realize/ctx ctx joint)))))))
    (test "prod with a zero" '(1.0 8.0 0.0) (grad 'prod '(2.0 0.0 3.0)))
    ;; y = (0, 0, 0): dy1/dx0 = x1 = 5, dy2/dx0 = x1 x2 = 0
    (test "prod with two zeros" '(6.0 0.0 0.0) (grad 'prod '(0.0 5.0 0.0)))
    (test "max: each element gets the gradient of the outputs it holds" '(1.0 2.0 0.0 2.0 0.0)
          (grad 'max '(1.0 3.0 2.0 5.0 4.0)))
    (test "max: a tie keeps the earlier element" '(1.0 2.0 0.0) (grad 'max '(1.0 3.0 3.0)))
    (test "min" '(2.0 0.0 1.0) (grad 'min '(2.0 4.0 1.0))))

  ;; Eager backward! gives the gradients of ssa-vjp.
  (for-each
   (lambda (op)
     (let* ((dt 'f64) (shape '(3 4 5)) (axis 1)
            (xs (sc-vals 60 1 dt #t)) (c-vals (sc-vals 60 7 dt #f))
            (r (sc-ssa-grads* op dt shape axis #t #t #f #t xs c-vals))
            (xv (am:make-var (morph-from-list xs #(3 4 5) dt) #t))
            (iv (am:make-var (morph-from-list (caddr r) #(3 5) dt) #t))
            (cv (am:make-var (morph-from-list c-vals #(3 4 5) dt) #f))
            (loss (am:var-sum (am:var* (am:var-scan op xv axis exclusive?: #t reverse?: #t init: iv) cv))))
       (am:backward! loss)
       (test-assert (sprintf "~A: eager gradients equal ssa-vjp's" op)
         (and (sc-same? (sc-elements (am:var-grad xv)) (car (car r)))
              (sc-same? (sc-elements (am:var-grad iv)) (cadr (car r)))))))
   '(prod max min))

  ;; The product against the step-by-step program (batch-scan/slices with
  ;; morph*), whose gradient adds in another order.
  (let* ((xs (map (lambda (i) (+ 0.5 (* 0.1 (modulo (* 7 i) 11)))) (iota 15)))
         (X (morph-from-list xs #(5 3) 'f64))
         (I (morph-from-list '(1.5 -0.5 2.0) #(3) 'f64))
         (c (morph-from-list (sc-vals 15 7 'f64 #f) #(5 3) 'f64))
         (grads (lambda (scan)
                  (let* ((fwd (morphism-to-ssa (am:make-var (morph-reduce 'sum (morph* (scan X I) c)) #f)))
                         (joint (ssa-vjp fwd (list (ssa-constant-id fwd X) (ssa-constant-id fwd I)) (ssa-loss-binding-val fwd)))
                         (ctx (make-morphism-context)))
                    (map sc-concrete-elements (cdr (ssa-realize/ctx ctx joint))))))
         (a (grads (lambda (x i) (morph-scan 'prod x 0 init: i))))
         (b (grads (lambda (x i) (batch-scan/slices morph* i x))))
         (close (lambda (u v) (every (lambda (p q) (<= (abs (- p q)) (* 1e-12 (max 1.0 (abs q))))) u v))))
    (test-assert "prod gradients agree with the step-by-step program within rounding"
      (and (close (car a) (car b)) (close (cadr a) (cadr b)))))

  ;; Fusion policies.
  (for-each
   (lambda (op)
     (let* ((xs (sc-vals 60 1 'f32 #t)) (c-vals (sc-vals 60 7 'f32 #f))
            (run (lambda (policy) (car (sc-ssa-grads* op 'f32 '(3 4 5) 2 #f #t #f #t xs c-vals policy)))))
       (test-assert (sprintf "~A: fusion policies give the same gradients" op)
         (let ((off (run #f)))
           (and (every sc-same? off (run #t)) (every sc-same? off (run 'all)))))))
   '(prod max min)))
