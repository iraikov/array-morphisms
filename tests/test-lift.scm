;;; test-lift.scm
;;; Tests for morph-lift.
;;;
;;; Each lifted result is compared, element by element with eqv? (NaN
;;; equal to NaN), with fn applied to every cell of the arguments, each
;;; cell copied into an array of its own.

(import scheme (chicken base) (chicken format) (chicken condition))
(import test)
(import (only srfi-1 every iota append-map make-list take drop fold))
(import srfi-69)
(import datatype)
(import array-morphisms-core)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-blas-exec)
(import array-morphisms-lift)
(import array-morphisms-batch-ops)

;;;; Helpers

(define (lf-vals n seed)
  (map (lambda (i)
         (case (modulo (+ i seed) 17)
           ((4) -0.0)
           (else (* 0.9 (sin (+ (* 1.37 i) seed))))))
       (iota n)))

(define (lf-arr shape dt seed)
  (morph-from-list (lf-vals (fold * 1 shape) seed) (list->vector shape) dt))

(define (lf-elements m)
  (cases array-morphism (realize m)
    (concrete-array (data shape strides offset dtype _ _)
      (map (lambda (i)
             (typed-vector-ref data dtype
                               (multi-to-linear-index (linear-to-multi-index i shape)
                                                      strides offset)))
           (iota (shape-size shape))))
    (else (error "lf-elements: not concrete"))))

(define (lf-same? xs ys)
  (and (= (length xs) (length ys))
       (every (lambda (x y)
                (if (and (flonum? x) (flonum? y) (nan? x) (nan? y)) #t (eqv? x y)))
              xs ys)))

;; The cells of argument a with frame rank fr, each copied into an array
;; of its own, in row-major order of the frame.
(define (lf-cells a fr)
  (let* ((shape (vector->list (get-morphism-shape a)))
         (cell  (drop shape fr))
         (size  (fold * 1 cell))
         (n     (fold * 1 (take shape fr)))
         (vals  (lf-elements a))
         (dt    (get-morphism-dtype a)))
    (map (lambda (c)
           (morph-from-list (take (drop vals (* c size)) size) (list->vector cell) dt))
         (iota n))))

;; fn applied to every cell: arguments with shorter frames repeat along
;; the principal frame's trailing axes.
(define (lf-oracle fn args frs)
  (let* ((frame (fold (lambda (a fr best)
                        (if (> fr (length best)) (take (vector->list (get-morphism-shape a)) fr) best))
                      '() args frs))
         (n     (fold * 1 frame))
         (cells (map lf-cells args frs)))
    (append-map
     (lambda (c)
       (lf-elements
        (apply fn (map (lambda (cs fr)
                         (let ((inner (fold * 1 (drop frame fr))))
                           (list-ref cs (quotient c inner))))
                       cells frs))))
     (iota n))))

;; (list lifted-elements oracle-elements) for fn over args with cell ranks.
(define (lf-case fn ranks . args)
  (let* ((frs (map (lambda (a r) (- (vector-length (get-morphism-shape a)) r))
                   args (if (list? ranks) ranks (make-list (length args) ranks)))))
    (list (lf-elements (apply morph-lift fn ranks args))
          (lf-oracle fn args frs))))

(define (lf-check name fn ranks . args)
  (test-assert name (let ((r (apply lf-case fn ranks args))) (lf-same? (car r) (cadr r)))))

;; The number of distinct nodes in morphism m.
(define (lf-node-count m)
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

(define (lf-error? thunk)
  (condition-case (begin (thunk) #f) (e (exn) #t)))

(define (softmax-rows x)
  (let* ((m (morph-reduce 'max x '(1) #t))
         (e (morph-exp (morph- x m)))
         (s (morph-reduce 'sum e '(1) #t)))
    (morph/ e s)))

;;;; Tests

(test-group "morph-lift: element-wise bodies"
  (for-each
   (lambda (dt)
     (for-each
      (lambda (frame)
        (let ((f (lambda (x y) (morph-relu (morph+ (morph* x y) x)))))
          (lf-check (sprintf "~A frame ~A: relu(x*y + x), cells [5]" dt frame) f 1
                    (lf-arr (append frame '(5)) dt 0) (lf-arr (append frame '(5)) dt 3))
          (lf-check (sprintf "~A frame ~A: cells [3 4]" dt frame) f 2
                    (lf-arr (append frame '(3 4)) dt 0) (lf-arr (append frame '(3 4)) dt 3)))
        (let ((w (lf-arr '(3 4) dt 7)) (b (lf-arr '(4) dt 9)))
          (lf-check (sprintf "~A frame ~A: x*W + b with W and b captured" dt frame)
                    (lambda (x) (morph-tanh-am (morph+ (morph* x w) b))) 1
                    (lf-arr (append frame '(4)) dt 1))
          (lf-check (sprintf "~A frame ~A: cast, comparison, map" dt frame)
                    (lambda (x) (morph* (morph-cast (morph> x b) dt)
                                        (morph-map (lambda (v) (* v v)) x)))
                    1 (lf-arr (append frame '(4)) dt 2))))
      '((4) (2 3))))
   '(f32 f64)))

(test-group "broadcast with operands of another dtype"
  (let ((x (morph-from-list '(1.0 -2.0 3.0 0.5 1.0 2.0 3.0 4.0) #(2 4) 'f32))
        (b (morph-from-list '(0.0 0.0 1.0 1.0) #(4) 'f32)))
    (test "f32 rows compared with an f32 row give f64 values"
          '(1.0 0.0 1.0 0.0 1.0 1.0 1.0 1.0)
          (lf-elements (morph> x b)))))

(test-group "morph-lift: shorter frames and captured operands"
  (lf-check "x with frame (2 3) plus y with frame (2)"
            (lambda (x y) (morph- x y)) 1
            (lf-arr '(2 3 5) 'f64 0) (lf-arr '(2 5) 'f64 4))
  (lf-check "cell [5] plus captured [3 5]"
            (let ((w (lf-arr '(3 5) 'f64 8))) (lambda (x) (morph+ x w))) 1
            (lf-arr '(4 5) 'f64 0))
  (lf-check "a result that depends on no argument repeats along the frame"
            (let ((w (lf-arr '(3) 'f64 8))) (lambda (x) (morph-exp w))) 1
            (lf-arr '(4 5) 'f64 0))
  (let* ((w   (lf-arr '(6 6) 'f64 5))
         (f   (lambda (x) (morph-matmul (morph-exp w) (morph+ x (morph-negate w)))))
         (n4  (lf-node-count (morph-lift f 2 (lf-arr '(4 6 6) 'f64 0))))
         (n64 (lf-node-count (morph-lift f 2 (lf-arr '(64 6 6) 'f64 0)))))
    (test "the lifted tree has the same nodes for 4 and 64 cells" n4 n64)
    (lf-check "nodes of captured operands are kept as they are" f 2 (lf-arr '(4 6 6) 'f64 0))))

(test-group "morph-lift: reductions"
  (for-each
   (lambda (dt)
     (lf-check (sprintf "~A softmax over the rows of [5 9] cells" dt) softmax-rows 2
               (lf-arr '(4 5 9) dt 0))
     (lf-check (sprintf "~A softmax with frame (2 3)" dt) softmax-rows 2
               (lf-arr '(2 3 5 9) dt 1))
     (lf-check (sprintf "~A mean of [64] cells" dt)
               (lambda (v) (morph-reduce 'mean v '(0))) 1 (lf-arr '(6 64) dt 2))
     (lf-check (sprintf "~A column sums of [5 9] cells, keepdims" dt)
               (lambda (x) (morph-reduce 'sum x '(0) #t)) 2 (lf-arr '(4 5 9) dt 3))
     (lf-check (sprintf "~A sum of all of a [5 9] cell" dt)
               (lambda (x) (morph-reduce 'sum x)) 2 (lf-arr '(4 5 9) dt 4)))
   '(f32 f64)))

(test-group "morph-lift: matrix products"
  (for-each
   (lambda (dt)
     (let ((w (lf-arr '(8 10) dt 6)))
       (lf-check (sprintf "~A x W with W captured" dt) (lambda (x) (morph-matmul x w)) 2
                 (lf-arr '(4 9 8) dt 0))
       (lf-check (sprintf "~A W^T x with W captured" dt)
                 (lambda (x) (morph-matmul (morph-transpose w '(1 0)) x)) 2
                 (lf-arr '(4 8 3) dt 0)))
     (lf-check (sprintf "~A a b with both framed" dt) morph-matmul 2
               (lf-arr '(2 3 9 8) dt 0) (lf-arr '(2 3 8 10) dt 1))
     (lf-check (sprintf "~A x^T x" dt) (lambda (x) (morph-matmul (morph-transpose x '(1 0)) x)) 2
               (lf-arr '(5 9 8) dt 2))
     (lf-check (sprintf "~A a b with frames (2 3) and (2)" dt) morph-matmul 2
               (lf-arr '(2 3 9 8) dt 0) (lf-arr '(2 8 10) dt 1)))
   '(f32 f64)))

(test-group "morph-lift: views, stacks and padding"
  (lf-check "slice, transpose and reshape"
            (lambda (x) (morph-reshape (morph-transpose (morph-slice x '(1 0) '(4 6)) '(1 0))
                                       #(3 6)))
            2 (lf-arr '(4 5 6) 'f64 0))
  (lf-check "concat of a cell with itself" (lambda (x) (morph-concat (list x (morph-exp x)) 0)) 2
            (lf-arr '(4 3 5) 'f64 0))
  (lf-check "stack with a captured operand"
            (let ((w (lf-arr '(3 5) 'f64 3))) (lambda (x) (morph-stack (list x w) 1))) 2
            (lf-arr '(4 3 5) 'f64 0))
  (lf-check "padding" (lambda (x) (morph-pad x '((1 0) (0 2)))) 2 (lf-arr '(4 3 5) 'f64 0)))

(test-group "morph-lift: attention and convolution"
  (lf-check "attention per head"
            (lambda (q k v) (attention-morphism q k v)) 2
            (lf-arr '(4 6 8) 'f32 0) (lf-arr '(4 6 8) 'f32 1) (lf-arr '(4 6 8) 'f32 2))
  (lf-check "attention with frame (2 3)"
            (lambda (q k v) (attention-morphism q k v)) 2
            (lf-arr '(2 3 6 8) 'f64 0) (lf-arr '(2 3 6 8) 'f64 1) (lf-arr '(2 3 6 8) 'f64 2))
  (let ((wt (lf-arr '(4 18) 'f32 5)))
    (lf-check "im2col of one image per cell, times a captured weight"
              (lambda (x) (morph-matmul wt (im2col-morph x '(3 3) 1 1))) 3
              (lf-arr '(3 2 5 5) 'f32 0))
    (lf-check "im2col of a batch of images per cell"
              (lambda (x) (im2col-morph x '(3 3) 1 1)) 4
              (lf-arr '(3 2 2 5 5) 'f32 0))
    (lf-check "col2im of one image per cell"
              (lambda (c) (col2im-morph c #(2 5 5) '(3 3) 1 1)) 2
              (lf-arr '(3 18 25) 'f32 0))))

(test-group "morph-lift: arguments and errors"
  (let ((x (lf-arr '(4 5) 'f64 0)))
    (test-assert "an argument without a frame gives fn's result"
      (lf-same? (lf-elements (morph-lift morph-exp 2 x)) (lf-elements (morph-exp x))))
    (test-assert "frames that are not prefixes are an error"
      (lf-error? (lambda () (morph-lift morph+ 1 (lf-arr '(4 5) 'f64 0) (lf-arr '(3 5) 'f64 0)))))
    (test-assert "a cell rank above the rank is an error"
      (lf-error? (lambda () (morph-lift morph-exp 3 x))))
    (test-assert "fn must return a morphism"
      (lf-error? (lambda () (morph-lift (lambda (v) 1) 1 x))))
    (let ((v (lf-arr '(5) 'f64 1)))
      (test "an op without a rule signals lift-unsupported naming it" 'matvec
            (condition-case (begin (morph-lift (lambda (a) (morph-matvec a v)) 2
                                               (lf-arr '(4 3 5) 'f64 0))
                                   #f)
              (c (lift-unsupported) (lift-unsupported-op c)))))))

(test-group "batch-map and batch-zip on morph-lift"
  (for-each
   (lambda (dt)
     (let ((m (lf-arr '(4 3 5) dt 0)) (w (lf-arr '(5) dt 2)))
       (for-each
        (lambda (axis)
          (let ((f (lambda (x) (morph-sigmoid (morph* x (morph-exp x))))))
            (test-assert (sprintf "~A batch-map axis ~A: values of batch-map/slices" dt axis)
              (lf-same? (lf-elements (batch-map f m axis))
                        (lf-elements (batch-map/slices f m axis))))
            (test (sprintf "~A batch-map axis ~A: batch axis" dt axis) axis
                  (get-morphism-batch-axis (batch-map f m axis)))))
        '(0 1 2))
       (test-assert (sprintf "~A batch-map of a softmax: the cells' own values" dt)
         (lf-same? (lf-elements (batch-map softmax-rows m))
                   (lf-oracle softmax-rows (list m) '(1))))
       (let ((wm (lf-arr '(5 2) dt 4)))
         (test-assert (sprintf "~A batch-map of x W: the cells' own values" dt)
           (lf-same? (lf-elements (batch-map (lambda (x) (morph-matmul x wm)) m))
                     (lf-oracle (lambda (x) (morph-matmul x wm)) (list m) '(1)))))
       (test-assert (sprintf "~A batch-zip morph*: values of batch-zip/slices" dt)
         (lf-same? (lf-elements (batch-zip morph* m (lf-arr '(4 3 5) dt 1)))
                   (lf-elements (batch-zip/slices morph* m (lf-arr '(4 3 5) dt 1)))))
       (test-assert (sprintf "~A batch-zip of matrix products: the cells' own values" dt)
         (let ((b (lf-arr '(4 5 2) dt 3)))
           (lf-same? (lf-elements (batch-zip morph-matmul m b))
                     (lf-oracle morph-matmul (list m b) '(1 1)))))))
   '(f32 f64))
  (let ((f (lambda (x) (morph-negate x))))
    (test "batch-map has the same nodes for 4 and 64 elements"
          (lf-node-count (batch-map f (lf-arr '(4 3) 'f64 0)))
          (lf-node-count (batch-map f (lf-arr '(64 3) 'f64 0))))
    (test-assert "batch-map/slices has a node per element"
      (> (lf-node-count (batch-map/slices f (lf-arr '(64 3) 'f64 0)))
         (lf-node-count (batch-map/slices f (lf-arr '(4 3) 'f64 0))))))
  (let ((m (lf-arr '(4 3 5) 'f64 0)) (v (lf-arr '(5) 'f64 1)))
    (test-assert "an fn morph-lift cannot widen runs on slices"
      (lf-same? (lf-elements (batch-map (lambda (a) (morph-matvec a v)) m))
                (lf-elements (batch-map/slices (lambda (a) (morph-matvec a v)) m))))
    (let ((data-dependent (lambda (a) (morph-from-list (map (lambda (x) (* 2.0 x)) (lf-elements a))
                                                       (get-morphism-shape a) 'f64))))
      (test-assert "an fn that reads its argument's values runs on slices"
        (lf-same? (lf-elements (batch-map data-dependent m))
                  (lf-elements (batch-map/slices data-dependent m)))))
    (test-assert "batch-zip falls back as well"
      (lf-same? (lf-elements (batch-zip (lambda (a b) (morph-matvec a (morph-reduce 'sum b '(0)))) m m))
                (lf-elements (batch-zip/slices
                              (lambda (a b) (morph-matvec a (morph-reduce 'sum b '(0)))) m m))))))

;;;; Scans inside lifted functions

(test-group "morph-lift: scans"
  (for-each
   (lambda (dt)
     (lf-check (sprintf "~A cumsum of each cell along its axis 1" dt)
               (lambda (x) (morph-cumsum x 1)) 2 (lf-arr '(3 4 5) dt 1))
     (lf-check (sprintf "~A reverse exclusive max with a captured init" dt)
               (let ((init (lf-arr '(5) dt 2)))
                 (lambda (x) (morph-scan 'max x 0 reverse?: #t exclusive?: #t init: init)))
               2 (lf-arr '(3 4 5) dt 3))
     (lf-check (sprintf "~A a combine expression with a framed init" dt)
               (lambda (x i) (morph-scan '(lambda (a b) (+ (* a 0.5) b)) x 0 init: i))
               (list 2 1) (lf-arr '(2 3 4 5) dt 4) (lf-arr '(2 3 5) dt 5))
     (lf-check (sprintf "~A a scan after an element-wise op, frame of rank 2" dt)
               (lambda (x) (morph-cumprod (morph-exp x) 0)) 1 (lf-arr '(2 3 4) dt 6)))
   '(f32 f64))
  (test-assert "a lifted scan is one scan node"
    (= (lf-node-count (morph-lift (lambda (x) (morph-cumsum x 0)) 1 (lf-arr '(4 5) 'f64 1)))
       (lf-node-count (morph-lift (lambda (x) (morph-cumsum x 0)) 1 (lf-arr '(64 5) 'f64 1))))))

(test-group "morph-lift: scans with aux arrays"
  (for-each
   (lambda (dt)
     (lf-check (sprintf "~A a linear recurrence with a framed aux array" dt)
               (lambda (x c) (morph-scan '(lambda (a b c) (+ b (* c a))) x 1 aux: (list c)))
               2 (lf-arr '(3 4 5) dt 1) (lf-arr '(3 4 5) dt 2))
     (lf-check (sprintf "~A a captured aux array" dt)
               (let ((c (lf-arr '(4 5) dt 3)))
                 (lambda (x) (morph-scan '(lambda (a b c) (- b (* c a))) x 0 aux: (list c) reverse?: #t)))
               2 (lf-arr '(2 4 5) dt 4)))
   '(f32 f64)))
