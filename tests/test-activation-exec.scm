;;; tests/test-activation-exec.scm
;;; Test suite for the activation-kernel registry (activation-exec.scm) and
;;; its use by the SSA replay engine (ri-activation-unary in ssa.scm,
;;; execute-activation-unary-compute in realization.scm).
;;;
;;; The kernels registered here are plain Scheme procedures standing in for
;;; compiled ones, so the routing can be checked without any native backend:
;;; a counting kernel records how often the replay engine calls it, and a
;;; marker kernel writes a value no combiner produces, which shows whether
;;; an output really came from the kernel.
;;;
;;; Organisation:
;;;   Group 1 - Registry: op names and backend records
;;;   Group 2 - execute-activation-unary-compute: kernel use and fallback
;;;   Group 3 - Replay-plan compilation: which instruction each binding gets
;;;   Group 4 - Replay through registered kernels: values and gradients

(import scheme (chicken base))
(import test)
(import (only srfi-1 iota every filter-map))
(import srfi-4)
(import datatype)
(import array-morphisms-core)
(import array-morphisms-basic-ops)
(import array-morphisms-realization)
(import array-morphisms-context)
(import array-morphisms-activation-exec)
(import (prefix array-morphisms-grad am:))
(import array-morphisms-ssa)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (ae-approx= a b) (< (abs (- a b)) 1e-9))

(define (ae-lists-approx= l1 l2)
  (and (= (length l1) (length l2)) (every ae-approx= l1 l2)))

;; Elements of a concrete array in logical (row-major) order, as flonums.
(define (ae-values m)
  (cases array-morphism (realize m)
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (map (lambda (i)
             (exact->inexact
              (typed-vector-ref data dtype
                                (multi-to-linear-index (linear-to-multi-index i shape)
                                                       strides offset))))
           (iota (shape-size shape))))
    (else (error "ae-values: not concrete"))))

(define (ae-f64-kernel f)
  (lambda (n in out)
    (do ((i 0 (+ i 1))) ((= i n))
      (f64vector-set! out i (f (f64vector-ref in i))))))

(define (ae-f32-kernel f)
  (lambda (n in out)
    (do ((i 0 (+ i 1))) ((= i n))
      (f32vector-set! out i (f (f32vector-ref in i))))))

;; The formulas of basic-ops.scm and of the ssa-vjp derivative maps.
(define (ae-relu x)    (max 0.0 x))
(define (ae-sigmoid x) (/ 1.0 (+ 1.0 (exp (- x)))))
(define (ae-tanh x)    (let ((e2 (exp (* 2.0 x)))) (/ (- e2 1.0) (+ e2 1.0))))
(define ae-formulas
  `((relu . ,ae-relu)
    (sigmoid . ,ae-sigmoid)
    (tanh . ,ae-tanh)
    (relu-deriv . ,(lambda (x) (if (> x 0.0) 1.0 0.0)))
    (sigmoid-deriv . ,(lambda (s) (* s (- 1.0 s))))
    (tanh-deriv . ,(lambda (t) (- 1.0 (* t t))))))

;; A backend with correct kernels for every default activation op and both
;; float dtypes.  Each kernel call is counted under its op name.
(define (make-counting-backend counts)
  (let ((be (make-activation-backend 'counting)))
    (for-each
     (lambda (entry)
       (let ((op (car entry)) (f (cdr entry)))
         (for-each
          (lambda (dtype mk)
            (let ((k (mk f)))
              (activation-backend-add-kernel!
               be op dtype
               (lambda (n in out)
                 (let ((p (assq op (cdr counts))))
                   (if p
                       (set-cdr! p (+ 1 (cdr p)))
                       (set-cdr! counts (cons (cons op 1) (cdr counts)))))
                 (k n in out)))))
          '(f64 f32) (list ae-f64-kernel ae-f32-kernel))))
     ae-formulas)
    be))

(define (count-of counts op)
  (let ((p (assq op (cdr counts)))) (if p (cdr p) 0)))

;; Trace, finalize and replay the joint forward+backward program of loss
;; with respect to params; returns (values joint replay-results) where
;; replay-results is a list of value lists (loss first, then gradients).
;; The replay step is the one that compiles and runs the replay plan.
(define (trace-and-replay loss params)
  (let* ((ctx    (make-morphism-context))
         (fwd    (morphism-to-ssa loss))
         (p-vals (filter-map (lambda (p) (ssa-constant-id fwd (am:var-value p)))
                             params))
         (joint  (ssa-vjp fwd p-vals (ssa-loss-binding-val fwd))))
    (ssa-realize/ctx ctx joint)
    (finalize-context! ctx)
    (reset-context! ctx)
    (let ((results (ssa-realize/ctx ctx joint)))
      (values joint (map ae-values results)))))

(define (plan-count joint tag)
  (let* ((counts (cdr (assq 'counts (replay-plan-stats joint))))
         (p      (assq tag counts)))
    (if p (cdr p) 0)))

(define (param-var lst shape dtype)
  (am:make-var (morph-from-list lst (list->vector shape) dtype) #t))

;; Make sure no backend left over from another test file is active.
(register-activation-backend! #f)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1 - Registry: op names and backend records
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "activation registry: op names"
  (test-assert "default activation ops are registered"
    (every activation-op-registered?
           '(relu sigmoid tanh relu-deriv sigmoid-deriv tanh-deriv)))
  (test "generic ops are not activation ops"
    '(#f #f #f #f)
    (map activation-op-registered? '(map negate exp (reduce sum))))
  (test "register-activation-op! adds a name once"
    '(#t 1)
    (begin
      (register-activation-op! 'ae-test-op)
      (register-activation-op! 'ae-test-op)
      (list (activation-op-registered? 'ae-test-op)
            (length (filter-map (lambda (o) (eq? o 'ae-test-op))
                                (activation-ops))))))
  (test-error "register-activation-op! rejects a non-symbol"
    (register-activation-op! "relu")))

(test-group "activation registry: backends"
  (test "a new backend has a name and no kernels"
    '(empty ())
    (let ((be (make-activation-backend 'empty)))
      (list (activation-backend-name be) (activation-backend-kernels be))))
  (test "kernels are found by op and dtype, misses give #f"
    '(#t #f #f #f)
    (let* ((be (make-activation-backend 'one))
           (k  (ae-f64-kernel ae-relu)))
      (activation-backend-add-kernel! be 'relu 'f64 k)
      (list (eq? k (lookup-activation-kernel be 'relu 'f64))
            (lookup-activation-kernel be 'relu 'f32)
            (lookup-activation-kernel be 'sigmoid 'f64)
            (lookup-activation-kernel #f 'relu 'f64))))
  (test "a later kernel for the same op and dtype replaces the earlier one"
    #t
    (let ((be (make-activation-backend 'two))
          (k2 (ae-f64-kernel ae-sigmoid)))
      (activation-backend-add-kernel! be 'relu 'f64 (ae-f64-kernel ae-relu))
      (activation-backend-add-kernel! be 'relu 'f64 k2)
      (and (eq? k2 (lookup-activation-kernel be 'relu 'f64))
           (= 1 (length (activation-backend-kernels be))))))
  (test-error "only f32 and f64 kernels are accepted"
    (activation-backend-add-kernel! (make-activation-backend 'x) 'relu 's32
                                    (lambda (n in out) #f)))
  (test-error "a kernel must be a procedure"
    (activation-backend-add-kernel! (make-activation-backend 'x) 'relu 'f64 42))
  (test "registering #f removes the active backend"
    '(#t #f)
    (let ((be (make-activation-backend 'reg)))
      (register-activation-backend! be)
      (let ((was (eq? be (active-activation-backend))))
        (register-activation-backend! #f)
        (list was (active-activation-backend)))))
  (test-error "register-activation-backend! rejects other values"
    (register-activation-backend! 'not-a-backend)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2 - execute-activation-unary-compute: kernel use and fallback
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; A backend whose relu kernels write 42.0 everywhere, which relu never does.
(define (make-marker-backend)
  (let ((be (make-activation-backend 'marker)))
    (activation-backend-add-kernel! be 'relu 'f64 (ae-f64-kernel (lambda (x) 42.0)))
    (activation-backend-add-kernel! be 'relu 'f32 (ae-f32-kernel (lambda (x) 42.0)))
    be))

(test-group "execute-activation-unary-compute"
  (define in64 (f64vector -1.5 0.0 2.5))
  (define in32 (f32vector -1.5 0.0 2.5))

  (test "no backend: same result as execute-flat-unary-compute"
    #t
    (let ((a (make-f64vector 3 0.0)) (b (make-f64vector 3 0.0)))
      (register-activation-backend! #f)
      (execute-activation-unary-compute 'relu ae-relu in64 'f64 a 3 'f64)
      (execute-flat-unary-compute ae-relu in64 'f64 b 3 'f64)
      (equal? (f64vector->list a) (f64vector->list b))))

  (test "backend kernel is used for f64 -> f64"
    '(42.0 42.0 42.0)
    (let ((out (make-f64vector 3 0.0)))
      (register-activation-backend! (make-marker-backend))
      (execute-activation-unary-compute 'relu ae-relu in64 'f64 out 3 'f64)
      (register-activation-backend! #f)
      (f64vector->list out)))

  (test "backend kernel is used for f32 -> f32"
    '(42.0 42.0 42.0)
    (let ((out (make-f32vector 3 0.0)))
      (register-activation-backend! (make-marker-backend))
      (execute-activation-unary-compute 'relu ae-relu in32 'f32 out 3 'f32)
      (register-activation-backend! #f)
      (f32vector->list out)))

  (test "mixed dtypes f32 -> f64 fall back to the combiner"
    '(0.0 0.0 2.5)
    (let ((out (make-f64vector 3 -9.0)))
      (register-activation-backend! (make-marker-backend))
      (execute-activation-unary-compute 'relu ae-relu in32 'f32 out 3 'f64)
      (register-activation-backend! #f)
      (f64vector->list out)))

  (test "integer dtypes fall back to the combiner"
    '(0 0 3)
    (let ((out (make-s32vector 3 -9)))
      (register-activation-backend! (make-marker-backend))
      (execute-activation-unary-compute 'relu (lambda (x) (max 0 x))
                                        (s32vector -2 0 3) 's32 out 3 's32)
      (register-activation-backend! #f)
      (s32vector->list out)))

  (test "an op without a kernel falls back to the combiner"
    #t
    (let ((out (make-f64vector 3 0.0)))
      (register-activation-backend! (make-marker-backend))
      (execute-activation-unary-compute 'sigmoid ae-sigmoid in64 'f64 out 3 'f64)
      (register-activation-backend! #f)
      (equal? (f64vector->list out) (map ae-sigmoid (f64vector->list in64)))))

  (test "size 0 leaves the output untouched"
    '(7.0)
    (let ((out (f64vector 7.0)))
      (register-activation-backend! (make-marker-backend))
      (execute-activation-unary-compute 'relu ae-relu in64 'f64 out 0 'f64)
      (register-activation-backend! #f)
      (f64vector->list out))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3 - Replay-plan compilation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "replay plan: instruction selection"
  (register-activation-backend! #f)

  (test "relu forward and relu-deriv compile to ri-activation-unary"
    2
    (let* ((xv (param-var '(-2.0 -1.0 0.0 1.0 2.0) '(5) 'f64))
           (loss (am:var-mean (am:var-relu xv))))
      (let-values (((joint _) (trace-and-replay loss (list xv))))
        (plan-count joint 'ri-activation-unary))))

  (test "sigmoid forward and sigmoid-deriv compile to ri-activation-unary"
    2
    (let* ((xv (param-var '(1.0 2.0) '(2) 'f64))
           (loss (am:var-mean (am:var-sigmoid xv))))
      (let-values (((joint _) (trace-and-replay loss (list xv))))
        (plan-count joint 'ri-activation-unary))))

  (test "a non-activation unary op still compiles to ri-flat-unary"
    '(0 #t)
    (let* ((xv (param-var '(1.0 2.0 3.0) '(3) 'f64))
           (loss (am:var-mean (am:var-exp xv))))
      (let-values (((joint _) (trace-and-replay loss (list xv))))
        (list (plan-count joint 'ri-activation-unary)
              (> (plan-count joint 'ri-flat-unary) 0)))))

  (test "sigmoid fused with its producer is not an ri-activation-unary"
    ;; negate's output feeds only sigmoid, so the fusion pass merges the
    ;; two into one binding with op sigmoid and combiner sigmoid(-x).
    ;; Only the backward sigmoid-deriv may use the activation instruction.
    1
    (let* ((xv (param-var '(0.5 -1.0 2.0) '(3) 'f64))
           (loss (am:var-mean (am:var-sigmoid (am:var-negate xv)))))
      (let-values (((joint _) (trace-and-replay loss (list xv))))
        (plan-count joint 'ri-activation-unary))))

  (test "an activation right after a matmul is still a GEMM epilogue"
    '(#t 1)
    ;; loss = mean(sigmoid(x @ W)).  The matmul output feeds only sigmoid
    ;; (sigmoid-deriv reads sigmoid's output), so sigmoid becomes the GEMM's
    ;; epilogue and sigmoid-deriv is the one activation instruction left.
    (let* ((xv (am:make-var (morph-from-list '(1.0 -2.0 3.0 -4.0 5.0 -6.0) #(2 3) 'f64) #f))
           (Wv (param-var '(0.1 -0.2 0.3 0.4 -0.5 0.6) '(3 2) 'f64))
           (loss (am:var-mean (am:var-sigmoid (am:var-matmul xv Wv)))))
      (let-values (((joint _) (trace-and-replay loss (list Wv))))
        (list (> (plan-count joint 'ri-gemm-epilogue) 0)
              (plan-count joint 'ri-activation-unary))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4 - Replay through registered kernels
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Replay loss once with no backend and once with a counting backend;
;; returns (list results-without results-with counts).
(define (replay-both make-loss)
  (register-activation-backend! #f)
  (let-values (((j0 without) (make-loss)))
    (let ((counts (list 'counts)))
      (register-activation-backend! (make-counting-backend counts))
      (let-values (((j1 with) (make-loss)))
        (register-activation-backend! #f)
        (list without with counts)))))

(test-group "replay through activation kernels"

  (test "relu: kernels are called and the gradient is unchanged"
    '(#t 1 1 (0.0 0.0 0.0 0.2 0.2))
    (let* ((r (replay-both
               (lambda ()
                 (let ((xv (param-var '(-2.0 -1.0 0.0 1.0 2.0) '(5) 'f64)))
                   (trace-and-replay (am:var-mean (am:var-relu xv)) (list xv))))))
           (without (car r)) (with (cadr r)) (counts (caddr r)))
      (list (equal? without with)
            (count-of counts 'relu) (count-of counts 'relu-deriv)
            (cadr with))))

  (test "sigmoid: kernels are called and the gradient is s(1-s)/n"
    #t
    (let* ((xs '(1.0 2.0))
           (r (replay-both
               (lambda ()
                 (let ((xv (param-var xs '(2) 'f64)))
                   (trace-and-replay (am:var-mean (am:var-sigmoid xv)) (list xv))))))
           (without (car r)) (with (cadr r)) (counts (caddr r)))
      (and (equal? without with)
           (= 1 (count-of counts 'sigmoid))
           (= 1 (count-of counts 'sigmoid-deriv))
           (ae-lists-approx= (cadr with)
                             (map (lambda (x) (let ((s (ae-sigmoid x))) (* 0.5 s (- 1.0 s))))
                                  xs)))))

  (test "tanh: kernels are called and the gradient is (1-t^2)/n"
    #t
    (let* ((xs '(0.5 -0.5))
           (r (replay-both
               (lambda ()
                 (let ((xv (param-var xs '(2) 'f64)))
                   (trace-and-replay (am:var-mean (am:var-tanh xv)) (list xv))))))
           (without (car r)) (with (cadr r)) (counts (caddr r)))
      (and (equal? without with)
           (= 1 (count-of counts 'tanh))
           (= 1 (count-of counts 'tanh-deriv))
           (ae-lists-approx= (cadr with)
                             (map (lambda (x) (let ((t (ae-tanh x))) (* 0.5 (- 1.0 (* t t)))))
                                  xs)))))

  (test "f32 graph: kernels are called and results are unchanged"
    '(#t #t)
    (let* ((r (replay-both
               (lambda ()
                 (let ((xv (param-var '(-1.0 0.25 0.5 3.0) '(4) 'f32)))
                   (trace-and-replay (am:var-mean (am:var-tanh (am:var-relu xv)))
                                     (list xv))))))
           (without (car r)) (with (cadr r)) (counts (caddr r)))
      (list (equal? without with)
            (> (count-of counts 'tanh-deriv) 0))))

  (test "matmul with a sigmoid epilogue: results unchanged with kernels"
    '(#t 0 1)
    (let* ((r (replay-both
               (lambda ()
                 (let ((xv (am:make-var (morph-from-list '(1.0 -2.0 3.0 -4.0 5.0 -6.0)
                                                         #(2 3) 'f64) #f))
                       (Wv (param-var '(0.1 -0.2 0.3 0.4 -0.5 0.6) '(3 2) 'f64)))
                   (trace-and-replay (am:var-mean (am:var-sigmoid (am:var-matmul xv Wv)))
                                     (list Wv))))))
           (without (car r)) (with (cadr r)) (counts (caddr r)))
      (list (equal? without with)
            (count-of counts 'sigmoid)
            (count-of counts 'sigmoid-deriv))))

  (test "fused sigmoid(-x) is computed by its combiner, not the sigmoid kernel"
    #t
    (let* ((xs '(0.5 -1.0 2.0))
           (r (replay-both
               (lambda ()
                 (let ((xv (param-var xs '(3) 'f64)))
                   (trace-and-replay (am:var-mean (am:var-sigmoid (am:var-negate xv)))
                                     (list xv))))))
           (without (car r)) (with (cadr r)) (counts (caddr r))
           (expected-loss (/ (apply + (map (lambda (x) (ae-sigmoid (- x))) xs)) 3.0)))
      (and (equal? without with)
           (= 0 (count-of counts 'sigmoid))
           (ae-approx= (car (car with)) expected-loss))))

  (test "the kernel's output is what the replay returns"
    ;; With the marker backend, relu's forward values all become 42.0.
    '(42.0)
    (begin
      (register-activation-backend! (make-marker-backend))
      (let-values (((joint results)
                    (let ((xv (param-var '(-2.0 1.0 3.0) '(3) 'f64)))
                      (trace-and-replay (am:var-mean (am:var-relu xv)) (list xv)))))
        (register-activation-backend! #f)
        (car results)))))

(register-activation-backend! #f)
(test-exit)
