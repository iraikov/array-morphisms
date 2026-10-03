;;; test-scalar-ops.scm
;;;
;;; Tests of the scalar-op table (scalar-ops-table.scm) and of the combiners
;;; built from it (scalar-ops.scm).
;;;
;;; Organisation:
;;;   Group 1 - Table structure: names, arities, expression syntax
;;;   Group 2 - Expressions agree with the generic combiners on flonums
;;;   Group 3 - Combiners agree bit for bit with the earlier hand-written ones
;;;   Group 4 - Exact arguments: integer dtypes and large comparisons
;;;   Group 5 - Combiner identity, as relied on by the realization fast paths

(import scheme (chicken base) test
        (only srfi-1 every iota filter)
        datatype
        array-morphisms-core
        array-morphisms-basic-ops
        array-morphisms-scalar-ops
        array-morphisms-scalar-ops-table)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Bitwise equality of results: NaNs compare equal to each other, while
;; 0.0 and -0.0 are distinguished.
(define (so-same? a b)
  (cond ((and (flonum? a) (flonum? b))
         (or (and (nan? a) (nan? b)) (eqv? a b)))
        (else (eqv? a b))))

;; Float inputs covering signed zeros, infinities, NaN, subnormals, values
;; around the overflow of exp, and values of single-precision magnitude.
(define so-specials
  (list 0.0 -0.0 1.0 -1.0 0.5 -0.5 2.0 -3.25 1e-310 -1e-310 1e-30 -1e-30
        1e30 -1e30 3.4028234663852886e38 1e300 -1e300 709.0 710.0 -745.0
        -746.0 20.0 -20.0 0.1 -0.7 123.456 +inf.0 -inf.0 +nan.0))

;; (op x ...) evaluated from the table expression of op.
(eval '(import (only (chicken flonum) fpabs)))
(define (so-expr-proc op)
  (eval `(lambda ,(scalar-op-params op) ,(scalar-op-expr op))))

;; True when f and g agree on every argument tuple of args.
(define (so-agree? f g args)
  (every (lambda (xs) (so-same? (apply f xs) (apply g xs))) args))

(define (so-unary-args pred) (map list (filter pred so-specials)))

(define (so-binary-args)
  (apply append (map (lambda (x) (map (lambda (y) (list x y)) so-specials))
                     so-specials)))

(define (so-combiner-of m)
  (cases array-morphism m
    (morphism-expr (_ _ _ ifn _ _ _ _) (compute-index-fn-combiner ifn))
    (else (error "so-combiner-of: not a morphism-expr"))))

(define so-m (morph-from-list '(1.0 2.0) #(2) 'f64))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1 - Table structure
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; True when expr uses only the parameters, let*-bound variables, numeric
;; literals, if, let* and the primitives of scalar expressions.
(define (so-valid-expr? expr vars)
  (cond ((number? expr) #t)
        ((symbol? expr) (and (memq expr vars) #t))
        ((and (pair? expr) (eq? (car expr) 'if))
         (and (= (length expr) 4)
              (every (lambda (e) (so-valid-expr? e vars)) (cdr expr))))
        ((and (pair? expr) (eq? (car expr) 'let*))
         (let loop ((bs (cadr expr)) (vars vars))
           (if (null? bs)
               (and (= (length expr) 3) (so-valid-expr? (caddr expr) vars))
               (and (so-valid-expr? (cadar bs) vars)
                    (loop (cdr bs) (cons (caar bs) vars))))))
        ((pair? expr)
         (and (scalar-primitive? (car expr))
              (every (lambda (e) (so-valid-expr? e vars)) (cdr expr))))
        (else #f)))

(test-group "scalar-op table structure"
  (test-assert "every op SSA bindings can hold has an entry"
    (every scalar-op?
           '(add sub mul div pow negate sqrt exp log sin cos tan abs floor ceiling
             relu sigmoid tanh relu-deriv sigmoid-deriv tanh-deriv
             gt lt eq ge le)))
  (test "op names are unique"
    (length (scalar-op-names))
    (length (let loop ((ns (scalar-op-names)) (acc '()))
              (cond ((null? ns) acc)
                    ((memq (car ns) acc) (loop (cdr ns) acc))
                    (else (loop (cdr ns) (cons (car ns) acc)))))))
  (test "scalar-op-names follows the table order"
    (map car scalar-ops-table) (scalar-op-names))
  (test-assert "every expression uses only the expression subset"
    (every (lambda (op) (so-valid-expr? (scalar-op-expr op) (scalar-op-params op)))
           (scalar-op-names)))
  (test "unary ops have arity 1" '(1 1 1 1)
    (map scalar-op-arity '(relu tanh negate tanh-deriv)))
  (test "binary ops have arity 2" '(2 2 2)
    (map scalar-op-arity '(add pow ge)))
  (test-assert "map has no entry" (not (scalar-op? 'map)))
  (test-assert "a non-symbol is not an op" (not (scalar-op? "relu")))
  (test-error "an unknown op is an error" (scalar-op-combiner 'no-such-op))
  (test-assert "fpabs is a primitive, abs is not"
    (and (scalar-primitive? 'fpabs) (not (scalar-primitive? 'abs)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2 - Expressions agree with the generic combiners
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; sqrt and log of a negative number give a complex number in Scheme and
;; NaN in compiled code, so those arguments are excluded.
(define (so-nonnegative? x) (or (nan? x) (and (>= x 0.0) (not (eqv? x -0.0)))))

(test-group "table expressions agree with generic combiners on flonums"
  (for-each
   (lambda (op)
     (test-assert (string-append (symbol->string op) " expression = combiner")
       (so-agree? (so-expr-proc op) (scalar-op-combiner op)
                  (so-unary-args (if (memq op '(sqrt log)) so-nonnegative? (lambda (x) #t))))))
   '(exp log sqrt sin cos tan negate abs floor ceiling))
  (for-each
   (lambda (op)
     (test-assert (string-append (symbol->string op) " expression = combiner")
       (so-agree? (so-expr-proc op) (scalar-op-combiner op) (so-binary-args))))
   '(add sub mul div gt lt eq ge le))
  (test-assert "pow expression = combiner on non-negative bases"
    (so-agree? (so-expr-proc 'pow) (scalar-op-combiner 'pow)
               (filter (lambda (xy) (so-nonnegative? (car xy))) (so-binary-args))))
  (test-assert "the generated activations equal their expressions"
    (every (lambda (op)
             (so-agree? (so-expr-proc op) (scalar-op-combiner op) (so-unary-args (lambda (x) #t))))
           '(relu sigmoid tanh relu-deriv sigmoid-deriv tanh-deriv))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3 - Agreement with the earlier hand-written combiners
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; The combiners basic-ops and ssa-vjp used before the table existed.
(define so-legacy
  (list
   (cons 'relu    (lambda (x) (max 0.0 (exact->inexact x))))
   (cons 'sigmoid (lambda (x) (let ((xf (exact->inexact x)))
                                (/ 1.0 (+ 1.0 (exp (- xf)))))))
   (cons 'tanh    (lambda (x)
                    (let* ((xf (exact->inexact x))
                           (e  (exp (* -2.0 (abs xf))))
                           (t  (/ (- 1.0 e) (+ 1.0 e))))
                      (if (< xf 0.0) (- t) t))))
   (cons 'relu-deriv    (lambda (xv) (if (> xv 0.0) 1.0 0.0)))
   (cons 'sigmoid-deriv (lambda (sv) (* sv (- 1.0 sv))))
   (cons 'tanh-deriv    (lambda (tv) (- 1.0 (* tv tv))))
   (cons 'negate        (lambda (x) (- x)))))

(define so-legacy-cmp
  (list (cons 'gt >) (cons 'lt <) (cons 'eq =) (cons 'ge >=) (cons 'le <=)))

(test-group "table combiners agree with the earlier combiners"
  (for-each
   (lambda (entry)
     (test-assert (string-append (symbol->string (car entry)) " is bit-identical on flonums")
       (so-agree? (cdr entry) (scalar-op-combiner (car entry))
                  (so-unary-args (lambda (x) #t)))))
   so-legacy)
  (test-assert "relu, sigmoid and tanh agree on exact integers"
    (every (lambda (op)
             (so-agree? (cdr (assq op so-legacy)) (scalar-op-combiner op)
                        (map list '(0 1 -1 5 -7 40 -40))))
           '(relu sigmoid tanh)))
  (for-each
   (lambda (entry)
     (test-assert (string-append (symbol->string (car entry)) " comparison is unchanged")
       (so-agree? (lambda (x y) (if ((cdr entry) x y) 1.0 0.0))
                  (scalar-op-combiner (car entry))
                  (so-binary-args))))
   so-legacy-cmp)
  (test "relu of -0.0 is 0.0" 0.0 ((scalar-op-combiner 'relu) -0.0))
  (test-assert "relu of NaN is 0.0" (eqv? 0.0 ((scalar-op-combiner 'relu) +nan.0)))
  (test "tanh saturates to -1.0" -1.0 ((scalar-op-combiner 'tanh) -1e300))
  (test "negate of 0.0 is -0.0" -0.0 ((scalar-op-combiner 'negate) 0.0)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4 - Exact arguments
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "exact arguments"
  (test "negate, abs, floor and ceiling keep exact integers"
    (quote (-7 7 -7 -7))
    (map (lambda (op) ((scalar-op-combiner op) (if (eq? op 'negate) 7 -7)))
         '(negate abs floor ceiling)))
  (test-assert "add, sub and mul keep exact integers"
    (every exact? (map (lambda (op) ((scalar-op-combiner op) 6 4)) '(add sub mul))))
  (test "relu of an exact integer is a flonum" 3.0 ((scalar-op-combiner 'relu) 3))
  (test "relu of a negative exact integer is 0.0" 0.0 ((scalar-op-combiner 'relu) -3))
  (test "comparisons stay exact beyond 2^53"
    '(1.0 0.0 0.0)
    (map (lambda (op) ((scalar-op-combiner op) 9007199254740993 9007199254740992))
         '(gt lt eq))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5 - Combiner identity
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "combiner identity"
  (test-assert "generic combiners are the standard procedures"
    (and (eq? (scalar-op-combiner 'add) +) (eq? (scalar-op-combiner 'sub) -)
         (eq? (scalar-op-combiner 'mul) *) (eq? (scalar-op-combiner 'div) /)
         (eq? (scalar-op-combiner 'pow) expt) (eq? (scalar-op-combiner 'exp) exp)
         (eq? (scalar-op-combiner 'sqrt) sqrt) (eq? (scalar-op-combiner 'negate) -)))
  (test-assert "generic? is set exactly for the (generic name) entries"
    (and (scalar-op-generic? 'add) (scalar-op-generic? 'abs)
         (not (scalar-op-generic? 'relu)) (not (scalar-op-generic? 'gt))))
  (test-assert "a combiner is the same procedure on every call"
    (eq? (scalar-op-combiner 'relu) (scalar-op-combiner 'relu)))
  (test-assert "unary morphisms use the table combiners"
    (every (lambda (op)
             (eq? (so-combiner-of (create-unary-morphism op so-m)) (scalar-op-combiner op)))
           '(exp sqrt relu sigmoid tanh negate abs)))
  (test-assert "binary and comparison morphisms use the table combiners"
    (every (lambda (op)
             (eq? (so-combiner-of (if (memq op '(gt lt eq ge le))
                                      (create-comparison-morphism op so-m so-m)
                                      (create-binary-morphism op so-m so-m)))
                  (scalar-op-combiner op)))
           '(add sub mul div pow gt lt eq ge le))))

(test-exit)
