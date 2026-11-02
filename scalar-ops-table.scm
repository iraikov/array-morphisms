;;; scalar-ops-table.scm
;;;
;;; The scalar operations that element-wise array operations apply to each
;;; element, written once as data.  The basic-ops module builds its Scheme
;;; combiners from this table, segop bodies refer to its entries by name,
;;; and kernel generators (such as the crunch egg) read it at syntax time to
;;; emit compiled kernels, so that every implementation of an op computes
;;; the same formula.
;;;
;;; Each entry has the form
;;;
;;;   (op (param ...) expr prop ...)
;;;
;;; op is the op name used by the morphism constructors and by ssa-vjp.
;;; expr computes the op's value from its parameters, in double precision,
;;; and uses only this subset of Scheme, which every kernel generator must
;;; accept:
;;;
;;;   numeric literals, the parameters, let*-bound variables,
;;;   (+ e e) (- e e) (- e) (* e e) (/ e e),
;;;   (exp e) (log e) (sqrt e) (sin e) (cos e) (tan e) (floor e) (ceiling e),
;;;   (expt e e) (fpabs e),
;;;   (< e e) (> e e) (<= e e) (>= e e) (= e e),
;;;   (if e e e) and (let* ((v e) ...) e).
;;;
;;; fpabs is used rather than abs because some kernel compilers translate
;;; abs on a float to an integer absolute value.  negate is written as
;;; (* -1.0 x), which gives -0.0 for 0.0 as (- x) does.  relu is written as
;;; a comparison, which gives 0.0 for NaN and for -0.0.  tanh is computed
;;; from e = exp(-2|x|), which cannot overflow, so that a large |x| gives
;;; +-1.0 rather than NaN.
;;;
;;; The props say how the Scheme combiner of the op is obtained:
;;;
;;;   (generic name)  the combiner is the procedure name itself: a standard
;;;                   procedure, or scalar-add, which the scalar-ops module
;;;                   defines.  It accepts exact as well as inexact numbers,
;;;                   so the op keeps integer dtypes, and procedures that
;;;                   recognize common combiners by identity (eq? to
;;;                   scalar-add, exp, ...) keep working.  expr must agree
;;;                   with name on flonums.  add uses scalar-add rather than
;;;                   + so that -0.0 + -0.0 gives -0.0 (see scalar-ops.scm).
;;;   exact-args      the combiner is (lambda (param ...) expr), applied to
;;;                   the arguments as given.
;;;   (none)          the combiner is (lambda (param ...) expr) applied to
;;;                   the arguments converted with exact->inexact, so that
;;;                   integer inputs give flonum results.
;;;
;;; The derivative entries are the maps ssa-vjp emits for the activations:
;;; relu-deriv takes the forward input, sigmoid-deriv and tanh-deriv take
;;; the forward output.

(module array-morphisms-scalar-ops-table

  (scalar-ops-table)

  (import scheme)

  (define scalar-ops-table
    '(;; Activations and their derivatives
      (relu          (x) (if (> x 0.0) x 0.0))
      (sigmoid       (x) (/ 1.0 (+ 1.0 (exp (- x)))))
      (tanh          (x) (let* ((e (exp (* -2.0 (fpabs x))))
                                (t (/ (- 1.0 e) (+ 1.0 e))))
                           (if (< x 0.0) (- t) t)))
      (relu-deriv    (x) (if (> x 0.0) 1.0 0.0))
      (sigmoid-deriv (s) (* s (- 1.0 s)))
      (tanh-deriv    (t) (- 1.0 (* t t)))

      ;; Unary arithmetic and transcendental functions
      (exp     (x) (exp x)       (generic exp))
      (log     (x) (log x)       (generic log))
      (sqrt    (x) (sqrt x)      (generic sqrt))
      (sin     (x) (sin x)       (generic sin))
      (cos     (x) (cos x)       (generic cos))
      (tan     (x) (tan x)       (generic tan))
      (negate  (x) (* -1.0 x)    (generic -))
      (abs     (x) (fpabs x)     (generic abs))
      (floor   (x) (floor x)     (generic floor))
      (ceiling (x) (ceiling x)   (generic ceiling))

      ;; Binary arithmetic
      (add (x y) (+ x y)    (generic scalar-add))
      (sub (x y) (- x y)    (generic -))
      (mul (x y) (* x y)    (generic *))
      (div (x y) (/ x y)    (generic /))
      (pow (x y) (expt x y) (generic expt))

      ;; Comparisons, giving 1.0 for true and 0.0 for false
      (gt (x y) (if (> x y) 1.0 0.0)  exact-args)
      (lt (x y) (if (< x y) 1.0 0.0)  exact-args)
      (eq (x y) (if (= x y) 1.0 0.0)  exact-args)
      (ge (x y) (if (>= x y) 1.0 0.0) exact-args)
      (le (x y) (if (<= x y) 1.0 0.0) exact-args)))

) ;; end module array-morphisms-scalar-ops-table
