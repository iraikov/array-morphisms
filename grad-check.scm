;;; array-morphisms-grad-check.scm
;;; Numerical Gradient Verification
;;;
;;; Implements finite-difference gradient checking for verifying
;;; backward rules in array-morphisms-grad.  Necessarily materializes
;;; arrays to perturb individual elements.
;;;
;;; This module is for testing and debugging only.
;;;
;;; Usage:
;;;   (check-grad f vars)          ; #t if all pass
;;;   (grad-check-report f vars)   ; print per-element comparison

(module array-morphisms-grad-check

  (check-grad
   numerical-jacobian
   grad-check-report)

  (import scheme (chicken base))
  (import (only srfi-1 iota every map fold filter-map))
  (import (only srfi-4
                f32vector? f64vector? s32vector? s64vector?
                f32vector-length f64vector-length
                s32vector-length s64vector-length
                f32vector-ref f64vector-ref s32vector-ref s64vector-ref
                f32vector-set! f64vector-set!
                s32vector-set! s64vector-set!))
  (import datatype matchable)
  (import array-morphisms-core)
  (import array-morphisms-realization)
  (import array-morphisms-grad)


;;;; ============================================================
;;;; Internal: typed vector utilities
;;;; ============================================================

(define (typed-vector-length data dtype)
  "Return element count of a typed vector."
  (case dtype
    ((f32) (f32vector-length data))
    ((f64) (f64vector-length data))
    ((s32) (s32vector-length data))
    ((s64) (s64vector-length data))
    (else (error "typed-vector-length: unsupported dtype" dtype))))

(define (clone-typed-vector data dtype)
  "Return a fresh copy of a typed vector."
  (let* ((n    (typed-vector-length data dtype))
         (copy (allocate-typed-vector dtype n)))
    (let loop ((i 0))
      (when (< i n)
        (typed-vector-set! copy dtype i (typed-vector-ref data dtype i))
        (loop (+ i 1))))
    copy))


;;;; ============================================================
;;;; Internal: perturbed-var
;;;;
;;;; Creates a new make-var from v with element idx perturbed by delta.
;;;; v's value must be a concrete-array (realize first if needed).
;;;; The new variable has requires-grad=#f (used only for f evaluation).
;;;; ============================================================

(define (perturbed-var v idx delta)
  "Build a new morph-variable from v with data[idx] += delta.
   v must hold a concrete-array value.  Returns a leaf with no grad."
  (let* ((m (var-value v)))
    (cases array-morphism m
      (concrete-array (data shape strides offset dtype alloc-id batch-axis)
        (let* ((copy    (clone-typed-vector data dtype))
               (old-val (typed-vector-ref data dtype idx)))
          (typed-vector-set! copy dtype idx
            (+ old-val (exact->inexact delta)))
          (make-var
           (make-morphism copy (vector->list shape) dtype)
           #f)))
      (else
       (error "perturbed-var: var-value must be a concrete-array; realize first" v)))))


;;;; ============================================================
;;;; Internal: morph-scalar-value
;;;; ============================================================

(define (morph-scalar-value m)
  "Realize m and return its single scalar value as a number."
  (let ((c (realize m)))
    (cases array-morphism c
      (concrete-array (data shape strides offset dtype alloc-id batch-axis)
        (exact->inexact (typed-vector-ref data dtype 0)))
      (else (error "morph-scalar-value: result is not a concrete-array")))))


;;;; ============================================================
;;;; Internal: morph-to-f64-list
;;;; ============================================================

(define (morph-to-f64-list m)
  "Realize m and return its elements as a flat list of f64 values."
  (let ((c (realize m)))
    (cases array-morphism c
      (concrete-array (data shape strides offset dtype alloc-id batch-axis)
        (let ((n (shape-size shape)))
          (map (lambda (i) (exact->inexact (typed-vector-ref data dtype i)))
               (iota n))))
      (else (error "morph-to-f64-list: result is not a concrete-array")))))


;;;; ============================================================
;;;; Internal: replace-nth
;;;; ============================================================

(define (replace-nth lst k new-elem)
  "Return a copy of lst with the k-th element replaced by new-elem."
  (let loop ((i 0) (remaining lst))
    (if (null? remaining) '()
        (cons (if (= i k) new-elem (car remaining))
              (loop (+ i 1) (cdr remaining))))))


;;;; ============================================================
;;;; Internal: abs-f64
;;;; ============================================================

(define (abs-f64 x)
  (if (< x 0.0) (- x) x))


;;;; ============================================================
;;;; numerical-jacobian
;;;;
;;;; Computes the gradient vector of f w.r.t. vars[var-idx] using
;;;; central finite differences.
;;;; f: (list of morph-variable) -> morph-variable  (scalar output)
;;;; ============================================================

(define (numerical-jacobian f vars var-idx eps)
  "Compute numerical gradient of scalar-output f w.r.t. vars[var-idx].
   Returns a list of f64 values, one per element of vars[var-idx].
   vars[var-idx] must hold a concrete-array value."
  (let* ((v (list-ref vars var-idx))
         (n (morph-size (var-value v))))
    (map (lambda (i)
           (let* ((v+    (perturbed-var v i eps))
                  (v-    (perturbed-var v i (- eps)))
                  (vars+ (replace-nth vars var-idx v+))
                  (vars- (replace-nth vars var-idx v-))
                  (f+    (morph-scalar-value (var-value (f vars+))))
                  (f-    (morph-scalar-value (var-value (f vars-)))))
             (/ (- f+ f-) (* 2.0 eps))))
         (iota n))))


;;;; ============================================================
;;;; check-grad
;;;; ============================================================

(define (check-grad f vars #!key (eps 1e-5) (atol 1e-4) (rtol 1e-3))
  "Verify analytical gradients against central finite differences.

   f   : (list of morph-variable) -> morph-variable  (scalar output)
   vars: list of requires-grad=#t variables with concrete values
         (call (make-var (realize (var-value v)) #t) first if needed)
   eps : perturbation (default 1e-5)
   atol: absolute tolerance (default 1e-4)
   rtol: relative tolerance (default 1e-3)

   Returns #t if |analytical - numerical| <= atol + rtol*|analytical|
   for every element of every variable, #f otherwise."

  ;; Materialize all variable values so perturbed-var can clone them
  (let ((concrete-vars
         (map (lambda (v)
                (make-var (realize (var-value v))
                          (var-requires-grad? v)))
              vars)))

    ;; Forward + backward for analytical gradients
    (let ((out (f concrete-vars)))
      (backward! out))

    ;; Compare each variable
    (every
     (lambda (k)
       (let* ((v        (list-ref concrete-vars k))
              (ana-list (if (var-grad v)
                            (morph-to-f64-list (var-grad v))
                            (map (lambda (_) 0.0)
                                 (iota (morph-size (var-value v))))))
              (num-list (numerical-jacobian f concrete-vars k eps)))
         (every
          (lambda (ana num)
            (let ((err (abs-f64 (- ana num)))
                  (tol (+ atol (* rtol (abs-f64 ana)))))
              (<= err tol)))
          ana-list num-list)))
     (iota (length concrete-vars)))))


;;;; ============================================================
;;;; grad-check-report
;;;; ============================================================

(define (grad-check-report f vars #!key (eps 1e-5) (atol 1e-4) (rtol 1e-3))
  "Print a per-element gradient comparison table.
   Shows variable index, element index, analytical grad, numerical grad,
   absolute error, and PASS/FAIL status."

  (let ((concrete-vars
         (map (lambda (v)
                (make-var (realize (var-value v))
                          (var-requires-grad? v)))
              vars)))

    (let ((out (f concrete-vars)))
      (backward! out))

    (let loop-var ((k 0))
      (when (< k (length concrete-vars))
        (let* ((v        (list-ref concrete-vars k))
               (ana-list (if (var-grad v)
                             (morph-to-f64-list (var-grad v))
                             (map (lambda (_) 0.0)
                                  (iota (morph-size (var-value v))))))
               (num-list (numerical-jacobian f concrete-vars k eps)))

          (display (string-append "Variable " (number->string k) ":\n"))
          (display "  idx   analytical       numerical       abs-err         pass?\n")

          (let loop-elem ((i 0) (alist ana-list) (nlist num-list))
            (unless (or (null? alist) (null? nlist))
              (let* ((ana (car alist))
                     (num (car nlist))
                     (err (abs-f64 (- ana num)))
                     (tol (+ atol (* rtol (abs-f64 ana))))
                     (ok? (<= err tol)))
                (display
                 (string-append
                  "  " (number->string i)
                  "     " (number->string ana)
                  "    " (number->string num)
                  "    " (number->string err)
                  "    " (if ok? "PASS" "FAIL")
                  "\n")))
              (loop-elem (+ i 1) (cdr alist) (cdr nlist)))))

        (loop-var (+ k 1))))))

) ; end module array-morphisms-grad-check
