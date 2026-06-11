;;; test-grad-check.scm
;;; Test Suite for array-morphisms-grad-check
;;;
;;; Tests the numerical gradient checker against known functions
;;; where the exact gradient is analytically known, plus a
;;; deliberate failure case to verify the checker detects errors.

(import scheme (chicken base))
(import test)
(import (only srfi-1 iota map every))
(import (only srfi-4 f64vector))
(import datatype matchable)
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-blas-exec)
(import array-morphisms-realization)
(import array-morphisms-grad)
(import array-morphisms-grad-check)


;;;; ============================================================
;;;; Test Utilities
;;;; ============================================================

(define (make-f64-var lst shape rg)
  "Make a morph-variable from a flat f64 list with given shape."
  (make-var (morph-from-list lst (list->vector shape) 'f64) rg))

(define (make-concrete-f64-var lst shape rg)
  "Make a morph-variable with a realized (concrete) value."
  (make-var (realize (morph-from-list lst (list->vector shape) 'f64)) rg))


;;;; ============================================================
;;;; Group 1: scalar square  f(x) = x^2  (single element)
;;;; ============================================================

(test-group "check-grad: scalar square"

  ;; f(x) = x^2  =>  df/dx = 2x
  (let* ((vars (list (make-concrete-f64-var '(3.0) '(1) #t)))
         (f    (lambda (vs)
                 (let ((v (car vs)))
                   (var* v v))))
         (result (check-grad f vars)))
    (test-assert "f(x)=x^2: check-grad passes"
      result)))


;;;; ============================================================
;;;; Group 2: vector sum-of-squares  f(x) = sum(x*x)
;;;; ============================================================

(test-group "check-grad: vector sum-of-squares"

  ;; f(x) = sum(x_i^2)  =>  df/dx_i = 2*x_i
  (let* ((vars (list (make-concrete-f64-var '(1.0 2.0 3.0) '(3) #t)))
         (f    (lambda (vs)
                 (let ((v (car vs)))
                   (var-sum (var* v v)))))
         (result (check-grad f vars)))
    (test-assert "f(x)=sum(x^2): check-grad passes"
      result)))


;;;; ============================================================
;;;; Group 3: matmul  f(A,B) = sum(A @ B)
;;;; ============================================================

(test-group "check-grad: matmul"

  ;; f(A, B) = sum(A @ B)
  ;; dA = ones @ B^T
  ;; dB = A^T @ ones
  (let* ((A-data '(1.0 2.0 3.0 4.0 5.0 6.0))   ; 2x3
         (B-data '(0.5 1.0 0.5 1.0 0.5 1.0))    ; 3x2
         (vars   (list
                  (make-concrete-f64-var A-data '(2 3) #t)
                  (make-concrete-f64-var B-data '(3 2) #t)))
         (f      (lambda (vs)
                   (let ((vA (car vs))
                         (vB (cadr vs)))
                     (var-sum (var-matmul vA vB)))))
         (result (check-grad f vars)))
    (test-assert "f(A,B)=sum(A@B): check-grad passes for both inputs"
      result)))


;;;; ============================================================
;;;; Group 4: exp-log composition  f(x) = sum(log(exp(x)))
;;;; ============================================================

(test-group "check-grad: exp-log"

  ;; f(x) = sum(log(exp(x))) = sum(x)  =>  df/dx_i = 1.0
  (let* ((vars (list (make-concrete-f64-var '(0.5 1.0 1.5) '(3) #t)))
         (f    (lambda (vs)
                 (let ((v (car vs)))
                   (var-sum (var-log (var-exp v))))))
         (result (check-grad f vars)))
    (test-assert "f(x)=sum(log(exp(x))): check-grad passes (grad=ones)"
      result)))


;;;; ============================================================
;;;; Group 5: grad-check-report (smoke test)
;;;; ============================================================

(test-group "grad-check-report smoke test"

  (let* ((vars (list (make-concrete-f64-var '(1.0 2.0) '(2) #t)))
         (f    (lambda (vs) (var-sum (var* (car vs) (car vs))))))
    ;; Should print without error
    (test-assert "grad-check-report runs without error"
      (begin
        (grad-check-report f vars)
        #t))))


;;;; ============================================================
;;;; Group 6: failure case
;;;; ============================================================

(test-group "check-grad: failure case"

  ;; Build a variable with a deliberately wrong grad-fn
  ;; (returns zero gradient instead of the correct one)
  ;; check-grad should return #f
  (let* ((concrete-vars
          (list (make-concrete-f64-var '(2.0 3.0) '(2) #t)))
         ;; f = sum(x^2), but we'll inject a wrong backward
         (f (lambda (vs)
              (let* ((v   (car vs))
                     (x   (var-value v))
                     ;; Compute correct value but wrong gradient
                     (out (make-var (var-value (var-sum (var* v v))) #f)))
                ;; Set wrong grad-fn: returns zeros instead of 2x
                (morph-variable-grad-fn-set! out
                  (lambda (g)
                    (accumulate-grad! v
                      (morph-from-list '(0.0 0.0) #(2) 'f64))))
                (morph-variable-parents-set! out (list v))
                out)))
         (result (check-grad f concrete-vars)))
    (test-assert "deliberately wrong grad-fn: check-grad returns #f"
      (not result))))


(test-exit)
