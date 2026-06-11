;;; example-03-algebraic-simplification.scm
;;; Demonstrates Advantage 3: Algebraic Simplification (Psi Calculus)
;;; Domain: NCHW <-> NHWC tensor layout round-trip
;;;
;;; Applying permutation p and then its inverse is mathematically the
;;; identity, but in SRFI-231 the two transposes create two nested
;;; closures with no knowledge that they cancel.
;;;
;;; Array morphisms represent permutations as inspectable algebraic data
;;; (permutation-fn records with known structure).  The system can detect
;;; that composing two permutations yields the identity and eliminate the
;;; redundant steps.  Observable consequence: both transposes together
;;; cost 0 buffer allocations, and the realized values are bit-for-bit
;;; identical to the original.
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s examples/example-03-algebraic-simplification.scm

(import scheme (chicken base) (chicken format))
(import (only srfi-1 iota))
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-context)

(define (stats-ref stats key)
  (let ((p (assq key stats)))
    (if p (cdr p) 0)))

(format #t "=== Example 3: Algebraic Simplification (Double Permutation) ===~%~%")

;;; Tensor x: shape [2, 3, 4]  (N=2, C=3, HW=4)
(define x
  (morph-from-list (map exact->inexact (iota 24)) #(2 3 4) 'f64))

;;; Permute [0, 2, 1]: swap the C and HW axes
;;; NCHW [2, 3, 4] -> NHWC-like [2, 4, 3]
(define p (morph-transpose x '(0 2 1)))

;;; Apply the same permutation again: [0, 2, 1] is its own inverse
;;; [2, 4, 3] -> [2, 3, 4]  (back to original layout)
(define q (morph-transpose p '(0 2 1)))

;;; Both transposes are structural: 0 allocations
(define ctx (make-morphism-context))
(realize/ctx ctx q)

(format #t "Original shape:          ~a~%" (get-morphism-shape x))
(format #t "After permute [0,2,1]:   ~a~%" (get-morphism-shape p))
(format #t "After inverse permute:   ~a~%" (get-morphism-shape q))
(format #t "~%Allocations for both transposes: ~a  (expected 0)~%"
        (stats-ref (context-stats ctx) 'allocations))

;;; Values must be identical after the round-trip
(define x-vals (morph->list (realize x)))
(define q-vals (morph->list (realize q)))
(format #t "Round-trip preserves all values: ~a~%~%" (equal? x-vals q-vals))

(format #t "First row of original x  [N=0, C=0]: ~a~%" (car (car x-vals)))
(format #t "First row of round-trip q [N=0, C=0]: ~a~%" (car (car q-vals)))
(format #t "~%Intermediate permuted shape [N=0]:~%  ~a~%"
        (car (morph->list (realize p))))
