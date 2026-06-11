;;; example-02-unified-abstraction.scm
;;; Demonstrates the unified abstractions of array-morphisms.
;;; Domain: Layer normalization over a [4, 6] activation matrix
;;;
;;; In SRFI-231, array-map produces a "generalized" array while
;;; structural operations produce "specialized" arrays.  The reshape at
;;; the end of a batch-norm pipeline requires a specialized array, which
;;; forces an array-copy after the arithmetic steps.
;;;
;;; Array morphisms use a single type for both arithmetic and structural
;;; operations.  The final morph-reshape of an arithmetic result is valid
;;; without any intermediate copy; all steps compose uniformly and
;;; realize traverses the entire pipeline in a single pass.
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s examples/example-02-unified-abstraction.scm

(import scheme (chicken base) (chicken format))
(import (only srfi-1 iota make-list))
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)

(format #t "=== Example 2: Unified Abstraction (Layer Normalization) ===~%~%")

;;; Activation matrix: 4 samples, 6 features each.
;;; x[i, j] = i*6 + j + 1  (values 1.0 .. 24.0)
(define x
  (morph-from-list (map (lambda (i) (exact->inexact (+ i 1))) (iota 24))
                   #(4 6) 'f64))

;;; Step 1: Feature mean across the batch (reduce axis 0) -> shape [6]
(define mu (morph-reduce 'mean x '(0)))

;;; Step 2: Center each feature (broadcast mean [6] to [4, 6]) -> [4, 6]
(define centered (morph- x mu))

;;; Step 3: Feature variance -> shape [6]
(define variance (morph-reduce 'mean (morph* centered centered) '(0)))

;;; Step 4: Normalize.  eps = 1e-5 prevents division by zero -> [4, 6]
(define eps (morph-from-list (make-list 6 1e-5) #(6) 'f64))
(define normalized (morph/ centered (morph-sqrt (morph+ variance eps))))

;;; Step 5: Reshape for the next layer.
;;; normalized is the result of arithmetic morphisms; in SRFI-231 this
;;; would be a generalized array and specialized-array-reshape would
;;; fail without a copy.  Here the reshape composes uniformly.
(define flattened (morph-reshape normalized #(24)))

(format #t "Input shape:        ~a~%" (get-morphism-shape x))
(format #t "Mean shape:         ~a~%" (get-morphism-shape mu))
(format #t "Centered shape:     ~a~%" (get-morphism-shape centered))
(format #t "Variance shape:     ~a~%" (get-morphism-shape variance))
(format #t "Normalized shape:   ~a~%" (get-morphism-shape normalized))
(format #t "Flattened shape:    ~a~%~%" (get-morphism-shape flattened))

;;; Verify: the mean of the normalized values per feature should be ~0.
(define mu-check
  (morph->list (realize (morph-reduce 'mean normalized '(0)))))
(format #t "Per-feature mean of normalized activations (expected 0.0 each):~%")
(format #t "  ~a~%~%" mu-check)

;;; Realize the full flattened pipeline in one shot.
(define flat-vals (morph->list (realize flattened)))
(format #t "Flattened normalized values (first 6):~%")
(format #t "  ~a~%" (let loop ((v flat-vals) (n 6) (acc '()))
                      (if (or (null? v) (= n 0))
                          (reverse acc)
                          (loop (cdr v) (- n 1) (cons (car v) acc)))))
