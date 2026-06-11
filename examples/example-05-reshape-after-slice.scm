;;; example-05-reshape-after-slice.scm
;;; Demonstrates that reshape is always valid after non-contiguous operations.
;;; Domain: Signal downsampling followed by view reinterpretation
;;;
;;; In SRFI-231, specialized-array-reshape requires the array to be
;;; packed (contiguous in row-major order).  After array-sample or a
;;; stride-N slice the array is no longer packed and reshape raises an
;;; error -- unless the caller explicitly passes copy-on-failure?=#t,
;;; triggering a silent allocation.
;;;
;;; Array morphisms treat reshape as a pure affine index transformation
;;; that composes with any prior transformation regardless of contiguity.
;;; The realization engine resolves the compound index function at
;;; evaluation time, with no restriction on expression.
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s examples/example-05-reshape-after-slice.scm

(import scheme (chicken base) (chicken format))
(import (only srfi-1 iota))
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)

(format #t "=== Example 5: Reshape After Non-Contiguous Slice ===~%~%")

;;; Signal x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
(define x (morph-from-list (map exact->inexact (iota 8)) #(8) 'f64))

;;; Stride-2 downsample: selects x[0], x[2], x[4], x[6]
;;; The resulting view has step=2 in the flat buffer -- non-contiguous.
(define downsampled (morph-slice x '(0) '(8) 2))

;;; Both reshapes below compose with the stride-2 index map.
;;; No copy is forced; both are valid affine compositions.
(define as-2x2 (morph-reshape downsampled #(2 2)))
(define as-1x4 (morph-reshape downsampled #(1 4)))

(format #t "Original:            shape ~a  -> ~a~%"
        (get-morphism-shape x)
        (morph->list (realize x)))
(format #t "Stride-2 slice:      shape ~a  -> ~a~%"
        (get-morphism-shape downsampled)
        (morph->list (realize downsampled)))
(format #t "Reshaped to [2, 2]:  shape ~a  -> ~a~%"
        (get-morphism-shape as-2x2)
        (morph->list (realize as-2x2)))
(format #t "Reshaped to [1, 4]:  shape ~a -> ~a~%"
        (get-morphism-shape as-1x4)
        (morph->list (realize as-1x4)))

;;; Further composition: a 2x2 matrix viewed as a pair of 2-vectors,
;;; both backed by the original non-contiguous stride-2 buffer.
(define row0 (morph-slice as-2x2 '(0 0) '(1 2)))
(define row1 (morph-slice as-2x2 '(1 0) '(2 2)))
(format #t "~%Row 0 of [2,2] view: ~a  (original x[0], x[2])~%"
        (morph->list (realize row0)))
(format #t "Row 1 of [2,2] view: ~a  (original x[4], x[6])~%"
        (morph->list (realize row1)))
