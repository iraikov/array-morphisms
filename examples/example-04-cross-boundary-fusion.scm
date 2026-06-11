;;; example-04-cross-boundary-fusion.scm
;;; Demonstrates Advantage 4: Computational Morphisms Unified with Structural
;;; Domain: Windowed signal analysis (STFT-style envelope extraction)
;;;
;;; im2col-morph extracts overlapping signal windows as a structural
;;; (zero-copy) morphism.  Multiplying the window matrix by a weighting
;;; function is a computational morphism.  In SRFI-231 there is no path
;;; to fuse across the structural/generalized boundary; here both
;;; operations belong to the same algebra and the window matrix is never
;;; materialised as a separate buffer.
;;;
;;; Observable: im2col itself allocates 1 buffer (it is a window op,
;;; not a purely affine index transformation).  The transpose of the
;;; window matrix is zero-copy.  The element-wise multiply adds 1 more.
;;; Total: 2.  Without cross-boundary fusion the result would be 3
;;; (separate buffers for windows, transposed windows, and product).
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s examples/example-04-cross-boundary-fusion.scm

(import scheme (chicken base) (chicken format))
(import (only srfi-1 iota make-list))
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-context)

(define (stats-ref stats key)
  (let ((p (assq key stats)))
    (if p (cdr p) 0)))

(format #t "=== Example 4: Cross-Boundary Fusion (Windowed Analysis) ===~%~%")

;;; Signal x[n] = n,  n = 0..31
(define x (morph-from-list (map exact->inexact (iota 32)) #(32) 'f64))

;;; im2col-morph expects (C, H, W) input; reshape to [C=1, H=1, W=32]
(define x-3d (morph-reshape x #(1 1 32)))

;;; Extract overlapping windows:
;;;   kernel = (1, 8) -- window length 8
;;;   stride = (1, 4) -- hop size 4
;;;   OH = 1 + (1-1)/1 = 1, OW = 1 + (32-8)/4 = 7
;;;   Output shape: [C*KH*KW, OH*OW] = [8, 7]
(define windows (im2col-morph x-3d '(1 8) '(1 4)))

;;; Transpose to [7, 8]: each row is one 8-sample window
;;; Default morph-transpose on rank-2 reverses axes: [8,7] -> [7,8]
(define windows-t (morph-transpose windows))

;;; Measure structural pipeline cost alone (reshape + im2col + transpose)
(define ctx-struct (make-morphism-context))
(realize/ctx ctx-struct windows-t)

(format #t "Signal shape:            ~a~%" (get-morphism-shape x))
(format #t "Window matrix shape:     ~a  (7 windows x 8 samples)~%"
        (get-morphism-shape windows-t))
(format #t "Structural allocs (reshape + im2col + transpose): ~a  (expected 1: im2col + 0 for transpose)~%"
        (stats-ref (context-stats ctx-struct) 'allocations))

;;; Rectangular window weights broadcast across all 7 windows: [1, 8]
(define rect-win (morph-from-list (make-list 8 1.0) #(1 8) 'f64))

;;; Element-wise multiply: [7, 8] * [1, 8] -> [7, 8]
;;; This is 1 more allocation (the multiply result).
;;; Total 2, not 3 -- the transposed window view is still zero-copy.
(define windowed (morph* windows-t rect-win))

;;; Full pipeline: 2 allocations (im2col buffer + multiply result)
(define ctx-full (make-morphism-context))
(realize/ctx ctx-full windowed)

(format #t "~%Full pipeline allocs (im2col + window weights): ~a  (expected 2)~%"
        (stats-ref (context-stats ctx-full) 'allocations))

;;; Realize: verify window contents
(define result (morph->list (realize windowed)))
(format #t "~%Window 0 (x[0..7] weighted by 1.0):  ~a~%" (car result))
(format #t "Window 1 (x[4..11] weighted by 1.0): ~a~%" (cadr result))
(format #t "Window 6 (x[24..31] weighted by 1.0):~a~%"
        (list-ref result 6))
