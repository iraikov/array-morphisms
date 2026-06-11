;;; example-06-buffer-reuse.scm
;;; Demonstrates Advantage 6: Deferred Evaluation and Buffer Reuse
;;; Domain: Per-sample RMS energy of two audio channels
;;;
;;; The computation rms = sqrt(0.5*(a^2 + b^2)) has five non-trivial
;;; allocations: a^2, b^2, sum, scaled, sqrt.  After the addition step
;;; both a^2 and b^2 are dead, so they can share a single physical
;;; buffer with later intermediates.  Liveness analysis in
;;; finalize-context! reduces the 5 logical allocations to 3 physical
;;; buffers; the pool is reused on every subsequent replay run.
;;;
;;; In SRFI-231 every array-copy or array-assign! allocates fresh memory
;;; with no equivalent of this lifecycle.
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s examples/example-06-buffer-reuse.scm

(import scheme (chicken base) (chicken format))
(import (only srfi-1 make-list map))
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-context)

(define (stats-ref stats key)
  (let ((p (assq key stats)))
    (if p (cdr p) 0)))

(format #t "=== Example 6: Deferred Evaluation and Buffer Reuse ===~%~%")

;;; Two 4-sample audio channels
(define a (morph-from-list '(3.0 4.0 0.0 1.0) #(4) 'f64))
(define b (morph-from-list '(4.0 3.0 5.0 2.0) #(4) 'f64))
(define half (morph-from-list '(0.5 0.5 0.5 0.5) #(4) 'f64))

;;; Computation graph for per-sample RMS:
;;;   rms[i] = sqrt(0.5 * (a[i]^2 + b[i]^2))
;;;
;;; Logical allocations (in evaluation order):
;;;   a2     = a * a            alloc 0  (dies when sum is formed)
;;;   b2     = b * b            alloc 1  (dies when sum is formed)
;;;   sum-sq = a2 + b2          alloc 2  (dies when scaled is formed)
;;;   scaled = sum-sq * half    alloc 3  (dies when rms is formed)
;;;   rms    = sqrt(scaled)     alloc 4  (final result, lives until caller drops it)
(define a2     (morph* a a))
(define b2     (morph* b b))
(define sum-sq (morph+ a2 b2))
(define scaled (morph* sum-sq half))
(define rms    (morph-sqrt scaled))

;;; --- Phase 1: Trace ---
;;; realize/ctx in trace mode walks the morphism tree and records one
;;; allocation-rec per non-zero-copy node.
(define ctx (make-morphism-context))
(realize/ctx ctx rms)

(format #t "Trace phase:~%")
(format #t "  Logical allocations recorded: ~a~%"
        (stats-ref (context-stats ctx) 'allocations))

;;; --- Phase 2: Finalize (liveness analysis + buffer assignment) ---
;;; finalize-context! performs interval-graph colouring over the alloc
;;; lifetimes.  a2 and b2 are simultaneously live only up to the + step;
;;; afterwards one of their buffers can be reused for scaled and rms.
(finalize-context! ctx)

(format #t "~%After finalize (liveness analysis):~%")
(format #t "  Physical buffers in pool:     ~a  (fewer than ~a logical allocs)~%"
        (stats-ref (context-stats ctx) 'buffers)
        (stats-ref (context-stats ctx) 'allocations))

;;; --- Phase 3: Replay (reuses existing pool) ---
(reset-context! ctx)
(define result (realize/ctx ctx rms))

(format #t "~%Replay result -- rms per sample:~%")
(format #t "  Computed:  ~a~%" (morph->list result))
(format #t "  Expected:  ~a~%"
        (map (lambda (ai bi)
               (sqrt (* 0.5 (+ (* ai ai) (* bi bi)))))
             '(3.0 4.0 0.0 1.0)
             '(4.0 3.0 5.0 2.0)))

;;; Repeated replays reuse the same pool without new allocations.
(let loop ((i 1))
  (when (<= i 4)
    (reset-context! ctx)
    (realize/ctx ctx rms)
    (loop (+ i 1))))
(format #t "~%Buffer pool size unchanged after 5 total replay runs: ~a buffers~%"
        (stats-ref (context-stats ctx) 'buffers))
