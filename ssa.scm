;;; array-morphisms-ssa.scm
;;;
;;; SSA IR for fused forward+backward computation.
;;;
;;; Compiles a morph-variable loss node into a flat SSA program once,
;;; then replays the joint forward+backward computation every training step.
;;; Eliminates the lazy/eager boundary, fresh-copy overhead, and missed
;;; fusion opportunities that arise when forward and backward pass each
;;; build separate lazy trees.
;;;
;;; Phases:
;;;   1.  morphism-to-ssa  -- DFS over morphism-expr tree -> SSA program
;;;   2.  ssa-vjp          -- symbolic VJP over SSA bindings -> extended program
;;;   3.  ssa-realize      -- sequential executor (no context pooling)
;;;   4.  ssa-realize/ctx  -- context-pooled executor
;;;
;;; Key invariants:
;;;   - Constants (concrete-array leaves) are stored in a hash-table keyed by
;;;     gensym cid-symbols.  Parameters are concrete-arrays whose underlying
;;;     SRFI-4 vectors are updated in-place by the optimizer; the SSA constants
;;;     table sees the updated values automatically each step.
;;;   - Binding IDs are gensym symbols (bid-NNN for forward, adj-NNN for
;;;     backward adjoints) serving as hash-table keys in the value table.
;;;   - Reduction ops use (list 'reduce rop) as their op field to distinguish
;;;     them from plain element-wise ops.

(module array-morphisms-ssa

  (;; Data-type predicates/accessors
   ssa-value?
   binding-ref const-ref
   ssa-binding-ref? ssa-const-ref?
   ssa-value-id

   ;; SSA records
   make-ssa-binding
   ssa-binding?
   ssa-binding-name ssa-binding-op ssa-binding-inputs
   ssa-binding-shape ssa-binding-dtype ssa-binding-meta
   make-ssa-program ssa-program?
   ssa-program-constants ssa-program-morph-to-val
   ssa-program-bindings ssa-program-outputs ssa-program-n-params
   ssa-program-replay-plan ssa-program-fused?

   ;; Compilation
   morphism-to-ssa
   ssa-constant-id
   ssa-loss-binding-val

   ;; VJP
   ssa-vjp

   ;; Execution
   ssa-realize
   ssa-realize/ctx

   ;; Replay plan ADTs
   replay-ref?
   rr-val rr-const
   replay-instruction replay-instruction?
   ri-view
   ri-segop
   ri-im2col ri-col2im
   ri-conv-fwd ri-conv-bwd-data ri-conv-bwd-weights

   ;; Segop fusion
   segop-fusion
   ssa-segop-fusion-pass
   ssa-fusion-graph
   segop-fusion-max-rounds
   ssa-fusion-pass-stats
   make-fused-ssa-binding
   fused-red-binding?
   binding-out-names

   ;; Segop lowering
   binding-scalar-body
   standard-reduction
   standard-gemm

   ;; Replay plan compilation and execution
   compile-replay-plan
   execute-replay-plan

    ;; Diagnostics
    replay-plan-stats

    ;; Per-instruction timing
    replay-timing-reset!
    replay-timing-results)

  (import scheme (chicken base) (chicken time) (chicken format)
          (only (chicken fixnum) fx+ fx- fx* fx= fx< fx>=)
          (only (chicken flonum) fp+ fp- fp* fp/ fp= fpexpt)
          (only (scheme time) current-jiffy jiffies-per-second)
          (only (scheme base) make-parameter)
          (only srfi-1 iota fold filter map for-each append-map filter-map every any
                take drop list-index delete-duplicates)
          (only srfi-4
                f64vector f64vector? f64vector-ref f64vector-set! f64vector-length
                f32vector f32vector? f32vector-ref f32vector-set! f32vector-length
                s32vector-set! s64vector-set!)
          (only srfi-69
                make-hash-table
                hash-table-ref
                hash-table-ref/default
                hash-table-set!
                hash-table-walk
                hash-table->alist
                eq?-hash)
          datatype matchable)

  ;; Mutable-cell and stack utilities (plain define files, kept private
  ;; to this module).
  (include "box.scm")
  (include "stack.scm")

  (import array-morphisms-core
          array-morphisms-index-fn
          array-morphisms-basic-ops
          array-morphisms-structural-ops
          array-morphisms-blas-exec
          array-morphisms-blas-compat
          array-morphisms-activation-exec
          array-morphisms-realization
          array-morphisms-context
          array-morphisms-morph-env
          array-morphisms-scalar-ops
          array-morphisms-segop
          (prefix array-morphisms-grad am:))
          

;;; ============================================================
;;; SSA Value ADT
;;;
;;; An ssa-value identifies either a binding result (binding-ref) or
;;; a constant array (const-ref).  IDs are gensym symbols.
;;; ============================================================

(define-datatype ssa-value ssa-value?
  (binding-ref (id symbol?))
  (const-ref   (cid symbol?)))

(define (ssa-binding-ref? v)
  (cases ssa-value v (binding-ref (_) #t) (else #f)))

(define (ssa-const-ref? v)
  (cases ssa-value v (const-ref (_) #t) (else #f)))

(define (ssa-value-id v)
  "Return the gensym symbol identifying this ssa-value."
  (cases ssa-value v
    (binding-ref (id)  id)
    (const-ref   (cid) cid)))


;;; ============================================================
;;; SSA Records
;;; ============================================================

;; One binding in the SSA program.
;;   name   : gensym symbol (bid-NNN or adj-NNN); key in value table
;;   op     : symbol (e.g. 'add) or list (e.g. '(reduce mean))
;;   inputs : list of ssa-value
;;   shape  : vector (result shape)
;;   dtype  : symbol
;;   meta   : alist of extra info (index-fn, axes, perm, fn, ...)
(define-record ssa-binding name op inputs shape dtype meta)

;; The compiled SSA program.
;;   constants   : hash-table cid-symbol -> concrete-array
;;   morph-to-val: morph-env stable-id (symbol) -> ssa-value  (for ssa-constant-id)
;;   bindings    : list of ssa-binding in topological order
;;   outputs     : list of ssa-value  (loss first, then param grads)
;;   n-params    : count of trainable parameter constants
;;   replay-plan   : #f until compiled; vector of replay-instruction (one per binding)
;;   trace-info    : #f until trace; morph-env bid-sym -> (concrete-array . is-pool?);
;;                   released (#f again) once the replay-plan is compiled
;;   output-specs  : #f until compiled; list of (integer | (pos . buffer) | concrete-array)
;;                   per output: integer = index into vals vector; (pos . buffer) = the
;;                   same, copied into buffer when strided; concrete-array = const-ref value
;;   fused?        : true once the fusion policy has been applied to the bindings
;;                   (by ssa-vjp, or by ssa-realize/ctx at the first trace)
(define-record-type ssa-program
  (%make-ssa-program constants morph-to-val bindings outputs n-params
                     replay-plan trace-info output-specs fused?)
  ssa-program?
  (constants    ssa-program-constants)
  (morph-to-val ssa-program-morph-to-val)
  (bindings     ssa-program-bindings ssa-program-bindings-set!)
  (outputs      ssa-program-outputs)
  (n-params     ssa-program-n-params)
  (replay-plan  ssa-program-replay-plan ssa-program-replay-plan-set!)
  (trace-info   ssa-program-trace-info ssa-program-trace-info-set!)
  (output-specs ssa-program-output-specs ssa-program-output-specs-set!)
  (fused?       ssa-program-fused? ssa-program-fused?-set!))

(define (make-ssa-program constants morph-to-val bindings outputs n-params
                          replay-plan trace-info output-specs #!optional (fused? #f))
  (%make-ssa-program constants morph-to-val bindings outputs n-params
                     replay-plan trace-info output-specs fused?))


;;; ============================================================
;;; Replay Plan ADTs
;;;
;;; Pre-compiled dispatch structures produced by compile-replay-plan.
;;; Used by execute-replay-plan to avoid per-step morphism rebuilding,
;;; BLAS eligibility re-evaluation, and context-vector allocation.
;;; ============================================================

;; Input reference: previous binding result (by 0-based position in vals vector)
;; or a constant (by cid-symbol from constants hash-table).
(define-datatype replay-ref replay-ref?
  (rr-val   (val-idx integer?))
  (rr-const (cid symbol?)))

;; Predicate for SRFI-4 float typed vectors (col scratch buffers).
(define (typed-float-vec? x) (or (f32vector? x) (f64vector? x)))

;; One pre-compiled instruction per SSA binding.
(define-datatype replay-instruction replay-instruction?

  ;; Zero-copy view (transpose, reshape, slice): shares pool buffer with source.
  (ri-view
    (view-fn  procedure?)
    (in-ref   replay-ref?))

  ;; A binding lowered to a segop (see segop-lower.scm).  executor
  ;; computes the output into the pool buffer out-pool-idx; segop is the
  ;; segop it was selected for, and tag names the backend pattern, for
  ;; timing profiles and replay-plan-stats.  extra-outs lists the outputs
  ;; after the first of a fused binding with several results, each a
  ;; vector #(val-idx pool-idx shape strides dtype): its position in the
  ;; vector of values and its pool buffer.  The executor of such a segop
  ;; takes their buffers as a third argument, a list.
  (ri-segop
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (segop        segop?)
    (executor     procedure?)
    (tag          symbol?)
    (extra-outs   list?))

  ;; Fast im2col: all index parameters baked in at compile time from morph meta.
  ;; col-layout: 'nchw-standard (unbatched), 'nchw-classic (batched NCHW), 'nhwc-classic (batched NHWC).
  (ri-im2col
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (src-ref      replay-ref?)
    (N            integer?)
    (C            integer?)
    (H            integer?)
    (W            integer?)
    (KH           integer?)
    (KW           integer?)
    (SH           integer?)
    (SW           integer?)
    (PH           integer?)
    (PW           integer?)
    (OH           integer?)
    (OW           integer?)
    (col-layout   symbol?))

  ;; Fast col2im: all index parameters baked in at compile time from VJP meta.
  ;; col-layout: 'nchw-standard (unbatched), 'nchw-classic (batched NCHW), 'nhwc-classic (batched NHWC).
  (ri-col2im
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (col-ref      replay-ref?)
    (N            integer?)
    (C            integer?)
    (H            integer?)
    (W            integer?)
    (KH           integer?)
    (KW           integer?)
    (SH           integer?)
    (SW           integer?)
    (PH           integer?)
    (PW           integer?)
    (col-layout   symbol?))

  ;; Implicit GEMM forward: fused im2col+GEMM+bias, no col buffer.
  ;; src: [N,C,H,W] or [N,H,W,C]; wt: [fan_in, out_ch]; b: [out_ch]
  ;; out: [N*OH_OW, out_ch]
  (ri-conv-fwd
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (src-ref      replay-ref?)
    (wt-ref       replay-ref?)
    (b-ref        replay-ref?)
    (col-buf      typed-float-vec?)
    (N   integer?) (C   integer?) (H   integer?) (W   integer?)
    (KH  integer?) (KW  integer?) (SH  integer?) (SW  integer?)
    (PH  integer?) (PW  integer?) (OH  integer?) (OW  integer?)
    (out-ch     integer?)
    (col-layout symbol?))

  ;; Backward w.r.t. input: dX = scatter(g x WT), no col buffer.
  ;; g: [N*OH_OW, out_ch]; wt: [fan_in, out_ch]; out: x-shape
  (ri-conv-bwd-data
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (g-ref        replay-ref?)
    (wt-ref       replay-ref?)
    (col-buf      typed-float-vec?)
    (N   integer?) (C   integer?) (H   integer?) (W   integer?)
    (KH  integer?) (KW  integer?) (SH  integer?) (SW  integer?)
    (PH  integer?) (PW  integer?) (OH  integer?) (OW  integer?)
    (out-ch     integer?)
    (col-layout symbol?))

  ;; Backward w.r.t. weights: dWT = outer product sum, no col buffer.
  ;; g: [N*OH_OW, out_ch]; src: x-shape; out: [fan_in, out_ch]
  (ri-conv-bwd-weights
    (out-pool-idx integer?)
    (out-shape    vector?)
    (out-strides  vector?)
    (out-dtype    symbol?)
    (g-ref        replay-ref?)
    (src-ref      replay-ref?)
    (col-buf      typed-float-vec?)
    (N   integer?) (C   integer?) (H   integer?) (W   integer?)
    (KH  integer?) (KW  integer?) (SH  integer?) (SW  integer?)
    (PH  integer?) (PW  integer?) (OH  integer?) (OW  integer?)
    (out-ch     integer?)
    (col-layout symbol?)))


;;; ============================================================
;;; Binding helpers
;;;
;;; Queries on element-wise bindings, for the segop fusion pass.
;;; ============================================================

(define (ssa-binding-combiner b)
  "Extract element-wise combiner from an SSA binding's meta.
   Forward bindings carry (index-fn . compute-index-fn); backward bindings
   carry explicit (combiner . proc) embedded during ssa-vjp."
  (let* ((meta (ssa-binding-meta b))
         (cp   (assq 'combiner meta)))
    (if cp
        (cdr cp)
        (let ((ip (assq 'index-fn meta)))
          (if (and ip (compute-index-fn? (cdr ip)))
              (compute-index-fn-combiner (cdr ip))
              (error "ssa-binding-combiner: no combiner in meta"
                     (ssa-binding-name b) (ssa-binding-op b)))))))

(define (ssa-binding-has-combiner? b)
  "True when b carries an extractable element-wise combiner in meta."
  (let ((meta (ssa-binding-meta b)))
    (or (and (assq 'combiner meta) #t)
        (let ((ip (assq 'index-fn meta)))
          (and ip (compute-index-fn? (cdr ip)))))))

;; The fusion policy of ssa-vjp and ssa-realize/ctx: #t (the default)
;; fuses the bindings whose fusion is expected to be faster, all fuses
;; every pair that may legally be fused, and #f runs no fusion pass.  Fusion does not change results,
;; so a program run with all and with #f gives the same values, which
;; tests use as an oracle.
(define segop-fusion
  (make-parameter #t (lambda (x) (cond ((eq? x 'all) 'all) (x #t) (else #f)))))

;;; ============================================================
;;; Segop fusion
;;;
;;; Fuses map bindings into the map bindings that consume them, before
;;; the program is traced: ssa-vjp runs the pass on the joint program it
;;; builds, and ssa-realize/ctx on a forward program at its first trace.
;;; A fused binding has op fused and computes its
;;; members in one seg-map, described by a fusion template (see
;;; segop-lower.scm); it takes the place of its last member, the
;;; consumer, whose name and result it keeps.
;;;
;;; The pass works on a graph of uses.  Every read of a binding's value is
;;; an edge from the reader, with a kind: fusible when a map binding reads
;;; the value at the iteration point itself (the value has the reader's
;;; shape), infusible for any other read, and result when the value is a
;;; program output.  A producer P is fused into a consumer C when P is a
;;; map binding with a single output, its only use is a fusible edge from
;;; C, and C's body reads it once.  The value of P then never exists: every
;;; point of the fused binding computes it, once, from P's inputs, which
;;; are read at C's position.  Either binding may read inputs that
;;; broadcast to its shape; the fused binding reads them through their
;;; broadcast maps.
;;;
;;; The fused body is C's body with P's body in place of its reads of P,
;;; rounded to P's dtype as storing P's value would round it, so fusion
;;; never changes a result.  Inputs read by both bindings are read once.
;;;
;;; Legality (the graph conditions, the shapes of the inputs, bounds on
;;; the number of inputs and the size of the body) is decided apart from
;;; profitability.  The host executors of fused segops run the compiled
;;; body as Scheme procedures, and measured slower than the separate passes
;;; they replace (whose loops are specialized for each op, or run on
;;; backend kernels).  Under the default policy a fusion is therefore made
;;; only when some segop backend reports that it would run the fused segop
;;; natively, such as a catalogue of compiled fused kernels; the query is
;;; asked of a layout-free template, with each input's layout predicted
;;; (values of bindings that allocate, and reshapes of them, are expected
;;; to be flat; other views are not).  The policy all (see segop-fusion)
;;; fuses every legal pair.
;;;
;;; The driver visits the bindings from last to first and fuses into each
;;; consumer until nothing more fuses, so a chain collapses in one visit.
;;; Each round fuses maps into maps first, then fuses with reductions, then
;;; makes the multi-output and horizontal fusions (see below).  Rounds are
;;; repeated while they fuse something, at most (segop-fusion-max-rounds)
;;; times.
;;; ============================================================

;; The largest number of rounds of the fusion driver.  Stopping at this
;; bound leaves a correct, less fused program; ssa-fusion-pass-stats
;; reports it.
(define segop-fusion-max-rounds
  (make-parameter 16 (lambda (n)
                       (if (and (integer? n) (exact? n) (> n 0))
                           n
                           (error 'segop-fusion-max-rounds "not a positive integer" n)))))

;; Statistics of the last run of ssa-segop-fusion-pass.
(define *fusion-pass-stats* '())

(define (ssa-fusion-pass-stats)
  "An alist describing the last run of ssa-segop-fusion-pass: rounds (the
   rounds run), fusions (the fusions made), bound-reached? (true when the
   last round still fused and the round bound stopped the pass); kept,
   horizontal and dead-outputs (the fusions that kept a producer as a
   second output, the horizontal fusions, and the outputs removed for
   having no uses); candidates, the bodies of the fused seg-maps made, each a list
   (canonical-body dtype count elements): how many bindings have that
   body and dtype, and their total number of elements; red-candidates,
   the fused seg-reds made, each a list (canonical-body combine-op path
   map-output? dtype count elements), elements counting the reduced
   values (combine-op and path are gemm for a GEMM epilogue); and missed, an alist from reason to the number of pairs the
   last round left unfused for that reason:
     not-native           no backend runs the fused segop natively
     broadcast-input      the same, for a fused segop with a broadcast input
     size-cap             the fused body or its inputs exceed the bounds
     view                 the producer is read through a reshape or a
                          transpose, or a reduction's source is a view
     reduction-order      the reduction's order is left to its backend
     multi-use-reduction  a reduction result has other uses, or is read twice
     placement            no position fits a fused binding: a finalize's
                          inputs come after a reduction that keeps a map
                          output, or a kept producer has a use before its
                          consumer, whose inputs come after it
     horizontal-placement no position fits two siblings
     gemm-operands        a matrix product's operands are not two matrices
                          of its dtype, f32 or f64
     gemm-epilogue-form   an epilogue would change the product's dtype"
  *fusion-pass-stats*)

;; Views: their values share the buffer of their source, in a layout that
;; is in general not flat.
(define fusion-view-ops '(transpose reshape broadcast-expand slice))

;; The candidates entry of ssa-fusion-pass-stats for bindings bs.
(define (fusion-candidates bs)
  (let ((table (make-hash-table equal?)))
    (for-each
     (lambda (b)
       (when (and (fused-binding? b) (not (fused-red-binding? b)))
         (for-each
          (lambda (o)
            (let* ((key (cons (call-with-values
                                  (lambda () (canonical-scalar-body (fused-out-body o)
                                                                    (length (ssa-binding-inputs b))))
                                (lambda (c perm) c))
                              (fused-out-dtype o)))
                   (old (hash-table-ref/default table key '(0 . 0))))
              (hash-table-set! table key (cons (+ 1 (car old))
                                               (+ (shape-size (fused-out-shape o)) (cdr old))))))
          (fused-binding-outs b))))
     bs)
    (map (lambda (kv) (list (caar kv) (cdar kv) (cadr kv) (cddr kv)))
         (hash-table->alist table))))

;; The red-candidates entry of ssa-fusion-pass-stats for bindings bs.
(define (fusion-red-candidates bs)
  (let ((table (make-hash-table equal?)))
    (for-each
     (lambda (b)
       (when (fused-red-binding? b)
         (let* ((outs (fused-binding-outs b))
                (o    (car outs))
                (key  (list (call-with-values
                                (lambda () (canonical-scalar-body (fused-out-body o)
                                                                  (length (ssa-binding-inputs b))))
                              (lambda (c perm) c))
                            (let ((c (car (fused-out-red o))))
                              (if (eq? c 'gemm) 'gemm (combine-spec-op c)))
                            (cdr (assq 'path (ssa-binding-meta b)))
                            (and (any (lambda (x) (not (fused-out-red x))) outs) #t)
                            (fused-out-dtype o)))
                (old  (hash-table-ref/default table key '(0 . 0))))
           (hash-table-set! table key
                            (cons (+ 1 (car old))
                                  (+ (shape-size (cdr (assq 'src-shape (ssa-binding-meta b))))
                                     (cdr old)))))))
     bs)
    (map (lambda (kv) (append (car kv) (list (cadr kv) (cddr kv))))
         (hash-table->alist table))))

;; Ops that never take part in a fusion: contractions, views and the
;; convolution family read their operands in their own patterns.
(define fusion-barrier-ops
  '(matmul reshape transpose broadcast-expand slice
    im2col col2im conv2d-fwd conv2d-bwd-data conv2d-bwd-weights attention))

;; The largest number of nodes in a fused body, and of its inputs.
(define fusion-max-body-size 64)
(define fusion-max-inputs 8)

(define (fused-binding? b) (eq? (ssa-binding-op b) 'fused))

;; The number of reduced dimensions of fused binding b: 0 for a seg-map.
(define (fused-binding-nred b)
  (let ((p (assq 'nred (ssa-binding-meta b)))) (if p (cdr p) 0)))

(define (fused-red-binding? b) (and (fused-binding? b) (> (fused-binding-nred b) 0)))

(define (reduce-binding? b)
  (let ((op (ssa-binding-op b))) (and (pair? op) (eq? (car op) 'reduce))))

(define (fused-binding-template b)
  (cdr (assq 'template (ssa-binding-meta b))))

(define (fused-binding-outs b)
  (fusion-template-outs (fused-binding-template b)))

(define (binding-out-names b)
  (if (fused-binding? b) (map fused-out-name (fused-binding-outs b)) (list (ssa-binding-name b))))

(define (binding-members b)
  (if (fused-binding? b) (cdr (assq 'members (ssa-binding-meta b))) (list (ssa-binding-name b))))

(define (make-fused-ssa-binding inputs space in-maps outs members #!optional (nred 0) (meta '()))
  "A fused binding reading inputs (ssa-vtab) through in-maps, logical
   index maps from the iteration space space, of which the last nred
   dimensions are reduced.  outs lists its outputs as (name shape dtype
   body) or (name shape dtype body map red), the binding's own result
   first (see fused-out); each body reads the inputs as (in 0) ... in the
   order of inputs.  members names the bindings it computes, in program
   order; meta is added to the binding's meta."
  (let* ((n   (length inputs))
         (fos (map (lambda (o)
                     (make-fused-out (car o) (cadr o) (caddr o) (cadddr o)
                                     (scalar-expr->procedure (cadddr o) n)
                                     (and (> (length o) 4) (list-ref o 4))
                                     (and (> (length o) 5) (list-ref o 5))))
                   outs))
         (p   (car fos)))
    (make-ssa-binding (fused-out-name p) 'fused inputs (fused-out-shape p) (fused-out-dtype p)
                      `((template . ,(make-fusion-template space in-maps fos nred))
                        (members . ,members)
                        (body . ,(fused-out-body p))
                        (combiner . ,(fused-out-combiner p))
                        (nred . ,nred)
                        ,@meta))))

;; The output o of a fusion template in the list form of
;; make-fused-ssa-binding.
(define (fused-out->list o)
  (list (fused-out-name o) (fused-out-shape o) (fused-out-dtype o) (fused-out-body o)
        (fused-out-map o) (fused-out-red o)))

(define (fusion-map-binding? b)
  "True when b computes one value per point of its shape from its inputs:
   an element-wise binding with a combiner, or a fused binding with a
   single output."
  (let ((op (ssa-binding-op b)))
    (cond ((eq? op 'fused) (and (= 0 (fused-binding-nred b)) (null? (cdr (fused-binding-outs b)))))
          ((pair? op) #f)
          ((memq op fusion-barrier-ops) #f)
          (else (ssa-binding-has-combiner? b)))))

;; Shape and dtype table of the values of prog: binding results, extra
;; outputs of fused bindings and constants.
(define (fusion-value-table prog)
  (let ((t (make-hash-table eq?)))
    (for-each (lambda (b)
                (if (fused-binding? b)
                    (for-each (lambda (o)
                                (hash-table-set! t (fused-out-name o)
                                                 (cons (fused-out-shape o) (fused-out-dtype o))))
                              (fused-binding-outs b))
                    (hash-table-set! t (ssa-binding-name b)
                                     (cons (ssa-binding-shape b) (ssa-binding-dtype b)))))
              (ssa-program-bindings prog))
    (hash-table-walk (ssa-program-constants prog)
      (lambda (k v) (hash-table-set! t k (cons (morph-shape v) (morph-dtype v)))))
    t))

(define (fusion-edge-kind c shape)
  (cond ((fusion-map-binding? c) (if (equal? shape (ssa-binding-shape c)) 'fusible 'infusible))
        ;; a reduction reads its source at every point of its space
        ((reduce-binding? c) 'fusible)
        ((fused-red-binding? c)
         (if (equal? shape (cdr (assq 'src-shape (ssa-binding-meta c)))) 'fusible 'infusible))
        (else 'infusible)))

;; Adds to uses the edges of binding c's reads.
(define (add-fusion-edges! uses c vtab)
  (let loop ((vs (ssa-binding-inputs c)) (k 0))
    (when (pair? vs)
      (when (ssa-binding-ref? (car vs))
        (let ((id (ssa-value-id (car vs))))
          (hash-table-set! uses id
                           (append (hash-table-ref/default uses id '())
                                   (list (list (ssa-binding-name c) k
                                               (fusion-edge-kind
                                                c (car (hash-table-ref vtab id)))))))))
      (loop (cdr vs) (+ k 1)))))

(define (fusion-uses prog vtab)
  (let ((uses (make-hash-table eq?)))
    (for-each (lambda (b) (add-fusion-edges! uses b vtab)) (ssa-program-bindings prog))
    (for-each (lambda (ov)
                (when (ssa-binding-ref? ov)
                  (let ((id (ssa-value-id ov)))
                    (hash-table-set! uses id
                                     (append (hash-table-ref/default uses id '())
                                             (list (list #f #f 'result)))))))
              (ssa-program-outputs prog))
    uses))

(define (ssa-fusion-graph prog)
  "The uses of the values of prog, as an alist from the name of a value
   to its uses in program order, each (consumer position kind).  kind is
   fusible, infusible or result; a result use has no consumer or position."
  (hash-table->alist (fusion-uses prog (fusion-value-table prog))))

(define (can-fuse? uses p c k)
  "True when the graph allows fusing producer p into consumer c, of which
   p is input k: p is a map binding whose single use is a fusible read by
   c, a map binding whose body reads input k once.  A body reading it
   twice (after its inputs were merged) would compute p twice."
  (and (fusion-map-binding? p)
       (fusion-map-binding? c)
       (let ((u (hash-table-ref/default uses (ssa-binding-name p) '())))
         (and (= (length u) 1)
              (eq? (car (car u)) (ssa-binding-name c))
              (eq? (caddr (car u)) 'fusible)))
       (= 1 (scalar-body-input-reads (binding-scalar-body c (ssa-binding-combiner c)) k))))

;; The number of occurrences of (in k) in body e.
(define (scalar-body-input-reads e k)
  (cond ((not (pair? e)) 0)
        ((eq? (car e) 'in) (if (eqv? (cadr e) k) 1 0))
        ((eq? (car e) 'opaque) (fold + 0 (map (lambda (x) (scalar-body-input-reads x k)) (cddr e))))
        ((eq? (car e) 'cast) (scalar-body-input-reads (caddr e) k))
        ((eq? (car e) 'let*)
         (+ (fold + 0 (map (lambda (b) (scalar-body-input-reads (cadr b) k)) (cadr e)))
            (scalar-body-input-reads (caddr e) k)))
        (else (fold + 0 (map (lambda (x) (scalar-body-input-reads x k)) (cdr e))))))

(define (scalar-body-size e)
  (cond ((not (pair? e)) 1)
        ((eq? (car e) 'in) 1)
        ((eq? (car e) 'opaque) (+ 1 (fold + 0 (map scalar-body-size (cddr e)))))
        ((eq? (car e) 'cast) (+ 1 (scalar-body-size (caddr e))))
        ((eq? (car e) 'let*)
         (+ 1 (fold + 0 (map (lambda (b) (scalar-body-size (cadr b))) (cadr e)))
            (scalar-body-size (caddr e))))
        (else (+ 1 (fold + 0 (map scalar-body-size (cdr e)))))))

(define (fused-result-legal? f vtab)
  "True when every input of fused binding f has its shape or broadcasts to
   it (the input is then read through its broadcast map), and its inputs
   and body are within the size bounds."
  (let ((shape (ssa-binding-shape f)))
    (and (<= (length (ssa-binding-inputs f)) fusion-max-inputs)
         (every (lambda (v) (broadcastable? (car (hash-table-ref vtab (ssa-value-id v))) shape))
                (ssa-binding-inputs f))
         (<= (scalar-body-size (cdr (assq 'body (ssa-binding-meta f)))) fusion-max-body-size))))

(define (fused-binding-query f vtab flat?)
  "The segop template of fused binding f: its space, the dtype, logical
   map and predicted layout of each input (flat? tells whether an input
   value is expected to be stored flat), and the dtype and body of each
   output."
  (let ((tpl (fused-binding-template f)))
    (make-segop-template
     (fusion-template-space tpl)
     (map (lambda (v m)
            (make-template-input (cdr (hash-table-ref vtab (ssa-value-id v))) m (flat? v)))
          (ssa-binding-inputs f) (fusion-template-in-maps tpl))
     (map (lambda (o)
            (let ((red (fused-out-red o)))
              (make-template-output (fused-out-dtype o) (fused-out-body o)
                                    (and red (car red)) (and red (cdr red)) (fused-out-map o))))
          (fusion-template-outs tpl))
     (fusion-template-nred tpl))))

(define (fusion-profitable? f vtab flat?)
  "True when making fused binding f should be done under the current
   policy: always under all; under the default policy when some segop
   backend would run f natively (see segop-native?).  A fused segop that
   no backend runs natively would run on the host executors, which are
   slower than the separate bindings' (see above)."
  (or (eq? (segop-fusion) 'all)
      (and (segop-native? (fused-binding-query f vtab flat?)) #t)))

;; ins without repeated values, in order of first occurrence.
(define (distinct-values ins)
  (let loop ((ins ins) (acc '()))
    (cond ((null? ins) (reverse acc))
          ((any (lambda (v) (eq? (ssa-value-id v) (ssa-value-id (car ins)))) acc)
           (loop (cdr ins) acc))
          (else (loop (cdr ins) (cons (car ins) acc))))))

(define (fuse-into p c k vtab #!optional (kept? #f))
  "The fused binding computing producer p and consumer c, of which p is
   input k.  Its inputs are c's with p's in place of input k, each read
   once; its body is c's body with p's body, rounded to p's dtype, for
   (in k).  When kept? is true, it also stores p's value as a second
   output, whose body is the same object as p's body inside c's, so that
   it is computed once."
  (let* ((p-ins  (ssa-binding-inputs p))
         (c-ins  (ssa-binding-inputs c))
         (ins    (distinct-values (append (take c-ins k) p-ins (drop c-ins (+ k 1)))))
         (pos    (lambda (v) (list-index (lambda (u) (eq? (ssa-value-id u) (ssa-value-id v))) ins)))
         (renum  (lambda (vs) (lambda (j) `(in ,(pos (list-ref vs j))))))
         (p-body* (scalar-expr-substitute-inputs (rounded-map-body p) (renum p-ins)))
         (body   (scalar-expr-substitute-inputs
                  (binding-scalar-body c (ssa-binding-combiner c))
                  (lambda (j) (if (= j k) p-body* ((renum c-ins) j)))))
         (shape  (ssa-binding-shape c))
         (rank   (vector-length shape)))
    (make-fused-ssa-binding
     ins shape
     (map (lambda (v) (broadcast-index-map (car (hash-table-ref vtab (ssa-value-id v))) rank))
          ins)
     (cons (list (ssa-binding-name c) shape (ssa-binding-dtype c) body)
           (if kept?
               (list (list (ssa-binding-name p) (ssa-binding-shape p) (ssa-binding-dtype p) p-body*))
               '()))
     (append (binding-members p) (binding-members c)))))

;;; ------------------------------------------------------------
;;; Multi-output and horizontal fusion
;;; ------------------------------------------------------------
;;;
;;; A map P with several uses may still be fused into a map C that reads
;;; it: the fused binding also stores P's value, as a second output, for
;;; the other uses.  Two maps of the same shape that read a common input,
;;; and neither of which depends on the other, may be fused into one
;;; seg-map with an output for each (horizontal fusion), which reads the
;;; common input once.  Outputs of fused bindings that are left without
;;; uses are removed at the end of each round.

;; The largest number of outputs of a fused binding.
(define fusion-max-outputs 4)

;; True when b computes one or more values at every point of its shape:
;; a map binding, or a fused seg-map with any number of outputs.
(define (fusion-group-binding? b)
  (or (fusion-map-binding? b)
      (and (fused-binding? b) (= 0 (fused-binding-nred b)))))

;; The parts of map binding or fused seg-map b, as (inputs in-maps outs),
;; outs in the list form of make-fused-ssa-binding.
(define (map-parts b vtab)
  (if (fused-binding? b)
      (let ((tpl (fused-binding-template b)))
        (list (ssa-binding-inputs b) (fusion-template-in-maps tpl)
              (map fused-out->list (fusion-template-outs tpl))))
      (let ((rank (vector-length (ssa-binding-shape b))))
        (list (ssa-binding-inputs b)
              (map (lambda (v) (broadcast-index-map (car (hash-table-ref vtab (ssa-value-id v))) rank))
                   (ssa-binding-inputs b))
              (list (list (ssa-binding-name b) (ssa-binding-shape b) (ssa-binding-dtype b)
                          (binding-scalar-body b (ssa-binding-combiner b))))))))

(define (fuse-siblings s1 s2 vtab)
  "The fused seg-map computing map bindings or fused seg-maps s1 and s2,
   of the same shape, with the outputs of s1 first.  An input that both
   read through the same map is read once."
  (let* ((p1    (map-parts s1 vtab))
         (p2    (map-parts s2 vtab))
         (reads (distinct-reads (append (map cons (car p1) (cadr p1)) (map cons (car p2) (cadr p2)))))
         (renum (lambda (parts)
                  (let ((memo (list '()))
                        (subst (lambda (j)
                                 `(in ,(read-position reads (list-ref (car parts) j)
                                                      (list-ref (cadr parts) j))))))
                    (map (lambda (o) (list (car o) (cadr o) (caddr o)
                                           (renumber-body (cadddr o) subst memo)))
                         (caddr parts))))))
    (make-fused-ssa-binding (map car reads) (ssa-binding-shape s1) (map cdr reads)
                            (append (renum p1) (renum p2))
                            (append (binding-members s1) (binding-members s2)))))

(define (drop-fused-outputs b dead)
  "Fused binding b without its outputs named in dead, and without the
   inputs that only those outputs read."
  (let* ((tpl    (fused-binding-template b))
         (outs   (filter (lambda (o) (not (memq (car o) dead)))
                         (map fused-out->list (fusion-template-outs tpl))))
         (ins    (ssa-binding-inputs b))
         (maps   (fusion-template-in-maps tpl))
         (used   (delete-duplicates
                  (append-map (lambda (o)
                                (let ((red (list-ref o 5)))
                                  (append (scalar-expr-inputs (cadddr o))
                                          (if (and red (cdr red)) (scalar-expr-inputs (cdr red)) '()))))
                              outs)))
         (keep   (filter (lambda (j) (memv j used)) (iota (length ins))))
         (subst  (lambda (j) `(in ,(list-index (lambda (x) (= x j)) keep))))
         (memo   (list '()))
         (outs*  (map (lambda (o)
                        (let ((red (list-ref o 5)))
                          (list (car o) (cadr o) (caddr o) (renumber-body (cadddr o) subst memo)
                                (list-ref o 4)
                                (and red (cons (car red)
                                               (and (cdr red) (scalar-expr-substitute-inputs (cdr red) subst)))))))
                      outs))
         (meta   (filter (lambda (kv) (memq (car kv) '(src-shape path))) (ssa-binding-meta b))))
    (make-fused-ssa-binding (map (lambda (j) (list-ref ins j)) keep) (fusion-template-space tpl)
                            (map (lambda (j) (list-ref maps j)) keep)
                            outs* (binding-members b) (fusion-template-nred tpl) meta)))

;;; ------------------------------------------------------------
;;; Fusion with reductions
;;; ------------------------------------------------------------
;;;
;;; A map P read by a reduction C is fused into it: the fused seg-red
;;; computes P's body, rounded to P's dtype, at each point and combines it
;;; as the unfused reduction would.  When P has other uses, the fused
;;; binding also stores P's value as a map output and takes P's position,
;;; because those uses may come before C.
;;;
;;; A map C that is the only use of a reduction R's result, and reads it at
;;; C's own shape, becomes R's finalize: C's body, with R's stored value in
;;; place of the read.  The fused binding takes R's position when C's other
;;; inputs exist there, and otherwise C's position if R has no map outputs.
;;;
;;; A reduction whose order is left to its backend (the 1d sum, which may
;;; use the BLAS dot product) is never fused.
;;;
;;; A matmul is a reduction over the space (M N K) with K reduced, whose
;;; result has red (gemm . finalize) (see segop-lower.scm): maps that read
;;; its product become its epilogue, as above, when the epilogue keeps the
;;; product's dtype, since it runs in place on the stored product.  No map
;;; is fused into its operands, which BLAS reads from memory.

(define (gemm-binding? b) (eq? (ssa-binding-op b) 'matmul))

;; The parts of matmul binding g, in the form of reduction-parts; or #f and
;; the reason when its operands are not two matrices of its dtype, f32 or
;; f64.
(define (gemm-parts g vtab)
  (let* ((ins (ssa-binding-inputs g))
         (dt  (ssa-binding-dtype g))
         (sds (map (lambda (v) (hash-table-ref vtab (ssa-value-id v))) ins)))
    (if (and (= 2 (length ins))
             (memq dt '(f32 f64))
             (every (lambda (sd) (and (= 2 (vector-length (car sd))) (eq? (cdr sd) dt))) sds))
        (let ((space (vector (vector-ref (car (car sds)) 0)
                             (vector-ref (car (cadr sds)) 1)
                             (vector-ref (car (car sds)) 1))))
          (values (list ins space
                        (list (make-index-map gemm-A-exprs #f 0.0 #f)
                              (make-index-map gemm-B-exprs #f 0.0 #f))
                        (list (list (ssa-binding-name g) (ssa-binding-shape g) dt gemm-body
                                    (make-index-map gemm-out-exprs #f 0.0 #f) (cons 'gemm #f)))
                        1 space 'gemm)
                  #f))
        (values #f 'gemm-operands))))

;; The parts of reduction binding r (a reduce binding, a matmul or a fused
;; seg-red) as a list (inputs space in-maps outs nred src-shape path), outs
;; in the list form of make-fused-ssa-binding; or #f and the reason r
;; cannot be fused.  A reduce binding's source must be expected flat
;; (flat?), since its combine spec depends on its layout.
(define (reduction-parts r vtab flat?)
  (cond
   ((fused-binding? r)
      (let ((tpl (fused-binding-template r)) (meta (ssa-binding-meta r)))
        (values (list (ssa-binding-inputs r) (fusion-template-space tpl) (fusion-template-in-maps tpl)
                      (map fused-out->list (fusion-template-outs tpl)) (fusion-template-nred tpl)
                      (cdr (assq 'src-shape meta)) (cdr (assq 'path meta)))
                #f)))
   ((gemm-binding? r) (gemm-parts r vtab))
   (else
      (let* ((meta  (ssa-binding-meta r))
             (rop   (cadr (ssa-binding-op r)))
             (src   (car (ssa-binding-inputs r)))
             (sd    (hash-table-ref vtab (ssa-value-id src)))
             (s     (car sd))
             (sdt   (cdr sd))
             (rank  (vector-length s))
             (axes  (cdr (assq 'axes meta)))
             (keep? (let ((p (assq 'keepdims? meta))) (and p (cdr p))))
             (red   (filter (lambda (k) (memv k axes)) (iota rank)))
             (par   (filter (lambda (k) (not (memv k axes))) (iota rank)))
             (order (append par red))
             (dim-of (lambda (k) `(dim ,(list-index (lambda (j) (= j k)) order))))
             (in-map (make-index-map (list->vector (map dim-of (iota rank))) #f 0.0 #f))
             (out-map (if keep?
                          (make-index-map (list->vector (map (lambda (k) (if (memv k red) 0 (dim-of k)))
                                                             (iota rank)))
                                          #f 0.0 #f)
                          (identity-index-map (length par))))
             (path  (reduction-path rop s (compute-strides s) 0 sdt red))
             (spec  (reduction-combine-spec rop path sdt (ssa-binding-dtype r)))
             (count (fold (lambda (k acc) (* acc (vector-ref s k))) 1 red)))
        (cond ((eq? (combine-spec-order spec) 'unspecified) (values #f 'reduction-order))
              ((not (flat? src)) (values #f 'view))
              (else
               (values (list (list src)
                             (list->vector (map (lambda (k) (vector-ref s k)) order))
                             (list in-map)
                             (list (list (ssa-binding-name r) (ssa-binding-shape r) (ssa-binding-dtype r)
                                         '(in 0) out-map
                                         (cons spec (and (eq? rop 'mean) `(/ (reduced) ,count)))))
                             (length red) s path)
                       #f)))))))

;; True when index maps a and b read the same element at every point.
(define (same-index-map? a b)
  (and (not (index-map-proc a)) (not (index-map-proc b))
       (not (index-map-guard a)) (not (index-map-guard b))
       (equal? (index-map-exprs a) (index-map-exprs b))))

;; The (value . map) pairs of pairs without repeats, in order of first
;; occurrence.  A value read through two different maps is kept twice.
(define (distinct-reads pairs)
  (let loop ((ps pairs) (acc '()))
    (cond ((null? ps) (reverse acc))
          ((any (lambda (q) (and (eq? (ssa-value-id (car q)) (ssa-value-id (caar ps)))
                                 (same-index-map? (cdr q) (cdar ps))))
                acc)
           (loop (cdr ps) acc))
          (else (loop (cdr ps) (cons (car ps) acc))))))

(define (read-position reads v m)
  (list-index (lambda (q) (and (eq? (ssa-value-id (car q)) (ssa-value-id v)) (same-index-map? (cdr q) m)))
              reads))

;; e with each (in j) replaced by (subst j).  memo keeps bodies that were
;; the same object the same object after the change.
(define (renumber-body e subst memo)
  (let ((hit (assq e (car memo))))
    (if hit
        (cdr hit)
        (let ((new (scalar-expr-substitute-inputs e subst)))
          (set-car! memo (cons (cons e new) (car memo)))
          new))))

;; The body of map binding p, rounded to p's dtype as storing it would.
(define (rounded-map-body p)
  (let ((b (binding-scalar-body p (ssa-binding-combiner p))) (dt (ssa-binding-dtype p)))
    (if (eq? dt 'f64) b `(cast ,dt ,b))))

;; The number of reads of input k by the bodies of outs, a body shared by
;; several outputs counted once, and by their finalize bodies.
(define (parts-input-reads outs k)
  (let loop ((os outs) (seen '()) (n 0))
    (if (null? os)
        n
        (let* ((o (car os)) (body (list-ref o 3)) (red (list-ref o 5))
               (fin-n (if (and red (cdr red)) (scalar-body-input-reads (cdr red) k) 0)))
          (if (memq body seen)
              (loop (cdr os) seen (+ n fin-n))
              (loop (cdr os) (cons body seen) (+ n fin-n (scalar-body-input-reads body k))))))))

(define (make-fused-reduction ins+maps space outs nred src-shape path members)
  (make-fused-ssa-binding (map car ins+maps) space (map cdr ins+maps) outs members nred
                          `((src-shape . ,src-shape) (path . ,path))))

(define (fuse-map-into-reduction p parts k kept? members vtab)
  "The fused seg-red computing map binding p and the reduction whose parts
   are parts (see reduction-parts), of which p is input k.  When kept? is
   true, p's value is also written, as a map output named after p."
  (let* ((r-ins   (list-ref parts 0))
         (in-maps (list-ref parts 2))
         (outs    (list-ref parts 3))
         (m-p     (list-ref in-maps k))
         (p-ins   (ssa-binding-inputs p))
         (p-rank  (vector-length (ssa-binding-shape p)))
         (p-maps  (map (lambda (v)
                         (compose-index-maps
                          (broadcast-index-map (car (hash-table-ref vtab (ssa-value-id v))) p-rank)
                          m-p))
                       p-ins))
         (reads   (distinct-reads (append (map cons (take r-ins k) (take in-maps k))
                                          (map cons p-ins p-maps)
                                          (map cons (drop r-ins (+ k 1)) (drop in-maps (+ k 1))))))
         (p-body  (scalar-expr-substitute-inputs
                   (rounded-map-body p)
                   (lambda (i) `(in ,(read-position reads (list-ref p-ins i) (list-ref p-maps i))))))
         (subst   (lambda (j)
                    (if (= j k)
                        p-body
                        `(in ,(read-position reads (list-ref r-ins j) (list-ref in-maps j))))))
         (memo    (list '()))
         (outs*   (map (lambda (o)
                         (let ((red (list-ref o 5)))
                           (list (car o) (cadr o) (caddr o) (renumber-body (list-ref o 3) subst memo)
                                 (list-ref o 4)
                                 (and red (cons (car red)
                                                (and (cdr red)
                                                     (scalar-expr-substitute-inputs (cdr red) subst)))))))
                       outs)))
    (make-fused-reduction
     reads (list-ref parts 1)
     (if kept?
         (append outs* (list (list (ssa-binding-name p) (ssa-binding-shape p) (ssa-binding-dtype p)
                                   p-body m-p #f)))
         outs*)
     (list-ref parts 4) (list-ref parts 5) (list-ref parts 6) members)))

(define (fuse-epilogue parts c k members vtab)
  "The fused seg-red computing the reduction whose parts are parts and map
   binding c, which reads the reduction's result as input k: c's body
   becomes the result's finalize."
  (let* ((r-ins   (list-ref parts 0))
         (in-maps (list-ref parts 2))
         (outs    (list-ref parts 3))
         (res     (car outs))
         (om      (list-ref res 4))
         (red     (list-ref res 5))
         (r-dtype (list-ref res 2))
         (c-ins   (ssa-binding-inputs c))
         (c-rank  (vector-length (ssa-binding-shape c)))
         (c-maps  (map (lambda (v)
                         (compose-index-maps
                          (broadcast-index-map (car (hash-table-ref vtab (ssa-value-id v))) c-rank)
                          om))
                       c-ins))
         (others  (filter (lambda (j) (not (= j k))) (iota (length c-ins))))
         (reads   (distinct-reads (append (map cons r-ins in-maps)
                                          (map (lambda (j) (cons (list-ref c-ins j) (list-ref c-maps j)))
                                               others))))
         (fin     (or (cdr red) '(reduced)))
         (stored  (if (eq? r-dtype 'f64) fin `(cast ,r-dtype ,fin)))
         (fin*    (scalar-expr-substitute-inputs
                   (binding-scalar-body c (ssa-binding-combiner c))
                   (lambda (j)
                     (if (= j k)
                         stored
                         `(in ,(read-position reads (list-ref c-ins j) (list-ref c-maps j))))))))
    (make-fused-reduction
     reads (list-ref parts 1)
     (cons (list (ssa-binding-name c) (ssa-binding-shape c) (ssa-binding-dtype c)
                 (list-ref res 3) om (cons (car red) fin*))
           (cdr outs))
     (list-ref parts 4) (list-ref parts 5) (list-ref parts 6) members)))

(define (fused-reduction-legal? f)
  "True when fused seg-red f is within the bounds on the number of inputs
   and the size of its bodies, finalize bodies included."
  (and (<= (length (ssa-binding-inputs f)) fusion-max-inputs)
       (every (lambda (o)
                (and (<= (scalar-body-size (fused-out-body o)) fusion-max-body-size)
                     (let ((red (fused-out-red o)))
                       (or (not red) (not (cdr red))
                           (<= (scalar-body-size (cdr red)) fusion-max-body-size)))))
              (fused-binding-outs f))))

(define (ssa-segop-fusion-pass prog)
  "Fuse the map bindings of prog into their consumers, maps into the
   reductions that read them, and reductions into the maps that read their
   results (see above).  Returns a new program; prog is not changed."
  (let* ((bv     (list->vector (ssa-program-bindings prog)))
         (n      (vector-length bv))
         (pos    (make-hash-table eq?))
         (vtab (fusion-value-table prog))
         (uses   (fusion-uses prog vtab))
         (layouts (make-hash-table eq?))
         (max-rounds (segop-fusion-max-rounds)))

    ;; True when value v is expected to be stored row-major, at offset 0:
    ;; a constant with that layout, the result of a binding that allocates
    ;; (pool buffers are row-major), or a reshape of such a value, which
    ;; keeps its buffer and its row-major order.  Other views are not
    ;; expected to be flat.
    (define (flat? v)
      (cases ssa-value v
        (const-ref (cid) (concrete-row-major? (hash-table-ref (ssa-program-constants prog) cid)))
        (binding-ref (bid) (hash-table-ref/default layouts bid #t))))
    (for-each (lambda (b)
                (let ((op (ssa-binding-op b)))
                  (cond ((eq? op 'reshape)
                         (hash-table-set! layouts (ssa-binding-name b)
                                          (flat? (car (ssa-binding-inputs b)))))
                        ((memq op fusion-view-ops)
                         (hash-table-set! layouts (ssa-binding-name b) #f)))))
              (ssa-program-bindings prog))
    ;; pos maps the name of every value a binding computes to the
    ;; binding's position; pos-of-binding maps each binding's name.
    (define pos-of-binding (make-hash-table eq?))
    (do ((i 0 (+ i 1))) ((= i n))
      (let ((b (vector-ref bv i)))
        (hash-table-set! pos-of-binding (ssa-binding-name b) i)
        (for-each (lambda (name) (hash-table-set! pos name i)) (binding-out-names b))))

    (define (binding-of v)
      (and (ssa-binding-ref? v)
           (let ((i (hash-table-ref/default pos (ssa-value-id v) #f)))
             (and i (vector-ref bv i)))))

    ;; Replaces the bindings olds by f at position at, and updates the uses
    ;; of their inputs.
    (define (commit-at! olds f at)
      (let ((names (map ssa-binding-name olds))
            (keep  (map fused-out-name (fused-binding-outs f))))
        (for-each (lambda (v)
                    (when (ssa-binding-ref? v)
                      (let ((id (ssa-value-id v)))
                        (hash-table-set! uses id
                                         (filter (lambda (u) (not (memq (car u) names)))
                                                 (hash-table-ref/default uses id '()))))))
                  (append-map ssa-binding-inputs olds))
        (for-each (lambda (b)
                    (for-each (lambda (name)
                                (unless (memq name keep) (hash-table-set! uses name '()))
                                (hash-table-set! pos name #f))
                              (binding-out-names b))
                    (vector-set! bv (hash-table-ref pos-of-binding (ssa-binding-name b)) #f))
                  olds)
        (vector-set! bv at f)
        (for-each (lambda (name) (hash-table-set! pos name at)) keep)
        (hash-table-set! pos-of-binding (ssa-binding-name f) at)
        (add-fusion-edges! uses f vtab)))

    ;; Replaces p and c by f at c's position.
    (define (commit! p c f)
      (commit-at! (list p c) f (hash-table-ref pos-of-binding (ssa-binding-name c))))

    ;; The reasons for which the current round left pairs unfused, as an
    ;; alist from reason to count.
    (define missed '())
    (define (miss! reason)
      (let ((e (assq reason missed)))
        (if e (set-cdr! e (+ (cdr e) 1)) (set! missed (cons (cons reason 1) missed)))))

    ;; True when the only use of the value named name is a fusible read by
    ;; the binding named c-name.
    (define (single-fusible-use? name c-name)
      (let ((u (hash-table-ref/default uses name '())))
        (and (= (length u) 1)
             (eq? (car (car u)) c-name)
             (eq? (caddr (car u)) 'fusible))))

    ;; Records why the fusion of p into c, of which p is input k, is not
    ;; made, when p is a candidate that the rules or the policy refuse.
    (define (note-missed! p c k)
      (cond ((can-fuse? uses p c k)
             (let ((f (fuse-into p c k vtab)))
               (cond ((not (fused-result-legal? f vtab)) (miss! 'size-cap))
                     ((fusion-profitable? f vtab flat?))
                     ((every (lambda (v) (equal? (car (hash-table-ref vtab (ssa-value-id v)))
                                                 (ssa-binding-shape f)))
                             (ssa-binding-inputs f))
                      (miss! 'not-native))
                     (else (miss! 'broadcast-input)))))
            ((and (memq (ssa-binding-op p) '(reshape transpose))
                  (fusion-map-binding? c)
                  (single-fusible-use? (ssa-binding-name p) (ssa-binding-name c))
                  (let ((q (binding-of (car (ssa-binding-inputs p)))))
                    (and q
                         (fusion-map-binding? q)
                         (let ((u (hash-table-ref/default uses (ssa-binding-name q) '())))
                           (and (= (length u) 1)
                                (eq? (car (car u)) (ssa-binding-name p)))))))
             (miss! 'view))))

    ;; Counts of the fusions that keep a producer, of horizontal fusions
    ;; and of the outputs removed.
    (define kept-count 0)
    (define horizontal-count 0)
    (define dead-count 0)

    ;; The position of the binding that computes value v, or #f for a
    ;; constant.
    (define (value-position v)
      (and (ssa-binding-ref? v) (hash-table-ref/default pos (ssa-value-id v) #f)))

    ;; Fuses producer p, which has other uses, into map c, of which it is
    ;; input k, keeping p's value as a second output; true if it did.  The
    ;; fused binding takes p's position when c's other inputs exist there,
    ;; and otherwise c's position when no other use of p comes before c.
    (define (fuse-kept! p c k)
      (let* ((p-name (ssa-binding-name p))
             (c-name (ssa-binding-name c))
             (u      (hash-table-ref/default uses p-name '())))
        (and (fusion-map-binding? p)
             (fusion-map-binding? c)
             (eq? p-name (ssa-value-id (list-ref (ssa-binding-inputs c) k)))
             (> (length u) 1)
             (= 1 (length (filter (lambda (x) (eq? (car x) c-name)) u)))
             (any (lambda (x) (and (eq? (car x) c-name) (eq? (caddr x) 'fusible))) u)
             (= 1 (scalar-body-input-reads (binding-scalar-body c (ssa-binding-combiner c)) k))
             (let* ((p-at (hash-table-ref pos-of-binding p-name))
                    (c-at (hash-table-ref pos-of-binding c-name))
                    (at   (cond ((every (lambda (v j)
                                          (or (= j k)
                                              (let ((q (value-position v))) (or (not q) (< q p-at)))))
                                        (ssa-binding-inputs c) (iota (length (ssa-binding-inputs c))))
                                 p-at)
                                ((every (lambda (x)
                                          (or (not (car x)) (eq? (car x) c-name)
                                              (> (hash-table-ref pos-of-binding (car x)) c-at)))
                                        u)
                                 c-at)
                                (else #f))))
               (if (not at)
                   (begin (miss! 'placement) #f)
                   (let ((f (fuse-into p c k vtab #t)))
                     (cond ((not (fused-result-legal? f vtab)) (miss! 'size-cap) #f)
                           ((not (fusion-profitable? f vtab flat?)) (miss! 'not-native) #f)
                           (else (commit-at! (list p c) f at)
                                 (set! kept-count (+ kept-count 1))
                                 #t))))))))

    ;; Fuses one producer into the binding at position i; true if it did.
    (define (fuse-one! i)
      (let ((c (vector-ref bv i)))
        (let loop ((vs (ssa-binding-inputs c)) (k 0))
          (and (pair? vs)
               (or (let ((p (binding-of (car vs))))
                     (and p
                          (or (and (can-fuse? uses p c k)
                                   (let ((f (fuse-into p c k vtab)))
                                     (and (fused-result-legal? f vtab)
                                          (fusion-profitable? f vtab flat?)
                                          (begin (commit! p c f) #t))))
                              (begin (note-missed! p c k) #f))))
                   (loop (cdr vs) (+ k 1)))))))

    ;; Fuses into the map at position i one producer that has other uses,
    ;; keeping it as an output; true if it did.
    (define (fuse-one-kept! i)
      (let ((c (vector-ref bv i)))
        (let loop ((vs (ssa-binding-inputs c)) (k 0))
          (and (pair? vs)
               (or (let ((p (binding-of (car vs)))) (and p (fuse-kept! p c k)))
                   (loop (cdr vs) (+ k 1)))))))

    ;; True when the binding at position j depends on a value computed by
    ;; the binding at position i < j.
    (define (depends? j i)
      (let ((names (binding-out-names (vector-ref bv i))))
        (let walk ((stack (list j)) (seen '()))
          (cond ((null? stack) #f)
                ((memv (car stack) seen) (walk (cdr stack) seen))
                (else
                 (let ((ins (ssa-binding-inputs (vector-ref bv (car stack)))))
                   (or (any (lambda (v) (and (ssa-binding-ref? v) (memq (ssa-value-id v) names))) ins)
                       (walk (append (filter (lambda (q) (and q (> q i)))
                                             (map value-position ins))
                                     (cdr stack))
                             (cons (car stack) seen)))))))))

    ;; Fuses the map binding or fused seg-map at position i with an earlier
    ;; sibling: one of the same shape that reads a common input through the
    ;; same map and on which it does not depend; true if it did.  The fused
    ;; binding takes position i when no use of the sibling comes before i,
    ;; and otherwise the sibling's position when the inputs at i all exist
    ;; there.
    (define (fuse-horizontal! i)
      (let* ((s2    (vector-ref bv i))
             (pairs (map cons (ssa-binding-inputs s2) (cadr (map-parts s2 vtab)))))
        (let loop ((j (- i 1)))
          (and (>= j 0)
               (or (let ((s1 (vector-ref bv j)))
                     (and s1
                          (fusion-group-binding? s1)
                          (equal? (ssa-binding-shape s1) (ssa-binding-shape s2))
                          (<= (+ (length (binding-out-names s1)) (length (binding-out-names s2)))
                              fusion-max-outputs)
                          (let ((pairs1 (map cons (ssa-binding-inputs s1) (cadr (map-parts s1 vtab)))))
                            (any (lambda (p)
                                   (any (lambda (q) (and (eq? (ssa-value-id (car p)) (ssa-value-id (car q)))
                                                         (same-index-map? (cdr p) (cdr q))))
                                        pairs1))
                                 pairs))
                          (not (depends? i j))
                          (let ((at (cond ((every (lambda (name)
                                                    (every (lambda (x)
                                                             (or (not (car x))
                                                                 (> (hash-table-ref pos-of-binding (car x)) i)))
                                                           (hash-table-ref/default uses name '())))
                                                  (binding-out-names s1))
                                           i)
                                          ((every (lambda (v) (let ((q (value-position v))) (or (not q) (< q j))))
                                                  (ssa-binding-inputs s2))
                                           j)
                                          (else #f))))
                            (if (not at)
                                (begin (miss! 'horizontal-placement) #f)
                                (let ((f (fuse-siblings s1 s2 vtab)))
                                  (cond ((> (length (ssa-binding-inputs f)) fusion-max-inputs)
                                         (miss! 'size-cap) #f)
                                        ((not (fusion-profitable? f vtab flat?)) (miss! 'not-native) #f)
                                        (else (commit-at! (list s1 s2) f at)
                                              (set! horizontal-count (+ horizontal-count 1))
                                              #t)))))))
                   (loop (- j 1)))))))

    ;; Removes the outputs of fused bindings, other than their first, that
    ;; have no uses; returns how many it removed.
    (define (remove-dead-outputs!)
      (let loop ((i 0) (count 0))
        (if (= i n)
            count
            (let ((b (vector-ref bv i)))
              (if (and b (fused-binding? b))
                  (let ((dead (filter (lambda (name) (null? (hash-table-ref/default uses name '())))
                                      (cdr (binding-out-names b)))))
                    (if (null? dead)
                        (loop (+ i 1) count)
                        (begin
                          (commit-at! (list b) (drop-fused-outputs b dead) i)
                          (loop (+ i 1) (+ count (length dead))))))
                  (loop (+ i 1) count))))))

    ;; Puts fused binding f in place of olds at position at, if it is legal
    ;; and profitable; otherwise records why not.
    (define (try-commit! olds f at)
      (cond ((not (fused-reduction-legal? f)) (miss! 'size-cap) #f)
            ((not (fusion-profitable? f vtab flat?)) (miss! 'not-native) #f)
            (else (commit-at! olds f at) #t)))

    ;; Fuses a map producer into the reduction at position i; true if it
    ;; did.  A producer with other uses is kept as a map output, which only
    ;; a reduce binding can do.
    (define (fuse-into-reduction! i)
      (let ((c (vector-ref bv i)))
        (let loop ((vs (ssa-binding-inputs c)) (k 0))
          (and (pair? vs)
               (or (let ((p (binding-of (car vs))))
                     (and p
                          (eq? (ssa-binding-name p) (ssa-value-id (car vs)))
                          (fusion-map-binding? p)
                          (let* ((p-name (ssa-binding-name p))
                                 (u      (hash-table-ref/default uses p-name '()))
                                 (single (single-fusible-use? p-name (ssa-binding-name c)))
                                 (kept   (and (not single) (reduce-binding? c)
                                              (any (lambda (x) (and (eq? (car x) (ssa-binding-name c))
                                                                    (eq? (caddr x) 'fusible)))
                                                   u))))
                            (and (or single kept)
                                 (call-with-values (lambda () (reduction-parts c vtab flat?))
                                   (lambda (parts reason)
                                     (cond
                                      ((not parts) (miss! reason) #f)
                                      ((not (and (= 1 (parts-input-reads (list-ref parts 3) k))
                                                 (every (lambda (o)
                                                          (let ((red (list-ref o 5)))
                                                            (or (not red) (not (cdr red))
                                                                (= 0 (scalar-body-input-reads (cdr red) k)))))
                                                        (list-ref parts 3))))
                                       #f)
                                      (else
                                       (let ((f (fuse-map-into-reduction
                                                 p parts k kept
                                                 (append (binding-members p) (binding-members c))
                                                 vtab)))
                                         (try-commit! (list p c) f
                                                      (hash-table-ref pos-of-binding
                                                                      (if kept p-name (ssa-binding-name c)))))))))))))
                   (loop (cdr vs) (+ k 1)))))))

    ;; Fuses map binding c at position i into the reduction whose result
    ;; it reads (an epilogue); true if it did.
    (define (fuse-epilogue! i)
      (let ((c (vector-ref bv i)))
        (let loop ((vs (ssa-binding-inputs c)) (k 0))
          (and (pair? vs)
               (or (let ((r (binding-of (car vs))))
                     (and r
                          (or (reduce-binding? r) (fused-red-binding? r) (gemm-binding? r))
                          (eq? (ssa-binding-name r) (ssa-value-id (car vs)))
                          (let* ((r-name (ssa-binding-name r))
                                 (u      (hash-table-ref/default uses r-name '())))
                            (and (any (lambda (x) (and (eq? (car x) (ssa-binding-name c))
                                                       (eq? (caddr x) 'fusible)))
                                      u)
                                 (cond
                                  ((not (single-fusible-use? r-name (ssa-binding-name c)))
                                   (miss! 'multi-use-reduction) #f)
                                  ;; a second read would compute the finalize twice
                                  ((not (= 1 (scalar-body-input-reads
                                              (binding-scalar-body c (ssa-binding-combiner c)) k)))
                                   (miss! 'multi-use-reduction) #f)
                                  ;; c also reads another output of r
                                  ((any (lambda (v)
                                          (and (ssa-binding-ref? v)
                                               (not (eq? (ssa-value-id v) r-name))
                                               (memq (ssa-value-id v) (binding-out-names r))))
                                        (ssa-binding-inputs c))
                                   #f)
                                  (else
                                   (call-with-values (lambda () (reduction-parts r vtab flat?))
                                     (lambda (parts reason)
                                       (cond
                                        ((not parts) (miss! reason) #f)
                                        ;; a GEMM epilogue is stored in the product's buffer
                                        ((and (eq? (list-ref parts 6) 'gemm)
                                              (not (eq? (ssa-binding-dtype c)
                                                        (list-ref (car (list-ref parts 3)) 2))))
                                         (miss! 'gemm-epilogue-form) #f)
                                        (else
                                           (let* ((r-at (hash-table-ref pos-of-binding r-name))
                                                  (early? (every (lambda (v)
                                                                   (or (not (ssa-binding-ref? v))
                                                                       (eq? (ssa-value-id v) r-name)
                                                                       (let ((j (hash-table-ref/default
                                                                                 pos (ssa-value-id v) #f)))
                                                                         (or (not j) (< j r-at)))))
                                                                 (ssa-binding-inputs c)))
                                                  (at (cond (early? r-at)
                                                            ((null? (cdr (list-ref parts 3))) i)
                                                            (else #f))))
                                             (if (not at)
                                                 (begin (miss! 'placement) #f)
                                                 (try-commit!
                                                  (list r c)
                                                  (fuse-epilogue parts c k
                                                                 (append (binding-members r) (binding-members c))
                                                                 vtab)
                                                  at)))))))))))))
                   (loop (cdr vs) (+ k 1)))))))

    ;; Tries rule on each binding that satisfies pred, from last to first,
    ;; until it fuses nothing more there; returns the number of fusions.
    (define (run-phase pred rule)
      (let loop ((i (- n 1)) (count 0))
        (if (< i 0)
            count
            (let ((b (vector-ref bv i)))
              (if (and b (pred b))
                  (let fuse ((count count))
                    (if (and (vector-ref bv i) (pred (vector-ref bv i)) (rule i))
                        (fuse (+ count 1))
                        (loop (- i 1) count)))
                  (loop (- i 1) count))))))

    ;; Maps are fused first, so that a chain of maps is one map before it
    ;; meets its reduction.  Fusions that keep a producer as a second output,
    ;; and horizontal fusions, come after the fusions with reductions, since
    ;; a binding with several outputs no longer fuses into a reduction.
    (define (run-round)
      (set! missed '())
      (let* ((maps (run-phase fusion-map-binding? fuse-one!))
             (reds (run-phase (lambda (b) (or (reduce-binding? b) (fused-red-binding? b)))
                              fuse-into-reduction!))
             (epis (run-phase fusion-map-binding? fuse-epilogue!))
             (kept (run-phase fusion-map-binding? fuse-one-kept!))
             (horz (run-phase fusion-group-binding? fuse-horizontal!))
             (dead (remove-dead-outputs!)))
        (set! dead-count (+ dead-count dead))
        (set! round-dead dead)
        (+ maps reds epis kept horz)))

    ;; The outputs the last round removed: a binding left with one output
    ;; may fuse in the next round.
    (define round-dead 0)

    (let rounds ((r 1) (total 0))
      (let* ((fused (run-round)) (total (+ total fused)))
        (if (or (= 0 fused round-dead) (>= r max-rounds))
            (set! *fusion-pass-stats*
                  `((rounds . ,r) (fusions . ,total) (bound-reached? . ,(> fused 0))
                    (kept . ,kept-count) (horizontal . ,horizontal-count) (dead-outputs . ,dead-count)
                    (candidates . ,(fusion-candidates (filter (lambda (b) b) (vector->list bv))))
                    (red-candidates . ,(fusion-red-candidates (filter (lambda (b) b) (vector->list bv))))
                    (missed . ,(reverse missed))))
            (rounds (+ r 1) total))))

    (make-ssa-program
     (ssa-program-constants prog)
     (ssa-program-morph-to-val prog)
     (filter (lambda (b) b) (vector->list bv))
     (ssa-program-outputs prog)
     (ssa-program-n-params prog)
     #f #f #f #t)))

(define (realize-fused-binding ctx b inputs)
  "Realize fused binding b on the concrete arrays inputs, through its segop
   executor, the one a replay plan would select.  The outputs are
   allocated through context ctx, or as fresh arrays when ctx is #f.
   Returns one array per output."
  (let* ((tpl   (fused-binding-template b))
         (outs  (fusion-template-outs tpl))
         (in-v  (list->vector inputs))
         (sg    (lower-fused-binding tpl inputs (iota (length inputs)) (iota (length outs))))
         (exec  (cdr (or (select-segop-executor sg)
                         (error "realize-fused-binding: no backend accepts the segop"
                                (ssa-binding-name b)))))
         (deref (lambda (k) (vector-ref in-v k)))
         (run!  (lambda (bufs)
                  (if (null? (cdr bufs))
                      (exec deref (car bufs))
                      (exec deref (car bufs) (cdr bufs)))))
         (specs (map (lambda (o) (cons (fused-out-shape o) (fused-out-dtype o))) outs)))
    (if ctx
        (realize-segop/ctx ctx inputs specs run!)
        (let ((bufs (map (lambda (s) (allocate-typed-vector (cdr s) (shape-size (car s)))) specs)))
          (run! bufs)
          (map (lambda (s buf) (concrete-array buf (car s) (compute-strides (car s)) 0 (cdr s) -1 -1))
               specs bufs)))))


;;; ============================================================
;;; Phase 1: morphism-to-ssa
;;;
;;; Post-order DFS over the morphism-expr tree rooted at loss-mv.
;;; concrete-array leaves become constants; morphism-expr / reduction-morphism
;;; nodes become SSA bindings.
;;; ============================================================

(define (morphism-to-ssa loss-mv)
  "Compile the morph-variable graph rooted at loss-mv into an SSA program.
   Returns an ssa-program with outputs = (list loss-binding-val) and n-params = 0."
  (let* ((constants   (make-hash-table))  ; cid-symbol -> concrete-array
         (visited-eb  (make-env-builder)) ; stable-id (symbol) -> ssa-value
         (bindings-stack (make-stack)))   ; bindings, newest first (reverse topo order)

    (define (emit-binding! op inputs shape dtype meta)
      (let* ((bid (gensym 'bid-))
             (b   (make-ssa-binding bid op inputs shape dtype meta)))
        (stack-push! bindings-stack b)  ; O(1); reversed at end
        (binding-ref bid)))

    (define (visit m)
      (or (env-builder-lookup visited-eb (morph-stable-id m))
          (cases array-morphism m

            (concrete-array (data shape strides offset dtype alloc-id batch-axis)
              (let* ((cid (gensym 'cid-))
                     (v   (const-ref cid)))
                (hash-table-set! constants cid m)
                (env-builder-extend! visited-eb (morph-stable-id m) v)
                v))

            (morphism-expr (morph-id op operands index-fn shape dtype metadata batch-axis)
              (let* ((input-vals (map visit operands))
                     ;; Normalize: morph-transpose stores key 'permutation; we use 'perm.
                     ;; morph-map stores its procedure under 'function; the map
                     ;; bindings of ssa-vjp, and rebuild-morphism, use 'fn.
                     (norm-meta  (case op
                                   ((transpose)
                                    (let ((pe (assq 'permutation metadata)))
                                      (if pe
                                          (cons (cons 'perm (cdr pe))
                                                (filter (lambda (x)
                                                          (not (eq? (car x) 'permutation)))
                                                        metadata))
                                          metadata)))
                                   ((map)
                                    (let ((fe (assq 'function metadata)))
                                      (if (and fe (not (assq 'fn metadata)))
                                          (cons (cons 'fn (cdr fe)) metadata)
                                          metadata)))
                                   (else metadata)))
                     (meta       (cons (cons 'index-fn index-fn) norm-meta))
                     (v          (emit-binding! op input-vals shape dtype meta)))
                (env-builder-extend! visited-eb (morph-stable-id m) v)
                v))

            (reduction-morphism (morph-id rop operand reduce-axes index-fn shape dtype batch-axis)
              (let* ((src-val (visit operand))
                     (meta    (list (cons 'axes reduce-axes)
                                    (cons 'index-fn index-fn)
                                    (cons 'src-shape (morph-shape operand))
                                    (cons 'keepdims? (reduction-index-fn-keepdims? index-fn))))
                     (v       (emit-binding! (list 'reduce rop) (list src-val) shape dtype meta)))
                (env-builder-extend! visited-eb (morph-stable-id m) v)
                v)))))

    (let* ((loss-m   (am:var-value loss-mv))
           (loss-val (visit loss-m)))
      (make-ssa-program constants (env-builder-env visited-eb)
                        (reverse (stack->list bindings-stack)) (list loss-val) 0 #f #f #f))))


;;; ============================================================
;;; Accessors into a compiled program
;;; ============================================================

(define (ssa-constant-id prog m)
  "Return (const-ref cid) for morphism m if it is a constant in prog, else #f."
  (let ((v (morph-env-lookup (ssa-program-morph-to-val prog) (morph-stable-id m))))
    (and v (ssa-const-ref? v) v)))

(define (ssa-loss-binding-val prog)
  "Return the ssa-value for the loss node (first output)."
  (car (ssa-program-outputs prog)))


;;; ============================================================
;;; Phase 2: ssa-vjp
;;;
;;; Symbolic reverse-mode AD over SSA bindings.
;;; Appends backward bindings to fwd-prog's bindings and returns an
;;; extended ssa-program whose outputs are:
;;;   (loss-binding-val . param-grad-binding-vals)
;;; ============================================================

(define (ssa-vjp fwd-prog param-const-vals loss-binding-val)
  "Compute symbolic VJP of fwd-prog with respect to params.
   param-const-vals: list of (const-ref cid) ssa-values for trainable params.
   loss-binding-val: (binding-ref bid) ssa-value for the loss node.
   Returns extended ssa-program.  fwd-prog must not contain fused
   bindings, which have no derivative rule: a forward program realized
   through a context is fused at its first trace, so differentiate it
   before realizing it, or build it again with morphism-to-ssa."

  (when (any fused-binding? (ssa-program-bindings fwd-prog))
    (error "ssa-vjp: the forward program contains fused bindings"))

  (let* (;; Build shape/dtype lookup from forward bindings (single env-builder)
         (fwd-bindings  (ssa-program-bindings fwd-prog))
         (binding-sd-eb (make-env-builder)) ; sym -> (shape . dtype)
         ;; Accumulate backward bindings newest-first; reverse at end
         (bwd-bindings-stack (make-stack))
         ;; Adjoint table: bid-symbol -> ssa-value (the accumulated dL/d(bid))
         (adjoint-eb    (make-env-builder))
         ;; Param gradient table: cid-symbol -> ssa-value
         (param-grad-eb (make-env-builder))
         ;; Set of trainable param cid-symbols for fast membership test
         (param-cid-eb  (make-env-builder))
         ;; Loss shape/dtype
         (loss-bid-sym  (ssa-value-id loss-binding-val)))

    ;; Index forward binding shapes and dtypes
    (for-each (lambda (b)
                (env-builder-extend! binding-sd-eb
                                     (ssa-binding-name b)
                                     (cons (ssa-binding-shape b) (ssa-binding-dtype b))))
              fwd-bindings)

    ;; Also index constant shapes/dtypes
    (hash-table-walk (ssa-program-constants fwd-prog)
      (lambda (cid m)
        (env-builder-extend! binding-sd-eb cid (cons (morph-shape m) (morph-dtype m)))))

    ;; Register trainable param cid-symbols
    (for-each (lambda (v)
                (env-builder-extend! param-cid-eb (ssa-value-id v) #t))
              param-const-vals)

    ;; Forward bindings whose value depends on a trainable parameter
    (define param-dependent-eb (make-env-builder))
    (define (param-dependent? v)
      (cases ssa-value v
        (binding-ref (bid) (and (env-builder-lookup param-dependent-eb bid) #t))
        (const-ref (cid) (and (env-builder-lookup param-cid-eb cid) #t))))
    (for-each (lambda (b)
                (when (any param-dependent? (ssa-binding-inputs b))
                  (env-builder-extend! param-dependent-eb (ssa-binding-name b) #t)))
              fwd-bindings)

    ;; Lookup helpers
    (define (val-shape v)
      (let ((sd (env-builder-lookup binding-sd-eb (ssa-value-id v))))
        (and sd (car sd))))
    (define (val-dtype v)
      (let ((sd (env-builder-lookup binding-sd-eb (ssa-value-id v))))
        (and sd (cdr sd))))

    ;; The dtype of the morphism rebuild-morphism makes for a binding with
    ;; op and inputs, by the type rules of the morphism constructors:
    ;; arithmetic promotes its operand dtypes, the transcendental ops and
    ;; activations make integers f64, reductions follow
    ;; infer-reduction-dtype, and structural ops keep their input's dtype.
    ;; Ops not covered here (cast) keep the dtype the caller gives.
    (define (emitted-dtype op inputs dtype)
      (let ((in-dtypes (map val-dtype inputs)))
        (cond
         ((not (every values in-dtypes)) dtype)
         ((memq op '(add sub mul div pow matmul conv2d-bwd-data conv2d-bwd-weights))
          (fold promote-types (car in-dtypes) (cdr in-dtypes)))
         ((and (pair? op) (eq? (car op) 'reduce))
          (infer-reduction-dtype (cadr op) (car in-dtypes)))
         ((memq op '(reshape transpose col2im map relu-deriv sigmoid-deriv tanh-deriv))
          (car in-dtypes))
         ((memq op '(negate abs floor ceiling sqrt exp log sin cos tan relu sigmoid tanh))
          (infer-unary-dtype op (car in-dtypes)))
         (else dtype))))

    ;; emit! -- create a new backward SSA binding (O(1) push; reversed at end).
    ;; The binding's dtype is the one the trace computes for it (see
    ;; emitted-dtype), whatever dtype the caller passes, so that replay
    ;; writes each pool buffer in the dtype the trace allocated.
    (define (emit! op inputs shape dtype-hint meta)
      (let* ((adj-id (gensym 'adj-))
             (dtype  (emitted-dtype op inputs dtype-hint))
             (b      (make-ssa-binding adj-id op inputs shape dtype meta)))
        (stack-push! bwd-bindings-stack b)
        (env-builder-extend! binding-sd-eb adj-id (cons shape dtype))
        (binding-ref adj-id)))

    ;; Accumulate adjoint for a binding-ref input
    (define (accumulate-adjoint! bid-sym new-val shape dtype)
      (let ((existing (env-builder-lookup adjoint-eb bid-sym)))
        (if existing
            (let ((sum (emit! 'add (list existing new-val) shape dtype '())))
              (env-builder-extend! adjoint-eb bid-sym sum))
            (env-builder-extend! adjoint-eb bid-sym new-val))))

    ;; Accumulate gradient for a const-ref param input
    (define (accumulate-param-grad! cid-sym new-val shape dtype)
      (let ((existing (env-builder-lookup param-grad-eb cid-sym)))
        (if existing
            (let ((sum (emit! 'add (list existing new-val) shape dtype '())))
              (env-builder-extend! param-grad-eb cid-sym sum))
            (env-builder-extend! param-grad-eb cid-sym new-val))))

    ;; Dispatch adjoint accumulation based on ssa-value type
    (define (accumulate-input-adjoint! input-val new-val)
      (let ((shape (val-shape input-val))
            (dtype (val-dtype input-val)))
        (cases ssa-value input-val
          (binding-ref (bid)
            (accumulate-adjoint! bid new-val shape dtype))
          (const-ref (cid)
            (when (env-builder-lookup param-cid-eb cid)
              (accumulate-param-grad! cid new-val shape dtype))))))

    ;; emit-reduce-sum-to! -- reduce g-val to target-shape
    (define (emit-reduce-sum-to! g-val g-shape target-shape dtype)
      (let* ((g-rank (vector-length g-shape))
             (t-vec  (if (vector? target-shape)
                         target-shape
                         (list->vector target-shape)))
             (t-rank (vector-length t-vec)))
        ;; Step 1: sum leading extra dims
        (let* ((extra (- g-rank t-rank))
               (cur   (if (> extra 0)
                          (emit! (list 'reduce 'sum) (list g-val)
                                 (vector-drop-left g-shape extra)
                                 dtype
                                 (list (cons 'axes (iota extra))
                                       (cons 'keepdims? #f)
                                       (cons 'src-shape g-shape)))
                          g-val))
               (cur-shape (val-shape cur)))
          ;; Step 2: sum broadcast (size-1) dims
          (let loop ((k 0) (cur cur) (cur-shape cur-shape))
            (if (>= k t-rank)
                cur
                (let ((t-dim (vector-ref t-vec k))
                      (g-dim (vector-ref cur-shape k)))
                  (if (and (= t-dim 1) (> g-dim 1))
                      (let* ((new-shape (let* ((len (vector-length cur-shape))
                                              (nv  (make-vector len)))
                                         (do ((i 0 (+ i 1))) ((= i len) nv)
                                           (vector-set! nv i (vector-ref cur-shape i)))))
                             (_ (vector-set! new-shape k 1))
                             (r (emit! (list 'reduce 'sum) (list cur)
                                       new-shape dtype
                                       (list (cons 'axes (list k))
                                             (cons 'keepdims? #t)
                                             (cons 'src-shape cur-shape)))))
                        (loop (+ k 1) r new-shape))
                      (loop (+ k 1) cur cur-shape))))))))

    ;; vector-drop-left -- remove first n elements of a vector
    (define (vector-drop-left v n)
      (let* ((len (vector-length v))
             (new-len (- len n))
             (result (make-vector new-len)))
        (do ((i 0 (+ i 1)))
            ((= i new-len) result)
          (vector-set! result i (vector-ref v (+ i n))))))

    ;; emit-broadcast-grad! -- broadcast reduced gradient back to src-shape
    ;; src-shape: original input shape; reduced-axes: axes that were reduced
    ;; keepdims?: whether the reduction kept dimensions
    (define (emit-broadcast-grad! g-val g-shape src-shape reduced-axes keepdims? dtype)
      (let* ((in-rank (vector-length src-shape))
             ;; Compute keepdims shape (insert 1s at reduced positions)
             (kd-shape (list->vector
                        (map (lambda (i)
                               (if (member i reduced-axes) 1
                                   (vector-ref src-shape i)))
                             (iota in-rank))))
             ;; Reshape g to keepdims shape if needed
             (g-kd (if keepdims?
                       g-val
                       (emit! 'reshape (list g-val) kd-shape dtype '())))
             ;; Emit ones-like constant for src-shape
             (ones-data (allocate-typed-vector dtype (shape-size src-shape)))
             (_ (let loop ((i 0) (n (shape-size src-shape)))
                  (when (< i n)
                    (typed-vector-set! ones-data dtype i 1.0)
                    (loop (+ i 1) n))))
             (ones-const (make-morphism ones-data (vector->list src-shape) dtype))
             (ones-cid   (gensym 'cid-))
             (_ (hash-table-set! (ssa-program-constants fwd-prog) ones-cid ones-const))
             (_ (env-builder-extend! binding-sd-eb ones-cid (cons src-shape dtype)))
             (ones-val (const-ref ones-cid)))
        ;; morph* broadcasts: g-kd * ones -> src-shape
        (emit! 'mul (list g-kd ones-val) src-shape dtype `((combiner . ,*)))))

    ;; emit-scalar-const! -- emit a scalar constant of given value
    (define (emit-scalar-const! value dtype)
      (let* ((data (allocate-typed-vector dtype 1))
             (_ (typed-vector-set! data dtype 0 value))
             (m   (make-morphism data '(1) dtype))
             (cid (gensym 'cid-)))
        (hash-table-set! (ssa-program-constants fwd-prog) cid m)
        (env-builder-extend! binding-sd-eb cid (cons (vector 1) dtype))
        (const-ref cid)))

    ;; Seed: dL/dL = ones-like(loss) stored as a constant
    (let* ((loss-bid    (find-fwd-binding fwd-bindings loss-bid-sym))
           (loss-shape  (ssa-binding-shape loss-bid))
           (loss-dtype  (ssa-binding-dtype loss-bid))
           (seed-data   (allocate-typed-vector loss-dtype (shape-size loss-shape)))
           (_ (let loop ((i 0) (n (shape-size loss-shape)))
                (when (< i n)
                  (typed-vector-set! seed-data loss-dtype i 1.0)
                  (loop (+ i 1) n))))
           (seed-const  (make-morphism seed-data (vector->list loss-shape) loss-dtype))
           (seed-cid    (gensym 'cid-))
           (_ (hash-table-set! (ssa-program-constants fwd-prog) seed-cid seed-const))
           (_ (env-builder-extend! binding-sd-eb seed-cid (cons loss-shape loss-dtype)))
           (seed-val    (const-ref seed-cid)))
      (env-builder-extend! adjoint-eb loss-bid-sym seed-val))

    ;; Backward sweep: iterate bindings in reverse order
    (for-each
     (lambda (b)
       (let* ((bid-sym  (ssa-binding-name b))
              (g-val    (env-builder-lookup adjoint-eb bid-sym)))
         (when g-val
           (let* ((op      (ssa-binding-op b))
                  (inputs  (ssa-binding-inputs b))
                  (shape   (ssa-binding-shape b))
                  (dtype   (ssa-binding-dtype b))
                  (meta    (ssa-binding-meta b))
                  (g-shape (val-shape g-val)))

             (cond

               ;; add(x, y) -- dx = reduce-sum-to(g, shape-x); dy = reduce-sum-to(g, shape-y)
               ((eq? op 'add)
                (let* ((x-val (list-ref inputs 0))
                       (y-val (list-ref inputs 1))
                       (x-shape (val-shape x-val))
                       (y-shape (val-shape y-val)))
                  (let ((dx (emit-reduce-sum-to! g-val g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))
                  (let ((dy (emit-reduce-sum-to! g-val g-shape y-shape dtype)))
                    (accumulate-input-adjoint! y-val dy))))

               ;; sub(x, y) -- dx = reduce-sum-to(g, shape-x); dy = reduce-sum-to(-g, shape-y)
               ((eq? op 'sub)
                (let* ((x-val (list-ref inputs 0))
                       (y-val (list-ref inputs 1))
                       (x-shape (val-shape x-val))
                       (y-shape (val-shape y-val)))
                  (let ((dx (emit-reduce-sum-to! g-val g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))
                  (let* ((neg-g (emit! 'negate (list g-val) g-shape dtype
                                        `((combiner . ,(scalar-op-combiner 'negate)))))
                         (dy    (emit-reduce-sum-to! neg-g g-shape y-shape dtype)))
                    (accumulate-input-adjoint! y-val dy))))

               ;; mul(x, y) -- dx = reduce-sum-to(g*y, shape-x); dy = reduce-sum-to(g*x, shape-y)
               ((eq? op 'mul)
                (let* ((x-val (list-ref inputs 0))
                       (y-val (list-ref inputs 1))
                       (x-shape (val-shape x-val))
                       (y-shape (val-shape y-val)))
                  (let* ((gy    (emit! 'mul (list g-val y-val) g-shape dtype
                                        `((combiner . ,*))))
                         (dx    (emit-reduce-sum-to! gy g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))
                  (let* ((gx    (emit! 'mul (list g-val x-val) g-shape dtype
                                        `((combiner . ,*))))
                         (dy    (emit-reduce-sum-to! gx g-shape y-shape dtype)))
                    (accumulate-input-adjoint! y-val dy))))

               ;; div(x, y) -- dx = g/y; dy = -g*x/y^2
               ((eq? op 'div)
                (let* ((x-val (list-ref inputs 0))
                       (y-val (list-ref inputs 1))
                       (x-shape (val-shape x-val))
                       (y-shape (val-shape y-val)))
                  (let* ((g-over-y (emit! 'div (list g-val y-val) g-shape dtype
                                           `((combiner . ,/))))
                         (dx       (emit-reduce-sum-to! g-over-y g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))
                  (let* ((y2      (emit! 'mul (list y-val y-val) y-shape dtype
                                          `((combiner . ,*))))
                         (x-over-y2 (emit! 'div (list x-val y2) g-shape dtype
                                            `((combiner . ,/))))
                         (neg-dy  (emit! 'mul (list g-val x-over-y2) g-shape dtype
                                          `((combiner . ,*))))
                         (neg-dy2 (emit! 'negate (list neg-dy) g-shape dtype
                                          `((combiner . ,(scalar-op-combiner 'negate)))))
                         (dy      (emit-reduce-sum-to! neg-dy2 g-shape y-shape dtype)))
                    (accumulate-input-adjoint! y-val dy))))

               ;; pow(x, n) -- dx = g * n * x^(n-1); dn skipped (n usually not a param)
               ((eq? op 'pow)
                (let* ((x-val (list-ref inputs 0))
                       (n-val (list-ref inputs 1))
                       (x-shape (val-shape x-val)))
                  ;; n_minus_1 = n - 1  (using scalar 1 constant)
                  (let* ((one-val (emit-scalar-const! 1.0 dtype))
                         (nm1     (emit! 'sub (list n-val one-val)
                                         (val-shape n-val) dtype
                                         `((combiner . ,(scalar-op-combiner 'sub)))))
                         (xpow    (emit! 'pow (list x-val nm1) g-shape dtype
                                          `((combiner . ,expt))))
                         (n-xpow  (emit! 'mul (list n-val xpow) g-shape dtype
                                          `((combiner . ,*))))
                         (dx-raw  (emit! 'mul (list g-val n-xpow) g-shape dtype
                                          `((combiner . ,*))))
                         (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                    (accumulate-input-adjoint! x-val dx))))

               ;; negate(x) -- dx = -g
               ((eq? op 'negate)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (neg-g   (emit! 'negate (list g-val) g-shape dtype
                                        `((combiner . ,(scalar-op-combiner 'negate)))))
                       (dx      (emit-reduce-sum-to! neg-g g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; abs(x) -- dx = g * sign(x)
               ((eq? op 'abs)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (sign-fn (lambda (xv)
                                  (cond ((> xv 0.0)  1.0)
                                        ((< xv 0.0) -1.0)
                                        (else         0.0))))
                       (sign-x  (emit! 'map (list x-val) x-shape dtype
                                       (list (cons 'fn sign-fn))))
                       (dx-raw  (emit! 'mul (list g-val sign-x) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; sqrt(x) -- dx = g / (2 * sqrt(x))  i.e. g / (2 * fwd-output)
               ;; fwd-output is (binding-ref bid-sym) at this point in execution
               ((eq? op 'sqrt)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (two-val (emit-scalar-const! 2.0 dtype))
                       ;; fwd output = this binding
                       (fwd-out (binding-ref bid-sym))
                       (two-out (emit! 'mul (list two-val fwd-out) g-shape dtype
                                        `((combiner . ,*))))
                       (dx-raw  (emit! 'div (list g-val two-out) g-shape dtype
                                        `((combiner . ,/))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; exp(x) -- dx = g * exp(x) = g * fwd-output
               ((eq? op 'exp)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (fwd-out (binding-ref bid-sym))
                       (dx-raw  (emit! 'mul (list g-val fwd-out) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; log(x) -- dx = g / x
               ((eq? op 'log)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (dx-raw  (emit! 'div (list g-val x-val) g-shape dtype
                                        `((combiner . ,/))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; sin(x) -- dx = g * cos(x)
               ((eq? op 'sin)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (cos-x   (emit! 'cos (list x-val) x-shape dtype
                                        `((combiner . ,cos))))
                       (dx-raw  (emit! 'mul (list g-val cos-x) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; cos(x) -- dx = -g * sin(x)
               ((eq? op 'cos)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (sin-x   (emit! 'sin (list x-val) x-shape dtype
                                        `((combiner . ,sin))))
                       (neg-sin (emit! 'negate (list sin-x) x-shape dtype
                                        `((combiner . ,(scalar-op-combiner 'negate)))))
                       (dx-raw  (emit! 'mul (list g-val neg-sin) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; relu(x) -- dx = g * heaviside(x)
               ;; Decomposed as: hx = relu-deriv(x); dx = mul(g, hx)
               ;; (abs-style decomposition keeps rebuild-morphism simple)
               ((eq? op 'relu)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (hx      (emit! 'relu-deriv (list x-val) x-shape dtype
                                        `((fn . ,(scalar-op-combiner 'relu-deriv))
                                          (combiner . ,(scalar-op-combiner 'relu-deriv)))))
                       (dx-raw  (emit! 'mul (list g-val hx) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; sigmoid(x) -- dx = g * deriv where deriv = s*(1-s), s = forward output
               ((eq? op 'sigmoid)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (fwd-out (binding-ref bid-sym))
                       (deriv   (emit! 'sigmoid-deriv (list fwd-out) g-shape dtype
                                        `((fn . ,(scalar-op-combiner 'sigmoid-deriv))
                                          (combiner . ,(scalar-op-combiner 'sigmoid-deriv)))))
                       (dx-raw  (emit! 'mul (list g-val deriv) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; tanh(x) -- dx = g * (1-t^2) where t = tanh(x) = forward output
               ((eq? op 'tanh)
                (let* ((x-val   (list-ref inputs 0))
                       (x-shape (val-shape x-val))
                       (fwd-out (binding-ref bid-sym))
                       (deriv   (emit! 'tanh-deriv (list fwd-out) g-shape dtype
                                        `((fn . ,(scalar-op-combiner 'tanh-deriv))
                                          (combiner . ,(scalar-op-combiner 'tanh-deriv)))))
                       (dx-raw  (emit! 'mul (list g-val deriv) g-shape dtype
                                        `((combiner . ,*))))
                       (dx      (emit-reduce-sum-to! dx-raw g-shape x-shape dtype)))
                  (accumulate-input-adjoint! x-val dx)))

               ;; cast(x) -- dx = g converted back to the dtype of x
               ((eq? op 'cast)
                (let* ((x-val   (list-ref inputs 0))
                       (x-dtype (val-dtype x-val))
                       (dx      (if (eq? (val-dtype g-val) x-dtype)
                                    g-val
                                    (emit! 'cast (list g-val) (val-shape x-val) x-dtype
                                           `((dtype . ,x-dtype)
                                             (combiner . ,(cast-combiner x-dtype)))))))
                  (accumulate-input-adjoint! x-val dx)))

               ;; matmul(A, B) -- dA = g @ B^T; dB = A^T @ g
               ((eq? op 'matmul)
                (let* ((a-val    (list-ref inputs 0))
                       (b-val    (list-ref inputs 1))
                       (a-shape  (val-shape a-val))
                       (b-shape  (val-shape b-val))
                       ;; Determine 2D shapes; shape = [M K] [K N] -> [M N]
                       (rank     (vector-length a-shape))
                       (perm-t   (transpose-perm rank))
                       (b-t-shape (permute-shape b-shape perm-t))
                       (a-t-shape (permute-shape a-shape perm-t))
                       (bt-val  (emit! 'transpose (list b-val) b-t-shape dtype
                                       (list (cons 'perm perm-t))))
                       (at-val  (emit! 'transpose (list a-val) a-t-shape dtype
                                       (list (cons 'perm perm-t))))
                       (da-val  (emit! 'matmul (list g-val bt-val) a-shape dtype '()))
                       (db-val  (emit! 'matmul (list at-val g-val) b-shape dtype '())))
                  (accumulate-input-adjoint! a-val da-val)
                  (accumulate-input-adjoint! b-val db-val)))

               ;; transpose(x, perm) -- dx = transpose(g, inv-perm)
               ((eq? op 'transpose)
                (let* ((x-val    (list-ref inputs 0))
                       (x-shape  (val-shape x-val))
                       (perm     (cdr (assq 'perm meta)))
                       (inv-perm (invert-permutation perm))
                       (dx       (emit! 'transpose (list g-val) x-shape dtype
                                        (list (cons 'perm inv-perm)))))
                  (accumulate-input-adjoint! x-val dx)))

                ;; reshape(x) -- dx = reshape(g, original-shape)
                ((eq? op 'reshape)
                 (let* ((x-val   (list-ref inputs 0))
                        (x-shape (val-shape x-val))
                        (dx      (emit! 'reshape (list g-val) x-shape dtype '())))
                   (accumulate-input-adjoint! x-val dx)))

                ;; im2col(x) -- dx = col2im(g, shape(x), kernel, stride, padding)
                ((eq? op 'im2col)
                 (let* ((x-val       (list-ref inputs 0))
                        (x-shape     (val-shape x-val))
                        (kernel-size (cdr (assq 'kernel-size meta)))
                        (stride      (cdr (assq 'stride meta)))
                        (padding     (cdr (assq 'padding meta)))
                        ;; Propagate layout (nhwc) so col2im backward uses correct kernel
                        (layout-entry (assq 'layout meta))
                        (vjp-meta    (append
                                       (list (cons 'output-shape x-shape)
                                             (cons 'kernel-size kernel-size)
                                             (cons 'stride stride)
                                             (cons 'padding padding))
                                       (if layout-entry (list layout-entry) '())))
                        (dx          (emit! 'col2im (list g-val) x-shape dtype vjp-meta)))
                   (accumulate-input-adjoint! x-val dx)))

                ;; conv2d-fwd(x, wt, b) -- emit conv2d-bwd-data, conv2d-bwd-weights, reduce-sum for db
                ((eq? op 'conv2d-fwd)
                 (let* ((x-val    (list-ref inputs 0))
                        (wt-val   (list-ref inputs 1))
                        (b-val    (list-ref inputs 2))
                        (x-shape  (val-shape x-val))
                        (wt-shape (val-shape wt-val))
                        (b-shape  (val-shape b-val))
                        (bwd-meta (filter (lambda (kv)
                                            (memq (car kv)
                                                  '(kernel-size stride padding spatial-output layout)))
                                          meta))
                        (dx  (emit! 'conv2d-bwd-data
                                    (list g-val wt-val) x-shape dtype
                                    (cons (cons 'output-shape x-shape) bwd-meta)))
                        (dwt (emit! 'conv2d-bwd-weights
                                    (list g-val x-val) wt-shape dtype bwd-meta))
                        (db  (emit! '(reduce sum)
                                    (list g-val) b-shape dtype
                                    `((axes . (0)) (keepdims? . #f) (src-shape . ,g-shape)))))
                   (accumulate-input-adjoint! x-val  dx)
                   (accumulate-input-adjoint! wt-val dwt)
                   (accumulate-input-adjoint! b-val  db)))

                ;; (reduce mean) -- dx = broadcast(g / n, src-shape)
               ((equal? op '(reduce mean))
                (let* ((src-val    (list-ref inputs 0))
                       (src-shape  (cdr (assq 'src-shape meta)))
                       (axes       (cdr (assq 'axes meta)))
                       (keepdims?  (cdr (assq 'keepdims? meta)))
                       (n          (apply * (map (lambda (a) (vector-ref src-shape a)) axes)))
                       (n-val      (emit-scalar-const! (exact->inexact n) dtype))
                       (g-over-n   (emit! 'div (list g-val n-val) g-shape dtype '()))
                       (dx         (emit-broadcast-grad! g-over-n g-shape src-shape
                                                         axes keepdims? dtype)))
                  (accumulate-input-adjoint! src-val dx)))

               ;; (reduce sum) -- dx = broadcast(g, src-shape)
               ((equal? op '(reduce sum))
                (let* ((src-val   (list-ref inputs 0))
                       (src-shape (cdr (assq 'src-shape meta)))
                       (axes      (cdr (assq 'axes meta)))
                       (keepdims? (cdr (assq 'keepdims? meta)))
                       (dx        (emit-broadcast-grad! g-val g-shape src-shape
                                                        axes keepdims? dtype)))
                  (accumulate-input-adjoint! src-val dx)))

               ;; (reduce max) -- not needed for standard MLP; skip
               ((equal? op '(reduce max))
                (void))

               ;; map(x, fn) -- a user procedure has no derivative.  A map whose
               ;; input does not depend on a trainable parameter passes no
               ;; gradient on and needs none; otherwise the gradient cannot be
               ;; computed.
               ((eq? op 'map)
                (when (param-dependent? (car inputs))
                  (error "ssa-vjp: morph-map has no derivative, and its input depends on a trainable parameter"
                         bid-sym)))

               ;; The activation derivative ops are maps emitted only in the
               ;; backward pass, and have no gradient of their own.
               ((memq op '(relu-deriv sigmoid-deriv tanh-deriv))
                (void))

               (else
                (void)))))))
     (reverse fwd-bindings))

    ;; Emit add(loss, zero-const) as the final backward binding.
    ;;
    ;; The context's compute-last-uses! extends a buffer's lifetime only when
    ;; it appears as an INPUT to a later allocation-rec.  The loss forward binding
    ;; has no backward binding that references it directly (the seed is stored as
    ;; a pre-computed constant), so loss.alloc-id's last-use stays at its birth
    ;; step and backward bindings can reuse its buffer slot in replay mode.
    ;;
    ;; add(loss, zero) is a two-operand op -- can-zero-copy? returns #f so a real
    ;; allocation is recorded.  input-ids includes loss.alloc-id, which forces
    ;; compute-last-uses! to extend loss's lifetime to this final step, preventing
    ;; any backward binding from aliasing its buffer.
    (let* ((loss-bid-b  (find-fwd-binding fwd-bindings loss-bid-sym))
           (loss-shape  (ssa-binding-shape loss-bid-b))
           (loss-dtype  (ssa-binding-dtype loss-bid-b))
           ;; Zero constant with same shape as loss (alloc-id=-1, not context-tracked)
           (zero-data   (allocate-typed-vector loss-dtype (shape-size loss-shape)))
           (_ (let loop ((i 0) (n (shape-size loss-shape)))
                (when (< i n)
                  (typed-vector-set! zero-data loss-dtype i 0.0)
                  (loop (+ i 1) n))))
           (zero-const  (make-morphism zero-data (vector->list loss-shape) loss-dtype))
           (zero-cid    (gensym 'cid-))
           (_ (hash-table-set! (ssa-program-constants fwd-prog) zero-cid zero-const))
           (_ (env-builder-extend! binding-sd-eb zero-cid (cons loss-shape loss-dtype)))
           (zero-val    (const-ref zero-cid))
           (loss-out-val (emit! 'add (list loss-binding-val zero-val) loss-shape loss-dtype
                                      `((combiner . ,+))))
           ;; Collect param grad outputs in order of param-const-vals, one per
           ;; parameter.  A gradient computed in another dtype than its
           ;; parameter's (as when an f32 parameter meets f64 values) is cast
           ;; to the parameter's dtype.  A parameter no gradient reaches (as
           ;; through a max reduction, whose VJP is skipped) gets zeros,
           ;; computed into a fresh buffer by an add of a zero constant.
           (zero-grad   (lambda (pcv)
                          (let* ((shape (val-shape pcv))
                                 (dtype (val-dtype pcv))
                                 (n     (shape-size shape))
                                 (data  (allocate-typed-vector dtype n))
                                 (zero  (if (memq dtype '(f32 f64)) 0.0 0))
                                 (_     (do ((i 0 (+ i 1))) ((= i n))
                                          (typed-vector-set! data dtype i zero)))
                                 (cid   (gensym 'cid-)))
                            (hash-table-set! (ssa-program-constants fwd-prog) cid
                                             (make-morphism data (vector->list shape) dtype))
                            (env-builder-extend! binding-sd-eb cid (cons shape dtype))
                            (emit! 'add (list (const-ref cid) (const-ref cid)) shape dtype '()))))
           (grad-vals   (map
                         (lambda (pcv)
                           (let* ((cid-sym (ssa-value-id pcv))
                                  (g       (env-builder-lookup param-grad-eb cid-sym))
                                  (p-dtype (val-dtype pcv)))
                             (cond ((not g) (zero-grad pcv))
                                   ((not (eq? (val-dtype g) p-dtype))
                                    (emit! 'cast (list g) (val-shape pcv) p-dtype
                                           `((dtype . ,p-dtype)
                                             (combiner . ,(cast-combiner p-dtype)))))
                                   (else g))))
                         param-const-vals))
           (all-outputs  (cons loss-out-val grad-vals))
           (all-bindings (append fwd-bindings (reverse (stack->list bwd-bindings-stack)))))
      ;; Fuse the joint forward+backward program.  Fusion crosses the AD
      ;; boundary: backward element-wise bindings are fused under the same
      ;; rules as forward ones.
      ((if (segop-fusion) ssa-segop-fusion-pass (lambda (p) p))
       (make-ssa-program
        (ssa-program-constants fwd-prog)
        (ssa-program-morph-to-val fwd-prog)
        all-bindings
        all-outputs
        (length param-const-vals)
        #f #f #f #t)))))


;;; Helper: find a forward binding by its bid-symbol
(define (find-fwd-binding bindings bid-sym)
  (let loop ((bs bindings))
    (cond
      ((null? bs) (error "ssa-vjp: loss binding not found" bid-sym))
      ((eq? (ssa-binding-name (car bs)) bid-sym) (car bs))
      (else (loop (cdr bs))))))


;;; Helper: build the transpose permutation for a rank-r tensor
;;; For rank 2: (1 0).  For rank 3: (0 2 1) (transpose last two axes).
(define (transpose-perm rank)
  (if (= rank 2)
      '(1 0)
      (let ((base (iota (- rank 2))))
        (append base (list (- rank 1) (- rank 2))))))


;;; Helper: permute shape vector according to perm list
(define (permute-shape shape perm)
  (list->vector (map (lambda (i) (vector-ref shape i)) perm)))


;;; ============================================================
;;; Phase 3: rebuild-morphism
;;;
;;; Given an ssa-binding and a list of already-realized concrete-array
;;; inputs, construct a lazy morphism to pass to realize.
;;; ============================================================

(define (rebuild-morphism b inputs)
  "Reconstruct a lazy morphism from an SSA binding and concrete inputs."
  (let ((op   (ssa-binding-op b))
        (meta (ssa-binding-meta b))
        (shp  (ssa-binding-shape b)))
    (cond
      ((equal? op '(reduce mean))
       (let ((axes     (cdr (assq 'axes meta)))
             (keepdims? (cdr (assq 'keepdims? meta))))
         (morph-reduce 'mean (car inputs) axes keepdims?)))
      ((equal? op '(reduce sum))
       (let ((axes     (cdr (assq 'axes meta)))
             (keepdims? (cdr (assq 'keepdims? meta))))
         (morph-reduce 'sum (car inputs) axes keepdims?)))
      ((equal? op '(reduce max))
       (let ((axes     (cdr (assq 'axes meta)))
             (keepdims? (cdr (assq 'keepdims? meta))))
         (morph-reduce 'max (car inputs) axes keepdims?)))
      ((equal? op '(reduce min))
       (let ((axes     (cdr (assq 'axes meta)))
             (keepdims? (cdr (assq 'keepdims? meta))))
         (morph-reduce 'min (car inputs) axes keepdims?)))
      ((equal? op '(reduce prod))
       (let ((axes     (cdr (assq 'axes meta)))
             (keepdims? (cdr (assq 'keepdims? meta))))
         (morph-reduce 'prod (car inputs) axes keepdims?)))
      ((eq? op 'matmul)
       (morph-matmul (car inputs) (cadr inputs)))
      ((eq? op 'transpose)
       (let ((perm (cdr (assq 'perm meta))))
         (morph-transpose (car inputs) perm)))
      ((eq? op 'reshape)
       (morph-reshape (car inputs) shp))
      ((eq? op 'add)    (morph+ (car inputs) (cadr inputs)))
      ((eq? op 'sub)    (morph- (car inputs) (cadr inputs)))
      ((eq? op 'mul)    (morph* (car inputs) (cadr inputs)))
      ((eq? op 'div)    (morph/ (car inputs) (cadr inputs)))
      ((eq? op 'pow)    (morph-pow (car inputs) (cadr inputs)))
      ((eq? op 'negate) (morph-negate (car inputs)))
      ((eq? op 'abs)    (morph-abs (car inputs)))
      ((eq? op 'sqrt)   (morph-sqrt (car inputs)))
      ((eq? op 'exp)    (morph-exp (car inputs)))
      ((eq? op 'log)    (morph-log (car inputs)))
      ((eq? op 'sin)    (morph-sin (car inputs)))
      ((eq? op 'cos)    (morph-cos (car inputs)))
      ((eq? op 'relu)    (morph-relu    (car inputs)))
      ((eq? op 'sigmoid) (morph-sigmoid (car inputs)))
      ((eq? op 'tanh)    (morph-tanh-am (car inputs)))
      ((eq? op 'cast)    (morph-cast (car inputs) (ssa-binding-dtype b)))
       ;; map, and the activation derivative ops, which ssa-vjp emits as maps
       ;; under their own op names so that replay can use activation kernels for them.
       ((memq op '(map relu-deriv sigmoid-deriv tanh-deriv))
        (let ((fn (cdr (assq 'fn meta))))
          (morph-map fn (car inputs))))

       ;; im2col: forward image-to-column
       ((eq? op 'im2col)
        (let* ((kernel-size  (cdr (assq 'kernel-size meta)))
               (stride       (cdr (assq 'stride meta)))
               (padding      (cdr (assq 'padding meta)))
               (layout-entry (assq 'layout meta))
               (layout       (if layout-entry (cdr layout-entry) 'nchw)))
          (im2col-morph (car inputs) kernel-size stride padding layout: layout)))

       ;; col2im: column-to-image (backward of im2col)
       ((eq? op 'col2im)
        (let* ((output-shape  (cdr (assq 'output-shape meta)))
               (kernel-size   (cdr (assq 'kernel-size meta)))
               (stride        (cdr (assq 'stride meta)))
               (padding       (cdr (assq 'padding meta)))
               (layout-entry  (assq 'layout meta))
               (layout        (if layout-entry (cdr layout-entry) 'nchw)))
          (col2im-morph (car inputs) output-shape kernel-size stride padding layout: layout)))

       ;; conv2d-fwd: reference decomposition for trace (im2col + reshape + matmul + bias)
       ((eq? op 'conv2d-fwd)
        (let* ((kernel-size (cdr (assq 'kernel-size meta)))
               (stride      (cdr (assq 'stride meta)))
               (padding     (cdr (assq 'padding meta)))
               (layout-entry (assq 'layout meta))
               (layout      (if layout-entry (cdr layout-entry) 'nchw))
               (x-m   (list-ref inputs 0))
               (wt-m  (list-ref inputs 1))
               (b-m   (list-ref inputs 2))
               (col   (im2col-morph x-m kernel-size stride padding layout: layout))
               (col-s (get-morphism-shape col))
               (N     (vector-ref col-s 0))
               (fi    (vector-ref col-s 1))
               (ohow  (vector-ref col-s 2))
               ;; col is [N, fan_in, OH_OW]; bring OH_OW next to N before
               ;; flattening to the matmul-ready [N*OH_OW, fan_in].
               (col-r (morph-reshape (morph-transpose col '(0 2 1)) (vector (* N ohow) fi)))
               (res   (morph-matmul col-r wt-m)))
          (morph+ res b-m)))

       ;; conv2d-bwd-data: reference: g @ WT^T -> col2im
       ((eq? op 'conv2d-bwd-data)
        (let* ((kernel-size (cdr (assq 'kernel-size meta)))
               (stride      (cdr (assq 'stride meta)))
               (padding     (cdr (assq 'padding meta)))
               (x-shape     (cdr (assq 'output-shape meta)))
               (layout-entry (assq 'layout meta))
               (layout      (if layout-entry (cdr layout-entry) 'nchw))
               (meta-so     (assq 'spatial-output meta))
               (g-m   (list-ref inputs 0))
               (wt-m  (list-ref inputs 1))
               ;; wt is [fan_in, out_ch]; transpose to [out_ch, fan_in]
               (wt-t  (morph-transpose wt-m '(1 0)))
               ;; g [N*OH_OW, out_ch] @ wt-t [out_ch, fan_in] = g-col [N*OH_OW, fan_in]
               (g-col (morph-matmul g-m wt-t))
               (N     (vector-ref x-shape 0))
               (fi    (vector-ref (get-morphism-shape wt-m) 0))
               (OH    (if meta-so (car  (cdr meta-so)) (error "conv2d-bwd-data rebuild: no spatial-output")))
               (OW    (if meta-so (cadr (cdr meta-so)) (error "conv2d-bwd-data rebuild: no spatial-output")))
               ;; g-col rows are ordered (n, oh_ow); split them into
               ;; [N, OH_OW, fan_in], then transpose to the classic
               ;; [N, fan_in, OH_OW] layout that col2im expects.
               (g-col-3d (morph-transpose (morph-reshape g-col (vector N (* OH OW) fi))
                                          '(0 2 1))))
          (col2im-morph g-col-3d x-shape kernel-size stride padding layout: layout)))

       ;; conv2d-bwd-weights: reference: col^T @ g
       ((eq? op 'conv2d-bwd-weights)
        (let* ((kernel-size (cdr (assq 'kernel-size meta)))
               (stride      (cdr (assq 'stride meta)))
               (padding     (cdr (assq 'padding meta)))
               (layout-entry (assq 'layout meta))
               (layout      (if layout-entry (cdr layout-entry) 'nchw))
               (g-m   (list-ref inputs 0))
               (x-m   (list-ref inputs 1))
               (col   (im2col-morph x-m kernel-size stride padding layout: layout))
               (col-s (get-morphism-shape col))
               (N     (vector-ref col-s 0))
               (fi    (vector-ref col-s 1))
               (ohow  (vector-ref col-s 2))
               ;; col is [N, fan_in, OH_OW]; transpose before flattening to
               ;; the matmul-ready [N*OH_OW, fan_in].
               (col-r (morph-reshape (morph-transpose col '(0 2 1)) (vector (* N ohow) fi)))
               ;; col-r [N*OH_OW, fan_in] -> col-t [fan_in, N*OH_OW]
               (col-t (morph-transpose col-r '(1 0))))
          ;; col-t [fan_in, N*OH_OW] @ g [N*OH_OW, out_ch] = dwt [fan_in, out_ch]
          (morph-matmul col-t g-m)))

       (else
        (error "rebuild-morphism: unknown op" op)))))


;;; ============================================================
;;; Phase 3: ssa-realize
;;;
;;; Sequential executor.  Runs every binding once, returns outputs
;;; as a list of concrete-arrays.
;;; ============================================================

(define (ssa-realize prog)
  "Execute the SSA program sequentially.
   Returns a list of concrete-arrays for each output ssa-value."
  (let ((values (make-hash-table)))
    ;; Pre-populate constants
    (hash-table-walk (ssa-program-constants prog)
      (lambda (k v) (hash-table-set! values k v)))
    ;; Execute bindings in topological order
    (for-each
     (lambda (b)
       (let ((inputs (map (lambda (v)
                            (hash-table-ref values (ssa-value-id v)))
                          (ssa-binding-inputs b))))
         (if (fused-binding? b)
             (for-each (lambda (o r) (hash-table-set! values (fused-out-name o) r))
                       (fused-binding-outs b) (realize-fused-binding #f b inputs))
             (hash-table-set! values (ssa-binding-name b)
                              (realize (rebuild-morphism b inputs))))))
     (ssa-program-bindings prog))
    ;; Collect outputs
    (map (lambda (ov)
           (hash-table-ref values (ssa-value-id ov)))
         (ssa-program-outputs prog))))


;;; ============================================================
;;; Phase 4: ssa-realize/ctx
;;;
;;; Like ssa-realize but routes each allocation through a
;;; morphism context for buffer pooling.
;;; ============================================================

;; True iff m is a concrete-array with row-major strides and zero offset.
;; Transposed zero-copy views have permuted strides and fail this check.
;; Used to distinguish pool-allocated row-major results (which can be pinned
;; and returned directly) from zero-copy transposed views (which must be copied).
(define (concrete-row-major? m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (and (= offset 0) (equal? strides (compute-strides shape))))
    (else #f)))

(define (concrete-alloc-id m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis) alloc-id)
    (else -1)))

;; De-transpose a concrete-array to a fresh row-major non-pooled buffer (alloc-id=-1).
;; Used for output bindings that are zero-copy transposed views (no pool slot of their own).

(define (%copy-concrete-array-slow data shape strides offset dtype new-data size)
  "Generic element-by-element copy using multi-index conversion (fallback)."
  (do ((i 0 (+ i 1)))
      ((= i size))
    (let* ((multi (linear-to-multi-index i shape))
           (phys  (multi-to-linear-index multi strides offset)))
      (typed-vector-set! new-data dtype i (typed-vector-ref data dtype phys)))))

(define (%copy-concrete-array-fast-f32 data shape strides offset new-data)
  "Fast copy for f32 arrays up to rank 4 using direct stride arithmetic."
  (let ((rank (vector-length shape)))
    (case rank
      ((1)
       (let ((d0 (vector-ref shape 0))
             (s0 (vector-ref strides 0)))
         (do ((i0 0 (+ i0 1))) ((= i0 d0))
           (f32vector-set! new-data i0
             (f32vector-ref data (+ offset (* i0 s0)))))))
      ((2)
       (let ((d0 (vector-ref shape 0)) (d1 (vector-ref shape 1))
             (s0 (vector-ref strides 0)) (s1 (vector-ref strides 1)))
         (let ((idx 0))
           (do ((i0 0 (+ i0 1))) ((= i0 d0))
             (do ((i1 0 (+ i1 1))) ((= i1 d1))
               (f32vector-set! new-data idx
                 (f32vector-ref data (+ offset (* i0 s0) (* i1 s1))))
               (set! idx (+ idx 1)))))))
      ((3)
       (let ((d0 (vector-ref shape 0)) (d1 (vector-ref shape 1)) (d2 (vector-ref shape 2))
             (s0 (vector-ref strides 0)) (s1 (vector-ref strides 1)) (s2 (vector-ref strides 2)))
         (let ((idx 0))
           (do ((i0 0 (+ i0 1))) ((= i0 d0))
             (do ((i1 0 (+ i1 1))) ((= i1 d1))
               (do ((i2 0 (+ i2 1))) ((= i2 d2))
                 (f32vector-set! new-data idx
                   (f32vector-ref data (+ offset (* i0 s0) (* i1 s1) (* i2 s2))))
                 (set! idx (+ idx 1))))))))
      ((4)
       (let ((d0 (vector-ref shape 0)) (d1 (vector-ref shape 1)) (d2 (vector-ref shape 2)) (d3 (vector-ref shape 3))
             (s0 (vector-ref strides 0)) (s1 (vector-ref strides 1)) (s2 (vector-ref strides 2)) (s3 (vector-ref strides 3)))
         (let ((idx 0))
           (do ((i0 0 (+ i0 1))) ((= i0 d0))
             (do ((i1 0 (+ i1 1))) ((= i1 d1))
               (do ((i2 0 (+ i2 1))) ((= i2 d2))
                 (do ((i3 0 (+ i3 1))) ((= i3 d3))
                   (f32vector-set! new-data idx
                     (f32vector-ref data (+ offset (* i0 s0) (* i1 s1) (* i2 s2) (* i3 s3))))
                   (set! idx (+ idx 1)))))))))
      (else (error "%copy-concrete-array-fast-f32: unsupported rank" rank)))))

(define (%copy-concrete-array-fast-f64 data shape strides offset new-data)
  "Fast copy for f64 arrays up to rank 4 using direct stride arithmetic."
  (let ((rank (vector-length shape)))
    (case rank
      ((1)
       (let ((d0 (vector-ref shape 0))
             (s0 (vector-ref strides 0)))
         (do ((i0 0 (+ i0 1))) ((= i0 d0))
           (f64vector-set! new-data i0
             (f64vector-ref data (+ offset (* i0 s0)))))))
      ((2)
       (let ((d0 (vector-ref shape 0)) (d1 (vector-ref shape 1))
             (s0 (vector-ref strides 0)) (s1 (vector-ref strides 1)))
         (let ((idx 0))
           (do ((i0 0 (+ i0 1))) ((= i0 d0))
             (do ((i1 0 (+ i1 1))) ((= i1 d1))
               (f64vector-set! new-data idx
                 (f64vector-ref data (+ offset (* i0 s0) (* i1 s1))))
               (set! idx (+ idx 1)))))))
      ((3)
       (let ((d0 (vector-ref shape 0)) (d1 (vector-ref shape 1)) (d2 (vector-ref shape 2))
             (s0 (vector-ref strides 0)) (s1 (vector-ref strides 1)) (s2 (vector-ref strides 2)))
         (let ((idx 0))
           (do ((i0 0 (+ i0 1))) ((= i0 d0))
             (do ((i1 0 (+ i1 1))) ((= i1 d1))
               (do ((i2 0 (+ i2 1))) ((= i2 d2))
                 (f64vector-set! new-data idx
                   (f64vector-ref data (+ offset (* i0 s0) (* i1 s1) (* i2 s2))))
                 (set! idx (+ idx 1))))))))
      ((4)
       (let ((d0 (vector-ref shape 0)) (d1 (vector-ref shape 1)) (d2 (vector-ref shape 2)) (d3 (vector-ref shape 3))
             (s0 (vector-ref strides 0)) (s1 (vector-ref strides 1)) (s2 (vector-ref strides 2)) (s3 (vector-ref strides 3)))
         (let ((idx 0))
           (do ((i0 0 (+ i0 1))) ((= i0 d0))
             (do ((i1 0 (+ i1 1))) ((= i1 d1))
               (do ((i2 0 (+ i2 1))) ((= i2 d2))
                 (do ((i3 0 (+ i3 1))) ((= i3 d3))
                   (f64vector-set! new-data idx
                     (f64vector-ref data (+ offset (* i0 s0) (* i1 s1) (* i2 s2) (* i3 s3))))
                   (set! idx (+ idx 1)))))))))
      (else (error "%copy-concrete-array-fast-f64: unsupported rank" rank)))))

(define (copy-concrete-array m)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (copy-concrete-array-into m (allocate-typed-vector dtype (shape-size shape))))
    (else (error "copy-concrete-array: not a concrete-array" m))))

;; Copy the elements of concrete array m, in row-major order, into the
;; typed vector new-data, which holds at least as many elements as m, and
;; return a row-major array over new-data (alloc-id -1).
(define (copy-concrete-array-into m new-data)
  (cases array-morphism m
    (concrete-array (data shape strides offset dtype alloc-id batch-axis)
      (let* ((size     (shape-size shape))
             (new-strs (compute-strides shape))
             (rank     (vector-length shape)))
         (cond
          ((and (<= rank 4) (lookup-copy-kernel (active-activation-backend) dtype))
           => (lambda (kernel) (kernel data offset shape strides new-data)))
          ((and (eq? dtype 'f32) (<= rank 4))
           (%copy-concrete-array-fast-f32 data shape strides offset new-data))
          ((and (eq? dtype 'f64) (<= rank 4))
           (%copy-concrete-array-fast-f64 data shape strides offset new-data))
          (else
           (%copy-concrete-array-slow data shape strides offset dtype new-data size)))
        (concrete-array new-data shape new-strs 0 dtype -1 batch-axis)))
    (else (error "copy-concrete-array-into: not a concrete-array" m))))

;;; ============================================================
;;; Replay Plan: compile-time classification helpers
;;;
;;; These predicates inspect trace-time concrete arrays.  They run once
;;; inside compile-one-instruction and never during replay.
;;; ============================================================

(define (trace-arr-shape m)
  (cases array-morphism m
    (concrete-array (_ shape _ _ _ _ _) shape)
    (else (error "trace-arr-shape: not concrete" m))))


;; The (row-step col-step) pair of an operand of a broadcast binary op
;; whose output has shape OUT-SHAPE, viewed as [M, N] with N its last
;; dimension; #f when the operand's shape is none of the supported forms.
(define (broadcast-steps out-shape in-shape)
  (let* ((rank (vector-length out-shape))
         (N    (vector-ref out-shape (- rank 1)))
         (in-rank (vector-length in-shape))
         (in-size (shape-size in-shape)))
    (cond
     ((equal? in-shape out-shape) (list N 1))
     ((= in-size 1) (list 0 0))
     ;; [..., 1] with the output's leading dimensions
     ((and (= in-rank rank)
           (= (vector-ref in-shape (- rank 1)) 1)
           (let loop ((k 0))
             (or (= k (- rank 1))
                 (and (= (vector-ref in-shape k) (vector-ref out-shape k))
                      (loop (+ k 1))))))
      (list 1 0))
     ;; [N] or [1, ..., 1, N]
     ((and (= in-size N) (> in-rank 0) (= (vector-ref in-shape (- in-rank 1)) N))
      (list 0 1))
     (else #f))))

;; Loops over the [M, N] output, storing (F A[..] B[..]) at each
;; element.  F is a flonum operator or a procedure; the element reads are
;; nested directly in the call so that the compiler can unbox them.
(define-syntax broadcast-loop
  (syntax-rules ()
    ((_ ref set! out dataA dataB M N rsA csA rsB csB f)
     (do ((i 0 (fx+ i 1))) ((fx= i M))
       (let ((base (fx* i N)) (a0 (fx* i rsA)) (b0 (fx* i rsB)))
         (do ((j 0 (fx+ j 1))) ((fx= j N))
           (set! out (fx+ base j)
                 (f (ref dataA (fx+ a0 (fx* j csA)))
                    (ref dataB (fx+ b0 (fx* j csB)))))))))))

;; The operand mode of lookup-broadcast-kernel for the steps at
;; position K (0 for A, 2 for B) of a broadcast on N columns: 0 full,
;; 1 per row, 2 per column, 3 scalar.
(define (broadcast-mode steps k N)
  (let ((rs (vector-ref steps k)) (cs (vector-ref steps (+ k 1))))
    (cond ((and (= cs 1) (= rs N) (> N 0)) 0)
          ((= rs 1) 1)
          ((= cs 1) 2)
          (else 3))))

;; Integer powers of a scalar exponent are computed by multiplication.
(define (broadcast-pow-fn b)
  (cond ((fp= b 2.0) (lambda (a) (fp* a a)))
        ((fp= b 3.0) (lambda (a) (fp* a (fp* a a))))
        ((fp= b 1.0) (lambda (a) a))
        (else (lambda (a) (fpexpt a b)))))

(define (execute-flat-broadcast-compute combiner op-kind dataA dataB out M N steps dtype)
  (let ((rsA (vector-ref steps 0)) (csA (vector-ref steps 1))
        (rsB (vector-ref steps 2)) (csB (vector-ref steps 3)))
    (define-syntax dispatch
      (syntax-rules ()
        ((_ ref set!)
         (case op-kind
           ((add) (broadcast-loop ref set! out dataA dataB M N rsA csA rsB csB fp+))
           ((sub) (broadcast-loop ref set! out dataA dataB M N rsA csA rsB csB fp-))
           ((mul) (broadcast-loop ref set! out dataA dataB M N rsA csA rsB csB fp*))
           ((div) (broadcast-loop ref set! out dataA dataB M N rsA csA rsB csB fp/))
           ((pow)
            (if (and (fx= rsB 0) (fx= csB 0))
                (let ((f (broadcast-pow-fn (ref dataB 0))))
                  (broadcast-loop ref set! out dataA dataB M N rsA csA rsB csB
                                  (lambda (a b) (f a))))
                (broadcast-loop ref set! out dataA dataB M N rsA csA rsB csB fpexpt)))
           (else
            (broadcast-loop ref set! out dataA dataB M N rsA csA rsB csB
                            (lambda (a b) (exact->inexact (combiner a b)))))))))
    (case dtype
      ((f32) (dispatch f32vector-ref f32vector-set!))
      ((f64) (dispatch f64vector-ref f64vector-set!))
      (else (error "execute-flat-broadcast-compute: unsupported dtype" dtype)))))

(include "segop-lower.scm")


;;; ============================================================
;;; Replay Plan: compile-replay-ref
;;; ============================================================

(define (compile-replay-ref v name->pos-eb)
  "Compile an ssa-value to a replay-ref.
   v: ssa-value (binding-ref or const-ref)
   name->pos-eb: env-builder bid-sym -> integer (0-based binding position)"
  (cases ssa-value v
    (const-ref   (cid) (rr-const cid))
    (binding-ref (bid) (rr-val (or (env-builder-lookup name->pos-eb bid)
                                   (error "compile-replay-ref: unknown binding" bid))))))


;;; ============================================================
;;; Replay Plan: compile-one-instruction
;;; ============================================================

(define (compile-one-instruction b in-refs pool-idx trace-info-env constants
                                 #!optional (extra-outs '()))
  "Compile one SSA binding into a replay-instruction.
   b:               ssa-binding
   in-refs:         list of replay-ref? (already compiled from inputs)
   pool-idx:        integer (physical buffer slot) or -1 (zero-copy)
   trace-info-env:  morph-env bid-sym -> (concrete-array . is-pool?)
   constants:       hash-table cid-sym -> concrete-array
   extra-outs:      for a fused binding with several outputs, the outputs
                    after the first (see ri-segop)"
  (let* ((op        (ssa-binding-op b))
         (meta      (ssa-binding-meta b))
         (shape     (ssa-binding-shape b))
         (dtype     (ssa-binding-dtype b))
         (strides   (if (vector? shape) (compute-strides shape) '#()))
         (info      (or (morph-env-lookup trace-info-env (ssa-binding-name b))
                        (error "compile-one-instruction: no trace-info for binding"
                               (ssa-binding-name b))))
         (trace-arr (car info))
         (is-pool?  (cdr info)))

    ;; Look up trace-time concrete-array for any ssa-value input:
    ;; binding-ref -> trace-info-env; const-ref -> constants.
    (define (trace-arr-of v)
      (cases ssa-value v
        (const-ref   (cid) (hash-table-ref constants cid))
        (binding-ref (bid) (car (or (morph-env-lookup trace-info-env bid)
                                    (error "compile-one-instruction: no trace-info for input"
                                           bid))))))

    ;; The index function of b: from its meta when morphism-to-ssa stored
    ;; one, else from the morphism rebuilt over the trace-time inputs (the
    ;; bindings ssa-vjp emits carry none).
    (define (get-index-fn)
      (let ((ip (assq 'index-fn meta)))
        (if ip
            (cdr ip)
            (let ((trace-inputs (map trace-arr-of (ssa-binding-inputs b))))
              (cases array-morphism (rebuild-morphism b trace-inputs)
                (morphism-expr (_ _ _ ifn _ _ _ _) ifn)
                (reduction-morphism (_ _ _ _ ifn _ _ _) ifn)
                (else (error "compile-one-instruction: rebuild-morphism returned unexpected type"
                             op meta)))))))

    (cond
      ;; Fused binding: its template lowered against the trace-time inputs.
      ((fused-binding? b)
       (compile-fused-segop (fused-binding-template b)
                            (map trace-arr-of (ssa-binding-inputs b))
                            in-refs pool-idx extra-outs shape strides dtype))

      ;; Zero-copy view: any op where the trace did not allocate (counter not incremented).
      ;; Only transpose and reshape produce zero-copy views in practice.
      ((not is-pool?)
       (let* ((index-fn   (get-index-fn))
              (batch-axis (cases array-morphism trace-arr
                            (concrete-array (_ _ _ _ _ _ ba) ba)
                            (else -1)))
              (view-fn    (lambda (src) (create-view src index-fn shape batch-axis))))
         (ri-view view-fn (car in-refs))))

      ;; matmul: a seg-red over (M N K), routed to the BLAS backend through
      ;; a gemm-plan prepared from the trace-time operand layouts (see
      ;; lower-matmul-binding in segop-lower.scm).
      ((eq? op 'matmul)
       (or (compile-matmul-segop (trace-arr-of (car  (ssa-binding-inputs b)))
                                 (trace-arr-of (cadr (ssa-binding-inputs b)))
                                 (car in-refs) (cadr in-refs) pool-idx shape strides dtype)
           (error "compile-one-instruction: matmul not BLAS-compatible" (ssa-binding-name b))))

      ;; Reductions: op is (list 'reduce rop).
      ;; VJP-emitted reduce has 'axes and 'keepdims? in meta but no 'index-fn;
      ;; get-index-fn handles both via rebuild-morphism.
      ((and (pair? op) (eq? (car op) 'reduce))
       (compile-reduction-segop (cadr op)
                                (cdr (assq 'axes meta))
                                (cdr (assq 'keepdims? meta))
                                (trace-arr-of (car (ssa-binding-inputs b)))
                                (car in-refs) pool-idx shape strides dtype))

      ;; Fast im2col: extract kernel/stride/padding from morph meta; bake into ri-im2col.
      ;; The morph meta carries kernel-size/stride/padding/spatial-output from im2col-morph.
      ;; Source shape comes from the trace-time concrete-array for the input binding.
      ((eq? op 'im2col)
       (let* ((meta-ks  (cdr (assq 'kernel-size meta)))
              (meta-st  (cdr (assq 'stride meta)))
              (meta-pd  (cdr (assq 'padding meta)))
              (meta-so  (let ((p (assq 'spatial-output meta))) (and p (cdr p))))
              (KH       (car  meta-ks)) (KW (cadr meta-ks))
              (SH       (car  meta-st)) (SW (cadr meta-st))
              (PH       (car  meta-pd)) (PW (cadr meta-pd))
              (nhwc?    (let ((p (assq 'layout meta))) (and p (eq? (cdr p) 'nhwc))))
              (src-arr  (trace-arr-of (car (ssa-binding-inputs b))))
              (src-shape (cases array-morphism src-arr
                           (concrete-array (_ s _ _ _ _ _) s)
                           (else (error "compile-one-instruction im2col: source not concrete"
                                        (ssa-binding-name b)))))
              (rank     (vector-length src-shape))
              (batched? (= rank 4))
              (col-layout (cond (nhwc?    'nhwc-classic)
                                (batched? 'nchw-classic)
                                (else     'nchw-standard)))
              (N        (if batched? (vector-ref src-shape 0) 1))
              ;; For NHWC input [N,H,W,C]: C is at index 3, H at 1, W at 2
              (C        (cond (nhwc?    (vector-ref src-shape 3))
                              (batched? (vector-ref src-shape 1))
                              (else     (vector-ref src-shape 0))))
              (H        (cond (nhwc?    (vector-ref src-shape 1))
                              (batched? (vector-ref src-shape 2))
                              (else     (vector-ref src-shape 1))))
              (W        (cond (nhwc?    (vector-ref src-shape 2))
                              (batched? (vector-ref src-shape 3))
                              (else     (vector-ref src-shape 2))))
              (OH       (if meta-so (car  meta-so)
                           (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH))))
              (OW       (if meta-so (cadr meta-so)
                           (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW)))))
         (ri-im2col pool-idx shape strides dtype
                    (car in-refs) N C H W KH KW SH SW PH PW OH OW col-layout)))

      ;; Fast col2im: VJP-emitted backward of im2col.
      ;; Meta carries output-shape (= grad target shape), kernel-size, stride, padding.
      ;; The output shape (= col2im output = im2col input gradient shape) is ssa-binding-shape.
      ((eq? op 'col2im)
       (let* ((meta-ks  (cdr (assq 'kernel-size meta)))
              (meta-st  (cdr (assq 'stride meta)))
              (meta-pd  (cdr (assq 'padding meta)))
              (KH       (car  meta-ks)) (KW (cadr meta-ks))
              (SH       (car  meta-st)) (SW (cadr meta-st))
              (PH       (car  meta-pd)) (PW (cadr meta-pd))
              (nhwc?    (let ((p (assq 'layout meta))) (and p (eq? (cdr p) 'nhwc))))
              (rank     (vector-length shape))
              (batched? (= rank 4))
              (col-layout (cond (nhwc?    'nhwc-classic)
                                (batched? 'nchw-classic)
                                (else     'nchw-standard)))
              (N        (if batched? (vector-ref shape 0) 1))
              ;; For NHWC output [N,H,W,C]: C is at index 3, H at 1, W at 2
              (C        (cond (nhwc?    (vector-ref shape 3))
                              (batched? (vector-ref shape 1))
                              (else     (vector-ref shape 0))))
              (H        (cond (nhwc?    (vector-ref shape 1))
                              (batched? (vector-ref shape 2))
                              (else     (vector-ref shape 1))))
              (W        (cond (nhwc?    (vector-ref shape 2))
                              (batched? (vector-ref shape 3))
                              (else     (vector-ref shape 2)))))
         (ri-col2im pool-idx shape strides dtype
                    (car in-refs) N C H W KH KW SH SW PH PW col-layout)))

      ;; conv2d-fwd: fused im2col+GEMM+bias, no col buffer.
      ;; inputs: [x-val, wt-val, b-val]; shape = [N*OH_OW, out_ch]
      ((eq? op 'conv2d-fwd)
       (let* ((meta-ks (cdr (assq 'kernel-size meta)))
              (meta-st (cdr (assq 'stride meta)))
              (meta-pd (cdr (assq 'padding meta)))
              (meta-so (let ((p (assq 'spatial-output meta))) (and p (cdr p))))
              (KH (car meta-ks)) (KW (cadr meta-ks))
              (SH (car meta-st)) (SW (cadr meta-st))
              (PH (car meta-pd)) (PW (cadr meta-pd))
              (nhwc?    (let ((p (assq 'layout meta))) (and p (eq? (cdr p) 'nhwc))))
              (src-arr  (trace-arr-of (list-ref (ssa-binding-inputs b) 0)))
              (src-shape (cases array-morphism src-arr
                           (concrete-array (_ s _ _ _ _ _) s)
                           (else (error "conv2d-fwd: src not concrete"
                                        (ssa-binding-name b)))))
              (rank     (vector-length src-shape))
              (N   (if (= rank 4) (vector-ref src-shape 0) 1))
              (C   (cond (nhwc?    (vector-ref src-shape 3))
                         ((= rank 4) (vector-ref src-shape 1))
                         (else       (vector-ref src-shape 0))))
              (H   (cond (nhwc?    (vector-ref src-shape 1))
                         ((= rank 4) (vector-ref src-shape 2))
                         (else       (vector-ref src-shape 1))))
              (W   (cond (nhwc?    (vector-ref src-shape 2))
                         ((= rank 4) (vector-ref src-shape 3))
                         (else       (vector-ref src-shape 2))))
              (OH  (if meta-so (car  meta-so) (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH))))
              (OW  (if meta-so (cadr meta-so) (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW))))
              (out-ch     (vector-ref shape 1))
              (col-layout (if nhwc? 'nhwc-classic 'nchw-classic))
              (col-buf    (allocate-typed-vector dtype (* N OH OW C KH KW))))
         (ri-conv-fwd pool-idx shape strides dtype
                      (car in-refs) (cadr in-refs) (caddr in-refs)
                      col-buf
                      N C H W KH KW SH SW PH PW OH OW out-ch col-layout)))

      ;; conv2d-bwd-data: dX = scatter(g x WT), no col buffer.
      ;; inputs: [g-val, wt-val]; shape = x-shape
      ((eq? op 'conv2d-bwd-data)
       (let* ((meta-ks (cdr (assq 'kernel-size meta)))
              (meta-st (cdr (assq 'stride meta)))
              (meta-pd (cdr (assq 'padding meta)))
              (meta-so (let ((p (assq 'spatial-output meta))) (and p (cdr p))))
              (KH (car meta-ks)) (KW (cadr meta-ks))
              (SH (car meta-st)) (SW (cadr meta-st))
              (PH (car meta-pd)) (PW (cadr meta-pd))
              (nhwc?    (let ((p (assq 'layout meta))) (and p (eq? (cdr p) 'nhwc))))
              (rank     (vector-length shape))
              (N   (if (= rank 4) (vector-ref shape 0) 1))
              (C   (cond (nhwc?    (vector-ref shape 3))
                         ((= rank 4) (vector-ref shape 1))
                         (else       (vector-ref shape 0))))
              (H   (cond (nhwc?    (vector-ref shape 1))
                         ((= rank 4) (vector-ref shape 2))
                         (else       (vector-ref shape 1))))
              (W   (cond (nhwc?    (vector-ref shape 2))
                         ((= rank 4) (vector-ref shape 3))
                         (else       (vector-ref shape 2))))
              (OH  (if meta-so (car  meta-so) (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH))))
              (OW  (if meta-so (cadr meta-so) (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW))))
              (g-arr   (trace-arr-of (list-ref (ssa-binding-inputs b) 0)))
              (g-shape (cases array-morphism g-arr
                          (concrete-array (_ s _ _ _ _ _) s)
                          (else (error "conv2d-bwd-data: g not concrete"))))
              (out-ch     (vector-ref g-shape 1))
              (col-layout (if nhwc? 'nhwc-classic 'nchw-classic))
              (col-buf    (allocate-typed-vector dtype (* N OH OW C KH KW))))
         (ri-conv-bwd-data pool-idx shape strides dtype
                           (car in-refs) (cadr in-refs)
                           col-buf
                           N C H W KH KW SH SW PH PW OH OW out-ch col-layout)))

      ;; conv2d-bwd-weights: dWT = outer product, no col buffer.
      ;; inputs: [g-val, x-val]; shape = wt-shape [fan_in, out_ch]
      ((eq? op 'conv2d-bwd-weights)
       (let* ((meta-ks (cdr (assq 'kernel-size meta)))
              (meta-st (cdr (assq 'stride meta)))
              (meta-pd (cdr (assq 'padding meta)))
              (meta-so (let ((p (assq 'spatial-output meta))) (and p (cdr p))))
              (KH (car meta-ks)) (KW (cadr meta-ks))
              (SH (car meta-st)) (SW (cadr meta-st))
              (PH (car meta-pd)) (PW (cadr meta-pd))
              (nhwc?    (let ((p (assq 'layout meta))) (and p (eq? (cdr p) 'nhwc))))
              (src-arr  (trace-arr-of (list-ref (ssa-binding-inputs b) 1)))
              (src-shape (cases array-morphism src-arr
                            (concrete-array (_ s _ _ _ _ _) s)
                            (else (error "conv2d-bwd-weights: src not concrete"))))
              (rank     (vector-length src-shape))
              (N   (if (= rank 4) (vector-ref src-shape 0) 1))
              (C   (cond (nhwc?    (vector-ref src-shape 3))
                         ((= rank 4) (vector-ref src-shape 1))
                         (else       (vector-ref src-shape 0))))
              (H   (cond (nhwc?    (vector-ref src-shape 1))
                         ((= rank 4) (vector-ref src-shape 2))
                         (else       (vector-ref src-shape 1))))
              (W   (cond (nhwc?    (vector-ref src-shape 2))
                         ((= rank 4) (vector-ref src-shape 3))
                         (else       (vector-ref src-shape 2))))
              (OH  (if meta-so (car  meta-so) (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH))))
              (OW  (if meta-so (cadr meta-so) (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW))))
              (g-arr   (trace-arr-of (list-ref (ssa-binding-inputs b) 0)))
              (g-shape (cases array-morphism g-arr
                          (concrete-array (_ s _ _ _ _ _) s)
                          (else (error "conv2d-bwd-weights: g not concrete"))))
              (out-ch     (vector-ref g-shape 1))
              (col-layout (if nhwc? 'nhwc-classic 'nchw-classic))
              (col-buf    (allocate-typed-vector dtype (* N OH OW C KH KW))))
         (ri-conv-bwd-weights pool-idx shape strides dtype
                              (car in-refs) (cadr in-refs)
                              col-buf
                              N C H W KH KW SH SW PH PW OH OW out-ch col-layout)))

      ;; Element-wise ops, materialized affine views and all other
      ;; pool-backed ops.  VJP-emitted bindings have empty meta; get-index-fn
      ;; rebuilds via trace inputs.  Each binding is lowered to a segop
      ;; once, at compile time; an index function with no segop form runs on
      ;; the host through execute-index-fn.
      (else
       (let* ((index-fn  (get-index-fn))
              (in-traces (map trace-arr-of (ssa-binding-inputs b))))
         (or (cond
              ((compute-index-fn? index-fn)
               (compile-elementwise-segop b (compute-index-fn-combiner index-fn)
                                          in-traces in-refs pool-idx shape strides dtype))
              ((and (affine-index-fn? index-fn) (= (length in-traces) 1))
               (compile-affine-segop index-fn (car in-traces) (car in-refs)
                                     pool-idx shape strides dtype))
              (else #f))
             (compile-host-segop index-fn in-traces in-refs pool-idx shape strides dtype)))))))


;;; ============================================================
;;; Replay Plan: compile-replay-plan
;;; ============================================================

(define (compile-replay-plan prog ctx)
  "Compile the SSA program's bindings into a vector of replay-instructions.
   ctx must be in replay mode (finalize-context! already called).
   Reads ssa-program-trace-info to determine pool-idx and BLAS variant.
   Returns a vector of replay-instruction (one per binding, in binding order)."
  (let* ((bindings    (ssa-program-bindings prog))
         (n           (length bindings))
         (constants   (ssa-program-constants prog))
         (trace-info  (ssa-program-trace-info prog))
         (plan-vec    (make-vector n #f))
         ;; Map binding name -> 0-based position in bindings list.  The
         ;; outputs after the first of fused bindings take the positions
         ;; after the bindings'.
         (name->pos-eb (let ((eb (make-env-builder)))
                         (let loop ((bs bindings) (i 0))
                           (unless (null? bs)
                             (env-builder-extend! eb (ssa-binding-name (car bs)) i)
                             (loop (cdr bs) (+ i 1))))
                         (let loop ((bs bindings) (j n))
                           (unless (null? bs)
                             (loop (cdr bs)
                                   (if (fused-binding? (car bs))
                                       (fold (lambda (o j)
                                               (env-builder-extend! eb (fused-out-name o) j)
                                               (+ j 1))
                                             j (cdr (fused-binding-outs (car bs))))
                                       j))))
                         eb)))
    (let loop ((bs bindings) (i 0))
      (unless (null? bs)
        (let* ((b     (car bs))
               (bname (ssa-binding-name b))
               (instr
                (let* ((in-refs  (map (lambda (v) (compile-replay-ref v name->pos-eb))
                                      (ssa-binding-inputs b)))
                       (info     (or (morph-env-lookup trace-info bname)
                                     (error "compile-replay-plan: no trace-info for binding"
                                            bname)))
                       (is-pool? (cdr info))
                       (pool-idx (if is-pool?
                                     (context-alloc->pool-idx ctx (concrete-alloc-id (car info)))
                                     -1))
                       (extra-outs
                        (if (fused-binding? b)
                            (map (lambda (o)
                                   (let* ((name  (fused-out-name o))
                                          (oinfo (or (morph-env-lookup trace-info name)
                                                     (error "compile-replay-plan: no trace-info for output"
                                                            name)))
                                          (shape (fused-out-shape o)))
                                     (vector (env-builder-lookup name->pos-eb name)
                                             (context-alloc->pool-idx
                                              ctx (concrete-alloc-id (car oinfo)))
                                             shape (compute-strides shape) (fused-out-dtype o))))
                                 (cdr (fused-binding-outs b)))
                            '())))
                  (compile-one-instruction b in-refs pool-idx trace-info constants
                                           extra-outs))))
          (vector-set! plan-vec i instr)
          (loop (cdr bs) (+ i 1)))))

    ;; Pre-compute output specs to eliminate name->idx hash-table rebuild per replay step.
    ;; Each spec is one of
    ;;   an integer       the 0-based position in vals of a binding-ref output;
    ;;   (pos . buffer)   the same for an output that the trace produced as a
    ;;                    strided view, with a buffer of the plan that each
    ;;                    replay copies it into in row-major order;
    ;;   a concrete-array the constant value itself, for const-refs.
    (let ((output-specs
           (map (lambda (ov)
                  (cases ssa-value ov
                    (binding-ref (bid)
                      (let ((pos  (or (env-builder-lookup name->pos-eb bid)
                                      (error "compile-replay-plan: output binding not found" bid)))
                            (info (morph-env-lookup trace-info bid)))
                        (if (and info (concrete-array? (car info))
                                 (not (concrete-row-major? (car info))))
                            (cases array-morphism (car info)
                              (concrete-array (_ shape _ _ dtype _ _)
                                (cons pos (allocate-typed-vector dtype (shape-size shape))))
                              (else pos))
                            pos)))
                    (const-ref   (cid) (hash-table-ref constants cid))))
                (ssa-program-outputs prog))))
      (ssa-program-output-specs-set! prog output-specs))

    plan-vec))


;;; ============================================================
;;; Replay Plan: per-instruction timing accumulators
;;; ============================================================

;; Timing is accumulated in a hash table tag -> total-ms across all steps.
;; Call replay-timing-reset! before a timing run; replay-timing-results after.
;; Times are wall-clock milliseconds (microsecond resolution), so that
;; kernels running on several threads are not over-counted.  The
;; per-tag-counts entry gives how many instructions ran under each tag.
(define *timing-enabled?* #f)
(define *timing-table* (make-hash-table eq?))
(define *timing-counts* (make-hash-table eq?))
(define *timing-steps* 0)

(define (replay-timing-reset!)
  (set! *timing-table* (make-hash-table eq?))
  (set! *timing-counts* (make-hash-table eq?))
  (set! *timing-steps* 0)
  (set! *timing-enabled?* #t))

(define (replay-timing-results)
  (set! *timing-enabled?* #f)
  (list (cons 'steps *timing-steps*)
        (cons 'per-tag (hash-table->alist *timing-table*))
        (cons 'per-tag-counts (hash-table->alist *timing-counts*))))

(define (timing-wall-ms)
  (/ (* 1000.0 (current-jiffy)) (jiffies-per-second)))

(define-syntax time-instr
  (syntax-rules ()
    ((_ tag body ...)
     (if *timing-enabled?*
         (let* ((t0 (timing-wall-ms))
                (result (begin body ...))
                (elapsed (- (timing-wall-ms) t0)))
           (hash-table-set! *timing-table* tag
                            (+ elapsed (hash-table-ref/default *timing-table* tag 0)))
           (hash-table-set! *timing-counts* tag
                            (+ 1 (hash-table-ref/default *timing-counts* tag 0)))
           result)
         (begin body ...)))))


;;; ============================================================
;;; Replay Plan: execute-replay-plan
;;; ============================================================

(define (execute-replay-plan plan pool constants)
  "Execute a pre-compiled replay-plan.
   plan:      vector of replay-instruction (from compile-replay-plan)
   pool:      buffer-pool record (from morphism-context-pool ctx)
   constants: hash-table cid-sym -> concrete-array (from ssa-program-constants)
   Returns:   vector of concrete-array, one per binding position."
  (let* ((n         (vector-length plan))
         (pool-bufs (buffer-pool-buffers pool))
         ;; one value per instruction, then the outputs after the first of
         ;; the segops with several outputs
         (n-vals    (do ((i 0 (+ i 1))
                         (m n (cases replay-instruction (vector-ref plan i)
                                (ri-segop (_ _ _ _ _ _ _ extra-outs) (+ m (length extra-outs)))
                                (else m))))
                        ((= i n) m)))
         (vals      (make-vector n-vals #f))
         ;; alloc-ctr mirrors the trace-time context counter so that pool-backed
         ;; bindings get the same alloc-id here as they did during the trace run.
         ;; Zero-copy (ri-view) bindings don't increment this, matching the trace.
         (alloc-ctr 0))

    (define (deref ref)
      (cases replay-ref ref
        (rr-val   (i)   (vector-ref vals i))
        (rr-const (cid) (hash-table-ref constants cid))))

    ;; Build a concrete-array for a pool-backed instruction using the next
    ;; trace-aligned alloc-id (not pool-idx) so callers that inspect alloc-id
    ;; see the same value as they did in the trace run.
    ;; Strides are pre-computed at compile time; no compute-strides call per step.
    (define (make-pool-arr pool-idx shape strides dtype)
      (let ((aid alloc-ctr))
        (set! alloc-ctr (+ alloc-ctr 1))
        (concrete-array (vector-ref pool-bufs pool-idx)
                        shape strides 0 dtype aid -1)))

    (when *timing-enabled?*
      (set! *timing-steps* (+ *timing-steps* 1)))

    (do ((i 0 (+ i 1)))
        ((= i n))
      (let* ((instr (vector-ref plan i))
             (result
              (cases replay-instruction instr


                (ri-view (view-fn in-ref)
                  (time-instr 'ri-view
                    (view-fn (deref in-ref))))

                (ri-segop (pool-idx shape strides dtype sg executor tag extra-outs)
                  (time-instr tag
                    (if (null? extra-outs)
                        (begin
                          (executor deref (vector-ref pool-bufs pool-idx))
                          (make-pool-arr pool-idx shape strides dtype))
                        ;; the outputs take alloc-ids in order, as in the trace
                        (begin
                          (executor deref (vector-ref pool-bufs pool-idx)
                                    (map (lambda (e) (vector-ref pool-bufs (vector-ref e 1)))
                                         extra-outs))
                          (let ((first (make-pool-arr pool-idx shape strides dtype)))
                            (for-each (lambda (e)
                                        (vector-set! vals (vector-ref e 0)
                                                     (make-pool-arr (vector-ref e 1) (vector-ref e 2)
                                                                    (vector-ref e 3) (vector-ref e 4))))
                                      extra-outs)
                            first)))))

                (ri-im2col (pool-idx shape strides dtype src-ref
                             N C H W KH KW SH SW PH PW OH OW col-layout)
                  (time-instr 'ri-im2col
                    (let* ((src (deref src-ref))
                           (buf (vector-ref pool-bufs pool-idx)))
                      (cases array-morphism src
                        (concrete-array (_ _ _ _ _ _ _)
                          (let ((src-data (concrete-data/row-major src)))
                          (case col-layout
                            ((nchw-standard)
                             (execute-im2col-unbatched buf src-data C H W KH KW SH SW PH PW OH OW dtype))
                            ((nchw-classic)
                             (execute-im2col-batched buf src-data N C H W KH KW SH SW PH PW OH OW dtype))
                            ((nhwc-classic)
                             (execute-im2col-nhwc-classic buf src-data N C H W KH KW SH SW PH PW OH OW dtype))
                            (else (error "ri-im2col: unknown col-layout" col-layout))))
                          (make-pool-arr pool-idx shape strides dtype))
                        (else (error "ri-im2col: source not concrete" src))))))

                (ri-col2im (pool-idx shape strides dtype col-ref
                             N C H W KH KW SH SW PH PW col-layout)
                  (time-instr 'ri-col2im
                    (let* ((col (deref col-ref))
                           (buf (vector-ref pool-bufs pool-idx)))
                      (cases array-morphism col
                        (concrete-array (_ col-shape _ _ _ _ _)
                          (let ((col-data (concrete-data/row-major col)))
                          (case col-layout
                            ((nchw-standard)
                             (execute-col2im-unbatched buf shape col-data col-shape KH KW SH SW PH PW dtype))
                            ((nchw-classic)
                             (execute-col2im-batched buf shape col-data col-shape KH KW SH SW PH PW dtype))
                            ((nhwc-classic)
                             (execute-col2im-nhwc-classic buf shape col-data col-shape KH KW SH SW PH PW dtype))
                            (else (error "ri-col2im: unknown col-layout" col-layout))))
                          (make-pool-arr pool-idx shape strides dtype))
                        (else (error "ri-col2im: col not concrete" col))))))

                (ri-conv-fwd (pool-idx shape strides dtype src-ref wt-ref b-ref col-buf
                              N C H W KH KW SH SW PH PW OH OW out-ch col-layout)
                  (time-instr 'ri-conv-fwd
                    (let* ((src  (deref src-ref)) (wt (deref wt-ref)) (b (deref b-ref))
                           (buf  (vector-ref pool-bufs pool-idx))
                           (src-d (concrete-data/row-major src))
                           (wt-d  (concrete-data/row-major wt))
                           (b-d   (concrete-data/row-major b)))
                                  (if (and (blas-enabled?) (blas-available?))
                                      (case col-layout
                                        ((nchw-classic)
                                         (execute-conv-fwd-blas buf src-d wt-d b-d N C H W KH KW SH SW PH PW OH OW out-ch dtype col-buf))
                                        ((nhwc-classic)
                                         (execute-conv-fwd-nhwc-blas buf src-d wt-d b-d N C H W KH KW SH SW PH PW OH OW out-ch dtype col-buf))
                                        (else (error "ri-conv-fwd: unknown layout" col-layout)))
                                      (case col-layout
                                        ((nchw-classic)
                                         (execute-conv-fwd-nchw buf src-d wt-d b-d N C H W KH KW SH SW PH PW OH OW out-ch dtype))
                                        ((nhwc-classic)
                                         (execute-conv-fwd-nhwc buf src-d wt-d b-d N C H W KH KW SH SW PH PW OH OW out-ch dtype))
                                        (else (error "ri-conv-fwd: unknown layout" col-layout))))
                                  (make-pool-arr pool-idx shape strides dtype))))

                (ri-conv-bwd-data (pool-idx shape strides dtype g-ref wt-ref col-buf
                                   N C H W KH KW SH SW PH PW OH OW out-ch col-layout)
                  (time-instr 'ri-conv-bwd-data
                    (let* ((g  (deref g-ref)) (wt (deref wt-ref))
                           (buf (vector-ref pool-bufs pool-idx))
                           (g-shape (cases array-morphism g
                                      (concrete-array (_ s _ _ _ _ _) s)
                                      (else (error "ri-conv-bwd-data: g not concrete"))))
                           (g-d  (concrete-data/row-major g))
                           (wt-d (concrete-data/row-major wt)))
                              (if (and (blas-enabled?) (blas-available?))
                                  (case col-layout
                                    ((nchw-classic)
                                     (execute-conv-bwd-data-blas buf shape g-d g-shape wt-d N C H W KH KW SH SW PH PW OH OW out-ch dtype col-buf))
                                    ((nhwc-classic)
                                     (execute-conv-bwd-data-nhwc-blas buf shape g-d g-shape wt-d N C H W KH KW SH SW PH PW OH OW out-ch dtype col-buf))
                                    (else (error "ri-conv-bwd-data: unknown layout" col-layout)))
                                  (case col-layout
                                    ((nchw-classic)
                                     (execute-conv-bwd-data-nchw buf shape g-d g-shape wt-d N C H W KH KW SH SW PH PW OH OW out-ch dtype))
                                    ((nhwc-classic)
                                     (execute-conv-bwd-data-nhwc buf shape g-d g-shape wt-d N C H W KH KW SH SW PH PW OH OW out-ch dtype))
                                    (else (error "ri-conv-bwd-data: unknown layout" col-layout))))
                              (make-pool-arr pool-idx shape strides dtype))))

                (ri-conv-bwd-weights (pool-idx shape strides dtype g-ref src-ref col-buf
                                      N C H W KH KW SH SW PH PW OH OW out-ch col-layout)
                  (time-instr 'ri-conv-bwd-weights
                    (let* ((g   (deref g-ref)) (src (deref src-ref))
                           (buf (vector-ref pool-bufs pool-idx))
                           (g-shape (cases array-morphism g
                                      (concrete-array (_ s _ _ _ _ _) s)
                                      (else (error "ri-conv-bwd-weights: g not concrete"))))
                           (g-d   (concrete-data/row-major g))
                           (src-d (concrete-data/row-major src)))
                              (if (and (blas-enabled?) (blas-available?))
                                  (case col-layout
                                    ((nchw-classic)
                                     (execute-conv-bwd-weights-blas buf shape g-d g-shape src-d N C H W KH KW SH SW PH PW OH OW out-ch dtype col-buf))
                                    ((nhwc-classic)
                                     (execute-conv-bwd-weights-nhwc-blas buf shape g-d g-shape src-d N C H W KH KW SH SW PH PW OH OW out-ch dtype col-buf))
                                    (else (error "ri-conv-bwd-weights: unknown layout" col-layout)))
                                  (case col-layout
                                    ((nchw-classic)
                                     (execute-conv-bwd-weights-nchw buf shape g-d g-shape src-d N C H W KH KW SH SW PH PW OH OW out-ch dtype))
                                    ((nhwc-classic)
                                     (execute-conv-bwd-weights-nhwc buf shape g-d g-shape src-d N C H W KH KW SH SW PH PW OH OW out-ch dtype))
                                    (else (error "ri-conv-bwd-weights: unknown layout" col-layout))))
                              (make-pool-arr pool-idx shape strides dtype))))
                ))
             )
        (vector-set! vals i result)))
    
    vals))
      


;;; ============================================================
;;; Phase 4: ssa-realize/ctx (with lazy replay-plan compilation)
;;; ============================================================

(define (ssa-realize/ctx ctx prog)
  "Execute the SSA program through a morphism context.

   Trace run (first call, context in trace mode):
     Executes each binding via realize/ctx, recording per-binding trace-info
     (concrete-array + is-pool? flag). Output bindings with pool allocations
     are pinned via context-pin-output! so the greedy allocator never reuses
     their slot within a replay run.

   First replay call (context in replay mode, no plan yet):
     Lazily compiles the replay-plan from trace-info + finalized pool, then
     executes it via execute-replay-plan.  The trace-info, and with it the
     arrays of the trace run, is released once the plan exists.

   Subsequent replay calls:
     Executes the pre-compiled replay-plan directly.  No morphism rebuilding,
     no BLAS-compat re-evaluation, no context-vector allocation per binding.

   The outputs of a replay call live in pool buffers, or, for an output
   that is a strided view, in a buffer of the plan that receives its
   row-major copy; the next replay call of the program overwrites them, so
   a caller that keeps an output across calls copies it first.

   A program to which the fusion policy has not been applied yet (a
   forward program from morphism-to-ssa) is fused at its first trace,
   under the policy and the backends in effect then; its bindings are
   replaced by the fused ones."
  (when (and (eq? (context-mode ctx) 'trace) (not (ssa-program-fused? prog)))
    (when (segop-fusion)
      (ssa-program-bindings-set! prog (ssa-program-bindings (ssa-segop-fusion-pass prog))))
    (ssa-program-fused?-set! prog #t))
  (let* ((mode        (context-mode ctx))
         (trace-mode? (eq? mode 'trace))
         (constants   (ssa-program-constants prog))
         (bindings    (ssa-program-bindings prog)))

    (cond
      ;; ---- BINDING-LOOP PATH ----
      ;; Used for the trace run AND for any replay call of a program that
      ;; has neither a replay plan nor trace-info (e.g., when the caller's
      ;; joint-program state table missed a lookup due to GC-invalidated
      ;; eq?-hash keys and created a fresh program).  In both cases we run
      ;; every binding via realize/ctx, collect trace-info, and return the
      ;; outputs.  Output pinning via context-pin-output! is only performed
      ;; in true trace mode.
      ((or trace-mode?
           (not (or (ssa-program-replay-plan prog) (ssa-program-trace-info prog))))
       (let* ((values       (make-hash-table))
              (trace-info-eb (make-env-builder))
              (output-bids-eb (make-env-builder)))
         ;; Index output binding names for membership lookup
         (for-each (lambda (ov)
                     (when (ssa-binding-ref? ov)
                       (env-builder-extend! output-bids-eb (ssa-value-id ov) #t)))
                   (ssa-program-outputs prog))
         ;; Pre-populate constants (values is kept as hash-table: hot O(1) lookup path)
         (hash-table-walk constants
           (lambda (k v) (hash-table-set! values k v)))
         ;; Execute each binding; record trace-info.  A fused binding runs
         ;; its segop executor and allocates one buffer per output, each a
         ;; value of its own.
         (for-each
          (lambda (b)
           (if (fused-binding? b)
            (let ((inputs (map (lambda (v) (hash-table-ref values (ssa-value-id v)))
                               (ssa-binding-inputs b))))
              (for-each
               (lambda (o result)
                 (let ((name (fused-out-name o)))
                   (when (and trace-mode? (env-builder-lookup output-bids-eb name))
                     (context-pin-output! ctx (concrete-alloc-id result)))
                   (hash-table-set! values name result)
                   (env-builder-extend! trace-info-eb name (cons result #t))))
               (fused-binding-outs b) (realize-fused-binding ctx b inputs)))
            (let* ((inputs     (map (lambda (v) (hash-table-ref values (ssa-value-id v)))
                                    (ssa-binding-inputs b)))
                   (is-output? (env-builder-lookup output-bids-eb (ssa-binding-name b)))
                   (ctr-before (context-counter ctx))
                   (result     (realize/ctx ctx (rebuild-morphism b inputs)))
                   (is-pool?   (> (context-counter ctx) ctr-before))
                   (stored
                    (if is-output?
                        (begin
                          ;; Pin the pool allocation behind an output, whether
                          ;; this binding allocated it or the output is a
                          ;; zero-copy view of an earlier binding's buffer (a
                          ;; view inherits its source's allocation id).  Replay
                          ;; reads outputs after the whole plan has run, so the
                          ;; buffer must not be reused within the run.
                          ;; context-pin-output! is only valid in trace mode.
                          (when (and trace-mode? (concrete-array? result)
                                     (>= (concrete-alloc-id result) 0))
                            (context-pin-output! ctx (concrete-alloc-id result)))
                          (if (concrete-row-major? result)
                              result
                              (copy-concrete-array result)))
                        result)))
              (hash-table-set! values (ssa-binding-name b) stored)
              ;; Record per-binding (concrete-array . is-pool?) for replay-plan compilation
              (env-builder-extend! trace-info-eb (ssa-binding-name b) (cons result is-pool?)))))
          bindings)
         ;; Save morph-env snapshot for lazy replay-plan compilation
         (ssa-program-trace-info-set! prog (env-builder-env trace-info-eb))
         ;; Return outputs
         (map (lambda (ov) (hash-table-ref values (ssa-value-id ov)))
              (ssa-program-outputs prog))))

      ;; ---- REPLAY RUN ----
      (else
       ;; Lazily compile replay-plan on first replay call (pool is now
       ;; available).  The trace arrays are not needed once the plan exists.
       (unless (ssa-program-replay-plan prog)
         (ssa-program-replay-plan-set! prog (compile-replay-plan prog ctx))
         (ssa-program-trace-info-set! prog #f))
       ;; Execute the pre-compiled plan directly
       (let* ((plan  (ssa-program-replay-plan prog))
              (pool  (morphism-context-pool ctx))
              (vals  (execute-replay-plan plan pool constants))
              ;; output-specs pre-computed by compile-replay-plan: list of
              ;; (integer | (pos . buffer) | concrete-array) -- no hash-table
              ;; rebuild per step.
              (specs (ssa-program-output-specs prog)))
           (time-instr 'ri-output-mapping
             (map (lambda (spec)
                    (cond
                     ((integer? spec)
                      (let ((v (vector-ref vals spec)))
                        (if (concrete-row-major? v) v (copy-concrete-array v))))
                     ((pair? spec)
                      (let ((v (vector-ref vals (car spec))))
                        (if (concrete-row-major? v) v (copy-concrete-array-into v (cdr spec)))))
                     (else spec)))   ; const-ref: direct concrete-array from constants
                  specs)))))

    ))

  ;; How many fused bindings of a plan run as a native fused kernel and how
  ;; many on the host executors, as ((native . n) (host . m)).
  (define (fused-placement plan bindings)
    (let loop ((bs bindings) (i 0) (native 0) (host 0))
      (cond ((null? bs) `((native . ,native) (host . ,host)))
            ((fused-binding? (car bs))
             (if (cases replay-instruction (vector-ref plan i)
                   (ri-segop (_ _ _ _ _ _ tag _)
                     (memq tag '(segop/fused-kernel segop/colsum-kernel segop/gemm-epilogue)))
                   (else #f))
                 (loop (cdr bs) (+ i 1) (+ native 1) host)
                 (loop (cdr bs) (+ i 1) native (+ host 1))))
            (else (loop (cdr bs) (+ i 1) native host)))))

  (define (replay-plan-stats prog)
    "Return alist with:
       (counts . alist-of-tag.count)
       (ri-index-shapes . list-of-(i . shape-vector)), the positions of the
          segops executed by the generic strided loops, gathers, host index
          functions and the interpreter
       (ri-reduce-specs . list-of-(i axes keepdims? shape)), the reductions"
    (let ((plan (ssa-program-replay-plan prog)))
      (if (not plan)
          '()
          (let ((counts       (make-hash-table))
                (index-shapes (make-stack))
                (reduce-specs (make-stack)))
            (do ((i 0 (+ i 1))) ((= i (vector-length plan)))
              (let* ((instr (vector-ref plan i))
                     (tag (cases replay-instruction instr
                            (ri-view             (_ _)                   'ri-view)
                            (ri-segop            (_ shape _ _ sg _ tag _)
                              (when (memq tag '(segop/strided-map segop/gather
                                                segop/host-index-fn segop/interpreted))
                                (stack-push! index-shapes (cons i shape)))
                              (let ((r (and (eq? (segop-kind sg) 'seg-red) (standard-reduction sg))))
                                (when r
                                  (stack-push! reduce-specs (list i (cadr r) (caddr r) shape))))
                              tag)
                            (ri-im2col           (_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) 'ri-im2col)
                            (ri-col2im           (_ _ _ _ _ _ _ _ _ _ _ _ _ _ _)     'ri-col2im)
                            (ri-conv-fwd         (_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) 'ri-conv-fwd)
                            (ri-conv-bwd-data    (_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)  'ri-conv-bwd-data)
                            (ri-conv-bwd-weights (_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)  'ri-conv-bwd-weights))))
                (hash-table-set! counts tag
                                 (+ (hash-table-ref/default counts tag 0) 1))))
            `((counts . ,(hash-table->alist counts))
              (ri-index-shapes . ,(reverse (stack->list index-shapes)))
              (ri-reduce-specs . ,(reverse (stack->list reduce-specs)))
              (fused . ,(fused-placement plan (ssa-program-bindings prog)))))))

    )

) ; end module array-morphisms-ssa
