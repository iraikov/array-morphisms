;;; array-morphisms-context.scm
;;;
;;; Memory Reuse Integration
;;;
;;; Two-phase trace/replay execution context for amortising buffer
;;; allocations across repeated inferences.
;;;
;;; Usage:
;;;   (define ctx (make-morphism-context))
;;;   (realize/ctx ctx morphism)          ; trace run
;;;   (finalize-context! ctx)             ; lifetime analysis + buffer plan
;;;   (reset-context! ctx)
;;;   (realize/ctx ctx morphism)          ; replay run (reuses buffers)

(module array-morphisms-context

  (;; Context lifecycle
   make-morphism-context
   realize/ctx
   finalize-context!
   reset-context!

   ;; Inspection
   context-mode
   context-stats
   context-counter
   print-context-plan

   ;; Output pinning
   context-pin-output!

   ;; Pool accessors (for replay-plan compilation)
   morphism-context-pool
   buffer-pool-assignment
   buffer-pool-buffers
   context-alloc->pool-idx)

  (import scheme chicken.base chicken.format chicken.port)
  (import (only srfi-1 iota fold every filter find))
  (import srfi-4 srfi-69)
  (import array-morphisms-core)
  (import array-morphisms-realization)

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Record Types
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; One entry per non-zero-copy allocation recorded during the trace phase.
  ;; last-use is initialised to id (birth) and updated during lifetime analysis.
  ;; kind is 'normal or 'output-pinned; pinned allocs have last-use extended to n-1.
  (define-record allocation-rec id dtype size shape last-use inputs kind)

  ;; Immutable buffer pool created by finalize-context!.
  ;;   buffers    - #(typed-vector ...)  one physical buffer per logical slot
  ;;   dtypes     - #(symbol ...)        dtype of each logical buffer
  ;;   assignment - #(buf-id ...)        alloc-id -> buf-id
  ;;   sizes      - #(count ...)         buf-id -> max element count
  (define-record buffer-pool buffers dtypes assignment sizes)

  ;; Mutable execution context.
  ;;   mode    - symbol 'trace or 'replay
  ;;   allocs  - list (grows during trace, cons-prepended), then vector after finalize
  ;;   pool    - #f during trace, buffer-pool after finalize
  ;;   counter - integer; incremented once per non-zero-copy allocation
  ;;
  ;; Use define-record-type to give the raw constructor a private name,
  ;; allowing the public make-morphism-context to be a 0-arg wrapper.
  (define-record-type morphism-context
    (%make-morphism-context mode allocs pool counter)
    morphism-context?
    (mode    morphism-context-mode    morphism-context-mode-set!)
    (allocs  morphism-context-allocs  morphism-context-allocs-set!)
    (pool    morphism-context-pool    morphism-context-pool-set!)
    (counter morphism-context-counter morphism-context-counter-set!))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Context Creation
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (make-morphism-context)
    "Create a fresh trace-mode execution context."
    (%make-morphism-context 'trace '() #f 0))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Dispatch Vector Construction
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; Builds the opaque #(mode next-id! record! get-buf!) vector stored
  ;; in current-morphism-context.  Called fresh on each realize/ctx
  ;; invocation so that the mode symbol is always current.
  (define (make-context-vector ctx)

    (vector
     ;; slot 0: mode symbol - captured by value at call time
     (morphism-context-mode ctx)

     ;; slot 1: next-id! - returns current counter then increments
     (lambda ()
       (let ((id (morphism-context-counter ctx)))
         (morphism-context-counter-set! ctx (+ id 1))
         id))

     ;; slot 2: record! - prepend a new allocation-rec to the allocs list
     (lambda (alloc-id dtype size shape input-ids)
       (let ((rec (make-allocation-rec alloc-id dtype size shape
                                       alloc-id   ; last-use = birth initially
                                       input-ids
                                       'normal)))  ; kind
         (morphism-context-allocs-set!
          ctx (cons rec (morphism-context-allocs ctx)))))

     ;; slot 3: get-buf! - retrieve workspace buffer from pool
     (lambda (alloc-id dtype size)
       (get-workspace-buffer (morphism-context-pool ctx)
                             alloc-id dtype size))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Public API
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (realize/ctx ctx m)
    "Realize morphism m with context ctx active.
    On the first call (trace mode) allocations are recorded.
    After finalize-context! subsequent calls replay into workspace buffers.
    The caller must call reset-context! before each replay run."
    (parameterize ((current-morphism-context (make-context-vector ctx)))
      (realize m)))

  (define (finalize-context! ctx)
    "Analyse the recorded trace, compute buffer lifetimes, run greedy
    interval scheduling, and install a buffer pool.  After this call
    context mode switches to 'replay."
    (unless (eq? (morphism-context-mode ctx) 'trace)
      (error "finalize-context!: context is not in trace mode"))
    (let* ((allocs-list (reverse (morphism-context-allocs ctx)))
           (n           (length allocs-list))
           (allocs-vec  (list->vector allocs-list)))
      ;; Step 1: dependency-driven lifetime extension
      (compute-last-uses! allocs-vec)
      ;; Step 2: extend output-pinned allocations to end-of-program
      (do ((i 0 (+ i 1)))
          ((= i n))
        (let ((rec (vector-ref allocs-vec i)))
          (when (eq? (allocation-rec-kind rec) 'output-pinned)
            (allocation-rec-last-use-set! rec (- n 1)))))
      ;; Step 3: greedy interval scheduling
      (let ((pool (allocate-buffers allocs-vec n)))
        (morphism-context-allocs-set! ctx allocs-vec)
        (morphism-context-pool-set!   ctx pool)
        (morphism-context-mode-set!   ctx 'replay))))

  (define (reset-context! ctx)
    "Reset counter to 0 for a fresh replay run.
    Must only be called in replay mode (after finalize-context!)."
    (unless (eq? (morphism-context-mode ctx) 'replay)
      (error "reset-context!: context is not in replay mode"))
    (morphism-context-counter-set! ctx 0))

  (define (context-mode ctx)
    "Return the current mode: 'trace or 'replay."
    (morphism-context-mode ctx))

  (define (context-stats ctx)
    "Return an alist of statistics about the context."
    (let* ((mode   (morphism-context-mode ctx))
           (allocs (morphism-context-allocs ctx))
           (n      (if (vector? allocs)
                       (vector-length allocs)
                       (length allocs))))
      (case mode
        ((trace)
         `((mode . trace) (allocations . ,n)))
        ((replay)
         (let ((n-bufs (vector-length
                        (buffer-pool-buffers (morphism-context-pool ctx)))))
           `((mode . replay)
             (allocations . ,n)
             (buffers . ,n-bufs))))
        (else `((mode . ,mode))))))

  (define (context-counter ctx)
    "Return the current allocation counter.
    Incremented once per non-zero-copy realize/ctx call; unchanged for zero-copy views."
    (morphism-context-counter ctx))

  (define (context-pin-output! ctx alloc-id)
    "Mark alloc-id as output-pinned.  During finalize-context! its last-use
    will be extended to n-1 so the greedy allocator never reuses its buffer slot
    within a single program run.  Must be called in trace mode."
    (unless (eq? (morphism-context-mode ctx) 'trace)
      (error "context-pin-output!: context must be in trace mode" alloc-id))
    (let ((rec (find (lambda (r) (= (allocation-rec-id r) alloc-id))
                     (morphism-context-allocs ctx))))
      (unless rec
        (error "context-pin-output!: alloc-id not found" alloc-id))
      (allocation-rec-kind-set! rec 'output-pinned)))

  (define (context-alloc->pool-idx ctx alloc-id)
    "Return the physical buffer slot (pool-idx) for alloc-id.
    The context must be in replay mode (finalize-context! already called)."
    (let ((pool (morphism-context-pool ctx)))
      (unless pool
        (error "context-alloc->pool-idx: context not finalized" alloc-id))
      (let ((assignment (buffer-pool-assignment pool)))
        (when (>= alloc-id (vector-length assignment))
          (error "context-alloc->pool-idx: alloc-id out of range"
                 `((alloc-id ,alloc-id)
                   (pool-size ,(vector-length assignment)))))
        (vector-ref assignment alloc-id))))

  (define (print-context-plan ctx)
    "Print a human-readable allocation plan for debugging."
    (let ((allocs (morphism-context-allocs ctx))
          (pool   (morphism-context-pool ctx)))
      (display "=== Context Plan ===") (newline)
      (if (not (vector? allocs))
          (begin (display "(trace not yet finalized)") (newline))
          (begin
            (display (string-append "Allocations: "
                                    (number->string (vector-length allocs))))
            (newline)
            (when pool
              (display (string-append "Buffers:     "
                                      (number->string
                                       (vector-length
                                        (buffer-pool-buffers pool)))))
              (newline)
              (newline)
              (display "alloc  dtype  size  born  dies  buf  kind           inputs") (newline)
              (display "-----  -----  ----  ----  ----  ---  ----           ------") (newline)
              (do ((i 0 (+ i 1)))
                  ((= i (vector-length allocs)))
                (let* ((rec    (vector-ref allocs i))
                       (buf-id (vector-ref (buffer-pool-assignment pool) i)))
                  (display
                   (string-append
                    (number->string (allocation-rec-id rec)) "  "
                    (symbol->string (allocation-rec-dtype rec)) "  "
                    (number->string (allocation-rec-size rec)) "  "
                    (number->string (allocation-rec-id rec)) "  "
                    (number->string (allocation-rec-last-use rec)) "  "
                    (number->string buf-id) "  "
                    (symbol->string (allocation-rec-kind rec)) "  "
                    (with-output-to-string (lambda ()
                                            (write (allocation-rec-inputs rec))))))
                  (newline))))))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Workspace Buffer Retrieval
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (get-workspace-buffer pool alloc-id dtype size)
    "Return the pre-allocated workspace typed-vector for alloc-id.
    The buffer's dtype must match and its capacity must be >= size."
    (let* ((assignment (buffer-pool-assignment pool))
           (buffers    (buffer-pool-buffers    pool))
           (dtypes     (buffer-pool-dtypes     pool))
           (sizes      (buffer-pool-sizes      pool)))
      (when (>= alloc-id (vector-length assignment))
        (error "get-workspace-buffer: alloc-id out of range"
               `((alloc-id ,alloc-id)
                 (pool-size ,(vector-length assignment)))))
      (let* ((buf-id    (vector-ref assignment alloc-id))
             (buf-dtype (vector-ref dtypes buf-id))
             (buf-size  (vector-ref sizes  buf-id)))
        (unless (eq? buf-dtype dtype)
          (error "get-workspace-buffer: dtype mismatch"
                 `((alloc-id ,alloc-id) (expected ,dtype) (got ,buf-dtype))))
        (unless (>= buf-size size)
          (error "get-workspace-buffer: buffer too small"
                 `((alloc-id ,alloc-id) (need ,size) (have ,buf-size))))
        (vector-ref buffers buf-id))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Lifetime Analysis
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (compute-last-uses! allocs-vec)
    "Forward scan: for each allocation i, extend last-use of each of its
    inputs to at least i.  Inputs with id -1 (external user arrays) are
    ignored -- their lifetime is managed by the caller."
    (let ((n (vector-length allocs-vec)))
      (do ((i 0 (+ i 1)))
          ((= i n))
        (for-each
         (lambda (inp-id)
           (when (>= inp-id 0)
             (let ((inp (vector-ref allocs-vec inp-id)))
               (when (> i (allocation-rec-last-use inp))
                 (allocation-rec-last-use-set! inp i)))))
         (allocation-rec-inputs (vector-ref allocs-vec i))))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Greedy Buffer Allocation (Interval Scheduling)
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; Process allocations in birth order.  For each dtype, maintain:
  ;;   active - list of (buf-id . end-time) for live buffers
  ;;   free   - list of (buf-id . max-size) for expired and reclaimable buffers
  ;;
  ;; Buffers are never shared across dtypes.

  ;; active alist helpers: (dtype -> list of (buf-id . end))
  (define (active-get active dtype)
    (let ((p (assq dtype active)))
      (if p (cdr p) '())))

  (define (active-set active dtype entries)
    (cons (cons dtype entries)
          (filter (lambda (p) (not (eq? (car p) dtype))) active)))

  ;; free alist helpers: (dtype -> list of (buf-id . max-size))
  (define (free-get free dtype)
    (let ((p (assq dtype free)))
      (if p (cdr p) '())))

  (define (free-set free dtype entries)
    (cons (cons dtype entries)
          (filter (lambda (p) (not (eq? (car p) dtype))) free)))

  (define (allocate-buffers allocs-vec n)
    "Greedy interval graph colouring by birth order, dtype-aware.
    Returns a buffer-pool record."
    (let ((assignment  (make-vector n -1))
          (active      '())    ; dtype -> list of (buf-id . end)
          (free        '())    ; dtype -> list of (buf-id . max-size)
          (buf-info    '())    ; growing list of (buf-id dtype max-size)
          (next-buf-id 0))

      (do ((i 0 (+ i 1)))
          ((= i n))

        (let* ((rec   (vector-ref allocs-vec i))
               (dtype (allocation-rec-dtype rec))
               (size  (allocation-rec-size  rec))
               (end   (allocation-rec-last-use rec)))

          ;; Expire buffers whose interval ended before i
          (let* ((dtype-active  (active-get active dtype))
                 (still-live    (filter (lambda (e) (>= (cdr e) i)) dtype-active))
                 (expired       (filter (lambda (e) (<  (cdr e) i)) dtype-active)))

            ;; Move expired entries into the free list
            (let ((dtype-free (free-get free dtype)))
              (for-each
               (lambda (e)
                 ;; e = (buf-id . end); find max-size from buf-info
                 (let* ((bid       (car e))
                        (max-size  (let ((r (find (lambda (x) (= (car x) bid))
                                                  buf-info)))
                                     (if r (caddr r) 0))))
                   (set! dtype-free (cons (cons bid max-size) dtype-free))))
               expired)
              (set! free (free-set free dtype dtype-free))
              (set! active (active-set active dtype still-live)))

            ;; Find a free same-dtype buffer large enough to hold this alloc
            (let* ((dtype-free (free-get free dtype))
                   (reusable   (find (lambda (e) (>= (cdr e) size))
                                     dtype-free)))

              (if reusable
                  ;; Reuse: remove from free, add to active
                  (let ((bid (car reusable)))
                    (vector-set! assignment i bid)
                    ;; Update max-size in buf-info if current size is larger
                    (set! buf-info
                          (map (lambda (x)
                                 (if (= (car x) bid)
                                     (list bid (cadr x) (max (caddr x) size))
                                     x))
                               buf-info))
                    (set! free   (free-set   free   dtype
                                             (filter (lambda (e) (not (= (car e) bid)))
                                                     dtype-free)))
                    (set! active (active-set active dtype
                                             (cons (cons bid end)
                                                   (active-get active dtype)))))

                  ;; No reusable buffer: allocate a new logical slot
                  (let ((bid next-buf-id))
                    (set! next-buf-id (+ bid 1))
                    (set! buf-info    (cons (list bid dtype size) buf-info))
                    (vector-set! assignment i bid)
                    (set! active (active-set active dtype
                                             (cons (cons bid end)
                                                   (active-get active dtype))))))))))

      ;; Build immutable pool from buf-info
      (let* ((num-bufs    next-buf-id)
             (sizes-vec   (make-vector num-bufs 0))
             (dtypes-vec  (make-vector num-bufs 'f64))
             (buffers-vec (make-vector num-bufs #f)))

        (for-each (lambda (info)
                    (let ((bid (car info)) (dtype (cadr info)) (sz (caddr info)))
                      (vector-set! sizes-vec  bid sz)
                      (vector-set! dtypes-vec bid dtype)))
                  buf-info)

        ;; Allocate physical typed vectors
        (do ((i 0 (+ i 1)))
            ((= i num-bufs))
          (vector-set! buffers-vec i
                       (allocate-typed-vector (vector-ref dtypes-vec i)
                                              (vector-ref sizes-vec  i))))

        (make-buffer-pool buffers-vec dtypes-vec assignment sizes-vec))))

) ;; end module
