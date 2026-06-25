;;; array-morphisms-morph-env.scm
;;;
;;; Environment abstraction for array-morphisms.
;;;
;;; Three layers, each building on the previous:
;;;
;;;   Layer 1  morph-env       -- immutable, GC-safe associative map
;;;                               keyed by symbols (gensyms survive minor GC)
;;;
;;;   Layer 2  env-builder     -- mutable cell wrapping an immutable morph-env
;;;                               plus a reverse-order accumulation list;
;;;                               provides the "functional core / imperative shell"
;;;                               pattern used throughout the project (YASOS style)
;;;
;;;   Layer 3  env-monad       -- combined State+Writer monad over morph-env,
;;;                               with env-do do-notation macro;
;;;                               for new code that benefits from compositional style
;;;


(module array-morphisms-morph-env

  (;; ---- Layer 1: immutable morph-env ----
   morph-env?
   empty-morph-env
   morph-env-extend
   morph-env-lookup
   morph-env-remove
   morph-env-fold
   morph-env->alist
   alist->morph-env
   morph-env-size
   morph-env-merge

   ;; ---- morph-env stacks ----
   morph-env-stack?
   empty-morph-env-stack
   morph-env-stack-push
   morph-env-stack-pop
   morph-env-stack-peek
   morph-env-stack-lookup
   morph-env-stack-extend

   ;; ---- Layer 2: env-builder ----
   make-env-builder
   env-builder-lookup
   env-builder-extend!
   env-builder-emit!
   env-builder-env
   env-builder-items
   env-builder-snapshot

   ;; ---- Layer 3: State+Writer monad ----
   env-return
   env-bind
   env-then
   env-lookup-m
   env-extend-m
   env-emit-m
   env-map-m
   env-run
   env-run*

   ;; ---- do-notation macro ----
   env-do)

  (import scheme (chicken base) datatype srfi-1)


  ;;; ==========================================================
  ;;; Layer 1: morph-env  -- immutable GC-safe associative map
  ;;;
  ;;; Keys must be symbols.  Symbols in CHICKEN are interned and their
  ;;; eq? identity is stable across GC moves; gensym values are unique.
  ;;;
  ;;; Representation: linked list of (key . val) pairs, newest first.
  ;;; O(N) lookup, O(1) extend.  Acceptable for env sizes in this code
  ;;; (typically O(N-SSA-bindings), < 1000 entries).
  ;;; ==========================================================

  (define-datatype morph-env morph-env?
    (MorphEnv  (binding pair?) (rest morph-env?))
    (EmptyMorphEnv))

  (define empty-morph-env (EmptyMorphEnv))

  (define (morph-env-extend env key val)
    "Return new env with (key . val) prepended.  key must be a symbol."
    (MorphEnv (cons key val) env))

  (define (morph-env-lookup env key)
    "Return the value bound to key, or #f if absent."
    (cases morph-env env
      (MorphEnv (binding rest)
        (if (eq? key (car binding))
            (cdr binding)
            (morph-env-lookup rest key)))
      (EmptyMorphEnv () #f)))

  (define (morph-env-remove env key)
    "Return env with the first binding for key removed."
    (cases morph-env env
      (MorphEnv (binding rest)
        (if (eq? key (car binding))
            rest
            (MorphEnv binding (morph-env-remove rest key))))
      (EmptyMorphEnv () empty-morph-env)))

  (define (morph-env-fold env proc init)
    "Fold over (key . val) pairs from newest to oldest.
     proc: (key . val) accumulator -> accumulator"
    (cases morph-env env
      (MorphEnv (binding rest)
        (proc binding (morph-env-fold rest proc init)))
      (EmptyMorphEnv () init)))

  (define (morph-env->alist env)
    "Convert env to a list of (key . val) pairs, newest first."
    (morph-env-fold env cons '()))

  (define (alist->morph-env alist)
    "Build env from a list of (key . val) pairs.
     Last pair in the list becomes the newest (innermost) binding."
    (fold (lambda (kv e) (morph-env-extend e (car kv) (cdr kv)))
          empty-morph-env
          alist))

  (define (morph-env-size env)
    "Return the number of bindings in env."
    (let loop ((e env) (n 0))
      (cases morph-env e
        (MorphEnv (_ rest) (loop rest (+ n 1)))
        (EmptyMorphEnv () n))))

  (define (morph-env-merge base override)
    "Return a new env that searches override bindings before base.
     Bindings in override shadow those in base for the same key."
    (morph-env-fold override
      (lambda (kv acc) (morph-env-extend acc (car kv) (cdr kv)))
      base))


  ;;; ==========================================================
  ;;; morph-env stacks  -- for scoped lookup across nested frames
  ;;; ==========================================================

  (define-datatype morph-env-stack morph-env-stack?
    (MorphEnvStack (top morph-env?) (rest morph-env-stack?))
    (EmptyMorphEnvStack))

  (define empty-morph-env-stack (EmptyMorphEnvStack))

  (define (morph-env-stack-push env stack)
    "Push a new env frame onto the stack."
    (MorphEnvStack env stack))

  (define (morph-env-stack-pop stack)
    "Remove the top frame; error on empty stack."
    (cases morph-env-stack stack
      (MorphEnvStack (_ rest) rest)
      (EmptyMorphEnvStack ()
        (error 'morph-env-stack-pop "empty environment stack"))))

  (define (morph-env-stack-peek stack)
    "Return the top frame without removing it; error on empty."
    (cases morph-env-stack stack
      (MorphEnvStack (top _) top)
      (EmptyMorphEnvStack ()
        (error 'morph-env-stack-peek "empty environment stack"))))

  (define (morph-env-stack-lookup key stack)
    "Search each frame from top to bottom; return value or #f."
    (cases morph-env-stack stack
      (MorphEnvStack (top rest)
        (or (morph-env-lookup top key)
            (morph-env-stack-lookup key rest)))
      (EmptyMorphEnvStack () #f)))

  (define (morph-env-stack-extend key val stack)
    "Extend the top frame with (key . val); error on empty stack."
    (cases morph-env-stack stack
      (MorphEnvStack (top rest)
        (MorphEnvStack (morph-env-extend top key val) rest))
      (EmptyMorphEnvStack ()
        (error 'morph-env-stack-extend "empty environment stack"))))


  ;;; ==========================================================
  ;;; Layer 2: env-builder  -- functional core / imperative shell
  ;;;
  ;;; Holds a mutable pointer to an immutable morph-env, plus a
  ;;; reverse-order list for O(1) accumulation (items).
  ;;;
  ;;; The immutable env is updated by rebinding the internal pointer;
  ;;; the env VALUE itself is never mutated.  This reconciles the need
  ;;; for convenient imperative-style code with referential transparency
  ;;; of the environment data structure.
  ;;;
  ;;; Pattern (replaces correlated hash-table + list mutations):
  ;;;
  ;;;   (let ((eb (make-env-builder)))
  ;;;     (env-builder-extend! eb 'key value)   ; update env
  ;;;     (env-builder-emit!   eb item)          ; accumulate item
  ;;;     (env-builder-lookup  eb 'key)          ; => value
  ;;;     (env-builder-items eb))                ; => (item ...)
  ;;; ==========================================================

  (define (make-env-builder #!optional (initial-env empty-morph-env))
    "Create an env-builder initialised with initial-env (default: empty).
     Returns a YASOS-style dispatch procedure."
    (let ((current-env initial-env)
          (rev-items   '()))
      (lambda (msg . args)
        (case msg
          ;; Lookup key; returns value or #f
          ((lookup)
           (morph-env-lookup current-env (car args)))
          ;; Extend: replace internal env pointer with new immutable env
          ((extend!)
           (set! current-env
                 (morph-env-extend current-env (car args) (cadr args))))
          ;; Emit: cons item onto reverse list in O(1); returns item
          ((emit!)
           (let ((item (car args)))
             (set! rev-items (cons item rev-items))
             item))
          ;; Snapshot: return (values env items-in-emission-order)
          ((snapshot)
           (values current-env (reverse rev-items)))
          ;; Read current env without items
          ((env)   current-env)
          ;; Read items in emission order
          ((items) (reverse rev-items))
          (else
           (error "env-builder: unknown message" msg))))))

  (define (env-builder-lookup  b key)     (b 'lookup key))
  (define (env-builder-extend! b key val) (b 'extend! key val))
  (define (env-builder-emit!   b item)    (b 'emit! item))
  (define (env-builder-env     b)         (b 'env))
  (define (env-builder-items   b)         (b 'items))
  (define (env-builder-snapshot b)        (b 'snapshot))


  ;;; ==========================================================
  ;;; Layer 3: State+Writer monad over morph-env
  ;;;
  ;;; "Thread" (monad state) = (morph-env . rev-items-list)
  ;;; A computation M a     = Thread -> (a . Thread)
  ;;;
  ;;; State component  : morph-env      (read/extend via env-lookup-m, env-extend-m)
  ;;; Writer component : rev-items-list (append via env-emit-m)
  ;;;
  ;;; Use env-do for do-notation; env-run to execute.
  ;;; ==========================================================

  (define (env-return v)
    "Lift v into the monad without changing state or emitting."
    (lambda (thread) (cons v thread)))

  (define (env-bind m f)
    "Monadic bind: run m, pass its value to f, sequence the states."
    (lambda (thread)
      (let* ((r  (m thread))
             (v  (car r))
             (t2 (cdr r)))
        ((f v) t2))))

  (define (env-then m1 m2)
    "Sequence m1 then m2, discarding m1's result value."
    (env-bind m1 (lambda (_) m2)))

  (define (env-lookup-m key)
    "Monadic lookup: return value bound to key, or #f; state unchanged."
    (lambda (thread)
      (cons (morph-env-lookup (car thread) key) thread)))

  (define (env-extend-m key val)
    "Monadic extend: prepend (key . val) to the current env."
    (lambda (thread)
      (cons (void)
            (cons (morph-env-extend (car thread) key val)
                  (cdr thread)))))

  (define (env-emit-m item)
    "Monadic emit: accumulate item in the writer list; returns item."
    (lambda (thread)
      (cons item
            (cons (car thread)
                  (cons item (cdr thread))))))

  (define (env-map-m f lst)
    "Map monadic function f over lst, threading state; returns list of results."
    (if (null? lst)
        (env-return '())
        (env-bind (f (car lst))
                  (lambda (v)
                    (env-bind (env-map-m f (cdr lst))
                              (lambda (vs)
                                (env-return (cons v vs))))))))

  (define (env-run m #!optional (initial-env empty-morph-env))
    "Execute monadic computation m.
     Returns (values result final-env emitted-items-in-emission-order)."
    (let* ((thread (cons initial-env '()))
           (result (m thread))
           (v      (car result))
           (final  (cdr result)))
      (values v (car final) (reverse (cdr final)))))

  (define (env-run* m #!optional (initial-env empty-morph-env))
    "Execute m; returns (values final-env emitted-items).
     Discards the computation's return value."
    (call-with-values
      (lambda () (env-run m initial-env))
      (lambda (v env items) (values env items))))


  ;;; ==========================================================
  ;;; env-do  -- Haskell-style do-notation for the env monad
  ;;;
  ;;;   (env-do (v <- m) rest ...)
  ;;;       bind: run m, bind result to v, continue with rest
  ;;;
  ;;;   (env-do m rest ...)
  ;;;       sequence: run m ignoring result, continue with rest
  ;;;
  ;;;   (env-do e)
  ;;;       terminal: e is the final monadic computation
  ;;;
  ;;; Example:
  ;;;   (env-run
  ;;;     (env-do
  ;;;       (env-extend-m 'x 42)
  ;;;       (v <- (env-lookup-m 'x))
  ;;;       (env-emit-m v)
  ;;;       (env-return (* v 2))))
  ;;;   => (values 84 #<env with x=42> (42))
  ;;; ==========================================================

  (define-syntax env-do
    (syntax-rules (<-)
      ;; Terminal: single expression
      ((_ e)
       e)
      ;; Bind: v <- m, then rest
      ((_ (v <- m) rest ...)
       (env-bind m (lambda (v) (env-do rest ...))))
      ;; Sequence: m then rest, result discarded
      ((_ m rest ...)
       (env-then m (env-do rest ...)))))

) ; end module array-morphisms-morph-env
