;;;
;;; Registry of whole-array kernels for element-wise activation functions.
;;;
;;; The SSA replay engine normally evaluates an element-wise unary op
;;; by calling its Scheme combiner once per element.  For the ops
;;; named in this registry it instead emits a ri-activation-unary
;;; replay instruction, which at execution time asks the active
;;; activation backend for a kernel that processes the whole array in
;;; one call, and falls back to the combiner loop when there is no
;;; backend or no suitable kernel.
;;;
;;; Two separate tables are kept here:
;;;
;;;   * The set of activation op names.  It decides which SSA bindings
;;;     are compiled to ri-activation-unary.  It holds relu, sigmoid,
;;;     tanh and the derivative ops that ssa-vjp emits for them.
;;;     Membership alone changes no results: without a registered
;;;     backend every ri-activation-unary runs the same combiner loop
;;;     as ri-flat-unary.
;;;
;;;   * The active activation backend: a named table mapping
;;;     (op . dtype) to a kernel procedure (size in out) -> void, where in
;;;     and out are SRFI-4 vectors of that dtype holding at least size
;;;     elements.  A kernel must be a pure per-element map, so in and out
;;;     may be the same vector.
;;;
;;; A backend can also supply kernels for three other kinds of work that
;;; the engine otherwise does with Scheme loops:
;;;
;;;   * Binary element-wise ops, (op . dtype) -> (size a b out), used by
;;;     ri-activation-binary for the op names registered with
;;;     register-binary-op!.  No binary op names are registered by
;;;     default, so plans compiled without such a backend are unchanged.
;;;
;;;   * Reductions of a row-major 2-D array over axis 0 or 1,
;;;     (rop axis dtype) -> (rows cols src out), where rop is sum, mean,
;;;     max or min.  A kernel must round exactly as the Scheme fast path
;;;     of execute-reduction-morphism does.
;;;
;;;   * Strided copies, dtype -> (src offset shape strides dst), which copy
;;;     an array of rank at most 4, described by its shape and element
;;;     strides (Scheme vectors) and the offset of its first element, into
;;;     dst in row-major order.
;;;
;;; This registry is separate from the blas-backend record of blas-exec.scm
;;; because the set of activation ops is open-ended and every kernel has the
;;; same calling convention, whereas blas-backend is a fixed record whose
;;; slots each have their own signature.
;;;
;;;   (import array-morphisms-activation-exec)
;;;   (define bkend (make-activation-backend 'example))
;;;   (activation-backend-add-kernel! bkend 'relu 'f64 my-relu-f64-kernel)
;;;   (register-activation-backend! bkend)

(module array-morphisms-activation-exec

  (;; Backend records
   make-activation-backend
   activation-backend?
   activation-backend-name
   activation-backend-add-kernel!
   activation-backend-kernels

   ;; Backend registration
   register-activation-backend!
   active-activation-backend
   lookup-activation-kernel

   ;; Activation op names
   register-activation-op!
   activation-op-registered?
   activation-ops

   ;; Binary element-wise kernels and op names
   activation-backend-add-binary-kernel!
   lookup-binary-kernel
   register-binary-op!
   unregister-binary-op!
   binary-op-registered?
   binary-ops

   ;; Broadcast binary kernels
   activation-backend-add-broadcast-kernel!
   lookup-broadcast-kernel

   ;; Reduction kernels
   activation-backend-add-reduction-kernel!
   lookup-reduction-kernel

   ;; Strided copy kernels
   activation-backend-add-copy-kernel!
   lookup-copy-kernel)

  (import scheme (chicken base))
  (import (only srfi-69
                make-hash-table hash-table-ref/default hash-table-set!
                hash-table-keys))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Backend records
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; table: hash table (op . dtype) -> kernel.  Keys are compared with
  ;; equal?, so they hash by content rather than by address.
  ;; binary-table, reduction-table and copy-table hold the binary,
  ;; reduction and strided-copy kernels in the same way.
  ;; broadcast-table holds the broadcast binary kernels.
  (define-record-type activation-backend
    (%make-activation-backend name table binary-table reduction-table copy-table
                              broadcast-table)
    activation-backend?
    (name            activation-backend-name)
    (table           activation-backend-table)
    (binary-table    activation-backend-binary-table)
    (reduction-table activation-backend-reduction-table)
    (copy-table      activation-backend-copy-table)
    (broadcast-table activation-backend-broadcast-table))

  (define (make-activation-backend name)
    "Create an activation backend called name (a symbol) with no kernels."
    (%make-activation-backend name (make-hash-table equal?) (make-hash-table equal?)
                              (make-hash-table equal?) (make-hash-table equal?)
                              (make-hash-table equal?)))

  (define (check-kernel who dtype kernel)
    (unless (memq dtype '(f32 f64))
      (error who "unsupported dtype" dtype))
    (unless (procedure? kernel)
      (error who "kernel is not a procedure" kernel)))

  (define (activation-backend-add-binary-kernel! bkend op dtype kernel)
    "Installs activation kernel, a procedure (size a b out) -> void computing
    out[i] = op(a[i], b[i]) for i below size, as the backend's
    implementation of the binary op on dtype arrays.  out may be a or b."
    (check-kernel 'activation-backend-add-binary-kernel! dtype kernel)
    (hash-table-set! (activation-backend-binary-table bkend) (cons op dtype) kernel))

  (define (lookup-binary-kernel bkend op dtype)
    "The binary kernel bkend holds for op on dtype arrays, or #f."
    (and bkend
         (hash-table-ref/default (activation-backend-binary-table bkend)
                                 (cons op dtype) #f)))

  (define (activation-backend-add-broadcast-kernel! bkend op dtype kernel)
    "Installs broadcast kernel, a procedure (rows cols a mode-a b mode-b out) -> void,
    as the backend's broadcast binary op (add, sub, mul or div) on dtype
    arrays.  out is a row-major rows x cols array receiving
    out[i, j] = op(a[.], b[.]); the mode of an operand says how it is
    indexed: 0, a rows x cols array (index i*cols + j); 1, one value per
    row (index i); 2, one value per column (index j); 3, a single value
    (index 0).  out is distinct from a and b."
    (check-kernel 'activation-backend-add-broadcast-kernel! dtype kernel)
    (unless (memq op '(add sub mul div))
      (error 'activation-backend-add-broadcast-kernel! "unsupported op" op))
    (hash-table-set! (activation-backend-broadcast-table bkend) (cons op dtype) kernel))

  (define (lookup-broadcast-kernel bkend op dtype)
    "The broadcast binary kernel bkend holds for op on dtype arrays, or #f."
    (and bkend
         (hash-table-ref/default (activation-backend-broadcast-table bkend)
                                 (cons op dtype) #f)))

  (define (activation-backend-add-reduction-kernel! bkend rop axis dtype kernel)
    "Installs reduction kernel, a procedure (rows cols src out) -> void, as the
    backend's reduction rop (sum, mean, max or min) over axis (0 or 1) of a
    row-major rows x cols dtype array src.  out receives cols results for
    axis 0 and rows results for axis 1."
    (check-kernel 'activation-backend-add-reduction-kernel! dtype kernel)
    (unless (memq rop '(sum mean max min))
      (error 'activation-backend-add-reduction-kernel! "unsupported reduction" rop))
    (unless (memv axis '(0 1))
      (error 'activation-backend-add-reduction-kernel! "axis must be 0 or 1" axis))
    (hash-table-set! (activation-backend-reduction-table bkend) (list rop axis dtype) kernel))

  (define (lookup-reduction-kernel bkend rop axis dtype)
    "The reduction kernel bkend holds for rop over axis on dtype arrays, or #f."
    (and bkend
         (hash-table-ref/default (activation-backend-reduction-table bkend)
                                 (list rop axis dtype) #f)))

  (define (activation-backend-add-copy-kernel! bkend dtype kernel)
    "Install copy kernel, a procedure (src offset shape strides dst) -> void,
    as the backend's strided copy of dtype arrays of rank at most 4 into
    row-major order."
    (check-kernel 'activation-backend-add-copy-kernel! dtype kernel)
    (hash-table-set! (activation-backend-copy-table bkend) dtype kernel))

  (define (lookup-copy-kernel bkend dtype)
    "The strided copy kernel bkend holds for dtype arrays, or #f."
    (and bkend
         (hash-table-ref/default (activation-backend-copy-table bkend) dtype #f)))

  (define (activation-backend-add-kernel! bkend op dtype kernel)
    "Installs kernel, a procedure (size in out) -> void, as the backend's
    implementation of op for arrays of dtype ('f32 or 'f64).  A later call
    for the same op and dtype replaces the earlier kernel."
    (unless (memq dtype '(f32 f64))
      (error "activation-backend-add-kernel!: unsupported dtype" dtype))
    (unless (procedure? kernel)
      (error "activation-backend-add-kernel!: kernel is not a procedure" kernel))
    (hash-table-set! (activation-backend-table bkend) (cons op dtype) kernel))

  (define (activation-backend-kernels bkend)
    "List of the (op . dtype) pairs the backend has kernels for."
    (hash-table-keys (activation-backend-table bkend)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Backend registration
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define *active-activation-backend* #f)

  (define (register-activation-backend! bkend)
    "Make bkend the active activation backend, replacing any earlier one.
    Passing #f removes the active backend, so that every activation op runs
    its Scheme combiner again."
    (unless (or (not bkend) (activation-backend? bkend))
      (error "register-activation-backend!: not an activation backend" bkend))
    (set! *active-activation-backend* bkend))

  (define (active-activation-backend)
    "The active activation backend, or #f when none is registered."
    *active-activation-backend*)

  (define (lookup-activation-kernel bkend op dtype)
    "The kernel bkend holds for op on dtype arrays, or #f."
    (and bkend
         (hash-table-ref/default (activation-backend-table bkend)
                                 (cons op dtype) #f)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Activation op names
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; The derivative ops are emitted by ssa-vjp in the backward pass:
  ;; relu-deriv maps the forward input x to 1 or 0, while sigmoid-deriv and
  ;; tanh-deriv map the forward output to s(1-s) and 1-t^2 respectively.
  (define *activation-ops*
    (list 'relu 'sigmoid 'tanh 'relu-deriv 'sigmoid-deriv 'tanh-deriv))

  (define (register-activation-op! op)
    "Add op (a symbol) to the activation op names.  SSA bindings with this
    op are then compiled to ri-activation-unary.  Adding a name twice has no
    further effect."
    (unless (symbol? op)
      (error "register-activation-op!: op must be a symbol" op))
    (unless (memq op *activation-ops*)
      (set! *activation-ops* (cons op *activation-ops*))))

  (define (activation-op-registered? op)
    "True when op is one of the activation op names."
    (and (memq op *activation-ops*) #t))

  (define (activation-ops)
    "List of the activation op names."
    *activation-ops*)

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Binary op names
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define *binary-ops* '())

  (define (register-binary-op! op)
    "Add op (a symbol) to the binary op names.  SSA bindings with this op
    and two row-major operands of the output's shape are then compiled to
    ri-activation-binary.  Adding a name twice has no further effect."
    (unless (symbol? op)
      (error "register-binary-op!: op must be a symbol" op))
    (unless (memq op *binary-ops*)
      (set! *binary-ops* (cons op *binary-ops*))))

  (define (unregister-binary-op! op)
    "Remove op from the binary op names, so that its bindings compile to
    ri-flat-binary again."
    (set! *binary-ops* (let loop ((ops *binary-ops*))
                         (cond ((null? ops) '())
                               ((eq? (car ops) op) (cdr ops))
                               (else (cons (car ops) (loop (cdr ops))))))))

  (define (binary-op-registered? op)
    "True when op is one of the binary op names."
    (and (memq op *binary-ops*) #t))

  (define (binary-ops)
    "List of the binary op names."
    *binary-ops*)

) ;; end module array-morphisms-activation-exec
