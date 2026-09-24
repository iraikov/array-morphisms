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
   activation-ops)

  (import scheme (chicken base))
  (import (only srfi-69
                make-hash-table hash-table-ref/default hash-table-set!
                hash-table-keys))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Backend records
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; table: hash table (op . dtype) -> kernel.  Keys are compared with
  ;; equal?, so they hash by content rather than by address.
  (define-record-type activation-backend
    (%make-activation-backend name table)
    activation-backend?
    (name  activation-backend-name)
    (table activation-backend-table))

  (define (make-activation-backend name)
    "Create an activation backend called name (a symbol) with no kernels."
    (%make-activation-backend name (make-hash-table equal?)))

  (define (activation-backend-add-kernel! bkend op dtype kernel)
    "Install kernel, a procedure (size in out) -> void, as the backend's
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

) ;; end module array-morphisms-activation-exec
