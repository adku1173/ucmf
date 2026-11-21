# 0001: Explicit solve-local iteration state

Status: Superseded by [0003](0003-local-loops-and-ls-only.md).
The decision below is historical; state records and TLS support are removed.

## Context

FISTA captured its extrapolated point and Nesterov scalar inside a step
closure with `nonlocal`. This is valid, but understanding one iteration
required following both the closure and a separate callback loop. The source
unfolded solvers depend on the existing protected interface, especially
`_iterate_accelerated`, preparation tuples, and callback hooks.

## Decision

Deepen the iteration module around explicit solve-local `IterationState` and
`IterationContext` records. ISTA and FISTA share parameter selection and the
proximal step; FISTA specializes state initialization and its momentum update.
State transitions return new records and retain autograd graphs.

Keep the existing tensor-based `_run_iterations` interface. A private,
per-solve adapter advances state through it, preserving subclass overrides
and callback order. Both fixed and scheduled parameters use the same schedule
contract. Do not store momentum or controller tensors on the torch module.

The seam is parameter policy versus iteration, not a hypothetical universal
controller framework. Existing scheduled and recurrent inheritance styles
must remain usable; new hooks are optional for later migrations.

## Consequences

- Locality: the proximal step and state transition are visible together.
- Leverage: new parameter policies reuse iteration and callback behavior.
- Depth: the iteration module owns the changing state, not a hidden closure.
- Compatibility requires a small adapter and retains existing protected hooks.
- Existing callback attributes remain mutable; thread safety is not added.
- Preserve numerical operations, shapes, dtype/device, W/autograd semantics,
  and the warned legacy TLS behavior.
- Pure-helper extraction into `utils.py` and preparation redesign are deferred.

[0002](0002-classical-forward-without-weights.md) refines the public weight
interface: classical forward no longer accepts W; protected weighted reuse
remains available to subclass-owned forwards.

See [the architecture guide](../architecture.md) for the extension interface.
