# 0003: Explicit local loops and least squares only

Status: Accepted — requested by the author. Supersedes
[0001](0001-explicit-iteration-state.md).

## Context

State/context dataclasses and a runner replaced the original nonlocal FISTA
closure, but understanding the algorithm still required several abstractions.
They bundled local variables without providing enough depth to justify the
larger interface. Recurrent controller tensors also work as local variables.
The author requested the simpler iteration style and removal of TLS.

## Decision

Use explicit loops in `ISTA._iterate` and `FISTA._iterate_accelerated`, with
local x, z, tk. Remove the state/context records, runner, state-transition
hooks, and generic iteration drivers. A little duplicated loop control
improves locality and needs no compatibility adapter.

Share the actual mathematics, input preparation, validation, and callback
helpers. Retain the accelerated schedule interface for learned step parameters;
recurrent models can own a normal loop. These concrete seams provide leverage
without a speculative general controller framework.

Remove TLS entirely, including `tls_grad`, warnings, and the `ls_mode`
constructor argument. LS is the only data-fit objective; old mode keywords
raise TypeError rather than being silently accepted. Original unfolded
constructors will need to drop their obsolete ls_mode argument when ported.

## Consequences

- Preserve normalized LS operations, outputs, warm starts, batching,
  dtype/device, gradients, callbacks, and early stopping.
- Preserve `_iterate_accelerated`, preparation tuples, and the gradient,
  proximal, initialization, momentum, and callback helper interfaces.
- Newly introduced state-framework hooks and the generic `_run_iterations`
  interface are intentionally not retained.
- Autograd follows tensor operations; no state container is needed.
- The narrow public forward from [0002](0002-classical-forward-without-weights.md)
  stays unchanged. Weighted inputs belong to subclass-owned forwards.
- Tests cover algorithms and extension behavior, not removed record types.
- Pure-helper extraction into utils remains separate work.
