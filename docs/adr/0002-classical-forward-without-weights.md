# 0002: Keep independent weights out of classical forward

Status: Accepted — requested by the author. The tensor helper
`_setup_forward(..., W=...)` below is removed: since
[0007](0007-operators-instead-of-matrices.md) every model passes A and W as
operators. The public forward decision stands.

## Context

The migration exposed W in ISTA/FISTA forward to support later unfolded models.
An independent adjoint-side weight is not part of classical ISTA/FISTA and can
invalidate the stated LASSO objective and automatic step-size guarantee.

## Decision

The classical interface is
`forward(y, x0=None, t=1, A=None, lam=None, gamma=None)`.
Do not accept W, a catch-all keyword argument, or a compatibility fallback;
even W=None is rejected. FISTA inherits the same forward from ISTA.

A weighted unfolded subclass owns its forward and accepts or constructs its
weight explicitly. The protected `_setup_forward(..., W=...)`, its tuple
contract, and iteration hooks remain reusable. They validate weights and
preserve autograd without exposing weighted inputs on classical solvers.

Keep callback metadata compatible: W denotes the effective internal operator
and equals A for classical solves. Tests exercise weighted reuse through
subclasses that override forward, separately from classical solver tests.

## Consequences

- The public interface describes the classical algorithm, not a future model.
- Weighted models own their input semantics and step-size policy.
- Protected preparation, weighted iteration reuse, and callback behavior remain.
  [0003](0003-local-loops-and-ls-only.md) simplifies the iteration implementation
  to explicit local loops without changing this public forward decision.
- This intentionally narrows public compatibility from the initial migration;
  it does not remove the internal weighted-update capability.
- No unfolded production model is migrated in this change.
