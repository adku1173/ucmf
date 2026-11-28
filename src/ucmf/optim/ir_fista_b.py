"""Safeguarded working-set FISTA for normalized signed/nonnegative LASSO.

This is an adaptation of Wang et al. (2022), DOI 10.1137/20M1374328, not a
literal implementation of Algorithm 5. It uses a best-candidate first-step
safeguard instead of the paper's spectral inner-decrease test. No Gram matrix
or matrix-free measurement dictionary is constructed.
"""

import math
from numbers import Integral, Real

from .ista import FISTA
from .operator import LinearOperator
from .utils import resolve_batch_param, validate_tensor

import torch


def _finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise RuntimeError('Nonfinite arithmetic in IRFISTAB; no KKT certificate available.')


def _new_counts():
    keys = ['restriction_fallbacks', 'batch_fallback_calls']
    for kind in ('forward', 'adjoint'):
        keys.extend(
            [
                f'full_{kind}',
                f'restricted_{kind}',
                f'fallback_{kind}',
                f'underlying_{kind}_rows',
                f'underlying_{kind}_elements',
            ]
        )
    return dict.fromkeys(keys, 0)


class _RowOperator(LinearOperator):
    """One independent row, optionally restricted, without inspecting operator data."""

    def __init__(self, operator, row, counts, columns=None):
        self.operator = operator
        self.counts = counts
        self.restricted = columns is not None
        self.row = row
        self.columns = None
        if columns is not None:
            select = getattr(operator, 'select_columns', None)
            if callable(select):
                self.operator = select(columns)
            else:
                self.columns = columns
                counts['restriction_fallbacks'] += 1
        self.shape = (operator.shape[0], operator.shape[1] if columns is None else columns.numel())
        self.batch_size = None
        self.dtype, self.device, self.rows = operator.dtype, operator.device, operator.rows
        if self.rows is not None and self.rows.ndim == 2:
            self.rows = self.rows[row]

    def _apply(self, value, adjoint):
        op = self.operator
        kind = 'adjoint' if adjoint else 'forward'
        self.counts[('restricted_' if self.restricted else 'full_') + kind] += 1
        self.counts['underlying_' + kind + '_rows'] += op.batch_size or 1
        self.counts['underlying_' + kind + '_elements'] += (op.batch_size or 1) * op.shape[1]
        if self.columns is not None:
            self.counts['fallback_' + kind] += 1
        if op.batch_size is not None:
            self.counts['batch_fallback_calls'] += 1
        if not adjoint and self.columns is not None:
            value = value.new_zeros((1, op.shape[1])).index_copy(1, self.columns, value)
        if op.batch_size is not None:
            full = value.new_zeros((op.batch_size, value.shape[1]))
            value = full.index_copy(0, value.new_tensor([self.row], dtype=torch.long), value)
        result = op.adjoint(value) if adjoint else op.forward(value)
        if op.batch_size is not None:
            result = result[self.row : self.row + 1]
        if adjoint and self.columns is not None:
            result = result.index_select(1, self.columns)
        return result

    def forward(self, x):
        """Apply the logical row product, using B-wide work when necessary."""
        return self._apply(x, False)

    def adjoint(self, r):
        """Apply the logical row adjoint, gathering restricted coordinates."""
        return self._apply(r, True)

    def detached(self):
        """Return self: the parent solver has already detached the operator."""
        return self


class IRFISTAB(FISTA):
    """Safeguarded iterative-reduction FISTA-B for ``||Ax-y||²/(2m)+lam*|x|₁``.

    Parameters
    ----------
    positive : bool, optional
        Nonnegative constraint (default True); False solves signed LASSO.
    callbacks : list, optional
        Full-coordinate begin/end callbacks once per outer sweep, not inner step.
    nonnegative : bool, optional
        Compatibility alias; when supplied overrides ``positive``.
    tol, rtol : float, optional
        Stop on full natural residual infinity norm <= tol + rtol * initial norm.
        Defaults 1e-6 and 0. This is not an objective-gap/error bound. Tiny positive
        coefficients can have small natural but large raw subgradient residuals.
    tau : float, optional
        Near-violator selection margin in [0,1), default 1e-4. Support is retained.
    delta : float or Tensor, optional
        Fixed normalized proximal curvature, scalar or per row. None means 1e-3/m
        (paper units 1e-3). Explicit values are NOT divided by m. Zero is allowed.
    inner_max_iter : int, optional
        Positive reduced-solve step budget, default 100.
    inner_tol : float, optional
        Relative subproblem natural-residual target in (0,1), default 0.1.
    backtracking_factor : float, optional
        Curvature multiplier >1, default 2. Inner curvature never decreases.
    max_backtracking : int, optional
        Positive trial budget per step, default 60; exhaustion raises RuntimeError.
    min_lipschitz : float, optional
        Positive curvature floor, default 1e-12.

    Notes
    -----
    Batch rows are solved independently and sequentially, on the input device.
    Native column restriction is preferred; generic operators use scatter/gather.
    Batched operators use zero-injection and can compute unused batch rows.
    No compile or differentiability promise across selection/stopping boundaries.
    Use no_grad for inference to avoid retaining iteration graphs.

    ``diagnostics`` is replaced each solve. Final tensors have shape (B,) and
    are detached; ``rows`` contains per-row histories. Status is determined only
    by the full certificate of the returned point, never by callback stopping.
    """

    def __init__(
        self,
        positive=True,
        callbacks=None,
        *,
        nonnegative=None,
        tol=1e-6,
        rtol=0.0,
        tau=1e-4,
        delta=None,
        inner_max_iter=100,
        inner_tol=0.1,
        backtracking_factor=2.0,
        max_backtracking=60,
        min_lipschitz=1e-12,
    ):
        super().__init__(nonnegative=positive if nonnegative is None else nonnegative, callbacks=callbacks)
        for name, value, lower, upper, strict in [
            ('tol', tol, 0, math.inf, False),
            ('rtol', rtol, 0, math.inf, False),
            ('tau', tau, 0, 1, False),
            ('inner_tol', inner_tol, 0, 1, True),
            ('backtracking_factor', backtracking_factor, 1, math.inf, True),
            ('min_lipschitz', min_lipschitz, 0, math.inf, True),
        ]:
            if (
                not isinstance(value, Real)
                or not math.isfinite(value)
                or value < lower
                or (strict and value == lower)
                or value >= upper
            ):
                raise ValueError(f'Invalid {name}.')
            setattr(self, name, float(value))
        for name, value in [('inner_max_iter', inner_max_iter), ('max_backtracking', max_backtracking)]:
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
            setattr(self, name, int(value))
        self.delta = delta
        self.diagnostics = {}

    def _parameter(self, value, name, y, *, positive=False):
        if torch.is_tensor(value):
            validate_tensor(value, name=name, dtype=y.dtype, device=y.device)
        value = resolve_batch_param(value, y.shape[0], name)
        if value is None:
            raise ValueError(f'{name} must be provided.')
        result = torch.as_tensor(value, dtype=y.dtype, device=y.device).expand(y.shape[0], 1)
        if (
            not bool(torch.isfinite(result).all())
            or bool((result < 0).any())
            or (positive and bool((result == 0).any()))
        ):
            raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}.')
        return result

    def _natural(self, x, grad, lam):
        # Mandatory cancellation-free identities, also used for the inner residual.
        if self.nonnegative:
            return torch.minimum(x, grad + lam)
        return torch.minimum(torch.maximum(x, grad - lam), grad + lam)

    def _certificate(self, x, y, op, lam):
        residual = op.forward(x) - y
        grad = op.adjoint(residual) / y.shape[1]
        natural = self._natural(x, grad, lam).abs().amax()
        if self.nonnegative:
            s = grad + lam
            raw = torch.where(x > 0, s.abs(), (-s).clamp_min(0)).amax()
        else:
            raw = torch.where(x != 0, (grad + lam * x.sign()).abs(), (grad.abs() - lam).clamp_min(0)).amax()
        objective = residual.square().sum() / (2 * y.shape[1]) + lam * x.abs().sum()
        _finite(x, grad, natural, raw, objective)
        return grad, natural, raw, objective.squeeze()

    def _working_set(self, x, grad, lam):
        violation = grad < -(1 - self.tau) * lam if self.nonnegative else grad.abs() > (1 - self.tau) * lam
        return torch.nonzero(((x != 0) | violation).flatten(), as_tuple=True)[0]

    def _reduced_solve(self, a, y, op, lam, delta, lipschitz, target):
        v, previous, momentum = a, a, 1.0
        best = best_phi = best_residual = None
        rejected = 0
        accepted_lipschitz = []
        first_phi = None
        shrink = self.get_shrink_func()
        for _j in range(self.inner_max_iter):
            grad = op.adjoint(op.forward(v) - y) / y.shape[1] + delta * (v - a)
            _finite(grad, v)
            for _ in range(self.max_backtracking):
                _finite(lipschitz)
                z = shrink(v - grad / lipschitz, lam / lipschitz)
                displacement = z - v
                # The quadratic inequality is homogeneous in D. Rescale both sides
                # identically to avoid accepting 0 <= 0 when tiny D**2 underflows.
                scale = displacement.abs().amax()
                direction = displacement / torch.where(scale > 0, scale, torch.ones_like(scale))
                image = op.forward(direction)
                norm2 = direction.square().sum()
                left = image.square().sum() / y.shape[1] + delta * norm2
                right = lipschitz * norm2
                _finite(z, left, right)
                slack = 32 * torch.finfo(a.dtype).eps * torch.maximum(left.abs(), right.abs())
                if bool(left <= right + slack):
                    break
                rejected += 1
                lipschitz = lipschitz * self.backtracking_factor
            else:
                raise RuntimeError('IRFISTAB backtracking budget exhausted; no accepted step.')
            residual = op.forward(z) - y
            phi = residual.square().sum() / (2 * y.shape[1]) + delta * (z - a).square().sum() / 2 + lam * z.abs().sum()
            grad_z = op.adjoint(residual) / y.shape[1] + delta * (z - a)
            inner_residual = self._natural(z, grad_z, lam).abs().amax()
            _finite(phi, grad_z, inner_residual)
            accepted_lipschitz.append(float(lipschitz.detach()))
            if first_phi is None:
                first_phi = float(phi.detach())
            if best is None or bool(phi < best_phi):
                best, best_phi, best_residual = z, phi, inner_residual
            hit = bool(inner_residual <= target)
            if hit:
                break
            momentum, v = self._nesterov_momentum(momentum, previous, z)
            previous = z
        info = {
            'first_phi': first_phi,
            'best_phi': float(best_phi.detach()),
            'accepted_lipschitz': accepted_lipschitz,
            'iterations': _j + 1,
            'backtracking_rejections': rejected,
            'target_met': hit,
            'best_target_met': bool(best_residual <= target),
            'budget_exhausted': not hit,
            'best_residual': float(best_residual.detach()),
            'lipschitz': float(lipschitz.detach()),
        }
        return best, lipschitz, info

    def forward(self, y, x0=None, t=100, operator=None, lam=None, gamma=None):
        """Return a full ``(B,n)`` solution with a final global KKT certificate.

        ``y`` is (m,) or (B,m); ``x0`` is (n,) or (B,n). All tensors and the
        operator must share float32/float64 dtype and device; no implicit casts.
        Positive warm starts are projected to x>=0. ``t`` counts OUTER sweeps,
        unlike FISTA's proximal steps. t=0 certifies the prepared start only.
        ``lam`` and ``gamma`` accept scalars or per-row (B,)/(B,1) tensors.
        Gamma is an initial inverse-curvature guess, not a fixed step; None
        starts L at max(1,min_lipschitz). Callback gamma/theta are (B,1) most
        recent accepted inverse curvatures and thresholds (initial for untouched
        rows). Callbacks see full A=W and y, once per sweep. Stops cannot certify
        unconverged rows. Nonfinite arithmetic or line-search exhaustion raises.
        """
        self.diagnostics = {}
        self._start_iterations()
        if not isinstance(t, Integral) or isinstance(t, bool) or t < 0:
            raise ValueError('t must be a nonnegative integer.')
        y, x, operator = self._prepare_operator_inputs(y, x0, operator)
        if y.dtype not in (torch.float32, torch.float64):
            raise ValueError('IRFISTAB supports float32 and float64 only.')
        if y.shape[0] == 0 or not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(x).all()):
            raise ValueError('y and x0 must be finite and batch nonempty.')
        x = x.clamp_min(0) if self.nonnegative else x.clone()
        lam = self._parameter(lam, 'lam', y)
        delta = self._parameter(1e-3 / y.shape[1] if self.delta is None else self.delta, 'delta', y)
        gamma = self._parameter(1.0 if gamma is None else gamma, 'gamma', y, positive=True)
        floor = y.new_tensor(self.min_lipschitz)
        lipschitz = (1 / gamma).clamp_min(floor)
        if not bool(torch.isfinite(lipschitz).all()) or not bool((lipschitz > 0).all()) or floor <= 0:
            raise ValueError('Initial curvature and min_lipschitz must be finite, positive and representable.')
        batch = y.shape[0]
        rows = [
            {
                'outer_iterations': 0,
                'inner_iterations': 0,
                'working_set_sizes': [],
                'inner': [],
                'counts': _new_counts(),
            }
            for _ in range(batch)
        ]
        full_ops = [_RowOperator(operator, b, rows[b]['counts']) for b in range(batch)]
        cert = [self._certificate(x[b : b + 1], y[b : b + 1], full_ops[b], lam[b]) for b in range(batch)]
        initial = torch.stack([c[1] for c in cert]).detach()
        thresholds = self.tol + self.rtol * initial
        _finite(thresholds)
        for k in range(t):
            if all(bool(c[1] <= thresholds[b]) for b, c in enumerate(cert)):
                break
            version = None if torch.is_inference(x) else x._version
            if not self._on_iteration_begin(k, x):
                break
            if self.callbacks and (version is None or x._version != version):
                # Callbacks may modify x in place. Inference tensors have no version
                # counter, so conservatively rescan after callbacks in that mode.
                x = x.clamp_min(0) if self.nonnegative else x
                cert = [self._certificate(x[b : b + 1], y[b : b + 1], full_ops[b], lam[b]) for b in range(batch)]
            updated = list(x.split(1))
            for b in range(batch):
                if bool(cert[b][1] <= thresholds[b]):
                    continue
                columns = self._working_set(updated[b], cert[b][0], lam[b])
                if columns.numel() == 0:
                    continue
                op = _RowOperator(operator, b, rows[b]['counts'], columns)
                L = lipschitz[b]
                if rows[b]['outer_iterations']:
                    L = (L / self.backtracking_factor).clamp_min(floor)
                best, L, info = self._reduced_solve(
                    updated[b].index_select(1, columns),
                    y[b : b + 1],
                    op,
                    lam[b],
                    delta[b],
                    L,
                    self.inner_tol * cert[b][1],
                )
                updated[b] = torch.zeros_like(updated[b]).index_copy(1, columns, best)
                lipschitz = lipschitz.index_copy(0, columns.new_tensor([b]), L.reshape(1, 1))
                rows[b]['outer_iterations'] += 1
                rows[b]['inner_iterations'] += info['iterations']
                rows[b]['working_set_sizes'].append(columns.numel())
                rows[b]['inner'].append(info)
                cert[b] = self._certificate(updated[b], y[b : b + 1], full_ops[b], lam[b])
            x = torch.cat(updated)
            version = None if torch.is_inference(x) else x._version
            if not self._on_iteration_end(x, 1 / lipschitz, lam / lipschitz, A=operator, W=operator, y=y):
                break
            if self.callbacks and (version is None or x._version != version):
                x = x.clamp_min(0) if self.nonnegative else x
                cert = [self._certificate(x[b : b + 1], y[b : b + 1], full_ops[b], lam[b]) for b in range(batch)]
        # Always certify the returned point, including t=0 and callback-stopped solves.
        x = x.clamp_min(0) if self.nonnegative else x
        cert = [self._certificate(x[b : b + 1], y[b : b + 1], full_ops[b], lam[b]) for b in range(batch)]
        self.diagnostics = {
            'counts': {key: sum(row['counts'][key] for row in rows) for key in _new_counts()},
            'status': [
                'converged' if bool(c[1] <= thresholds[b]) else 'callback_stopped' if self.stop else 'max_iter'
                for b, c in enumerate(cert)
            ],
            'natural_residual': torch.stack([c[1] for c in cert]).detach(),
            'raw_residual': torch.stack([c[2] for c in cert]).detach(),
            'objective': torch.stack([c[3] for c in cert]).detach(),
            'initial_residual': initial,
            'threshold': thresholds.detach(),
            'rows': rows,
        }
        return x
