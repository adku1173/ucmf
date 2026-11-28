"""Reviewed IR-FISTA-B: analytic and cancellation-free certification regressions."""

from ucmf import IRFISTAB
from ucmf.optim import IterationCallback, SolutionChangeStopper
from ucmf.optim.operator import CMFDictionary, CMFPropagation, LinearOperator, MatrixMult

import numpy as np
import pytest
import torch


@pytest.mark.parametrize('positive', [True, False])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_identity(positive, dtype):
    y = torch.tensor([1.0, -2.0, 0.01, -0.01], dtype=dtype)
    model = IRFISTAB(positive=positive, tol=1e-6 if dtype == torch.float32 else 1e-10)
    x = model(y, operator=MatrixMult(torch.eye(4, dtype=dtype)), lam=0.1, t=100)
    expected = (y - 0.4).clamp_min(0) if positive else y.sign() * (y.abs() - 0.4).clamp_min(0)
    torch.testing.assert_close(x[0], expected, atol=5e-6 if dtype == torch.float32 else 1e-9, rtol=0)
    assert model.diagnostics['status'] == ['converged']


@pytest.mark.parametrize('sign', [-1, 1])
def test_m1_large_signed_warm_start(sign):
    model = IRFISTAB(positive=False)
    x0 = torch.tensor([sign * 2.0**60], dtype=torch.float64)
    x = model(torch.zeros_like(x0), x0=x0, operator=MatrixMult(torch.zeros(1, 1, dtype=x0.dtype)), lam=1.0, t=0)
    assert x.item() == x0.item()
    assert model.diagnostics['natural_residual'].item() == 1
    assert model.diagnostics['raw_residual'].item() == 1
    assert model.diagnostics['status'] == ['max_iter']


def test_projection_alias_zero_operator_and_zero_budget():
    model = IRFISTAB(positive=False, nonnegative=True)
    x0 = torch.tensor([-2.0, 3.0], dtype=torch.float64)
    x = model(
        torch.zeros(2, dtype=x0.dtype), x0=x0, operator=MatrixMult(torch.zeros(2, 2, dtype=x0.dtype)), lam=0.0, t=0
    )
    torch.testing.assert_close(x, torch.tensor([[0.0, 3.0]], dtype=x0.dtype))
    assert model.diagnostics['status'] == ['converged']
    assert x0[0] == -2


@pytest.mark.parametrize('positive', [True, False])
def test_lambda_max(positive):
    A = torch.tensor([[1.0, 2.0], [-3.0, 1.0]], dtype=torch.float64)
    y = torch.tensor([1.0, 2.0], dtype=A.dtype)
    c = A.T @ y / 2
    lam = c.max().clamp_min(0) if positive else c.abs().max()
    model = IRFISTAB(positive=positive)
    x = model(y, operator=MatrixMult(A), lam=lam)
    assert torch.count_nonzero(x) == 0
    assert model.diagnostics['status'] == ['converged']


@pytest.mark.parametrize('positive', [True, False])
@pytest.mark.parametrize('delta', [0.0, 0.001, 2.0])
@pytest.mark.parametrize('anchor', [0.0, 0.1])
def test_delta_scalar_d2_and_safeguard(positive, delta, anchor):
    # D2: literal paper (2.22) asks for decrease >16, though only .125 is available.
    model = IRFISTAB(positive=positive, delta=delta, inner_tol=1e-10, inner_max_iter=500, tol=1e-12)
    A = torch.tensor([[2.0]], dtype=torch.float64)
    y = torch.tensor([1.0], dtype=A.dtype)
    a = torch.tensor([anchor], dtype=A.dtype)
    x = model(y, x0=a, operator=MatrixMult(A), lam=1.0, gamma=100.0, t=1)
    torch.testing.assert_close(x.flatten(), (1 + delta * a) / (4 + delta), atol=1e-8, rtol=0)
    info = model.diagnostics['rows'][0]['inner'][0]
    assert info['backtracking_rejections'] > 0
    assert info['best_phi'] <= info['first_phi']
    assert info['accepted_lipschitz'] == sorted(info['accepted_lipschitz'])
    assert info['target_met'] != info['budget_exhausted']


def test_inner_m1_never_false_target_and_budget():
    model = IRFISTAB(positive=False, inner_max_iter=1)
    model(
        torch.zeros(1, dtype=torch.float64),
        x0=torch.tensor([2.0**60], dtype=torch.float64),
        operator=MatrixMult(torch.zeros(1, 1, dtype=torch.float64)),
        lam=1.0,
        t=1,
    )
    info = model.diagnostics['rows'][0]['inner'][0]
    assert not info['target_met'] and not info['best_target_met']
    assert info['budget_exhausted'] and info['best_residual'] == 1.0
    assert model.diagnostics['status'] == ['max_iter']


def test_backtracking_exhaustion_and_no_tiny_energy_slack():
    model = IRFISTAB(max_backtracking=1, delta=0, min_lipschitz=1e-30, tol=0)
    with pytest.raises(RuntimeError, match='backtracking budget'):
        model(
            torch.tensor([1e-20], dtype=torch.float64),
            operator=MatrixMult(torch.ones(1, 1, dtype=torch.float64)),
            lam=0.0,
            gamma=1e20,
            t=1,
        )
    assert model.diagnostics == {}


class CountingFallback(LinearOperator):
    def __init__(self, op, calls=None):
        self.op = op
        self.calls = [] if calls is None else calls
        self.shape, self.batch_size = op.shape, op.batch_size
        self.dtype, self.device, self.rows = op.dtype, op.device, op.rows

    def forward(self, x):
        self.calls.append(('forward', self.shape[1], x.shape[0]))
        return self.op.forward(x)

    def adjoint(self, r):
        self.calls.append(('adjoint', self.shape[1], r.shape[0]))
        return self.op.adjoint(r)

    def detached(self):
        return type(self)(self.op.detached(), self.calls)


class CountingNative(CountingFallback):
    def select_columns(self, columns):
        return type(self)(self.op.select_columns(columns), self.calls)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('wrapper', [CountingNative, CountingFallback])
def test_batch_independence_and_actual_counts(batched, wrapper):
    rng = torch.Generator().manual_seed(43)
    A = torch.randn((3, 7, 5) if batched else (7, 5), dtype=torch.float64, generator=rng)
    y = torch.randn(3, 7, dtype=A.dtype, generator=rng)
    y[0] = 0
    x0 = torch.randn(3, 5, dtype=A.dtype, generator=rng)
    x0[0] = 0
    lam = torch.tensor([0.1, 0.02, 0.2], dtype=A.dtype)
    gamma = torch.tensor([1.0, 2.0, 0.2], dtype=A.dtype)
    delta = torch.tensor([0.0, 0.001, 0.1], dtype=A.dtype)
    op = wrapper(MatrixMult(A))
    model = IRFISTAB(delta=delta, tol=1e-9)
    x = model(y, x0=x0, operator=op, lam=lam, gamma=gamma, t=100)
    for b in range(3):
        single = IRFISTAB(delta=delta[b], tol=1e-9)
        expected = single(
            y[b], x0=x0[b], operator=wrapper(MatrixMult(A[b] if batched else A)), lam=lam[b], gamma=gamma[b], t=100
        )
        torch.testing.assert_close(x[b : b + 1], expected, atol=1e-12, rtol=1e-12)
    counts = model.diagnostics['counts']
    for kind in ['forward', 'adjoint']:
        calls = [c for c in op.calls if c[0] == kind]
        assert counts['full_' + kind] + counts['restricted_' + kind] == len(calls)
        assert counts['underlying_' + kind + '_rows'] == sum(c[2] for c in calls)
        assert counts['underlying_' + kind + '_elements'] == sum(c[1] * c[2] for c in calls)
    assert counts['restriction_fallbacks'] > 0 if wrapper is CountingFallback else counts['restriction_fallbacks'] == 0
    assert counts['batch_fallback_calls'] == (len(op.calls) if batched else 0)
    assert model.diagnostics['rows'][0]['outer_iterations'] == 0
    assert model.diagnostics['status'] == ['converged'] * 3


class Trace(IterationCallback):
    def __init__(self, stop=None):
        self.begin, self.end, self.params = [], [], []
        self.stop_at = stop

    def on_step_begin(self, solver, x, **kwargs):
        if solver.iter == 0:
            self.begin, self.end, self.params = [], [], []
        self.begin.append(x.detach().clone())
        if self.stop_at == 'begin':
            solver.stop = True

    def on_step_end(self, solver, x, **kwargs):
        self.end.append(x.detach().clone())
        self.params.append(kwargs)
        if self.stop_at == 'end':
            solver.stop = True


@pytest.mark.parametrize('stop', ['begin', 'end', None])
def test_callbacks_and_repeated_solve(stop):
    cb = Trace(stop)
    model = IRFISTAB(callbacks=[cb], inner_max_iter=1, tol=0)
    y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    op = MatrixMult(torch.eye(2, dtype=y.dtype))
    x = model(y, operator=op, lam=0.1, gamma=0.01, t=2)
    first = model.diagnostics
    assert all(v.shape == (2, 2) for v in cb.begin + cb.end)
    for params in cb.params:
        assert params['gamma'].shape == params['theta'].shape == (2, 1)
        assert params['A'] is params['W']
        torch.testing.assert_close(params['theta'], 0.1 * params['gamma'])
    assert first['status'] == ['callback_stopped' if stop else 'max_iter'] * 2
    repeated = model(y, operator=op, lam=0.1, gamma=0.01, t=2)
    torch.testing.assert_close(x, repeated, rtol=0, atol=0)
    assert first['counts'] == model.diagnostics['counts']
    model(y, operator=op, lam=0.1, t=0)
    assert not model.stop
    assert model.diagnostics['rows'][0]['outer_iterations'] == 0


def test_solution_change_stopper_does_not_claim_kkt():
    stopper = SolutionChangeStopper(tol=100, check_every=1)
    model = IRFISTAB(callbacks=[stopper], inner_max_iter=1, tol=0)
    model(torch.ones(2), operator=MatrixMult(torch.eye(2)), lam=0.1, gamma=0.01)
    assert model.diagnostics['status'] == ['callback_stopped']


def test_subnormal_displacement_does_not_bypass_backtracking():
    model = IRFISTAB(max_backtracking=1, inner_max_iter=1, delta=0, tol=0)
    with pytest.raises(RuntimeError, match='backtracking budget'):
        model(
            torch.tensor([1e-30], dtype=torch.float32),
            operator=MatrixMult(torch.tensor([[2.0]], dtype=torch.float32)),
            lam=0.0,
            gamma=1.0,
            t=1,
        )


@pytest.mark.parametrize(
    'kw',
    [
        {'tol': -1},
        {'rtol': float('nan')},
        {'tau': 1},
        {'tau': -0.1},
        {'inner_tol': 0},
        {'inner_tol': 1},
        {'backtracking_factor': 1},
        {'min_lipschitz': 0},
        {'inner_max_iter': 0},
        {'inner_max_iter': 1.5},
        {'max_backtracking': False},
    ],
)
def test_invalid_constructor(kw):
    with pytest.raises(ValueError):
        IRFISTAB(**kw)


@pytest.mark.parametrize(
    'kw',
    [
        {'lam': -1},
        {'lam': float('inf')},
        {'lam': None},
        {'gamma': 0},
        {'gamma': float('nan')},
        {'gamma': 1e-40},
        {'t': 1.2},
        {'t': -1},
        {'t': True},
        {'x0': torch.tensor([float('inf')])},
        {'y': torch.tensor([float('nan')])},
        {'lam': torch.tensor(0.1, dtype=torch.float64)},
        {'x0': torch.ones(1, dtype=torch.float64)},
        {'gamma': torch.ones(3)},
        {'operator': MatrixMult(torch.ones(1, 1, dtype=torch.float64))},
    ],
)
def test_invalid_forward_even_at_t0(kw):
    inputs = {'y': torch.ones(1), 'operator': MatrixMult(torch.ones(1, 1)), 'lam': 0.1, 't': 0}
    inputs.update(kw)
    with pytest.raises(ValueError):
        IRFISTAB()(**inputs)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_rejected(dtype):
    with pytest.raises(ValueError, match='float32 and float64'):
        IRFISTAB()(torch.ones(1, dtype=dtype), operator=MatrixMult(torch.ones(1, 1, dtype=dtype)), lam=0, t=0)


@pytest.mark.parametrize('delta', [-1, float('nan'), torch.tensor(0.1, dtype=torch.float64)])
def test_invalid_delta(delta):
    with pytest.raises(ValueError):
        IRFISTAB(delta=delta)(torch.ones(1), operator=MatrixMult(torch.ones(1, 1)), lam=0, t=0)


def test_nonfinite_operator_raises_not_certifies():
    with pytest.raises(RuntimeError, match='Nonfinite'):
        IRFISTAB()(torch.ones(1), operator=MatrixMult(torch.tensor([[float('inf')]])), lam=0, t=0)


def oracle(A, y, x, lam, positive):
    residual = A @ x - y
    grad = A.T @ residual / len(y)
    if positive:
        natural = np.minimum(x, grad + lam)
        raw = np.where(x > 0, np.abs(grad + lam), np.maximum(-grad - lam, 0))
    else:
        # Independent scalar cases (not the solver's clamp expression).
        natural = np.array(
            [g + lam if z > g + lam else g - lam if z < g - lam else z for z, g in zip(x, grad, strict=True)]
        )
        raw = np.where(x != 0, np.abs(grad + lam * np.sign(x)), np.maximum(np.abs(grad) - lam, 0))
    return (residual @ residual / (2 * len(y)) + lam * np.abs(x).sum(), np.abs(natural).max(), raw.max())


def numpy_fista(A, y, lam, positive, tol=1e-10):
    L = np.linalg.norm(A, 2) ** 2 / len(y)
    x = np.zeros(A.shape[1])
    v, theta = x.copy(), 1.0
    for k in range(100000):
        w = v - A.T @ (A @ v - y) / (len(y) * L)
        z = np.maximum(w - lam / L, 0) if positive else np.sign(w) * np.maximum(np.abs(w) - lam / L, 0)
        theta_next = (1 + np.sqrt(1 + 4 * theta**2)) / 2
        v = z + (theta - 1) / theta_next * (z - x)
        x, theta = z, theta_next
        if k % 25 == 0 and oracle(A, y, x, lam, positive)[1] <= tol:
            return x
    pytest.fail('Independent reference not certified in its finite budget')


@pytest.mark.parametrize('positive', [True, False])
@pytest.mark.parametrize('family', ['random', 'coherent', 'rank_deficient'])
@pytest.mark.parametrize('support', [2, 8])
@pytest.mark.parametrize('warm', [False, True])
def test_random_and_challenging_references(positive, family, support, warm):
    rng = np.random.default_rng(196)
    A = rng.normal(size=(18, 8))
    if family == 'coherent':
        A = 0.99 * A[:, :1] + 0.1 * A
    elif family == 'rank_deficient':
        A[:, 4:] = A[:, :4]
        A[:, -1] = 0
    truth = np.zeros(8)
    truth[:support] = rng.uniform(0.5, 1.5, support) * (1 if positive else rng.choice([-1, 1], support))
    y = A @ truth + 0.01 * rng.normal(size=18)
    lam = 0.05
    ref = numpy_fista(A, y, lam, positive)
    cb = Trace()
    model = IRFISTAB(positive=positive, tol=1e-8, callbacks=[cb])
    x0 = torch.from_numpy(rng.normal(size=8)) if warm else None
    x = model(torch.from_numpy(y), x0=x0, operator=MatrixMult(torch.from_numpy(A)), lam=lam, t=300)
    obj, natural, raw = oracle(A, y, x[0].numpy(), lam, positive)
    assert model.diagnostics['status'] == ['converged']
    assert natural <= 1e-8 and raw <= 1e-7
    assert abs(obj - oracle(A, y, ref, lam, positive)[0]) < 2e-8
    np.testing.assert_allclose(model.diagnostics['natural_residual'].numpy(), natural, atol=1e-14)
    np.testing.assert_allclose(model.diagnostics['raw_residual'].numpy(), raw, atol=1e-14)
    values = [oracle(A, y, p[0].numpy(), lam, positive)[0] for p in cb.begin[:1] + cb.end]
    assert np.all(np.diff(values) <= 1e-13 * max(1, values[0]))
    for before, size in zip(cb.begin, model.diagnostics['rows'][0]['working_set_sizes'], strict=True):
        b = before[0].numpy()
        g = A.T @ (A @ b - y) / len(y)
        violation = g < -(1 - model.tau) * lam if positive else np.abs(g) > (1 - model.tau) * lam
        assert size == np.count_nonzero((b != 0) | violation)
    if positive:
        assert x.min() >= 0


@pytest.mark.parametrize('rank_deficient', [False, True])
def test_lambda_zero_nnls(rank_deficient):
    from scipy.optimize import nnls

    rng = np.random.default_rng(19)
    A = rng.normal(size=(20, 8))
    if rank_deficient:
        A[:, 4:] = A[:, :4]
    y = rng.normal(size=20)
    ref, _ = nnls(A, y)
    model = IRFISTAB(tol=1e-9)
    x = model(torch.from_numpy(y), operator=MatrixMult(torch.from_numpy(A)), lam=0.0, t=200)[0].numpy()
    obj, natural, raw = oracle(A, y, x, 0.0, True)
    assert natural <= 1e-9 and raw <= 1e-8
    assert abs(obj - oracle(A, y, ref, 0.0, True)[0]) < 1e-10


def test_reactivation_review_example():
    A = torch.tensor([[1.0, -1.0], [0.0, 1.0]], dtype=torch.float64)
    y = torch.ones(2, dtype=A.dtype)
    cb = Trace()
    model = IRFISTAB(tau=0, delta=0, tol=1e-9, callbacks=[cb])
    x = model(y, operator=MatrixMult(A), lam=0.1, t=100)
    assert model.diagnostics['rows'][0]['working_set_sizes'][:2] == [1, 2]
    assert cb.end[0][0, 1] == 0 and cb.end[1][0, 1] > 0
    torch.testing.assert_close(x, torch.tensor([[1.4, 0.6]], dtype=A.dtype), atol=1e-8, rtol=0)


@pytest.mark.parametrize('positive', [True, False])
def test_measurement_stacking_and_scaling(positive):
    rng = torch.Generator().manual_seed(35)
    A = torch.randn(11, 6, generator=rng, dtype=torch.float64)
    y = torch.randn(11, generator=rng, dtype=A.dtype)
    solutions = []
    for matrix, data, lam, delta, gamma in [
        (A, y, 0.03, 0.001, 1.0),
        (A.repeat(2, 1), y.repeat(2), 0.03, 0.001, 1.0),
        (3 * A, 3 * y, 0.27, 0.009, 1 / 9),
    ]:
        model = IRFISTAB(positive=positive, delta=delta, tol=1e-8)
        solutions.append(model(data, operator=MatrixMult(matrix), lam=lam, gamma=gamma, t=100))
    for solution in solutions[1:]:
        torch.testing.assert_close(solution, solutions[0], atol=1e-7, rtol=0)
    default = IRFISTAB(tol=0)
    explicit = IRFISTAB(delta=0.001 / len(y), tol=0)
    torch.testing.assert_close(
        default(y, operator=MatrixMult(A), lam=0.03, t=2),
        explicit(y, operator=MatrixMult(A), lam=0.03, t=2),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('positive', [False, True])
def test_cmf_matrix_free_equivalence(batched, positive):
    rng = torch.Generator().manual_seed(71)
    H = torch.randn((2, 4, 6) if batched else (4, 6), generator=rng, dtype=torch.complex128)
    dense = CMFDictionary(H, r_diag=True)
    propagation = CMFPropagation(H, r_diag=True)
    y = torch.randn(2, 12, generator=rng, dtype=torch.float64)
    values = []
    for op in [dense, propagation, CountingFallback(propagation)]:
        model = IRFISTAB(positive=positive, tol=1e-8)
        x = model(y, operator=op, lam=0.05, t=200)
        assert model.diagnostics['status'] == ['converged'] * 2
        r = dense.forward(x) - y
        values.append(r.square().sum(1) / 24 + 0.05 * x.abs().sum(1))
    for val in values[1:]:
        torch.testing.assert_close(val, values[0], atol=1e-9, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cuda_and_strict_device(dtype):
    y = torch.tensor([1.0, -2.0], device='cuda', dtype=dtype)
    op = MatrixMult(torch.eye(2, device='cuda', dtype=dtype))
    model = IRFISTAB(positive=False, tol=1e-6)
    x = model(y, operator=op, lam=0.1)
    torch.testing.assert_close(x[0], torch.tensor([0.8, -1.8], device='cuda', dtype=dtype), atol=3e-6, rtol=0)
    with pytest.raises(ValueError, match='device'):
        model(y, operator=op, lam=torch.tensor(0.1, dtype=dtype, device='cpu'))


def test_no_gram_or_dictionary_construction():
    import inspect

    import ucmf.optim.ir_fista_b as module

    source = inspect.getsource(module)
    for forbidden in ['torch.eye(', '.matrix', '.H', 'CMFDictionary(', 'torch.linalg.eig', 'torch.linalg.svd', ' @ ']:
        assert forbidden not in source
    # Runtime matrix-free test: reject dictionary/Gram-shaped allocations, including eye probes.
    from torch.utils._python_dispatch import TorchDispatchMode

    class Guard(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            tensors = result if isinstance(result, (tuple, list)) else [result]
            for value in tensors:
                if torch.is_tensor(value) and value.ndim >= 2:
                    assert tuple(value.shape[-2:]) not in [(31, 31), (12, 31), (31, 12)]
            return result

    H = torch.randn(4, 31, dtype=torch.complex128, generator=torch.Generator().manual_seed(4))
    with Guard():
        IRFISTAB()(torch.ones(12, dtype=torch.float64), operator=CMFPropagation(H, r_diag=True), lam=0.1, t=2)


def test_best_candidate_including_first_not_last_and_exact_ties():
    # A diagonal problem with acceleration overshoot: independent iterate replay.
    A = np.diag([1.0, 0.1])
    y = np.array([1.0, 1.0])
    a = np.zeros(2)
    lam, delta, L = 0.001, 0.003, 1.0
    z, v, theta = a.copy(), a.copy(), 1.0
    candidates, objectives = [], []
    for _ in range(50):
        grad = A.T @ (A @ v - y) / 2 + delta * (v - a)
        next_z = np.maximum(v - grad / L - lam / L, 0)
        next_theta = (1 + np.sqrt(1 + 4 * theta**2)) / 2
        v = next_z + (theta - 1) / next_theta * (next_z - z)
        z, theta = next_z, next_theta
        candidates.append(z)
        objectives.append(oracle(A, y, z, lam, True)[0] + delta * np.sum((z - a) ** 2) / 2)
    model = IRFISTAB(delta=delta, inner_max_iter=50, inner_tol=1e-15, tol=0)
    x = model(torch.from_numpy(y), operator=MatrixMult(torch.from_numpy(A)), lam=lam, gamma=1 / L, t=1)
    best_index = int(np.argmin(objectives))
    assert best_index < len(candidates) - 1  # deliberately not the last accelerated candidate
    np.testing.assert_allclose(x[0].numpy(), candidates[best_index], atol=1e-12)
    info = model.diagnostics['rows'][0]['inner'][0]
    assert info['iterations'] == 50 and info['budget_exhausted']
    # M1 has exact objective ties at every step: retain first, still not converged.
    huge = IRFISTAB(positive=False, inner_max_iter=3)
    huge(
        torch.zeros(1, dtype=torch.float64),
        x0=torch.tensor([2.0**60], dtype=torch.float64),
        operator=MatrixMult(torch.zeros(1, 1, dtype=torch.float64)),
        lam=1.0,
        t=1,
    )
    assert huge.diagnostics['rows'][0]['inner'][0]['best_residual'] == 1


def test_natural_vs_raw_tiny_support_and_relative_initial_scale():
    model = IRFISTAB(tol=0, rtol=0.5)
    model(torch.zeros(1), x0=torch.tensor([1e-10]), operator=MatrixMult(torch.zeros(1, 1)), lam=1.0, t=0)
    assert model.diagnostics['natural_residual'].item() == pytest.approx(1e-10)
    assert model.diagnostics['raw_residual'].item() == 1
    assert model.diagnostics['threshold'].item() == pytest.approx(5e-11)
    assert model.diagnostics['status'] == ['max_iter']


def test_autograd_operator_detached():
    A = torch.eye(2, dtype=torch.float64, requires_grad=True)
    y = torch.ones(2, dtype=torch.float64, requires_grad=True)
    IRFISTAB(delta=0)(y, operator=MatrixMult(A), lam=0.1, gamma=2.0, t=1).sum().backward()
    assert A.grad is None
    torch.testing.assert_close(y.grad, torch.ones_like(y))


def test_benchmark_harness_counts_and_independent_oracle():
    import runpy
    from pathlib import Path

    harness = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'examples/benchmark_ir_fista_b.py'))
    harness['check_harness']()
    A = torch.tensor([[1.0, 2.0], [-3.0, 1.0]], dtype=torch.float64)
    y = torch.tensor([[1.0, 2.0]], dtype=A.dtype)
    x = torch.tensor([[0.2, -0.1]], dtype=A.dtype)
    result = harness['evaluate'](MatrixMult(A), y, x, 0.1, False)
    expected = oracle(A.numpy(), y[0].numpy(), x[0].numpy(), 0.1, False)
    for key, val in zip(['objective', 'natural', 'raw'], expected, strict=True):
        assert result[key].item() == pytest.approx(val)


@pytest.mark.parametrize('positive', [True, False])
def test_support_retention_exact_selection_and_no_empty_restriction(positive):
    model = IRFISTAB(positive=positive, tau=0.1)
    x = torch.tensor([[1e-200, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    g = torch.tensor([[100.0, -0.95, 0.95, -0.9, 0.0]], dtype=x.dtype)
    expected = [0, 1] if positive else [0, 1, 2]
    assert model._working_set(x, g, x.new_tensor(1.0)).tolist() == expected
    op = CountingNative(MatrixMult(torch.eye(5, dtype=x.dtype)))
    model(torch.zeros(5, dtype=x.dtype), operator=op, lam=0.0)
    assert model.diagnostics['counts']['restricted_forward'] == 0
    assert model.diagnostics['counts']['full_forward'] == 2
    assert model.diagnostics['counts']['full_adjoint'] == 2


def test_final_full_certification_counts():
    model = IRFISTAB(inner_max_iter=1, tol=0)
    model(torch.ones(2), operator=MatrixMult(torch.eye(2)), lam=0.1, gamma=0.01, t=3)
    row = model.diagnostics['rows'][0]
    assert row['outer_iterations'] == 3
    assert row['counts']['full_forward'] == row['counts']['full_adjoint'] == 5
    # Per accepted inner step: gradient F+A, trial F, candidate F+A. All trials counted.
    rejected = sum(info['backtracking_rejections'] for info in row['inner'])
    assert row['counts']['restricted_forward'] == 3 * row['inner_iterations'] + rejected
    assert row['counts']['restricted_adjoint'] == 2 * row['inner_iterations']


def test_callback_mutation_is_recertified():
    class Modify(IterationCallback):
        def on_step_begin(self, solver, x, **kw):
            x.fill_(2)

        def on_step_end(self, solver, x, **kw):
            x.fill_(3)
            solver.stop = True

    model = IRFISTAB(callbacks=[Modify()], delta=0)
    x = model(torch.ones(2), operator=MatrixMult(torch.eye(2)), lam=0.1, t=1)
    assert torch.all(x == 3)
    assert model.diagnostics['status'] == ['callback_stopped']
    torch.testing.assert_close(model.diagnostics['natural_residual'], torch.tensor([1.1]))


def test_inference_mode_with_callbacks():
    with torch.inference_mode():
        model = IRFISTAB(delta=0, callbacks=[Trace()])
        x = model(torch.ones(2), operator=MatrixMult(torch.eye(2)), lam=0.1, gamma=2.0)
        torch.testing.assert_close(x, torch.full((1, 2), 0.8))
        assert model.diagnostics['status'] == ['converged']


def test_signed_lambda_zero_least_squares():
    rng = np.random.default_rng(32)
    A = rng.normal(size=(20, 7))
    y = rng.normal(size=20)
    ref = np.linalg.lstsq(A, y, rcond=None)[0]
    model = IRFISTAB(positive=False, tol=1e-9)
    x = model(torch.from_numpy(y), operator=MatrixMult(torch.from_numpy(A)), lam=0.0, t=100)[0].numpy()
    assert oracle(A, y, x, 0.0, False)[1] <= 1e-9
    assert abs(oracle(A, y, x, 0.0, False)[0] - oracle(A, y, ref, 0.0, False)[0]) < 1e-12
