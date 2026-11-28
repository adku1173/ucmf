"""Bounded correctness / exploratory common-KKT comparisons (not a statistical study).

Before Gate 2: --mode correctness only. After independent Gate 2 approval:
  uv run python examples/benchmark_ir_fista_b.py --mode benchmark --gate2-passed --large

Every baseline uses ONE forward call with callback stopping: FISTA momentum is
never accidentally restarted. Practical certification products and default step
setup are charged. Timings include Python instrumentation, selection, callbacks,
and final IR certification, but exclude construction, reference solves, and the
external final oracle. No speed ratios or general speed claims are produced.
CPU RSS is process lifetime high water (not isolated solver peak); allocator
retention and previous runs contaminate it. CUDA allocator peaks exclude driver
memory. Use fresh processes for memory comparisons. CLI budgets censor runs;
wall caps are checked only at public iteration boundaries (IR outer sweeps).
"""

import argparse
import hashlib
import json
import math
import platform
import resource
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from ucmf import FISTA, IRFISTAB, ISTA
from ucmf.optim import IterationCallback
from ucmf.optim.operator import CMFPropagation, LinearOperator, MatrixMult

import torch


class Counted(LinearOperator):
    """Independently count actual calls by phase, source width and batch work."""

    def __init__(self, op, state=None):
        self.op = op
        self.shape, self.batch_size = op.shape, op.batch_size
        self.dtype, self.device, self.rows = op.dtype, op.device, op.rows
        self.state = {'phase': 'solve', 'calls': Counter()} if state is None else state

    def _count(self, kind, value):
        self.state['calls'][(self.state['phase'], kind, self.shape[1], value.shape[0])] += 1

    def forward(self, x):
        self._count('forward', x)
        return self.op.forward(x)

    def adjoint(self, r):
        self._count('adjoint', r)
        return self.op.adjoint(r)

    def detached(self):
        return Counted(self.op.detached(), self.state)

    def select_columns(self, columns):
        return Counted(self.op.select_columns(columns), self.state)

    @contextmanager
    def phase(self, name):
        previous = self.state['phase']
        self.state['phase'] = name
        try:
            yield
        finally:
            self.state['phase'] = previous

    def total(self, phase, kind):
        return sum(count for (p, k, _width, _rows), count in self.state['calls'].items() if p == phase and k == kind)

    def summary(self):
        return [
            {'phase': p, 'kind': k, 'width': w, 'batch_rows': b, 'calls': count}
            for (p, k, w, b), count in sorted(self.state['calls'].items())
        ]


def evaluate(op, y, x, lam, positive):
    """Independent full operator oracle; stable signed scalar-case formula."""
    residual = op.forward(x) - y
    m = y.shape[-1]
    grad = op.adjoint(residual) / m
    if positive:
        natural = torch.where(x < grad + lam, x, grad + lam)
        raw = torch.where(x > 0, (grad + lam).abs(), (-grad - lam).clamp_min(0))
        dual_violation = (-grad).amax(1).clamp_min(0)
    else:
        natural = torch.where(x > grad + lam, grad + lam, torch.where(x < grad - lam, grad - lam, x))
        raw = torch.where(x != 0, (grad + lam * x.sign()).abs(), (grad.abs() - lam).clamp_min(0))
        dual_violation = grad.abs().amax(1)
    objective = residual.square().sum(1) / (2 * m) + lam * x.abs().sum(1)
    result = {
        'objective': objective,
        'natural': natural.abs().amax(1),
        'raw': raw.amax(1),
        'support': (x != 0).sum(1),
        'feasible': (x >= 0).all(1) if positive else torch.ones_like(objective).bool(),
    }
    if lam > 0:
        alpha_max = torch.minimum(
            torch.ones_like(dual_violation), lam / dual_violation.clamp_min(torch.finfo(dual_violation.dtype).tiny)
        )
        w = residual / m
        denom = m * w.square().sum(1)
        alpha = (-(w * y).sum(1) / denom.clamp_min(torch.finfo(denom.dtype).tiny)).clamp_min(0)
        alpha = torch.minimum(alpha, alpha_max)
        dual = -m / 2 * alpha.square() * w.square().sum(1) - alpha * (w * y).sum(1)
        result['dual_gap'] = objective - dual
    return result


class KKTStop(IterationCallback):
    """Baseline common-global-KKT stopping, preserving the accelerated trajectory."""

    def __init__(self, op, lam, positive, tol, every, cap):
        self.op, self.lam, self.positive, self.tol = op, lam, positive, tol
        self.every, self.cap = every, cap
        self.steps = self.checks = 0
        self.capped = False
        self.start = time.perf_counter()

    def on_step_end(self, solver, x, **kw):
        self.steps = solver.iter + 1
        if self.steps % self.every == 0:
            with self.op.phase('stopping'):
                quality = evaluate(self.op, kw['y'], x, self.lam, self.positive)
            self.checks += 1
            if bool((quality['natural'] <= self.tol).all()):
                solver.stop = True
        if time.perf_counter() - self.start >= self.cap:
            self.capped = True
            solver.stop = True


class WallStop(IterationCallback):
    def __init__(self, cap):
        self.cap, self.start, self.capped = cap, time.perf_counter(), False

    def on_step_end(self, solver, x, **kw):
        if time.perf_counter() - self.start >= self.cap:
            self.capped = True
            solver.stop = True


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def check_harness():
    """Untimed correctness checks of baseline counts, callback trajectory, oracle."""
    for solver_type in [ISTA, FISTA]:
        for gamma in [None, 1.0]:
            op = Counted(MatrixMult(torch.tensor([[1.0, 0.2], [0.3, 2.0]], dtype=torch.float64)))
            y = torch.tensor([[2.0, 3.0]], dtype=op.dtype)
            cb = KKTStop(op, 0.1, True, 0.0, 1, math.inf)
            model = solver_type(callbacks=[cb])
            x = model(y, operator=op, lam=0.1, gamma=gamma, t=7)
            assert op.total('solve', 'forward') == cb.steps + (51 if gamma is None else 0)
            assert op.total('solve', 'adjoint') == cb.steps + (50 if gamma is None else 0)
            assert op.total('stopping', 'forward') == cb.checks
            assert op.total('stopping', 'adjoint') == cb.checks
            replay = solver_type()(y, operator=op.op, lam=0.1, gamma=gamma, t=cb.steps)
            torch.testing.assert_close(x, replay, atol=0, rtol=0)
    # Literal signed prox subtraction fails here; the harness must not.
    quality = evaluate(
        MatrixMult(torch.zeros(1, 1, dtype=torch.float64)),
        torch.zeros(1, 1, dtype=torch.float64),
        torch.tensor([[2.0**60]], dtype=torch.float64),
        1.0,
        False,
    )
    assert quality['natural'].item() == quality['raw'].item() == 1.0


def make_problem(args, family):
    """Seeded Gaussian or synthetic realistic free-field covariance propagation.

    Free-field microphones lie on a radius-0.2m planar spiral; sources are on a
    [-.4,.4]^2 plane at z=.8m, 8kHz and sound speed 343m/s. H=exp(-ikr)/r,
    with explicit global 1/sqrt(c) amplitude scaling. No Acoular dependency or
    claim of matching captured Acoular measurements; off-diagonal hvec used.
    """
    rng = torch.Generator().manual_seed(args.seed)
    dtype, device = getattr(torch, args.dtype), torch.device(args.device)
    if family == 'dense':
        m, n = (args.m, args.n) if args.large else (32, 24)
        needed = m * n * 8
        if needed > args.max_mib * 2**20:
            raise ValueError('Dense dictionary exceeds --max-mib allocation guard.')
        status = Path('/proc/meminfo')
        if status.exists():
            available = (
                int(
                    next(
                        line.split()[1] for line in status.read_text().splitlines() if line.startswith('MemAvailable:')
                    )
                )
                * 1024
            )
            if available < 3 * needed + 2**30:
                raise ValueError('Insufficient available RAM for dense case and reference.')
        A = torch.randn(m, n, generator=rng, dtype=torch.float64) / math.sqrt(m)
        A = A.to(device=device, dtype=dtype)
        op = MatrixMult(A)
        ref_op = MatrixMult(A.to(torch.float64))
    else:
        c, side = (args.mics, args.grid) if args.large else (6, 5)
        angle = torch.arange(c, dtype=torch.float64) * (math.pi * (3 - math.sqrt(5)))
        radius = 0.2 * torch.sqrt((torch.arange(c, dtype=torch.float64) + 1) / c)
        mic = torch.stack([radius * angle.cos(), radius * angle.sin(), torch.zeros(c)], 1)
        q = torch.linspace(-0.4, 0.4, side, dtype=torch.float64)
        gx, gy = torch.meshgrid(q, q, indexing='ij')
        grid = torch.stack([gx.flatten(), gy.flatten(), torch.full((side * side,), 0.8)], 1)
        distances = torch.cdist(mic, grid)
        H = torch.exp(-2j * math.pi * 8000 / 343 * distances) / distances / math.sqrt(c)
        H = H.to(device=device, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64)
        op = CMFPropagation(H, r_diag=True)
        ref_op = CMFPropagation(H.to(torch.complex128), r_diag=True)
        m, n = op.shape
    truth = torch.zeros(1, n, dtype=torch.float64)
    support = min(args.support, n)
    index = torch.randperm(n, generator=rng)[:support]
    truth[:, index] = 0.5 + torch.rand(support, generator=rng, dtype=torch.float64)
    if args.signed:
        truth[:, index[::2]] *= -1
    y_ref = ref_op.forward(truth.to(device))
    noise = torch.randn(1, m, generator=rng, dtype=torch.float64).to(device)
    y_ref = y_ref + 0.01 * y_ref.norm() * noise / noise.norm()
    # Validate the same quantized problem in float64, not a different f64 input.
    y_ref = y_ref.to(dtype).to(torch.float64)
    corr = ref_op.adjoint(y_ref) / m
    scale = float(corr.abs().max() if args.signed else corr.max().clamp_min(0))
    return op, y_ref.to(dtype), ref_op, y_ref, args.lam_fraction * scale, max(scale, 1e-12)


def reference(op, y, lam, positive, tol, args):
    counted = Counted(op)
    cb = KKTStop(counted, lam, positive, tol, args.check_every, args.cap)
    x = FISTA(nonnegative=positive, callbacks=[cb])(y, operator=counted, lam=lam, t=args.reference_steps)
    quality = evaluate(op, y, x, lam, positive)
    return quality, cb.steps


def run_solver(name, op, y, ref_op, y_ref, lam, tol, args):
    counted = Counted(op)
    positive = not args.signed
    if name == 'IRFISTAB':
        cb = WallStop(args.cap)
        model = IRFISTAB(positive=positive, tol=tol, callbacks=[cb], inner_max_iter=args.inner_steps)
        budget = args.outer_steps
    else:
        cb = KKTStop(counted, lam, positive, tol, args.check_every, args.cap)
        model = {'FISTA': FISTA, 'ISTA': ISTA}[name](nonnegative=positive, callbacks=[cb])
        budget = args.steps
    device = y.device
    if device.type == 'cuda' and args.mode == 'benchmark':
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    cb.start = start
    x = model(y, operator=counted, lam=lam, t=budget)
    sync(device)
    elapsed = time.perf_counter() - start
    # Float64 full oracle is outside solve scope, independent of IR diagnostics.
    quality = evaluate(ref_op, y_ref, x.to(torch.float64), lam, positive)
    success = bool((quality['natural'] <= tol).all())
    record = {
        'solver': name,
        'success_oracle': success,
        'censored': not success,
        'wall_cap_hit': cb.capped,
        'quality': quality,
        'counts': counted.summary(),
        'budget': budget,
        'tol_abs': tol,
        'scope': 'public call + practical stopping + counting overhead',
    }
    if name == 'IRFISTAB':
        diag = model.diagnostics
        for kind in ['forward', 'adjoint']:
            assert counted.total('solve', kind) == diag['counts']['full_' + kind] + diag['counts']['restricted_' + kind]
        same_dtype = op.dtype == torch.float64
        if same_dtype:
            torch.testing.assert_close(diag['natural_residual'], quality['natural'], atol=1e-12, rtol=1e-8)
            torch.testing.assert_close(diag['raw_residual'], quality['raw'], atol=1e-12, rtol=1e-8)
        if any(s == 'converged' for s in diag['status']) and not success:
            # Only finite-precision f32 oracle discrepancy is allowed, explicitly recorded.
            if same_dtype:
                raise AssertionError('False IR convergence certificate')
            record['float64_oracle_disagrees'] = True
        record['diagnostics'] = diag
    else:
        assert counted.total('solve', 'forward') == cb.steps + 51
        assert counted.total('solve', 'adjoint') == cb.steps + 50
        assert counted.total('stopping', 'forward') == cb.checks
        assert counted.total('stopping', 'adjoint') == cb.checks
        record.update(iterations=cb.steps, stopping_checks=cb.checks)
    if args.mode == 'benchmark':
        record.update(
            wall_seconds=elapsed,
            cpu_lifetime_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            memory_caveat='Linux process lifetime high-water RSS; NOT isolated solver incremental peak',
        )
        if device.type == 'cuda':
            record.update(
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['correctness', 'benchmark'], default='correctness')
    parser.add_argument(
        '--gate2-passed', action='store_true', help='Assert independent Gate 2 approval before benchmarking'
    )
    parser.add_argument('--family', choices=['dense', 'cmf', 'both'], default='both')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', choices=['float64', 'float32'], default='float64')
    parser.add_argument(
        '--large', action='store_true', help='Use configurable large shapes rather than tiny correctness cases'
    )
    parser.add_argument('--m', type=int, default=512)
    parser.add_argument('--n', type=int, default=8192)
    parser.add_argument('--mics', type=int, default=16)
    parser.add_argument('--grid', type=int, default=32)
    parser.add_argument('--max-mib', type=float, default=512)
    parser.add_argument('--support', type=int, default=3)
    parser.add_argument('--lam-fraction', type=float, default=0.1)
    parser.add_argument('--signed', action='store_true')
    parser.add_argument('--seed', type=int, default=7957)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument(
        '--tol', type=float, default=1e-6, help='Relative to fixed initial correlation scale, all solvers'
    )
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--reference-steps', type=int, default=30000)
    parser.add_argument('--outer-steps', type=int, default=200)
    parser.add_argument('--inner-steps', type=int, default=100)
    parser.add_argument('--check-every', type=int, default=10)
    parser.add_argument('--cap', type=float, default=30.0, help='Seconds per public solve, boundary checks only')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.mode == 'benchmark' and not args.gate2_passed:
        parser.error('Performance comparisons require --gate2-passed after independent review.')
    if args.large and args.mode == 'correctness':
        parser.error('--large is reserved for post-Gate-2 exploratory benchmarks.')
    for field in [
        'm',
        'n',
        'mics',
        'grid',
        'support',
        'threads',
        'steps',
        'reference_steps',
        'outer_steps',
        'inner_steps',
        'check_every',
        'cap',
        'repeats',
        'tol',
        'max_mib',
    ]:
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            parser.error(f'--{field.replace("_", "-")} must be positive')
    if not math.isfinite(args.lam_fraction) or args.lam_fraction < 0:
        parser.error('--lam-fraction must be nonnegative')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        print(json.dumps({'status': 'not-run', 'reason': 'CUDA unavailable'}))
        return
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    with torch.no_grad():
        check_harness()
        results = []
        families = ['dense', 'cmf'] if args.family == 'both' else [args.family]
        for family in families:
            setup_start = time.perf_counter()
            op, y, ref_op, y_ref, lam, scale = make_problem(args, family)
            sync(args.device)
            setup_seconds = time.perf_counter() - setup_start
            tol = args.tol * scale
            ref, ref_steps = reference(ref_op, y_ref, lam, not args.signed, min(tol / 100, 1e-10 * scale), args)
            ref_certified = bool((ref['natural'] <= min(tol / 100, 1e-10 * scale)).all())
            repeats = args.repeats if args.mode == 'benchmark' else 1
            for repeat in range(repeats):
                # Rotate order to reduce simple ordering bias; no statistical claim.
                names = ['IRFISTAB', 'FISTA', 'ISTA']
                names = names[repeat % 3 :] + names[: repeat % 3]
                for name in names:
                    record = run_solver(name, op, y, ref_op, y_ref, lam, tol, args)
                    gap = record['quality']['objective'] - ref['objective']
                    record.update(
                        mode=args.mode,
                        family=family,
                        seed=args.seed,
                        shape=op.shape,
                        batch=y.shape[0],
                        dtype=args.dtype,
                        device=args.device,
                        threads=args.threads,
                        lam=lam,
                        reference_certified=ref_certified,
                        reference_steps=ref_steps,
                        reference_quality=ref,
                        objective_reference_gap=gap,
                        repeat=repeat,
                        torch_version=torch.__version__,
                        platform=platform.platform(),
                        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    )
                    if args.mode == 'correctness':
                        assert record['success_oracle'], f'{family}/{name} budget-limited; adjust explicit budgets'
                        assert bool(record['quality']['feasible'].all())
                        assert ref_certified, f'{family} reference uncertified'
                        assert bool((gap.abs() <= 1e-7 * max(1.0, float(ref['objective'].max()))).all())
                    else:
                        record['setup_seconds_excluded'] = setup_seconds
                        record['study'] = 'bounded exploratory; no warm-up or statistical study; repeats disclosed'
                    results.append(jsonable(record))
        text = '\n'.join(json.dumps(record) for record in results) + '\n'
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
        print(text, end='')


if __name__ == '__main__':
    main()
