"""Benchmark ucmf operators on CUDA; run with --help for filters.

Timing includes the solve and, for learned models, parameter gradients. Operator
construction, step-size estimation and the NA feature normalization are excluded. Distinct dictionaries use
two alternating same-shape batches; shared dictionaries reuse the operator.
Each case and execution mode runs in a fresh process. Peak allocated and
reserved memory are measured during warm timing, including live inputs and
any allocator reservations retained after warm-up.
"""

import argparse
from itertools import product
from math import sqrt
from multiprocessing import get_context
from statistics import median
from time import perf_counter

from ucmf import (
    FISTA,
    ISTA,
    AdaLFISTA,
    AdaLFISTAR,
    AdaLFISTASW,
    AdaLFISTASWR,
    NA_AdaLFISTA,
    NA_AdaLFISTAR,
    NA_AdaLFISTASW,
    NA_AdaLFISTASWR,
)
from ucmf.optim.operator import CMFDictionary, CMFPropagation
from ucmf.optim.utils import default_gamma_from_operator

import torch

MODELS = {model.__name__: model for model in (
    ISTA, FISTA, AdaLFISTASW, AdaLFISTA, AdaLFISTASWR, AdaLFISTAR,
    NA_AdaLFISTASW, NA_AdaLFISTA, NA_AdaLFISTASWR, NA_AdaLFISTAR,
)}
CLASSICAL = (ISTA, FISTA)
REDUCED = (AdaLFISTASWR, AdaLFISTAR, NA_AdaLFISTASWR, NA_AdaLFISTAR)
DENSE_WEIGHTS = (AdaLFISTASW, AdaLFISTA, NA_AdaLFISTASW, NA_AdaLFISTA)
NEURAL = (NA_AdaLFISTASW, NA_AdaLFISTA, NA_AdaLFISTASWR, NA_AdaLFISTAR)
OPERATORS = ('CMFDictionary', 'CMFPropagation')
GRID = {16: 16, 64: 51}  # grid side per array, as ucmf_data.parameters.default_focus_grid


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mics', nargs='+', type=int, choices=tuple(GRID), default=list(GRID), help='grid: see GRID')
    parser.add_argument('--batch', nargs='+', type=int, choices=(1, 32, 512), default=[1, 32, 512])
    parser.add_argument('--layout', nargs='+', choices=('shared', 'distinct'), default=['shared', 'distinct'])
    parser.add_argument('--model', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--operator', nargs='+', choices=OPERATORS, default=OPERATORS)
    parser.add_argument('--weight', nargs='+', choices=('factored', 'precomputed'), default=['factored', 'precomputed'])
    parser.add_argument('--mode', nargs='+', choices=('eager', 'compiled'), default=['eager', 'compiled'])
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--keep-diagonal', action='store_true', help='include the CSM diagonal (default: omit it)')
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error('--steps and --repeats must be positive')
    return args


def cases(args):
    for c, batch, layout, name, representation in product(
        args.mics, args.batch, args.layout, args.model, args.operator,
    ):
        grid = GRID[c]
        if (batch == 1 and layout == 'distinct') or (batch == 512 and c != 16):
            continue
        model_type = MODELS[name]
        if model_type not in DENSE_WEIGHTS:
            weights = ['—' if model_type in CLASSICAL else 'factored']
        else:
            weights = [w for w in args.weight if w == 'factored' or representation != 'CMFPropagation']
        for weight in weights:
            yield c, grid, batch, layout, model_type, representation, weight


def make_inputs(c, grid, batch, layout, r_diag, rng):
    n = grid**2
    shape = (batch, c, n) if layout == 'distinct' else (c, n)
    inputs = []
    for i in range(2):
        if i == 0 or layout == 'distinct':
            H = torch.randn(shape, device='cuda', dtype=torch.complex64, generator=rng) / sqrt(c)
            propagation = CMFPropagation(H, r_diag)
        x = torch.randn(batch, n, device='cuda', generator=rng).sub_(1).relu_()
        inputs.append((propagation.forward(x), propagation))
    return inputs


def represent(inputs, representation):
    if representation == 'CMFPropagation':
        return inputs
    dense = {}
    result = []
    for y, propagation in inputs:
        if propagation not in dense:
            dense[propagation] = CMFDictionary(propagation.H, r_diag=propagation.r_diag)
        result.append((y, dense[propagation]))
    return result


def make_model(model_type, c, propagation, steps, weight_mode, gamma):
    """Return the solver and its call arguments; learned weight matrices are perturbed."""
    theta = 0.01
    if model_type in CLASSICAL:
        return model_type(nonnegative=False), {'gamma': gamma, 'lam': theta / gamma}
    kwargs = {'weight_mode': weight_mode} if model_type in DENSE_WEIGHTS else {}
    if model_type not in NEURAL:
        kwargs['num_iterations'] = steps
    size = {'num_mics': c, 'r_diag': propagation.r_diag} if model_type in REDUCED else {
        'num_measurements': propagation.shape[0],
    }
    model = model_type(gamma_init=gamma, theta_init=theta, nonnegative=False, **size, **kwargs).cuda()
    rng = torch.Generator(device='cuda').manual_seed(8)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                noise = torch.randn(parameter.shape, device=parameter.device, generator=rng)
                parameter.add_(noise, alpha=0.05 / sqrt(parameter.shape[1]))
    return model, {}


def make_step(model, steps, kwargs):
    def solve(y, operator):
        return model(y, operator=operator, t=steps, **kwargs)

    params = tuple(model.parameters())

    def step(function, inputs):
        with torch.set_grad_enabled(bool(params)):
            x = function(*inputs)
            grads = torch.autograd.grad(x.square().mean(), params) if params else ()
        return x.detach(), *grads

    return solve, step


def measure(args, mode, solve, step, inputs, references):
    function = torch.compile(solve, mode='reduce-overhead', fullgraph=True) if mode == 'compiled' else solve
    torch.cuda.synchronize()

    def timed(i):
        if mode == 'compiled':
            torch.compiler.cudagraph_mark_step_begin()
        start = perf_counter()
        result = step(function, inputs[i])
        torch.cuda.synchronize()
        return perf_counter() - start, result

    first = None
    for i, reference in enumerate(references):
        elapsed, result = timed(i)
        if i == 0:
            first = elapsed
        for value, expected in zip(result, reference, strict=True):
            actual = value.cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-6)
            error = (actual - expected).norm().item()
            if error > 3e-3 * expected.norm().item():
                raise AssertionError(f'L2 error {error:.3g} exceeds the relative tolerance (0.3%).')
        del result, value
    for i in range(6):
        timed(i % 2)
    torch.cuda.reset_peak_memory_stats()
    times = [timed(i % 2)[0] for i in range(args.repeats)]
    allocated = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    return f'{1000 * median(times):.3f} | {first:.3f} | {allocated:.1f} | {reserved:.1f}'


def run(args, case, mode, *, header):
    if not torch.cuda.is_available():
        raise SystemExit('This benchmark requires CUDA.')
    torch.set_float32_matmul_precision('highest')
    if header:
        print_header(args)
    c, grid, batch, layout, model_type, representation, weight = case
    label = f'| {c} | {grid**2} | {batch} | {layout} | {model_type.__name__} | {representation} | {weight}'
    rng = torch.Generator(device='cuda').manual_seed(7)
    try:
        source = make_inputs(c, grid, batch, layout, not args.keep_diagonal, rng)
        propagation = source[0][1]
        gamma = 0.6 * float(default_gamma_from_operator(propagation).min())
        model, kwargs = make_model(model_type, c, propagation, args.steps, 'factored', gamma)
        if model_type in NEURAL:
            # Shared batches use one operator, so both enter the normalization (nonzero spread at batch 1).
            y = torch.cat([y for y, _ in source]) if layout == 'shared' else source[0][0]
            model.init_normalizers(y, propagation)
        solve, step = make_step(model, args.steps, kwargs)
        references = [tuple(value.cpu() for value in step(solve, data)) for data in source]
        if weight == 'precomputed':
            # Same parameters and feature normalization as the reference model.
            state = model.state_dict()
            del model, solve, step
            model, kwargs = make_model(model_type, c, propagation, args.steps, weight, gamma)
            model.load_state_dict(state)
            solve, step = make_step(model, args.steps, kwargs)
        inputs = represent(source, representation)
        del source, propagation
    except torch.OutOfMemoryError:
        metrics = 'OOM (setup) | — | — | —'
    else:
        torch.cuda.empty_cache()
        try:
            metrics = measure(args, mode, solve, step, inputs, references)
        except torch.OutOfMemoryError:
            metrics = 'OOM | — | — | —'
    print(f'{label} | {mode} | {metrics} |', flush=True)


def print_header(args):
    print(f'{torch.cuda.get_device_name()} | PyTorch {torch.__version__} | float32 | {args.steps} iterations')
    print(f'CSM diagonal: {"included" if args.keep_diagonal else "omitted"}; median of {args.repeats} runs.')
    print('Unsupported combinations are omitted (see --help for case filters).')
    print('| Mics | Grid points | Batch | Dictionaries | Model | Operator | Weight | Mode |'
          ' Warm ms | First s | Peak allocated MiB | Peak reserved MiB |')
    print('|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|', flush=True)


def main():
    args = parse_args()
    context = get_context('spawn')
    for index, (case, mode) in enumerate(product(cases(args), args.mode)):
        worker = context.Process(target=run, args=(args, case, mode), kwargs={'header': index == 0})
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            raise SystemExit(f'Benchmark worker failed for {case} ({mode}), exit status {worker.exitcode}.')
        worker.close()


if __name__ == '__main__':
    main()
