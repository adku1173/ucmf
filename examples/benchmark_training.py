"""Training throughput of ucmf's unfolded models on CUDA; run with --help for filters.

One training step is timed as in ``ucmf.train.train_model``: drawing the next batch, selecting
its dictionaries, objective (solve, loss), backward and Adam step, on a dataset of eight batches.
The operator is column normalized and the data are equiregularized beforehand (not timed).
Dictionary layouts: ``shared`` uses one operator for all samples; ``per_batch``
(``batch_sampling='per_dictionary'``) and ``distinct`` (``'random'``) select from a bank of
``--bank`` random dictionaries (e.g. one per stored frequency bin) one entry per batch or one
per sample. Throughput is
``batch / median step time`` at the first stage (t = 1) and the last stage (t = T).
Compiled steps compute all T iterations at every stage. Each case and execution mode
runs in a fresh process; peak memory is measured during warm timing at t = T.
"""

import argparse
import warnings
from itertools import product
from math import sqrt
from multiprocessing import get_context
from statistics import median
from time import perf_counter

from ucmf.optim.neural_augmentation import NeuralAugmentation
from ucmf.optim.operator import CMFDictionary, CMFPropagation
from ucmf.optim.utils import default_gamma_from_operator
from ucmf.scaling import normalize_operator
from ucmf.train import TrainingConfig
from ucmf.train.config import MODEL_NAMES, OPERATOR_NAMES
from ucmf.train.objectives import build_loss
from ucmf.train.utils import (
    batch_operator,
    build_model,
    build_operator,
    build_optimizer,
    build_solve,
    iterate_batches,
    prepare_data,
)

import torch

REDUCED = ('AdaLFISTAR', 'AdaLFISTASWR', 'NA_AdaLFISTAR', 'NA_AdaLFISTASWR')
LOSSES = ('nmse', 'lasso', 'bic', 'cv:rows', 'cv:complex_rows', 'cv:mics', 'cv:mics_cross')
LOSS_KWARGS = {'lasso': {'lam': 0.1}}
GRID = {16: 16, 64: 51}  # grid side per array, as ucmf_data.parameters.default_focus_grid
LAYOUTS = ('shared', 'per_batch', 'distinct')
NUM_BATCHES = 8  # dataset size in batches; the step draws batches as in train_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mics', nargs='+', type=int, choices=tuple(GRID), default=list(GRID), help='grid: see GRID')
    parser.add_argument('--layout', nargs='+', choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument('--bank', type=int, default=205, help='dictionaries in the bank (stored frequency bins)')
    parser.add_argument('--model', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--operator', nargs='+', choices=OPERATOR_NAMES, default=list(OPERATOR_NAMES))
    parser.add_argument('--weight', nargs='+', choices=('factored', 'precomputed'), default=['factored', 'precomputed'])
    parser.add_argument('--loss', nargs='+', choices=LOSSES, default=['nmse'])
    parser.add_argument('--w', nargs='+', type=float, default=[0.0], help='iterate weight base in [0, 1]')
    parser.add_argument('--mode', nargs='+', choices=('eager', 'compiled'), default=['eager', 'compiled'])
    parser.add_argument('--stages', type=int, default=15, help='T, the depth of the last stage')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--keep-diagonal', action='store_true', help='include the CSM diagonal (default: omit it)')
    args = parser.parse_args()
    if min(args.stages, args.batch, args.repeats, args.bank) < 1:
        parser.error('--stages, --batch, --repeats and --bank must be positive')
    return args


def supported(model, operator, loss):
    """Whether the reduced models, CV splits and operator fit together."""
    if operator == 'MatrixMult' and (model in REDUCED or loss in ('cv:complex_rows', 'cv:mics', 'cv:mics_cross')):
        return False  # no CMF metadata
    if loss in ('cv:rows', 'cv:complex_rows'):  # select_rows gives a MatrixMult without CMF metadata
        return operator != 'CMFPropagation' and model not in REDUCED
    return True


def cases(args):
    axes = (args.mics, args.layout, args.model, args.operator, args.loss, args.w)
    for c, layout, model, operator, loss, w in product(*axes):
        if not supported(model, operator, loss):
            continue
        if model in REDUCED:
            weights = ['—']
        else:
            weights = [weight for weight in args.weight if weight == 'factored' or operator != 'CMFPropagation']
        for weight in weights:
            yield c, layout, model, operator, weight, loss, w


def make_config(args, case, mode, H, r_diag):
    c, layout, model, operator, weight, loss, w = case
    gamma = float(default_gamma_from_operator(normalize_operator(CMFPropagation(H, r_diag))[0]).min())
    model_kwargs = {'gamma_init': gamma, 'theta_init': 0.01}
    if model in REDUCED:
        model_kwargs |= {'num_mics': c, 'r_diag': r_diag}
    else:
        model_kwargs |= {'num_measurements': c * (c - int(r_diag)), 'weight_mode': weight}
    if not model.startswith('NA_'):
        model_kwargs['num_iterations'] = args.stages
    matrix = {'matrix': CMFDictionary(H, r_diag).matrix} if operator == 'MatrixMult' else {'H': H, 'r_diag': r_diag}
    name, _, split = loss.partition(':')
    return TrainingConfig(
        model_name=model,
        operator_name=operator,
        model_kwargs=model_kwargs,
        operator_kwargs=matrix,
        stages=args.stages,
        batch_size=args.batch,
        loss=name,
        loss_kwargs={'split': split} if split else LOSS_KWARGS.get(name, {}),
        w=w,
        batch_sampling='per_dictionary' if layout == 'per_batch' else 'random',
        device='cuda',
        compile=mode == 'compiled',
    )


def make_dataset(args, layout, n, rng):
    """``NUM_BATCHES`` batches of sources ``q`` and their bank entries (None for shared).

    'per_batch' gives every batch its own entry, 'distinct' every sample a random one.
    """
    num_samples = NUM_BATCHES * args.batch
    q = torch.randn(num_samples, n, device='cuda', generator=rng).sub_(1).relu_()
    if layout == 'shared':
        return q, None
    if layout == 'per_batch':
        return q, torch.randint(args.bank, (NUM_BATCHES,), device='cuda', generator=rng).repeat_interleave(args.batch)
    return q, torch.randint(args.bank, (num_samples,), device='cuda', generator=rng)


def measure(args, config, q, index):
    bank, column_norms = normalize_operator(build_operator(config))
    y = []
    for i in range(0, q.shape[0], args.batch):  # measure every sample with its own dictionary, A q = D U q
        problems = None if index is None else index[i : i + args.batch]
        operator, norms = batch_operator(bank, column_norms, problems)
        y.append(operator.forward(q[i : i + args.batch] * norms))
    y = torch.cat(y)
    y, q, index = prepare_data(config, bank, y, q, index)
    model = build_model(config)
    if isinstance(model, NeuralAugmentation):
        problems = None if index is None else index[: args.batch]
        model.init_normalizers(y[: args.batch], batch_operator(bank, column_norms, problems)[0])
    objective, solve, optimizer = build_loss(config), build_solve(config, model), build_optimizer(config, model)
    batches = iterate_batches(config, y.shape[0], index)

    def timed(t):
        start = perf_counter()
        samples, problems = next(batches)  # batch drawing and bank selection are part of the step
        operator, norms = batch_operator(bank, column_norms, problems)
        optimizer.zero_grad()
        objective(solve, y[samples], q[samples], t, operator, norms).backward()
        optimizer.step()
        torch.cuda.synchronize()
        return perf_counter() - start

    first = timed(args.stages)
    for _ in range(6):
        timed(args.stages), timed(1)
    torch.cuda.reset_peak_memory_stats()
    last = median(timed(args.stages) for _ in range(args.repeats))
    allocated = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    stage1 = median(timed(1) for _ in range(args.repeats))
    return f'{args.batch / stage1:.0f} | {args.batch / last:.0f} | {first:.1f} | {allocated:.0f} | {reserved:.0f}'


def run(args, case, mode, *, header):
    if not torch.cuda.is_available():
        raise SystemExit('This benchmark requires CUDA.')
    torch.set_float32_matmul_precision('highest')
    warnings.filterwarnings('ignore', category=UserWarning)  # inductor hints on complex ops and TF32
    if header:
        print_header(args)
    c, layout = case[:2]
    n = GRID[c] ** 2
    rng = torch.Generator(device='cuda').manual_seed(7)
    shape = (c, n) if layout == 'shared' else (args.bank, c, n)
    H = torch.randn(shape, device='cuda', dtype=torch.complex64, generator=rng) / sqrt(c)
    r_diag = not args.keep_diagonal
    try:
        config = make_config(args, case, mode, H, r_diag)
        metrics = measure(args, config, *make_dataset(args, layout, n, rng))
    except torch.OutOfMemoryError:
        metrics = 'OOM | — | — | — | —'
    print(f'| {c} | {n} | ' + ' | '.join(map(str, case[1:])) + f' | {mode} | {metrics} |', flush=True)


def print_header(args):
    print(
        f'{torch.cuda.get_device_name()} | PyTorch {torch.__version__} | float32 | batch {args.batch} | '
        f'T = {args.stages} | bank {args.bank} | CSM diagonal {"included" if args.keep_diagonal else "omitted"}'
    )
    print(f'Throughput in samples/s (median of {args.repeats} steps); unsupported combinations are omitted.')
    print(
        '| Mics | Grid points | Dictionaries | Model | Operator | Weight | Loss | w | Mode | t = 1 | t = T | First s |'
        ' Peak allocated MiB | Peak reserved MiB |'
    )
    print('|---:|---:|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|', flush=True)


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
