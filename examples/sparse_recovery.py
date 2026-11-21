"""Run ISTA and FISTA without the FA2026 data-generation stack.

Run from the repository root: uv run python examples/sparse_recovery.py
"""

import torch

from ucmf import FISTA, ISTA
from ucmf.optim import SolutionChangeStopper, normalize_sensing_matrix_columns


def main():
    rng = torch.Generator().manual_seed(42)
    A_physical = torch.randn(32, 16, generator=rng, dtype=torch.float64)
    x_true = torch.zeros(1, 16, dtype=A_physical.dtype)
    x_true[0, [2, 7, 12]] = torch.tensor([1.0, 0.6, 0.8], dtype=x_true.dtype)
    y = x_true @ A_physical.T

    # Normalization is explicit: solve for z_j = ||a_j|| * x_j.
    A, column_norms = normalize_sensing_matrix_columns(A_physical)
    for solver_type in (ISTA, FISTA):
        stopper = SolutionChangeStopper(tol=1e-8)
        solver = solver_type(positive=True, callbacks=[stopper])
        with torch.no_grad():
            z_hat = solver(y, A=A, lam=1e-3, t=2000)
            x_hat = z_hat / column_norms
        relative_error = torch.linalg.vector_norm(x_hat - x_true) / torch.linalg.vector_norm(x_true)
        print(f'{solver_type.__name__}: {stopper.n_steps} steps, relative source error {relative_error:.4f}')


if __name__ == '__main__':
    main()
