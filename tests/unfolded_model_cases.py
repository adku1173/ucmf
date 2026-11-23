"""Model and problem cases for the properties shared by all unfolded models.

A model case returns ``(make_model, dense_weights)``:

- ``make_model(n_mics, r_diag, **kwargs)`` builds the model for a full array of
  ``n_mics`` microphones; ``kwargs`` are the common constructor arguments.
- ``dense_weights(model, operator)`` returns the dense matrices ``(M, P, Q)`` on
  the active measurement rows, differentiable in the model parameters. The
  model's update direction is ``(M.T @ y - P.T @ (Q @ z)) / m``.

Tags: ``dense`` (accepts ``MatrixMult``), ``cmf_only`` (requires CMF coordinates),
``weight_mode`` (factored/precomputed), ``two_weights`` (``W_recurrent`` callback).

A problem case returns ``(operator, y, x0)``. Its operator source (``H`` or the
matrix) requires gradients, so tests can check that models detach it.
Tags: ``cmf``, ``dictionary`` (``CMFDictionary``), ``rows`` (``MatrixMult`` row subsets).
"""

import math

from ucmf import AdaLFISTA, AdaLFISTASW
from ucmf.optim import AdaLFISTAR, AdaLFISTASWR
from ucmf.optim.operator import CMFDictionary, CMFPropagation, HermitianVectorization, MatrixMult

import torch
from pytest_cases import case, parametrize

N_MICS = 4
N_SOURCES = 3
BATCH = 2
T = 4


def build_model(make_model, r_diag=False, *, positive=False, gamma_init=0.3, theta_init=0.01, **kwargs):
    """Float64 model with identity auxiliary weights for the full array of ``N_MICS``."""
    model = make_model(
        N_MICS, r_diag, n_iterations=T, gamma_init=gamma_init, theta_init=theta_init, positive=positive, **kwargs
    )
    return model.double()


def perturb_weights(model):
    """Make every auxiliary weight (all 2D parameters) dense and non-identity."""
    rng = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.add_(0.2 * torch.randn(parameter.shape, dtype=parameter.dtype, generator=rng))
    return model


def r_diag_of(operator):
    """Diagonal setting a model needs for ``operator``; dense problems use the full CSM."""
    return isinstance(operator, (CMFDictionary, CMFPropagation)) and operator.r_diag


def operator_source(operator):
    """The tensor the operator is built from."""
    return operator.H if isinstance(operator, (CMFDictionary, CMFPropagation)) else operator.matrix


def dense_matrix(operator):
    """Dense active rows of the operator, shared ``(m, n)`` or batched ``(B, m, n)``."""
    if isinstance(operator, CMFPropagation):
        return CMFDictionary(operator.H, r_diag=operator.r_diag).matrix
    return operator.matrix


def restrict(B, rows):
    """Sub-block ``B[I, I]`` for the active rows ``I`` (shared or per problem)."""
    if rows is None:
        return B
    return B[rows.unsqueeze(-1), rows.unsqueeze(-2)]


def congruence_weight(Br_re, Br_im, n_mics, r_diag):
    """Dense ``B`` with ``B v = hvec(Br hvec*(v) Br^H)`` on the full CMF basis."""
    hvec = HermitianVectorization(n_mics, r_diag=r_diag)
    basis = torch.eye(hvec.n_measurements, dtype=Br_re.dtype)
    Br = torch.complex(Br_re, Br_im)
    weighted = Br @ torch.complex(*hvec.adjoint(basis)) @ Br.mH
    return hvec(weighted.real, weighted.imag).mT


def reference_iterates(M, P, Q, y, x0, gammas, thetas, positive):
    """Independent dense accelerated iteration; returns all iterates ``(T, B, n)``."""
    measurement = (M.mT @ y.unsqueeze(-1)).squeeze(-1)
    x, z, tk, iterates = x0, x0, 1.0, []
    for gamma, theta in zip(gammas, thetas, strict=True):
        feedback = (P.mT @ (Q @ z.unsqueeze(-1))).squeeze(-1)
        proposal = z + gamma * (measurement - feedback) / y.shape[-1]
        if positive:
            x_next = torch.relu(proposal - theta)
        else:
            x_next = proposal.sign() * torch.relu(proposal.abs() - theta)
        t_next = (1 + math.sqrt(1 + 4 * tk**2)) / 2
        z = x_next + (tk - 1) / t_next * (x_next - x)
        x, tk = x_next, t_next
        iterates.append(x)
    return torch.stack(iterates)


class UnfoldedModelCases:
    @case(tags=('dense', 'weight_mode'))
    def case_sw(self):
        def make_model(n_mics, r_diag, **kwargs):
            return AdaLFISTASW(n_measurements=n_mics * (n_mics - int(r_diag)), **kwargs)

        def dense_weights(model, operator):
            A = dense_matrix(operator)
            W = restrict(model.B, operator.rows) @ A
            return W, W, A

        return make_model, dense_weights

    @case(tags=('dense', 'weight_mode', 'two_weights'))
    def case_lfista(self):
        def make_model(n_mics, r_diag, **kwargs):
            return AdaLFISTA(n_measurements=n_mics * (n_mics - int(r_diag)), **kwargs)

        def dense_weights(model, operator):
            A = dense_matrix(operator)
            R = restrict(model.B1, operator.rows) @ A
            M = restrict(model.B2, operator.rows) @ A
            return M, R, R

        return make_model, dense_weights

    @case(tags='cmf_only')
    def case_swr(self):
        def make_model(n_mics, r_diag, **kwargs):
            return AdaLFISTASWR(n_mics=n_mics, r_diag=r_diag, **kwargs)

        def dense_weights(model, operator):
            A = dense_matrix(operator)
            B = congruence_weight(model.Br_re, model.Br_im, model.n_mics, model.r_diag)
            W = restrict(B, operator.rows) @ A
            return W, W, A

        return make_model, dense_weights

    @case(tags=('cmf_only', 'two_weights'))
    def case_r(self):
        def make_model(n_mics, r_diag, **kwargs):
            return AdaLFISTAR(n_mics=n_mics, r_diag=r_diag, **kwargs)

        def dense_weights(model, operator):
            A = dense_matrix(operator)
            B1 = congruence_weight(model.Br1_re, model.Br1_im, model.n_mics, model.r_diag)
            B2 = congruence_weight(model.Br2_re, model.Br2_im, model.n_mics, model.r_diag)
            R = restrict(B1, operator.rows) @ A
            M = restrict(B2, operator.rows) @ A
            return M, R, R

        return make_model, dense_weights


PER_PROBLEM_MICS = [[0, 2, 3], [0, 1, 2]]
CMF_LAYOUTS = parametrize(
    'batched_H, mics',
    [(False, None), (False, [0, 2, 3]), (False, PER_PROBLEM_MICS), (True, None), (True, PER_PROBLEM_MICS)],
    ids=['shared-H', 'shared-H-shared-mics', 'shared-H-per-problem-mics', 'batched-H', 'batched-H-per-problem-mics'],
)
R_DIAG = parametrize('r_diag', [False, True], ids=['with-diagonal', 'without-diagonal'])


def _with_data(operator, rng):
    m, n = operator.shape
    y = torch.randn(BATCH, m, dtype=operator.dtype, generator=rng)
    x0 = torch.randn(BATCH, n, dtype=operator.dtype, generator=rng)
    return operator, y, x0


def _cmf_problem(operator_type, batched_H, mics, r_diag):
    rng = torch.Generator().manual_seed(11)
    shape = (BATCH, N_MICS, N_SOURCES) if batched_H else (N_MICS, N_SOURCES)
    H = torch.randn(shape, dtype=torch.complex128, generator=rng).requires_grad_()
    operator = operator_type(H, r_diag=r_diag)
    if mics is not None:
        operator = operator.select_mics(torch.tensor(mics))
    return _with_data(operator, rng)


class ProblemCases:
    @case(tags=('cmf', 'dictionary'))
    @CMF_LAYOUTS
    @R_DIAG
    def case_dictionary(self, batched_H, mics, r_diag):
        return _cmf_problem(CMFDictionary, batched_H, mics, r_diag)

    @case(tags='cmf')
    @CMF_LAYOUTS
    @R_DIAG
    def case_propagation(self, batched_H, mics, r_diag):
        return _cmf_problem(CMFPropagation, batched_H, mics, r_diag)

    @case(tags='rows')
    @parametrize(
        'rows',
        [None, [0, 2, 3, 5, 8, 13], [[0, 2, 3, 5, 8, 13], [1, 2, 4, 5, 9, 15]]],
        ids=['full', 'shared-rows', 'per-problem-rows'],
    )
    def case_matrix(self, rows):
        rng = torch.Generator().manual_seed(12)
        A = torch.randn(N_MICS * N_MICS, N_SOURCES, dtype=torch.float64, generator=rng).requires_grad_()
        operator = MatrixMult(A)
        if rows is not None:
            operator = operator.select_rows(torch.tensor(rows))
        return _with_data(operator, rng)
