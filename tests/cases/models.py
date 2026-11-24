"""Model cases: every solver as a ``ModelSpec`` for the properties shared by all models.

Tags: ``classical`` (ISTA/FISTA) or ``unfolded``; unfolded models are ``learned``
or ``na``. ``dense`` (accepts ``MatrixMult``), ``cmf_only`` (requires CMF
coordinates), ``weight_mode`` (factored/precomputed), ``two_weights``
(``W_recurrent`` callback).
"""

from dataclasses import make_dataclass

from cases.problems import NUM_MICS
from ucmf import FISTA, ISTA, AdaLFISTA, AdaLFISTASW, NA_AdaLFISTA, NA_AdaLFISTASW
from ucmf.optim import AdaLFISTAR, AdaLFISTASWR, NA_AdaLFISTAR, NA_AdaLFISTASWR
from ucmf.optim.neural_augmentation import NeuralAugmentation
from ucmf.optim.operator import CMFDictionary, CMFPropagation, HermitianVectorization
from ucmf.optim.utils import softplus_inverse

import torch
from helpers import model_step_params
from pytest_cases import case

T = 4
LSTM_DIM = 5
# Fixed lam / gamma of the classical solvers.
LAM, GAMMA = 0.03, 0.3
CONTROLLER_STATE = [
    'controller.hidden_init',
    'controller.cell_init',
    'controller.feature_mean',
    'controller.feature_std',
    'controller.lll.weight',
    'controller.lll.bias',
    'controller.linear.weight',
    'controller.linear.bias',
    'controller.lstm.weight_ih',
    'controller.lstm.weight_hh',
    'controller.lstm.bias_ih',
    'controller.lstm.bias_hh',
]

ModelSpec = make_dataclass(
    'ModelSpec',
    [
        'make_model', 'model_kwargs', 'solve', 'dense_weights', 'step_params', 'accelerated',
        'state_names', 'init_weights',
    ],
    frozen=True,
    namespace={'__doc__': """How to build, run, and densely reproduce one model.

    - ``make_model(num_mics, r_diag, **kwargs)`` calls the constructor for a full
      array of ``num_mics`` microphones; ``model_kwargs`` are its default arguments.
    - ``solve(model, y, operator, x0=None, t=T)`` runs the forward.
    - ``dense_weights(model, operator)`` returns ``(M, P, Q)`` on the active rows,
      differentiable in the model parameters: the update direction at ``p`` is
      ``(M.T @ y - P.T @ (Q @ p)) / m``.
    - ``step_params(model, batch_size)`` gives the step parameters for
      ``helpers.reference_iterates``; ``accelerated`` selects FISTA momentum.
    - ``state_names`` are the expected ``state_dict`` keys; ``init_weights`` the
      constructor arguments for initial auxiliary weights.
    """},
)


def build_model(spec, r_diag=False, *, positive=False, **kwargs):
    """Float64 model with identity auxiliary weights for the full array of ``NUM_MICS``.

    NA models get the fixed feature normalization ``mean = 0``, ``std = 1``. The random
    controller init uses a fixed seed, independent of test order and global RNG state.
    """
    with torch.random.fork_rng(devices=[]):  # models are built on the CPU
        torch.manual_seed(0)
        model = spec.make_model(NUM_MICS, r_diag, positive=positive, **{**spec.model_kwargs, **kwargs}).double()
    if isinstance(model, NeuralAugmentation):
        model.controller.feature_mean.zero_()
        model.controller.feature_std.fill_(1.0)
    return model


def set_constant_step_params(model, gamma, theta):
    """Set every applied ``gamma`` / ``theta`` of an unfolded model exactly (in its dtype)."""
    with torch.no_grad():
        if isinstance(model, NeuralAugmentation):
            # The readout weight starts at zero, so the bias alone sets the output.
            bias = model.controller.linear.bias
            bias.copy_(torch.tensor([softplus_inverse(gamma), softplus_inverse(theta)], dtype=bias.dtype))
        else:
            model._gamma.fill_(softplus_inverse(gamma))
            model._theta.fill_(softplus_inverse(theta))


def perturb_weights(model):
    """Make every 2D parameter (auxiliary weights, NA controller matrices) dense and non-identity."""
    rng = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.add_(0.2 * torch.randn(parameter.shape, dtype=parameter.dtype, generator=rng))
    return model


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


def congruence_weight(Br_re, Br_im, num_mics, r_diag):
    """Dense ``B`` with ``B v = hvec(Br hvec*(v) Br^H)`` on the full CMF basis."""
    hvec = HermitianVectorization(num_mics, r_diag=r_diag)
    basis = torch.eye(hvec.num_measurements, dtype=Br_re.dtype)
    Br = torch.complex(Br_re, Br_im)
    weighted = Br @ torch.complex(*hvec.adjoint(basis)) @ Br.mH
    return hvec(weighted.real, weighted.imag).mT


def classical_weights(model, operator):
    A = dense_matrix(operator)
    return A, A, A


def sw_weights(model, operator):
    A = dense_matrix(operator)
    W = restrict(model.B, operator.rows) @ A
    return W, W, A


def lfista_weights(model, operator):
    A = dense_matrix(operator)
    R = restrict(model.B1, operator.rows) @ A
    M = restrict(model.B2, operator.rows) @ A
    return M, R, R


def swr_weights(model, operator):
    A = dense_matrix(operator)
    B = congruence_weight(model.Br_re, model.Br_im, model.num_mics, model.r_diag)
    W = restrict(B, operator.rows) @ A
    return W, W, A


def r_weights(model, operator):
    A = dense_matrix(operator)
    B1 = congruence_weight(model.Br1_re, model.Br1_im, model.num_mics, model.r_diag)
    B2 = congruence_weight(model.Br2_re, model.Br2_im, model.num_mics, model.r_diag)
    R = restrict(B1, operator.rows) @ A
    M = restrict(B2, operator.rows) @ A
    return M, R, R


def dense_size(num_mics, r_diag):
    return {'num_measurements': num_mics * (num_mics - int(r_diag))}


def reduced_size(num_mics, r_diag):
    return {'num_mics': num_mics, 'r_diag': r_diag}


def classical_spec(model_type, accelerated):
    return ModelSpec(
        make_model=lambda num_mics, r_diag, **kwargs: model_type(**kwargs),
        model_kwargs={},
        solve=lambda model, y, operator, x0=None, t=T: model(y, x0=x0, t=t, operator=operator, lam=LAM, gamma=GAMMA),
        dense_weights=classical_weights,
        step_params=lambda model, batch_size: (0, lambda residual, direction, k: (GAMMA, LAM * GAMMA, k + 1)),
        accelerated=accelerated,
        state_names=[],
        init_weights=(),
    )


def unfolded_spec(model_type, size, dense_weights, weight_names, init_weights=()):
    if issubclass(model_type, NeuralAugmentation):
        step_kwargs, step_names = {'lstm_dim': LSTM_DIM}, CONTROLLER_STATE
    else:
        step_kwargs, step_names = {'num_iterations': T}, ['_gamma', '_theta']
    return ModelSpec(
        make_model=lambda num_mics, r_diag, **kwargs: model_type(**size(num_mics, r_diag), **kwargs),
        model_kwargs={'gamma_init': 0.3, 'theta_init': 0.01, **step_kwargs},
        solve=lambda model, y, operator, x0=None, t=T: model(y, x0=x0, t=t, operator=operator),
        dense_weights=dense_weights,
        step_params=model_step_params,
        accelerated=True,
        state_names=[*weight_names, *step_names],
        init_weights=init_weights,
    )


class ModelCases:
    @case(tags=('classical', 'dense'))
    def case_ista(self):
        return classical_spec(ISTA, accelerated=False)

    @case(tags=('classical', 'dense'))
    def case_fista(self):
        return classical_spec(FISTA, accelerated=True)

    @case(tags=('unfolded', 'learned', 'dense', 'weight_mode'))
    def case_sw(self):
        return unfolded_spec(AdaLFISTASW, dense_size, sw_weights, ['B'], ('B',))

    @case(tags=('unfolded', 'learned', 'dense', 'weight_mode', 'two_weights'))
    def case_lfista(self):
        return unfolded_spec(AdaLFISTA, dense_size, lfista_weights, ['B1', 'B2'], ('B1', 'B2'))

    @case(tags=('unfolded', 'learned', 'cmf_only'))
    def case_swr(self):
        return unfolded_spec(AdaLFISTASWR, reduced_size, swr_weights, ['Br_re', 'Br_im'])

    @case(tags=('unfolded', 'learned', 'cmf_only', 'two_weights'))
    def case_r(self):
        return unfolded_spec(AdaLFISTAR, reduced_size, r_weights, ['Br1_re', 'Br1_im', 'Br2_re', 'Br2_im'])

    @case(tags=('unfolded', 'na', 'dense', 'weight_mode'))
    def case_na_sw(self):
        return unfolded_spec(NA_AdaLFISTASW, dense_size, sw_weights, ['B'], ('B',))

    @case(tags=('unfolded', 'na', 'dense', 'weight_mode', 'two_weights'))
    def case_na_lfista(self):
        return unfolded_spec(NA_AdaLFISTA, dense_size, lfista_weights, ['B1', 'B2'], ('B1', 'B2'))

    @case(tags=('unfolded', 'na', 'cmf_only'))
    def case_na_swr(self):
        return unfolded_spec(NA_AdaLFISTASWR, reduced_size, swr_weights, ['Br_re', 'Br_im'])

    @case(tags=('unfolded', 'na', 'cmf_only', 'two_weights'))
    def case_na_r(self):
        return unfolded_spec(NA_AdaLFISTAR, reduced_size, r_weights, ['Br1_re', 'Br1_im', 'Br2_re', 'Br2_im'])
