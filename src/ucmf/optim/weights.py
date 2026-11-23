"""Per-solve gradient weight helpers for unfolded sparse solvers."""

from .operator import (
    CMFDictionary,
    CMFPropagation,
    MatrixMult,
    _AuxiliaryOperator,
    _AuxiliaryPropagationOperator,
    _validate_index,
)
from .utils import validate_tensor

import torch


def _active_weight(operator, B):
    """Restrict ``B`` to the original rows represented by the operator."""
    validate_tensor(B, name='B', dtype=operator.dtype, device=operator.device)
    m, _ = operator.shape
    num_measurements_total = m if operator.rows is None else operator._num_measurements_total
    if num_measurements_total is not None and num_measurements_total != B.shape[0]:
        raise ValueError(f'operator has {num_measurements_total} measurements in total but B expects {B.shape[0]}.')
    if operator.rows is None:
        return B
    rows = _validate_index(operator.rows, 'operator.rows', operator.device, operator.batch_size)
    if rows.shape[-1] != m:
        raise ValueError(f'operator.rows length must match the measurement dimension {m}.')
    if rows.ndim == 2 and operator.batch_size is None:
        raise ValueError('Per-problem operator.rows requires a batched operator.')
    return B[rows.unsqueeze(-1), rows.unsqueeze(-2)]


def make_weight_operator(operator, B, weight_mode='factored'):
    """Build ``B_active @ A`` as a composition or precompute its dense product."""
    if weight_mode == 'precomputed':
        if not isinstance(operator, MatrixMult):
            raise ValueError("weight_mode='precomputed' requires a dense MatrixMult or CMFDictionary operator.")
        matrix = apply_auxiliary_weight(_active_weight(operator, B), operator.matrix)
        return operator._with_matrix(matrix, rows=operator.rows)
    return _AuxiliaryOperator(operator, MatrixMult(_active_weight(operator, B)))


def validate_cmf_operator(operator, num_mics, r_diag):
    """Require compatible CMF coordinates for a reduced auxiliary weight."""
    if not isinstance(operator, (CMFDictionary, CMFPropagation)):
        raise ValueError('operator must be a CMFDictionary or CMFPropagation.')
    if operator.r_diag != r_diag:
        raise ValueError('operator.r_diag must match model.r_diag.')
    if operator._num_mics_total != num_mics:
        raise ValueError(f'operator has {operator._num_mics_total} microphones in total but Br expects {num_mics}.')


def make_propagation_weight_operator(operator, Br_re, Br_im):
    """Compose the fixed CMF operator with a learned weight on its active microphones."""
    validate_tensor(Br_re, name='Br_re', dtype=operator.dtype, device=operator.device)
    validate_tensor(Br_im, name='Br_im', dtype=operator.dtype, device=operator.device)
    Br = torch.complex(Br_re, Br_im)
    mics = operator.mics
    if mics is not None:
        Br = Br[mics.unsqueeze(-1), mics.unsqueeze(-2)]
    return _AuxiliaryOperator(operator, _AuxiliaryPropagationOperator(Br, operator.hvec))


def apply_auxiliary_weight(B, A_tensor):
    """Form the effective weight from the auxiliary weight ``B``: ``W = B @ A``.

    Handles shared or per-sample ``B`` and shared or batched ``A``.
    """
    if B.ndim == 3:
        # Per-sample B: W_b = B_b @ A_b, (b, p, m) @ (b, m, n) -> (b, p, n).
        return torch.bmm(B, A_tensor)
    if A_tensor.ndim == 2:
        # Shared B and A: (p, m) @ (m, n) -> (p, n).
        return B @ A_tensor
    # Shared B, batched A: broadcast B over the batch, (p, m) @ (b, m, n) -> (b, p, n).
    return torch.bmm(B.unsqueeze(0).expand(A_tensor.shape[0], -1, -1), A_tensor)
