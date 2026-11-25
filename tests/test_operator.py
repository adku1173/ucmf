"""Shared linear-operator contracts and Hermitian CMF vectorization."""

import acoular as ac

from ucmf.optim.operator import (
    CMFDictionary,
    CMFPropagation,
    HermitianVectorization,
    MatrixMult,
    _AuxiliaryPropagationOperator,
)
from ucmf.optim.utils import default_gamma_from_operator

from cases.models import ModelCases, build_model
from cases.operators import BATCH_SIZE, OperatorCases, cmf_matrix
from cases.problems import NUM_MICS
from helpers import assert_operator_matches

import numpy as np
import pytest
import torch
from pytest_cases import parametrize_with_cases

C = 4
OPERATOR = 'operator, matrix'


def per_problem(tensor, index, dim):
    """Reference selection: problem ``b`` takes ``index_b`` along ``dim`` of its 2D tensor."""
    tensors = tensor if tensor.ndim == 3 else tensor.expand(BATCH_SIZE, -1, -1)
    indices = index if index.ndim == 2 else index.expand(BATCH_SIZE, -1)
    return torch.stack([t.index_select(dim, i) for t, i in zip(tensors, indices, strict=True)])


def assert_matches_on_random_inputs(operator, matrix):
    """``forward`` and ``adjoint`` of ``operator`` act like ``matrix`` (shared or per problem)."""
    rng = torch.Generator().manual_seed(42)
    m, n = matrix.shape[-2:]
    x = torch.randn(BATCH_SIZE, n, dtype=matrix.dtype, generator=rng)
    r = torch.randn(BATCH_SIZE, m, dtype=matrix.dtype, generator=rng)
    assert_operator_matches(operator, matrix, x, r)


@parametrize_with_cases(OPERATOR, cases=OperatorCases)
def test_forward_adjoint_and_metadata(operator, matrix):
    assert_matches_on_random_inputs(operator, matrix)
    rng = torch.Generator().manual_seed(1)
    x = torch.randn(BATCH_SIZE, matrix.shape[-1], dtype=matrix.dtype, generator=rng)
    r = torch.randn(BATCH_SIZE, matrix.shape[-2], dtype=matrix.dtype, generator=rng)
    torch.testing.assert_close((operator.forward(x) * r).sum(), (x * operator.adjoint(r)).sum())
    assert operator.shape == tuple(matrix.shape[-2:]) and operator.rows is None
    assert operator.batch_size == (matrix.shape[0] if matrix.ndim == 3 else None)
    assert operator.dtype == matrix.dtype and operator.device == matrix.device


@parametrize_with_cases(OPERATOR, cases=OperatorCases)
def test_column_norms_match_matrix(operator, matrix):
    torch.testing.assert_close(operator.column_norms(), torch.linalg.vector_norm(matrix, dim=-2))


@pytest.mark.parametrize('mics', [torch.tensor([4, 0, 2]), torch.tensor([[4, 0, 1], [0, 3, 2], [4, 3, 2]])])
@parametrize_with_cases(OPERATOR, cases=OperatorCases, has_tag='cmf')
def test_column_norms_of_mic_subset_use_active_rows(operator, matrix, mics):
    sub = operator.select_mics(mics)
    expected = torch.linalg.vector_norm(per_problem(matrix, sub.rows, dim=0), dim=-2)
    torch.testing.assert_close(sub.column_norms().expand_as(expected), expected)  # shared stays (n,)


@pytest.mark.parametrize('rows', [torch.tensor([5, 0, 2, 3]), torch.tensor([[0, 2, 5, 1], [1, 2, 3, 4], [4, 0, 1, 3]])])
@parametrize_with_cases(OPERATOR, cases=OperatorCases, has_tag='rows')
def test_column_norms_of_row_subset_use_active_rows(operator, matrix, rows):
    expected = torch.linalg.vector_norm(per_problem(matrix, rows, dim=0), dim=-2)
    torch.testing.assert_close(operator.select_rows(rows).column_norms().expand_as(expected), expected)


@pytest.mark.parametrize('index', [torch.tensor([2, 0, 2]), torch.tensor(1)], ids=['per-problem', 'shared'])
@parametrize_with_cases(OPERATOR, cases=OperatorCases)
def test_select_problems_takes_bank_entries(operator, matrix, index):
    if operator.batch_size is None:
        pytest.skip('select_problems needs a per-problem operator')
    selected = operator.select_problems(index)
    assert type(selected) is type(operator) and selected.rows is None
    assert selected.batch_size == (None if index.ndim == 0 else index.numel())
    assert_matches_on_random_inputs(selected, matrix[index])


def random_scale(matrix, per_problem):
    shape = (BATCH_SIZE, matrix.shape[-1]) if per_problem else (matrix.shape[-1],)
    return torch.rand(shape, dtype=matrix.dtype, generator=torch.Generator().manual_seed(3)) + 0.5


@pytest.mark.parametrize('per_problem_scale', [False, True], ids=['shared', 'per-problem'])
@parametrize_with_cases(OPERATOR, cases=OperatorCases)
def test_scale_columns_matches_scaled_matrix(operator, matrix, per_problem_scale):
    scale = random_scale(matrix, per_problem_scale)
    scaled = operator.scale_columns(scale)
    assert type(scaled) is type(operator) and scaled.rows is None
    assert scaled.batch_size == (BATCH_SIZE if operator.batch_size or per_problem_scale else None)
    assert_matches_on_random_inputs(scaled, matrix * scale.unsqueeze(-2))


@parametrize_with_cases(OPERATOR, cases=OperatorCases, has_tag='cmf')
def test_scale_columns_commutes_with_mic_selection(operator, matrix):
    scale, mics = random_scale(matrix, True), torch.tensor([[4, 0, 1], [0, 3, 2], [4, 3, 2]])
    scaled = operator.select_mics(mics).scale_columns(scale)
    reference = operator.scale_columns(scale).select_mics(mics)
    assert type(scaled) is type(operator)
    torch.testing.assert_close(scaled.rows, reference.rows)
    torch.testing.assert_close(scaled.mics, reference.mics)
    assert_matches_on_random_inputs(scaled, cmf_matrix(reference.H, operator.r_diag))


@parametrize_with_cases(OPERATOR, cases=OperatorCases, has_tag='rows')
def test_scale_columns_commutes_with_row_selection(operator, matrix):
    scale, rows = random_scale(matrix, True), torch.tensor([5, 0, 2, 3])
    scaled = operator.select_rows(rows).scale_columns(scale)
    torch.testing.assert_close(scaled.rows, rows)
    assert_matches_on_random_inputs(scaled, operator.scale_columns(scale).select_rows(rows).matrix)


@pytest.mark.parametrize(('shape', 'message'), [((3,), 'with n=4'), ((2, 4), 'batch size 2 but the operator has 3')])
@parametrize_with_cases(OPERATOR, cases=OperatorCases)
def test_scale_validation(operator, matrix, shape, message):
    batched = operator.select_columns(torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]]))  # n = 4, B = 3
    with pytest.raises(ValueError, match=message):
        batched.scale_columns(torch.ones(shape, dtype=matrix.dtype))


@pytest.mark.parametrize(
    'cols', [torch.tensor([3, 0]), torch.tensor([[0, 3], [2, 1], [3, 2]])], ids=['shared', 'per-problem']
)
@parametrize_with_cases(OPERATOR, cases=OperatorCases)
def test_select_columns_matches_column_subset(operator, matrix, cols):
    sub = operator.select_columns(cols)
    assert_matches_on_random_inputs(sub, per_problem(matrix, cols, dim=1))
    assert sub.shape == (matrix.shape[-2], cols.shape[-1]) and sub.rows is None
    assert sub.batch_size == (BATCH_SIZE if operator.batch_size or cols.ndim == 2 else None)


@pytest.mark.parametrize(
    'mics', [torch.tensor([4, 0, 2]), torch.tensor([[4, 0, 1], [0, 3, 2], [4, 3, 2]])], ids=['shared', 'per-problem']
)
@parametrize_with_cases(OPERATOR, cases=OperatorCases, has_tag='cmf')
def test_select_mics_matches_subarray_and_original_rows(operator, matrix, mics):
    """Selected microphones form the CSM block of the subarray; rows and mics refer to the full array."""
    H, r_diag = operator.H, operator.r_diag
    sub = operator.select_mics(mics)
    torch.testing.assert_close(operator.mic_rows(mics), sub.rows)
    sorted_mics = mics.sort(dim=-1).values  # the block keeps ascending microphone order
    torch.testing.assert_close(sub.mics, sorted_mics)
    assert_matches_on_random_inputs(sub, cmf_matrix(per_problem(H, sorted_mics, dim=0), r_diag))
    assert_matches_on_random_inputs(sub, per_problem(matrix, sub.rows, dim=0))

    nested = sub.select_mics(torch.tensor([2, 0]))  # local indices into the selected block
    torch.testing.assert_close(nested.mics, sorted_mics[..., torch.tensor([0, 2])])
    assert_matches_on_random_inputs(nested, cmf_matrix(per_problem(H, nested.mics, dim=0), r_diag))
    assert_matches_on_random_inputs(nested, per_problem(matrix, nested.rows, dim=0))

    reduced = nested.select_columns(torch.tensor([3, 0]))
    torch.testing.assert_close(reduced.rows, nested.rows)
    torch.testing.assert_close(reduced.mics, nested.mics)


@pytest.mark.parametrize(
    'rows',
    [torch.tensor([5, 0, 2, 3]), torch.tensor([[0, 2, 5, 1], [1, 2, 3, 4], [4, 0, 1, 3]])],
    ids=['shared', 'per-problem'],
)
@parametrize_with_cases(OPERATOR, cases=OperatorCases, has_tag='rows')
def test_select_rows_matches_row_subset_and_keeps_original_indices(operator, matrix, rows):
    sub = operator.select_rows(rows)
    assert type(sub) is MatrixMult
    assert_matches_on_random_inputs(sub, per_problem(matrix, rows, dim=0))
    torch.testing.assert_close(sub.rows, rows)

    local = torch.tensor([[0, 2], [3, 1], [1, 0]])
    nested = sub.select_rows(local)
    original = torch.stack([r[i] for r, i in zip(rows.expand(BATCH_SIZE, -1), local, strict=True)])
    torch.testing.assert_close(nested.rows, original)
    assert_matches_on_random_inputs(nested, per_problem(matrix, original, dim=0))
    torch.testing.assert_close(nested.select_columns(torch.tensor([0, 2])).rows, original)


@pytest.mark.parametrize('operator_type', [CMFDictionary, CMFPropagation])
@pytest.mark.parametrize('r_diag', [False, True])
@parametrize_with_cases('spec', cases=ModelCases, has_tag='learned')
def test_model_rejects_mic_subset_from_incompatible_full_array(spec, operator_type, r_diag):
    model = build_model(spec, r_diag)
    H = torch.ones(NUM_MICS + 1, 3, dtype=torch.complex128)
    operator = operator_type(H, r_diag=r_diag).select_mics(torch.tensor([0, 1, 2]))
    operator = operator.select_mics(torch.tensor([0, 1])).select_columns(torch.tensor([0, 2])).detached()
    y = torch.ones(operator.shape[0], dtype=operator.dtype)
    with pytest.raises(ValueError, match='but Br? expects'):
        model(y, operator=operator)


@parametrize_with_cases(OPERATOR, cases=OperatorCases, has_tag='rows')
@parametrize_with_cases('spec', cases=ModelCases, has_tag=('learned', 'dense'))
def test_model_rejects_row_subset_from_incompatible_full_matrix(spec, operator, matrix):
    model = build_model(spec)
    subset = operator.select_rows(torch.tensor([0, 1, 2])).select_rows(torch.tensor([0, 1]))
    subset = subset.select_columns(torch.tensor([0, 1]))
    y = matrix.new_ones((operator.batch_size or 1, subset.shape[0]))
    with pytest.raises(ValueError, match='but B expects'):
        model(y, operator=subset)

    # An already sliced matrix with explicit rows carries no full measurement count.
    model(y, operator=MatrixMult(subset.matrix, rows=subset.rows))


@pytest.mark.parametrize(
    ('index', 'message'),
    [(torch.tensor([0.0, 1.0]), 'integer tensor'), (torch.zeros(2, 2, 2, dtype=torch.long), '1D .shared. or 2D')],
)
@parametrize_with_cases(OPERATOR, cases=OperatorCases)
def test_index_validation(operator, matrix, index, message):
    with pytest.raises(ValueError, match=message):
        operator.select_columns(index)


@parametrize_with_cases(OPERATOR, cases=OperatorCases)
def test_per_problem_index_must_match_batch_size(operator, matrix):
    batched = operator.select_columns(torch.tensor([[0], [1], [2]]))
    with pytest.raises(ValueError, match='batch size 2 but the operator has 3'):
        batched.select_columns(torch.zeros(2, 1, dtype=torch.long))


@parametrize_with_cases(OPERATOR, cases=OperatorCases)
def test_detached_keeps_subset_and_input_gradients(operator, matrix):
    index = torch.tensor([3, 1, 0], device=matrix.device)
    if isinstance(operator, (CMFDictionary, CMFPropagation)):
        source = operator.H.clone().requires_grad_()
        subset = type(operator)(source, r_diag=operator.r_diag).select_mics(index)
    else:
        source = operator.matrix.clone().requires_grad_()
        subset = MatrixMult(source).select_rows(index)

    detached = subset.detached()
    assert detached is not subset and type(detached) is type(subset)
    assert detached.rows is subset.rows
    assert detached.shape == subset.shape and detached.batch_size == subset.batch_size
    assert detached.device == subset.device and detached.dtype == subset.dtype
    if isinstance(subset, (CMFDictionary, CMFPropagation)):
        assert detached.mics is subset.mics and detached.r_diag == subset.r_diag
        assert subset.H.requires_grad and not detached.H.requires_grad
        assert detached.H.data_ptr() == subset.H.data_ptr()
    if isinstance(subset, MatrixMult):
        assert subset.matrix.requires_grad and not detached.matrix.requires_grad
        assert detached.matrix.data_ptr() == subset.matrix.data_ptr()

    m, n = subset.shape
    x = matrix.new_ones((BATCH_SIZE, n), requires_grad=True)
    r = matrix.new_ones((BATCH_SIZE, m), requires_grad=True)
    forward, adjoint = detached.forward(x), detached.adjoint(r)
    torch.testing.assert_close(forward, subset.forward(x))
    torch.testing.assert_close(adjoint, subset.adjoint(r))
    (forward.sum() + adjoint.sum()).backward()
    assert source.grad is None
    torch.testing.assert_close(x.grad, adjoint.detach())
    torch.testing.assert_close(r.grad, forward.detach())


@parametrize_with_cases(OPERATOR, cases=OperatorCases)
def test_default_step_depends_only_on_the_linear_map(operator, matrix):
    gamma = default_gamma_from_operator(operator)
    torch.testing.assert_close(gamma, default_gamma_from_operator(MatrixMult(matrix)), rtol=1e-12, atol=0)


def test_operator_step_does_not_track_gradients():
    matrix = torch.eye(2, dtype=torch.float64, requires_grad=True)
    gamma = default_gamma_from_operator(MatrixMult(matrix))
    assert not gamma.requires_grad
    torch.testing.assert_close(gamma, matrix.new_tensor(2 / 1.05))


def test_operator_step_rejects_zero_operator():
    with pytest.raises(ValueError, match='finite, nonzero spectral norm'):
        default_gamma_from_operator(MatrixMult(torch.zeros(2, 2)))


@pytest.mark.parametrize('r_diag', [False, True], ids=['with-diagonal', 'without-diagonal'])
@pytest.mark.parametrize('batched', [False, True], ids=['shared-Br', 'batched-Br'])
def test_auxiliary_propagation_forward_and_adjoint_match_dense_reference(r_diag, batched):
    # Orthonormal Hermitian basis in CMF order for two microphones.
    basis = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]], [[0, 1j], [-1j, 0]]],
        dtype=torch.complex128,
    )
    basis[[1, 3]] /= np.sqrt(2)
    rng = torch.Generator().manual_seed(9)
    Br = torch.randn((BATCH_SIZE, 2, 2) if batched else (2, 2), dtype=basis.dtype, generator=rng)
    transformed = Br.unsqueeze(-3) @ basis @ Br.mH.unsqueeze(-3)
    full = torch.einsum('...kij,lij->...lk', transformed, basis.conj()).real
    # Removing the diagonal restricts both the input and the output of B.
    rows = torch.tensor([1, 3])
    matrix = full[..., rows, :][..., rows] if r_diag else full
    operator = _AuxiliaryPropagationOperator(Br, HermitianVectorization(2, r_diag=r_diag))
    x = torch.randn(BATCH_SIZE, matrix.shape[-1], dtype=matrix.dtype, generator=rng)
    r = torch.randn(x.shape, dtype=x.dtype, generator=rng)
    assert_operator_matches(operator, matrix, x, r)


def random_hermitian(*batch, r_diag=False):
    """Real (symmetric) and imaginary (antisymmetric) parts of random Hermitian matrices."""
    rng = torch.Generator().manual_seed(0)
    X = torch.randn(*batch, C, C, dtype=torch.float64, generator=rng)
    Y = torch.randn(*batch, C, C, dtype=torch.float64, generator=rng)
    real, imag = X + X.mT, Y - Y.mT
    if r_diag:
        real = real - torch.diag_embed(torch.diagonal(real, dim1=-2, dim2=-1))
    return real, imag


@pytest.mark.parametrize('r_diag', [False, True])
def test_dimensions(r_diag):
    assert HermitianVectorization(C, r_diag=r_diag).num_measurements == (C * (C - 1) if r_diag else C * C)


def test_diagonal_rows_select_the_r_diag_vectorization():
    real, imag = random_hermitian(3)
    full, reduced = HermitianVectorization(C), HermitianVectorization(C, r_diag=True)
    torch.testing.assert_close(reduced(real, imag), full(real, imag)[..., ~full.diagonal_rows()])
    assert full.diagonal_rows().sum() == C and not reduced.diagonal_rows().any()


@pytest.mark.parametrize('r_diag', [False, True])
def test_isometry_and_inverse_on_its_domain(r_diag):
    hvec = HermitianVectorization(C, r_diag=r_diag)
    real, imag = random_hermitian(3, r_diag=r_diag)
    v = hvec(real, imag)
    frobenius = (real.square().sum((-2, -1)) + imag.square().sum((-2, -1))).sqrt()
    torch.testing.assert_close(torch.linalg.vector_norm(v, dim=-1), frobenius)
    back_real, back_imag = hvec.adjoint(v)
    torch.testing.assert_close(back_real, real)
    torch.testing.assert_close(back_imag, imag)
    w = torch.randn(3, hvec.num_measurements, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(hvec(*hvec.adjoint(w)), w)


@pytest.mark.parametrize('r_diag', [False, True])
def test_adjoint_satisfies_dot_product_test(r_diag):
    """<hvec(S), v> = <S, hvec*(v)>_F for Hermitian S (with diagonal for r_diag=True)."""
    hvec = HermitianVectorization(C, r_diag=r_diag)
    real, imag = random_hermitian(5)
    v = torch.randn(5, hvec.num_measurements, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    adj_real, adj_imag = hvec.adjoint(v)
    lhs = (hvec(real, imag) * v).sum(-1)
    rhs = (real * adj_real).sum((-2, -1)) + (imag * adj_imag).sum((-2, -1))
    torch.testing.assert_close(lhs, rhs)


@pytest.mark.parametrize('r_diag', [False, True])
def test_matches_acoular_ordering_up_to_row_weights(r_diag):
    real, imag = random_hermitian()
    csm = (real + 1j * imag).numpy()
    freq_data = ac.PowerSpectraImport(csm=csm[None], frequencies=np.array([1000.0]))
    acoular_vector = ac.BeamformerCMF(freq_data=freq_data, r_diag=r_diag)._vectorize_csm(csm)[:, 0]
    hvec = HermitianVectorization(C, r_diag=r_diag)
    expected = hvec.row_weights(real) * torch.from_numpy(acoular_vector)
    torch.testing.assert_close(hvec(real, imag), expected)


def test_follows_device_and_dtype():
    hvec = HermitianVectorization(C)
    real, imag = (part.float() for part in random_hermitian())
    assert hvec(real, imag).dtype == torch.float32
    assert hvec.adjoint(hvec(real, imag))[0].dtype == torch.float32
    if torch.cuda.is_available():
        on_cuda = HermitianVectorization(C, device='cuda')
        torch.testing.assert_close(on_cuda(real.cuda(), imag.cuda()).cpu(), hvec(real, imag))
