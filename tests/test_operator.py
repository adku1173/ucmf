"""HermitianVectorization is a Frobenius isometry with Acoular's CMF ordering."""

import acoular as ac

from ucmf.optim.operator import HermitianVectorization

import numpy as np
import pytest
import torch

C = 4


def random_hermitian(*batch, r_diag=False):
    """Real (symmetric) and imaginary (antisymmetric) parts of random Hermitian matrices."""
    torch.manual_seed(0)
    X = torch.randn(*batch, C, C, dtype=torch.float64)
    Y = torch.randn(*batch, C, C, dtype=torch.float64)
    real, imag = X + X.mT, Y - Y.mT
    if r_diag:
        real = real - torch.diag_embed(torch.diagonal(real, dim1=-2, dim2=-1))
    return real, imag


@pytest.mark.parametrize('r_diag', [False, True])
def test_dimensions(r_diag):
    assert HermitianVectorization(C, r_diag=r_diag).n_measurements == (C * (C - 1) if r_diag else C * C)


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
    w = torch.randn(3, hvec.n_measurements, dtype=torch.float64)
    torch.testing.assert_close(hvec(*hvec.adjoint(w)), w)


@pytest.mark.parametrize('r_diag', [False, True])
def test_adjoint_satisfies_dot_product_test(r_diag):
    """<hvec(S), v> = <S, hvec*(v)>_F for Hermitian S (with diagonal for r_diag=True)."""
    hvec = HermitianVectorization(C, r_diag=r_diag)
    real, imag = random_hermitian(5)
    v = torch.randn(5, hvec.n_measurements, dtype=torch.float64)
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
    hvec = HermitianVectorization(C).to(torch.float32)
    real, imag = (part.float() for part in random_hermitian())
    assert hvec(real, imag).dtype == torch.float32
    assert hvec.adjoint(hvec(real, imag))[0].dtype == torch.float32
    if torch.cuda.is_available():
        hvec = hvec.cuda()
        torch.testing.assert_close(hvec(real.cuda(), imag.cuda()).cpu(), hvec.cpu()(real, imag))
