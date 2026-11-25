"""Problem scaling around the solvers: column normalization and back-transformation.

The solvers work on a normalized operator ``D = A U^-1`` with
``U = diag(||a_1||_2, ..., ||a_n||_2)`` and return the normalized solution
``x = U q``. These functions form the operator and map ``x`` back to the source
strengths ``q``; the solvers themselves never see ``U``.

Equiregularization divides measurements and source strengths per problem by
:func:`lambda_max` of the normalized operator, so that ``lam`` in ``[0, 1]`` has the
same relative meaning for all problems. Physical source strengths are
``lambda_max * to_source(x, u)``.
"""


def normalize_operator(operator):
    """Normalize the operator columns to unit L2 norm.

    Parameters
    ----------
    operator : LinearOperator
        Operator ``A`` providing ``column_norms`` and ``scale_columns``; all
        column norms must be positive.

    Returns
    -------
    D : LinearOperator
        Operator ``A U^-1`` of the same type with unit column norms.
    u : Tensor
        Column norms of ``A``, shared ``(n,)`` or per problem ``(B, n)``.
    """
    u = operator.column_norms()
    return operator.scale_columns(1 / u), u


def lambda_max(D, y):
    """Smallest regularization for which the nonnegative LASSO solution is zero.

    ``lambda_max = max_j d_j^T y / m`` for ``||y - D x||^2 / (2 m) + lam ||x||_1`` with
    ``x >= 0``. After dividing ``y`` by it, ``lambda_max = 1``. For cross-validation,
    compute it from the training subset and divide both subsets by it.

    Parameters
    ----------
    D : LinearOperator
        Normalized operator of the active measurements.
    y : Tensor
        Measurements of shape ``(B, m)``, aligned with ``D.rows``. The result must
        be positive (unchecked precondition).

    Returns
    -------
    Tensor
        ``lambda_max`` of shape ``(B,)``.
    """
    return D.adjoint(y).amax(dim=-1) / y.shape[-1]


def to_source(x, u):
    """Map a normalized solution back to the source strengths ``q = U^-1 x``.

    Parameters
    ----------
    x : Tensor
        Normalized solutions of shape ``(B, n)``.
    u : Tensor
        Column norms from :func:`normalize_operator`, ``(n,)`` or ``(B, n)``.

    Returns
    -------
    Tensor
        Source strengths of shape ``(B, n)``.
    """
    return x / u
