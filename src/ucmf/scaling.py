"""Problem scaling around the solvers: column normalization and back-transformation.

The solvers work on a normalized operator ``D = A U^-1`` with
``U = diag(||a_1||_2, ..., ||a_n||_2)`` and return the normalized solution
``x = U q``. These functions form the operator and map ``x`` back to the source
strengths ``q``; the solvers themselves never see ``U``.
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
