"""Construction of asset-level moments from factor-model parameters.

Shared by the benchmark routines so that the projection from factor space to
asset space is defined in exactly one place. Every estimator in the incremental
path differs only in how the loadings are obtained; the projection itself is
common to all of them.
"""

import numpy as np


def symmetrize(V):
    """Return the symmetric part of ``V``.

    Guards against the small asymmetries that accumulate in floating-point
    matrix products before the matrix is handed to a convex solver.
    """
    return (V + V.T) / 2


def fit_wls(X, Y, weights=None):
    """Weighted least squares of every column of ``Y`` on ``X``, solved jointly.

    All assets share the same design matrix, so the normal equations are formed
    once and solved for the whole cross-section at once rather than looping over
    assets.

    :param X: (T, K+1) design matrix, intercept in column 0.
    :param Y: (T, N) matrix of asset excess returns.
    :param weights: (T,) observation weights. Uniform (OLS) when None.
    :return: (N, K+1) coefficients, (N,) residual variances.
    """
    if weights is None:
        weights = np.ones(len(X))

    Xw = X * weights[:, None]
    coefs = np.linalg.solve(X.T @ Xw, Xw.T @ Y).T

    resid = Y - X @ coefs.T

    # A regime whose posterior mass is smaller than the parameter count would
    # otherwise be handed negative degrees of freedom, producing negative
    # variances and an asset covariance matrix that is not positive semidefinite.
    # Callers are expected to avoid that case (see core.estimation.guard_weights);
    # this floor is here so that a caller which does not cannot silently produce
    # an invalid covariance matrix.
    dof = max(weights.sum() - X.shape[1], 1.0)
    resid_var = np.maximum((weights[:, None] * resid**2).sum(axis=0) / dof, 0.0)

    return coefs, resid_var


def project(coefs, resid_var, factor_mean, factor_cov):
    """Project factor-model parameters into asset-level moments.

    Implements equation (8): the state-conditional mean vector and covariance
    matrix of the investable universe implied by a set of factor loadings and
    the factor distribution.

    :param coefs: (N, K+1) intercept in column 0, loadings in columns 1..K.
    :param resid_var: (N,) idiosyncratic variances.
    :param factor_mean: (K,) factor risk premia.
    :param factor_cov: (K, K) factor covariance matrix.
    :return: (N,) mean vector, (N, N) covariance matrix.
    """
    loadings = coefs[:, 1:]
    mu = coefs @ np.concatenate([[1.0], factor_mean])
    V = loadings @ factor_cov @ loadings.T + np.diag(resid_var)
    return mu, symmetrize(V)


def collapse(probs, means, covs):
    """Collapse regime-conditional moments into the moments of the mixture.

    Used by the estimators that optimize over a single Gaussian rather than
    over the full mixture: the M state-dependent distributions are replaced by
    the mean and covariance of the mixture they define.

    :param probs: (M,) mixing weights.
    :param means: list of M (N,) mean vectors.
    :param covs: list of M (N, N) covariance matrices.
    :return: (N,) mean vector, (N, N) covariance matrix.
    """
    mu = sum(p * m for p, m in zip(probs, means))
    V = sum(p * (S + np.outer(m, m)) for p, S, m in zip(probs, covs, means))
    return mu, symmetrize(V - np.outer(mu, mu))
