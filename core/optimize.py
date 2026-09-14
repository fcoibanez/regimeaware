"""Portfolio construction under expected exponential utility."""

import cvxpy as cvx
import numpy as np


def solve_portfolio(phi, probs, means, covs, n_assets, solver="MOSEK"):
    """Long-only allocation maximizing expected exponential utility.

    Minimizes the cumulant generating function of equation (10). Passing a single
    component reduces this to mean-variance optimization with risk aversion
    ``phi``, which is the objective used by the single-regime benchmarks.

    :param phi: absolute risk aversion.
    :param probs: (M,) forecast probability of each regime.
    :param means: list of M (N,) regime-conditional mean vectors.
    :param covs: list of M (N, N) regime-conditional covariance matrices.
    :param n_assets: size of the investable universe.
    :return: (N,) portfolio weights.
    """
    w = cvx.Variable(n_assets)
    terms = [
        np.log(probs[s])
        - phi * means[s] @ w
        + (phi**2 / 2) * cvx.quad_form(w, covs[s])
        for s in range(len(probs))
    ]
    objective = cvx.Minimize(cvx.log_sum_exp(cvx.vstack(terms)))
    constraints = [w >= 0, cvx.sum(w) == 1]
    cvx.Problem(objective, constraints).solve(solver=solver)
    return w.value
