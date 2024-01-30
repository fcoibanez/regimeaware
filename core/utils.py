"""Auxiliary function."""
import pandas as pd
import numpy as np
from regimeaware.routines.cfg import factor_set


def looking_fwd_prob(transmat: pd.DataFrame, current_emission_prob: pd.Series, horizon: int = 1):
    """Looking-forward emission probabilities.

    Combines the transition matrix and current emission probabilities to come up with
    ex-ante probabilities at a specific horizon (number of periods) used to get expected
    values.

    :param transmat:
    :param current_emission_prob:
    :param horizon:
    :return:
    """
    B = transmat.values
    B_h = np.linalg.matrix_power(B, horizon)
    g = current_emission_prob.values.reshape(-1, 1)
    g_t = B_h.T @ g

    return pd.Series(g_t.flatten(), index=transmat.index)


def unpack_betas(raw):
    """xyz.

    :param raw:
    :return:
    """
    res = pd.pivot_table(raw.reset_index(name='value'), index='id', columns='factor', values='value')
    return res[['const'] + factor_set]


def project_covariances(betas: pd.DataFrame, factor_covariance, residual_variances):
    """

    :param betas: n-by-k
    :param factor_covariance:
    :param residual_variances:
    """
    theta = betas.reindex(factor_covariance.index, axis=1).values.T
    sigma_f = factor_covariance.values
    sigma_e = np.diag(residual_variances.reindex(betas.index))
    sigma_s = theta.T @ sigma_f @ theta + sigma_e
    res = pd.DataFrame(sigma_s, index=betas.index, columns=betas.index)
    return res


def project_means(betas, factor_means):
    """

    :param betas:
    :param factor_means:
    """
    theta = betas.values.T
    mu_f = factor_means.reindex(betas.columns)
    if 'const' in mu_f.index:
        mu_f['const'] = 1
    mu = theta.T @ mu_f.values.reshape(-1, 1)
    res = pd.Series(mu.flatten(), index=betas.index)
    return res


def expected_covar(conditional_covars, probs, conditional_means):
    """Combines probabilities and regime covariance matrices.

    :param conditional_covars:
    :param conditional_means:
    :param probs:
    :return:
    """
    w = probs.copy()
    w.index.name = "state"
    V_st = np.zeros((conditional_covars.shape[1], conditional_covars.shape[1]))
    for s in range(len(probs)):
        V = conditional_covars.xs(s).values
        mu_s = conditional_means[conditional_covars.columns].xs(s).values.reshape(-1, 1)
        g = probs[s]
        V_st += g * (V + mu_s @ mu_s.T)
    mu_st = expected_means(conditional_means, probs).values.reshape(-1, 1)
    V_st -= mu_st @ mu_st.T
    V_st = pd.DataFrame(V_st, index=conditional_covars.columns, columns=conditional_means.columns)
    return V_st


def expected_means(conditional_means, probs):
    """Combines probabilities and regime means vector.

    :param conditional_means:
    :param probs:
    """
    res = conditional_means.mul(probs, axis=0).sum(axis=0)
    return res
