"""Regime identification and regime-conditional estimation.

Holds the pieces that are shared between the routines: fitting the hidden Markov
model on the information set available at a decision date, turning its posterior
probabilities into estimation weights, and computing the regime-conditional
factor moments implied by those weights.
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM

from regimeaware.constants import Factors, HMMParameters
from regimeaware.core.moments import symmetrize

EMISSIONS = {"uni": ["mktrf"], "multi": Factors._member_names_}


def build_hmm(n_states, warm_from=None):
    """Instantiate the HMM, warm-starting from the previous fit when available.

    Warm-starting keeps the per-period refit cheap and, as a side effect, holds
    the state labels stable across decision dates within a simulation path.

    The initial state distribution is held uniform and excluded from the update.
    Estimated from a single sequence it is not identified -- it collapses onto
    whichever state the path happens to start in -- and warm-starting from a
    vector containing exact zeros makes the forward pass degenerate as soon as
    the extended sample favours a different initial state. With T in the
    hundreds it has no material effect on the likelihood.
    """
    kwargs = dict(
        n_components=n_states,
        covariance_type=HMMParameters.COV.value,
        n_iter=HMMParameters.ITER.value,
        min_covar=HMMParameters.MINCOV.value,
        tol=HMMParameters.TOL.value,
        implementation=HMMParameters.IMPLEMENTATION.value,
    )
    if warm_from is None:
        mdl = GaussianHMM(
            random_state=HMMParameters.SEED.value,
            init_params="tmc",
            params="tmc",
            **kwargs,
        )
        mdl.startprob_ = np.full(n_states, 1 / n_states)
        return mdl

    transmat = np.clip(warm_from.transmat_, 1e-8, None)

    mdl = GaussianHMM(init_params="", params="tmc", **kwargs)
    mdl.startprob_ = np.full(n_states, 1 / n_states)
    mdl.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)
    mdl.means_ = warm_from.means_
    mdl.covars_ = np.diagonal(warm_from.covars_, axis1=1, axis2=2).copy()
    return mdl


def is_healthy(mdl):
    """True when every estimated parameter is finite and properly normalised."""
    params = [mdl.startprob_, mdl.transmat_, mdl.means_, mdl.covars_]
    return all(np.all(np.isfinite(p)) for p in params) and np.allclose(
        mdl.transmat_.sum(axis=1), 1.0
    )


def fit_regimes(X, n_states, warm_from=None):
    """Fit the HMM on ``X`` and return it together with its posteriors.

    ``X`` must contain only observations available as of the decision date. The
    last row of the returned posterior is then the *filtered* probability
    P(s_t | x_1..x_t): the backward lattice is initialised to one at the end of
    the sample, so it contributes nothing there. Earlier rows are smoothed given
    the same truncated sample, which is the correct Baum-Welch quantity for
    weighting historical observations and uses no future information.
    """
    mdl = build_hmm(n_states, warm_from)
    mdl.fit(X)

    if not is_healthy(mdl):
        # The warm start led into a degenerate region; discard it and refit from
        # the k-means initialisation instead.
        mdl = build_hmm(n_states, None)
        mdl.fit(X)
        if not is_healthy(mdl):
            raise RuntimeError("HMM failed to converge from a cold start")

    return mdl, mdl.predict_proba(X)


def forecast_probs(probs, transmat, floor=1e-12):
    """One-step-ahead state probabilities from the filtered probabilities."""
    pi = np.clip(probs[-1] @ transmat, floor, None)
    return pi / pi.sum()


def regime_weights(probs, loadings, window=None):
    """Observation weights defining each regime's estimation sample.

    Hard classification assigns every observation to its most likely state and
    keeps the most recent ``window`` of them, discarding the rest. Soft
    assignment keeps every observation and weights it by its posterior
    probability. With two states the hard rule coincides with the 0.5 threshold
    used by Costa & Kwon (2020).
    """
    n_obs, n_states = probs.shape

    if loadings == "rwls":
        return [probs[:, s] for s in range(n_states)]

    labels = probs.argmax(axis=1)
    weights = []
    for s in range(n_states):
        w = np.zeros(n_obs)
        idx = np.flatnonzero(labels == s)[-window:]
        w[idx] = 1.0
        weights.append(w)
    return weights


def align_states(mdl, reference):
    """Permutation matching a fitted model's states onto a reference model's.

    Expectation-maximisation numbers states arbitrarily: which of them comes out
    as state 0 depends on the initialisation, so the same economic regime carries
    different indices on different simulation paths. Anything that compares an
    estimated state against a known one -- a confusion matrix, a hit rate, a
    regime-conditional table -- has to undo that first, or it measures the
    labelling rather than the model.

    This is a reporting convention and nothing more. The portfolio problem is
    invariant to it, because the forecast probabilities and the state-conditional
    moments permute together, so no information from the reference model reaches
    the allocation.

    :param mdl: fitted model whose states are to be relabelled.
    :param reference: model whose labelling is treated as canonical.
    :return: array ``perm`` with ``perm[j] == k`` when state ``j`` of ``mdl``
        corresponds to state ``k`` of ``reference``.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    # Factors differ in scale by an order of magnitude, so distances are taken on
    # standardised means to stop the most volatile factor dominating the match.
    scale = reference.means_.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)

    cost = cdist(mdl.means_ / scale, reference.means_ / scale)
    rows, cols = linear_sum_assignment(cost)

    perm = np.empty(len(rows), dtype=int)
    perm[rows] = cols
    return perm


def guard_weights(weights, n_params, window):
    """Fall back when a regime carries too little mass to identify the loadings.

    A regime's effective sample size is the total posterior mass assigned to it.
    When that falls below the number of regression parameters the weighted
    residual variance is estimated on negative degrees of freedom, turns negative,
    and the projected asset covariance matrix stops being positive semidefinite --
    which the convex solver then rejects outright.

    In that situation the regime is simply not identified from the data, so the
    estimate falls back to the most recent ``window`` observations, the same rule
    the hard-classification benchmarks use when a state is sparsely populated.

    :param weights: (T,) observation weights for one regime.
    :param n_params: number of parameters in the regression, i.e. factors plus
        intercept.
    :param window: trailing observations to use when falling back.
    :return: the original weights, or the fallback.
    """
    if weights.sum() > n_params + 1:
        return weights

    fallback = np.zeros(len(weights))
    fallback[-window:] = 1.0
    return fallback


def factor_moments(X, w, diagonal=True):
    """Regime-conditional factor mean and covariance under weighting ``w``.

    For soft assignment this reproduces the M-step estimates of the HMM; for
    hard assignment it is the sample moment over the classified window. The
    covariance is restricted to be diagonal, consistent with the treatment of
    the state-dependent factor covariances elsewhere in the paper.
    """
    total = w.sum()
    mean = (w[:, None] * X).sum(axis=0) / total
    dev = X - mean
    cov = (w[:, None] * dev).T @ dev / total
    return mean, np.diag(np.diag(cov)) if diagonal else symmetrize(cov)
