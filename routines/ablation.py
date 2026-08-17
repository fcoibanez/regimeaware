"""Incremental path from the Costa & Kwon (2020) benchmark to the RWLS framework.

Each arm adds exactly one component to the previous one, so the contribution of
every methodological choice can be read off the difference between consecutive
arms:

    ck_uni        univariate HMM  | hard classification | collapsed MVO
    ck_multi      multivariate    | hard classification | collapsed MVO
    rwls_mvo      multivariate    | RWLS                | collapsed MVO
    rwls_mixture  multivariate    | RWLS                | full mixture utility

``ck_uni`` reproduces the estimator of Costa & Kwon (2020); ``rwls_mixture``
reproduces the framework proposed in this paper. The two intermediate arms
isolate the value of the multivariate emission distribution and of the
regime-weighted least squares estimator respectively.

Regime probabilities are always estimated on the data available as of the
decision date, and the factor moments of each regime are computed under that
arm's own regime assignment, so the emission dimension affects the results only
through the quality of the regime classification.
"""

import argparse
import random

import cvxpy as cvx
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm

from regimeaware.constants import (
    CK_WINDOW,
    DataConstants,
    Factors,
    HMMParameters,
    SimulationParameters,
)
from regimeaware.core.moments import collapse, fit_wls, project, symmetrize
from regimeaware.core.simdata import load_simulation

ARMS = {
    "ck_uni": dict(emission="uni", loadings="hard", objective="mvo"),
    "ck_multi": dict(emission="multi", loadings="hard", objective="mvo"),
    "rwls_mvo": dict(emission="multi", loadings="rwls", objective="mvo"),
    "rwls_mixture": dict(emission="multi", loadings="rwls", objective="mixture"),
}

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


def regime_weights(probs, loadings, window):
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


def solve_portfolio(phi, probs, means, covs, n_assets):
    """Long-only allocation maximizing expected exponential utility.

    Minimizes the cumulant generating function of equation (10). A single
    component reduces this to mean-variance optimization with risk aversion
    ``phi``, which is the objective used by the collapsed-moment arms.
    """
    w_var = cvx.Variable(n_assets)
    terms = [
        np.log(probs[s])
        - phi * means[s] @ w_var
        + (phi**2 / 2) * cvx.quad_form(w_var, covs[s])
        for s in range(len(probs))
    ]
    objective = cvx.Minimize(cvx.log_sum_exp(cvx.vstack(terms)))
    constraints = [w_var >= 0, cvx.sum(w_var) == 1]
    cvx.Problem(objective, constraints).solve(solver="MOSEK")
    return w_var.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--states", default=HMMParameters.STATES.value, type=int)
    parser.add_argument("--window", default=CK_WINDOW, type=int)
    parser.add_argument("--trials", default=SimulationParameters.TRIALS.value, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--shard",
        default="0/1",
        help="Process only iterations i with i %% n == k, given as 'k/n'. Lets "
        "several workers split one arm without duplicating any iteration.",
    )
    args = parser.parse_args()

    shard_k, shard_n = (int(x) for x in args.shard.split("/"))

    spec = ARMS[args.arm]
    emission = EMISSIONS[spec["emission"]]
    n_assets = SimulationParameters.NUM_STOCKS.value
    n_states = args.states
    is_periods = SimulationParameters.IS_PERIODS.value
    phis = SimulationParameters.RISK_AVERSION.value

    outdir = f"{DataConstants.WDIR.value}/results/{args.arm}"

    iterations = [i for i in range(args.trials) if i % shard_n == shard_k]
    random.shuffle(iterations) if args.shuffle else None

    sec_rt, fctr_rt = load_simulation(
        args.trials, n_assets, DataConstants.WDIR.value
    )

    for i in iterations:
        fnames = {phi: f"{outdir}/phi{phi}_iter{i}.pkl" for phi in phis}
        try:
            for fname in fnames.values():
                pd.read_pickle(fname)
            continue
        except FileNotFoundError:
            pass

        collect_wts = {phi: {} for phi in phis}
        warm_from = None

        for t in tqdm(
            range(SimulationParameters.OOS_PERIODS.value),
            desc=f"{args.arm} (iteration {i})",
        ):
            # Data available as of t
            Xt = fctr_rt[i].loc[: is_periods + t]
            Yt = sec_rt[i].loc[: is_periods + t].iloc[:, :n_assets]

            # Regime identification on the information set available at t. The
            # last row of the posterior is the filtered probability, since the
            # backward pass contributes nothing at the end of the sample.
            mdl_hmm = build_hmm(n_states, warm_from)
            mdl_hmm.fit(Xt[emission].values)

            if not is_healthy(mdl_hmm):
                # The warm start led into a degenerate region; discard it and
                # refit from the k-means initialisation instead.
                mdl_hmm = build_hmm(n_states, None)
                mdl_hmm.fit(Xt[emission].values)
                if not is_healthy(mdl_hmm):
                    raise RuntimeError(
                        f"HMM failed to converge at iteration {i}, period {t}"
                    )

            warm_from = mdl_hmm

            probs = mdl_hmm.predict_proba(Xt[emission].values)
            pi_t = np.clip(probs[-1] @ mdl_hmm.transmat_, 1e-12, None)
            pi_t /= pi_t.sum()

            # State-dependent asset moments
            design = np.column_stack([np.ones(len(Xt)), Xt.values])
            weights = regime_weights(probs, spec["loadings"], args.window)

            means, covs = [], []
            for s in range(n_states):
                w = weights[s]
                if w.sum() <= Xt.shape[1] + 2:
                    # Too few observations attributed to this state to identify
                    # the loadings; fall back to the most recent window.
                    w = np.zeros(len(Xt))
                    w[-args.window :] = 1.0

                coefs, resid_var = fit_wls(design, Yt.values, weights=w)
                lambda_s, omega_s = factor_moments(Xt.values, w)
                mu_s, V_s = project(coefs, resid_var, lambda_s, omega_s)
                means.append(mu_s)
                covs.append(V_s)

            if spec["objective"] == "mvo":
                mu, V = collapse(pi_t, means, covs)
                probs_opt, means_opt, covs_opt = np.array([1.0]), [mu], [V]
            else:
                probs_opt, means_opt, covs_opt = pi_t, means, covs

            for phi in phis:
                try:
                    w_opt = solve_portfolio(
                        phi, probs_opt, means_opt, covs_opt, n_assets
                    )
                except Exception as e:
                    print(f"Optimization failed (phi={phi}, t={t}): {e}")
                    w_opt = np.array([np.nan] * n_assets)
                collect_wts[phi][is_periods + t] = w_opt

        for phi in phis:
            wts_iter = pd.DataFrame.from_dict(collect_wts[phi], orient="index")
            wts_iter.index = pd.MultiIndex.from_product(
                [[i], wts_iter.index], names=["iteration", "period"]
            )
            wts_iter.to_pickle(fnames[phi])
