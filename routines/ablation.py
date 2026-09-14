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
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from regimeaware.constants import (
    CK_WINDOW,
    DataConstants,
    HMMParameters,
    SimulationParameters,
)
from regimeaware.core.estimation import (
    EMISSIONS,
    factor_moments,
    fit_regimes,
    forecast_probs,
    guard_weights,
    regime_weights,
)
from regimeaware.core.moments import collapse, fit_wls, project
from regimeaware.core.optimize import solve_portfolio
from regimeaware.core.simdata import load_simulation

ARMS = {
    "ck_uni": dict(emission="uni", loadings="hard", objective="mvo"),
    "ck_multi": dict(emission="multi", loadings="hard", objective="mvo"),
    "rwls_mvo": dict(emission="multi", loadings="rwls", objective="mvo"),
    "rwls_mixture": dict(emission="multi", loadings="rwls", objective="mixture"),
}


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

    # A non-default state count is written alongside the main result rather than
    # over it, so the faithful two-state replication of Costa & Kwon (2020) and
    # the three-state arm used for the incremental comparison can coexist.
    suffix = "" if n_states == HMMParameters.STATES.value else f"_{n_states}s"
    outdir = f"{DataConstants.WDIR.value}/results/{args.arm}{suffix}"
    os.makedirs(outdir, exist_ok=True)

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

            # Regime identification on the information set available at t
            mdl_hmm, probs = fit_regimes(Xt[emission].values, n_states, warm_from)
            warm_from = mdl_hmm
            pi_t = forecast_probs(probs, mdl_hmm.transmat_)

            # State-dependent asset moments
            design = np.column_stack([np.ones(len(Xt)), Xt.values])
            weights = regime_weights(probs, spec["loadings"], args.window)

            means, covs = [], []
            for s in range(n_states):
                w = guard_weights(weights[s], design.shape[1], args.window)

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
