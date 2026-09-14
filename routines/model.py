"""Backtest of the regime-aware framework: HMM regime identification with RWLS.

Constructs the portfolio of equation (10) over the Monte Carlo simulation paths,
maximizing expected exponential utility over the full Gaussian mixture of
regime-conditional return distributions.

The regime model can be supplied in two ways, selected with ``--regimes``:

    estimated   the HMM is fitted at every decision date on the observations
                available up to that date. This is the implementable variant and
                shares its information set with the regime-agnostic benchmark.
    oracle      the HMM parameters of the data-generating process are supplied
                directly. This is not implementable, but bounds what the
                framework could deliver if regime identification were perfect,
                so the gap between the two isolates how much of the
                outperformance comes from identifying regimes rather than from
                projecting them into asset space.

Based on the empirical section of "Incorporating Market Regimes into Large-Scale
Stock Portfolios: A Hidden Markov Model Approach".
"""

import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from regimeaware.constants import (
    CK_WINDOW,
    DataConstants,
    Factors,
    HMMParameters,
    SimulationParameters,
)
from regimeaware.core.estimation import (
    factor_moments,
    fit_regimes,
    forecast_probs,
    guard_weights,
    regime_weights,
)
from regimeaware.core.moments import fit_wls, project
from regimeaware.core.simdata import load_simulation
from regimeaware.core.optimize import solve_portfolio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", choices=["estimated", "oracle"], default="estimated")
    parser.add_argument("--states", default=HMMParameters.STATES.value, type=int)
    parser.add_argument("--trials", default=SimulationParameters.TRIALS.value, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--shard",
        default="0/1",
        help="Process only iterations i with i %% n == k, given as 'k/n'.",
    )
    args = parser.parse_args()

    shard_k, shard_n = (int(x) for x in args.shard.split("/"))

    n_assets = SimulationParameters.NUM_STOCKS.value
    n_states = args.states
    is_periods = SimulationParameters.IS_PERIODS.value
    phis = SimulationParameters.RISK_AVERSION.value
    factors = Factors._member_names_

    # results/model holds the output of the original pipeline and is left
    # untouched: it is the record of the submitted draft's numbers, needed to
    # account for what changed between rounds.
    outdir = f"{DataConstants.WDIR.value}/results/model_{args.regimes}"

    iterations = [i for i in range(args.trials) if i % shard_n == shard_k]
    random.shuffle(iterations) if args.shuffle else None

    sec_rt, fctr_rt = load_simulation(args.trials, n_assets, DataConstants.WDIR.value)

    # The data-generating parameters, used only by the oracle variant
    dgp = pd.read_pickle(f"{DataConstants.WDIR.value}/data/dist/mdl_hmm.pkl")

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
            desc=f"model[{args.regimes}] (iteration {i})",
        ):
            # Data available as of t
            Xt = fctr_rt[i].loc[: is_periods + t]
            Yt = sec_rt[i].loc[: is_periods + t].iloc[:, :n_assets]

            if args.regimes == "oracle":
                # Posteriors under the true parameters, but still computed only
                # on observations up to t so that the last row is the filtered
                # probability rather than a full-sample smoothed one.
                probs = dgp.predict_proba(Xt[factors].values)
                transmat = dgp.transmat_
            else:
                mdl_hmm, probs = fit_regimes(Xt[factors].values, n_states, warm_from)
                warm_from = mdl_hmm
                transmat = mdl_hmm.transmat_

            pi_t = forecast_probs(probs, transmat)

            # Regime-conditional asset moments
            design = np.column_stack([np.ones(len(Xt)), Xt.values])
            weights = regime_weights(probs, "rwls", window=None)

            means, covs = [], []
            for s in range(probs.shape[1]):
                w = guard_weights(weights[s], design.shape[1], CK_WINDOW)

                # Regime-weighted least squares. The residual variance is
                # weighted by the same posteriors as the regression itself,
                # which is what the model's sigma^2_{i,s} actually is.
                coefs, resid_var = fit_wls(design, Yt.values, weights=w)

                if args.regimes == "oracle":
                    lambda_s = dgp.means_[s]
                    omega_s = dgp.covars_[s]
                else:
                    lambda_s, omega_s = factor_moments(Xt.values, w)

                mu_s, V_s = project(coefs, resid_var, lambda_s, omega_s)
                means.append(mu_s)
                covs.append(V_s)

            # The moments do not depend on risk aversion, so they are built once
            # and reused across the whole grid of phi.
            for phi in phis:
                try:
                    w_opt = solve_portfolio(phi, pi_t, means, covs, n_assets)
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
