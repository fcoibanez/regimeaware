"""Backtest of the regime-agnostic benchmark.

Deliberately ignores latent state dynamics: factor loadings and factor moments
are estimated unconditionally over the full history available at each decision
date, which by construction averages across periods of high and low volatility.

The factor covariance is left unrestricted here, whereas the regime-aware
framework restricts its state-dependent factor covariances to be diagonal. That
asymmetry favours this benchmark, so the comparison remains conservative.
"""

import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from regimeaware.constants import DataConstants, SimulationParameters
from regimeaware.core.moments import fit_wls, project
from regimeaware.core.optimize import solve_portfolio
from regimeaware.core.simdata import load_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    is_periods = SimulationParameters.IS_PERIODS.value
    phis = SimulationParameters.RISK_AVERSION.value

    outdir = f"{DataConstants.WDIR.value}/results/baseline"

    iterations = [i for i in range(args.trials) if i % shard_n == shard_k]
    random.shuffle(iterations) if args.shuffle else None

    sec_rt, fctr_rt = load_simulation(args.trials, n_assets, DataConstants.WDIR.value)

    for i in iterations:
        fnames = {phi: f"{outdir}/phi{phi}_iter{i}.pkl" for phi in phis}
        try:
            for fname in fnames.values():
                pd.read_pickle(fname)
            continue
        except FileNotFoundError:
            pass

        collect_wts = {phi: {} for phi in phis}

        for t in tqdm(
            range(SimulationParameters.OOS_PERIODS.value),
            desc=f"baseline (iteration {i})",
        ):
            # Data available as of t
            Xt = fctr_rt[i].loc[: is_periods + t]
            Yt = sec_rt[i].loc[: is_periods + t].iloc[:, :n_assets]

            design = np.column_stack([np.ones(len(Xt)), Xt.values])
            coefs, resid_var = fit_wls(design, Yt.values)

            mu, V = project(
                coefs,
                resid_var,
                Xt.values.mean(axis=0),
                np.cov(Xt.values, rowvar=False),
            )

            # A single component reduces the objective to mean-variance with
            # risk aversion phi. The moments do not depend on phi, so they are
            # built once and reused across the grid.
            probs = np.array([1.0])
            for phi in phis:
                try:
                    w_opt = solve_portfolio(phi, probs, [mu], [V], n_assets)
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
