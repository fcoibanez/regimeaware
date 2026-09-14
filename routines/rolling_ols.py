"""Backtest of the rolling-window regime-agnostic benchmark.

A tighter control than the expanding-window baseline: re-estimating the factor
model on a short trailing window lets the loadings vary over time without
invoking regimes at all, so it captures parameter instability through a purely
mechanical channel. Any remaining advantage of the regime-aware framework has to
come from the regime structure itself rather than from time variation per se.

``--window`` sets the trailing window in months, which lets the sensitivity of
this benchmark to that choice be reported rather than assumed.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from regimeaware.constants import ROLLING_WINDOW, DataConstants, SimulationParameters
from regimeaware.core.moments import fit_wls, project
from regimeaware.core.optimize import solve_portfolio
from regimeaware.core.simdata import load_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default=ROLLING_WINDOW, type=int)
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

    # The default window keeps the canonical output location; any other window is
    # written alongside it so the sensitivity analysis does not overwrite it.
    suffix = "" if args.window == ROLLING_WINDOW else f"_{args.window}m"
    outdir = f"{DataConstants.WDIR.value}/results/rolling_ols{suffix}"
    os.makedirs(outdir, exist_ok=True)

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
            desc=f"rolling_ols[{args.window}m] (iteration {i})",
        ):
            # Trailing window of the data available as of t
            Xt = fctr_rt[i].loc[: is_periods + t].iloc[-args.window :]
            Yt = sec_rt[i].loc[: is_periods + t].iloc[-args.window :, :n_assets]

            design = np.column_stack([np.ones(len(Xt)), Xt.values])
            coefs, resid_var = fit_wls(design, Yt.values)

            mu, V = project(
                coefs,
                resid_var,
                Xt.values.mean(axis=0),
                np.cov(Xt.values, rowvar=False),
            )

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
