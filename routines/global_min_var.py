"""Backtest of the long-only global minimum-variance portfolio.

A canonical benchmark that uses no return forecast at all: it depends only on the
sample covariance of asset returns over the history available at each decision
date. Because it ignores expected returns it does not depend on risk aversion,
so the same weights are recorded for every value of phi.
"""

import argparse
import random

import cvxpy as cvx
import numpy as np
import pandas as pd
from tqdm import tqdm

from regimeaware.constants import DataConstants, SimulationParameters
from regimeaware.core.moments import symmetrize
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

    outdir = f"{DataConstants.WDIR.value}/results/global_min_var"

    iterations = [i for i in range(args.trials) if i % shard_n == shard_k]
    random.shuffle(iterations) if args.shuffle else None

    sec_rt, _ = load_simulation(args.trials, n_assets, DataConstants.WDIR.value)

    for i in iterations:
        fnames = {phi: f"{outdir}/phi{phi}_iter{i}.pkl" for phi in phis}
        try:
            for fname in fnames.values():
                pd.read_pickle(fname)
            continue
        except FileNotFoundError:
            pass

        collect_wts = {}

        for t in tqdm(
            range(SimulationParameters.OOS_PERIODS.value),
            desc=f"global_min_var (iteration {i})",
        ):
            # Data available as of t. Positional slicing on the columns: label
            # based slicing is inclusive of its endpoint and would take one
            # asset more than the rest of the experiment uses.
            Yt = sec_rt[i].loc[: is_periods + t].iloc[:, :n_assets]

            Sigma = symmetrize(np.cov(Yt.values, rowvar=False))

            w = cvx.Variable(n_assets)
            objective = cvx.Minimize(cvx.quad_form(w, Sigma))
            constraints = [cvx.sum(w) == 1, w >= 0]
            cvx.Problem(objective, constraints).solve(solver="MOSEK")

            collect_wts[is_periods + t] = w.value

        wts_iter = pd.DataFrame.from_dict(collect_wts, orient="index")
        wts_iter.index = pd.MultiIndex.from_product(
            [[i], wts_iter.index], names=["iteration", "period"]
        )

        # Independent of risk aversion, but recorded per phi so that the
        # aggregation and comparison code can treat every arm identically.
        for phi in phis:
            wts_iter.to_pickle(fnames[phi])
