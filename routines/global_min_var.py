"""
Backtesting script for baseline portfolio optimization.

Implements Monte Carlo simulations (5,000 trials) to evaluate out-of-sample performance
of regime-agnostic baseline approach. Manages 200-stock portfolios
"""

import cvxpy as cvx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
import random
import argparse
from regimeaware.constants import DataConstants, HMMParameters, SimulationParameters

if __name__ == "__main__":
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", default=False, type=bool)
    args = parser.parse_args()

    iterations = list(range(SimulationParameters.TRIALS.value))
    random.shuffle(iterations) if args.shuffle else None

    sec_rt = pd.read_pickle(f"{DataConstants.WDIR.value}/data/sim/sec_rt.pkl")
    fctr_rt = pd.read_pickle(f"{DataConstants.WDIR.value}/data/sim/fctr_rt.pkl")

    for phi in SimulationParameters.RISK_AVERSION.value:
        for i in iterations:
            fname = f"{DataConstants.WDIR.value}/results/global_min_var/phi{phi}_iter{i}.pkl"

            collect_wts = {}

            for t in tqdm(
                range(SimulationParameters.OOS_PERIODS.value),
                desc=f"Risk aversion φ={phi} (iteration {i})",
            ):
                # Data available as of t
                Yt = sec_rt[i].loc[: SimulationParameters.IS_PERIODS.value + t, :SimulationParameters.NUM_STOCKS.value]
                
                Sigma = np.cov(Yt.values.T)
                n = Sigma.shape[0]
                w = cvx.Variable(n)
                objective = cvx.Minimize(cvx.quad_form(w, Sigma))
                constraints = [cvx.sum(w) == 1, w >= 0]
                cvx.Problem(objective, constraints).solve()

                collect_wts[SimulationParameters.IS_PERIODS.value + t] = w.value

            wts_iter = pd.DataFrame.from_dict(collect_wts, orient="index")
            wts_iter.index = pd.MultiIndex.from_product(
                [[i], wts_iter.index], names=["iteration", "period"]
            )
            wts_iter.to_pickle(fname)
