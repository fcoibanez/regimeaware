"""
Backtesting script for equal-weighted portfolio.

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
            fname = f"{DataConstants.WDIR.value}/results/equalweighted/phi{phi}_iter{i}.pkl"
            wts_iter = pd.DataFrame(
                columns=range(SimulationParameters.NUM_STOCKS.value),
                index=range(
                    SimulationParameters.IS_PERIODS.value,
                    SimulationParameters.IS_PERIODS.value + SimulationParameters.OOS_PERIODS.value,
                ),
                dtype=float,
            )
            wts_iter.fillna(1 / SimulationParameters.NUM_STOCKS.value, inplace=True)

            wts_iter.index = pd.MultiIndex.from_product(
                [[i], wts_iter.index], names=["iteration", "period"]
            )
            wts_iter.to_pickle(fname)
