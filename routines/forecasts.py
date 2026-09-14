"""Recompute the one-step-ahead regime forecasts used at each decision date.

The backtests use the forecast probabilities but do not retain them, and they are
needed to say anything about how well regimes are actually predicted -- the hit
rate, the confusion against the realised state, and whether the forecast beats a
naive one.

Regenerating them is cheap: the costly part of the backtest is the portfolio
optimisation, not the regime model, whose refit is warm-started and takes about a
millisecond. The estimation follows exactly the same code path and the same order
as ``routines/model.py``, so the probabilities recovered here are the ones the
portfolios were actually built from rather than an approximation of them.
"""

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from regimeaware.constants import (
    DataConstants,
    Factors,
    HMMParameters,
    SimulationParameters,
)
from regimeaware.core.estimation import align_states, fit_regimes, forecast_probs
from regimeaware.core.simdata import load_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", choices=["estimated", "oracle"], default="estimated")
    parser.add_argument("--states", default=HMMParameters.STATES.value, type=int)
    parser.add_argument("--trials", default=SimulationParameters.TRIALS.value, type=int)
    args = parser.parse_args()

    n_states = args.states
    is_periods = SimulationParameters.IS_PERIODS.value
    oos_periods = SimulationParameters.OOS_PERIODS.value
    factors = Factors._member_names_

    _, fctr_rt = load_simulation(
        args.trials, SimulationParameters.NUM_STOCKS.value, DataConstants.WDIR.value
    )
    dgp = pd.read_pickle(f"{DataConstants.WDIR.value}/data/dist/mdl_hmm.pkl")

    collect_forecast = {}
    collect_filtered = {}

    for i in tqdm(range(args.trials), desc=f"forecasts[{args.regimes}]"):
        warm_from = None

        for t in range(oos_periods):
            Xt = fctr_rt[i].loc[: is_periods + t]

            if args.regimes == "oracle":
                probs = dgp.predict_proba(Xt[factors].values)
                transmat = dgp.transmat_
                perm = np.arange(n_states)
            else:
                mdl, probs = fit_regimes(Xt[factors].values, n_states, warm_from)
                warm_from = mdl
                transmat = mdl.transmat_

                # Relabel onto the generating model's states. Without this the
                # stored probabilities are indexed by whichever ordering EM
                # happened to settle on for this path, and comparing them against
                # the realised regime would score the labelling rather than the
                # forecast.
                perm = align_states(mdl, dgp)

            # The filtered probability of the current state, and the forecast it
            # implies for the state that will govern the period being traded.
            filtered = np.zeros(n_states)
            forecast = np.zeros(n_states)
            filtered[perm] = probs[-1]
            forecast[perm] = forecast_probs(probs, transmat)

            collect_filtered[(i, is_periods + t)] = filtered
            collect_forecast[(i, is_periods + t)] = forecast

    for name, collected in [("forecast", collect_forecast), ("filtered", collect_filtered)]:
        out = pd.DataFrame.from_dict(collected, orient="index")
        out.index = pd.MultiIndex.from_tuples(out.index, names=["iteration", "period"])
        out.columns.name = "state"
        out.to_pickle(
            f"{DataConstants.WDIR.value}/results/{name}_{args.regimes}.pkl"
        )
        print(f"wrote {out.shape} to results/{name}_{args.regimes}.pkl")
