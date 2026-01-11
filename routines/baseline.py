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
            fname = f"{DataConstants.WDIR.value}/results/baseline/phi{phi}_iter{i}.pkl"
            # try:
            #     wts_iter = pd.read_pickle(fname)
            #     continue
            # except FileNotFoundError:
            #     pass

            collect_wts = {}

            for t in tqdm(
                range(SimulationParameters.OOS_PERIODS.value),
                desc=f"Risk aversion φ={phi} (iteration {i})",
            ):
                # Data available as of t
                Xt = fctr_rt[i].loc[: SimulationParameters.IS_PERIODS.value + t]
                Yt = sec_rt[i].loc[: SimulationParameters.IS_PERIODS.value + t]

                Vt = {}
                Rt = {}

                # Single-regime benchmark
                B = np.zeros(
                    shape=(SimulationParameters.NUM_STOCKS.value, Xt.shape[1] + 1)
                )
                E = np.zeros(
                    shape=(
                        SimulationParameters.NUM_STOCKS.value,
                        SimulationParameters.NUM_STOCKS.value,
                    )
                )

                for j in range(SimulationParameters.NUM_STOCKS.value):
                    y = Yt[j]  # Endog
                    x = sm.add_constant(Xt)  # Exog
                    mdl_ols = sm.OLS(endog=y, exog=x, hasconst=True).fit()

                    B[j] = mdl_ols.params
                    E[j, j] = np.var(mdl_ols.resid)

                Mt = np.mean(x, axis=0).values.reshape(-1, 1)
                St = np.cov(x, rowvar=False)

                Vt = B @ St @ B.T + E  # Stock-level covariance matrix
                Vt = [
                    Vt + Vt.conj().T / 2
                ]  # Make sure it is a Hermitian symmetric matrix.
                Rt = [(B @ Mt).flatten()]  # Stock-level returns vector
                pi_t = [1]

                # Portfolio optimization (Regime-aware)
                try:

                    def K(w):
                        """
                        Objective function for regime-aware portfolio optimization (eq. 11).

                        Computes expected log-utility across regimes, weighted by pi_t.

                        Args:
                            w: Portfolio weights (cvx.Variable).

                        Returns:
                            Log-sum-exp of regime utilities (cvx.Expression).
                        """
                        u = cvx.vstack(
                            [
                                cvx.log(pi_t[i])
                                - phi * Rt[i] @ w
                                + (phi**2 / 2) * cvx.quad_form(w, Vt[i])
                                for i in range(len(pi_t))
                            ]
                        )
                        return cvx.log_sum_exp(u)

                    w = cvx.Variable(SimulationParameters.NUM_STOCKS.value)
                    objective = cvx.Minimize(K(w))
                    constraints = [w >= 0, cvx.sum(w) == 1]
                    egm_prob = cvx.Problem(objective, constraints)
                    egm_prob.solve(solver="MOSEK")
                    collect_wts[SimulationParameters.IS_PERIODS.value + t] = w.value

                except Exception as e:
                    print(f"Optimization failed: {e}")
                    collect_wts[SimulationParameters.IS_PERIODS.value + t] = np.array(
                        [np.nan] * SimulationParameters.NUM_STOCKS.value
                    )
                    continue

            wts_iter = pd.DataFrame.from_dict(collect_wts, orient="index")
            wts_iter.index = pd.MultiIndex.from_product(
                [[i], wts_iter.index], names=["iteration", "period"]
            )
            wts_iter.to_pickle(fname)
