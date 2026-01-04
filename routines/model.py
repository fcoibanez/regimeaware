"""
Backtesting script for regime-aware portfolio optimization using HMM and RWLS.

Implements Monte Carlo simulations (5,000 trials) to evaluate out-of-sample performance
of RWLS framework against regime-agnostic baselines. Manages 200-stock portfolios across
risk aversion levels (φ=1-50), using CVXPY for optimization.

Based on empirical section of "Regime-Aware Portfolio Optimization...".
Results show improved returns and diversification.
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
    mdl_hmm = pd.read_pickle(f"{DataConstants.WDIR.value}/data/dist/mdl_hmm.pkl")

    sec_rt = pd.read_pickle(f"{DataConstants.WDIR.value}/data/sim/sec_rt.pkl")
    fctr_rt = pd.read_pickle(f"{DataConstants.WDIR.value}/data/sim/fctr_rt.pkl")
    post_probs = pd.read_pickle(f"{DataConstants.WDIR.value}/data/sim/probs.pkl")

    Mt = mdl_hmm.means_
    St = mdl_hmm.covars_

    for phi in SimulationParameters.RISK_AVERSION.value:
        for i in iterations:
            fname = f"{DataConstants.WDIR.value}/results/model/phi{phi}_iter{i}.pkl"
            try:
                wts_iter = pd.read_pickle(fname)
                continue
            except FileNotFoundError:
                pass

            collect_wts = {}

            for t in tqdm(
                range(SimulationParameters.OOS_PERIODS.value),
                desc=f"Risk aversion φ={phi} (iteration {i})",
            ):
                # Data available as of t
                Gt = post_probs[i].loc[: SimulationParameters.IS_PERIODS.value + t]
                Xt = fctr_rt[i].loc[: SimulationParameters.IS_PERIODS.value + t]
                Yt = sec_rt[i].loc[: SimulationParameters.IS_PERIODS.value + t]

                Vt = {}
                Rt = {}

                # RWLS
                for s in range(HMMParameters.STATES.value):
                    g = Gt[s]  # Posterior probabilities
                    B_s = np.zeros(
                        shape=(SimulationParameters.NUM_STOCKS.value, Xt.shape[1] + 1)
                    )
                    E_s = np.zeros(
                        shape=(
                            SimulationParameters.NUM_STOCKS.value,
                            SimulationParameters.NUM_STOCKS.value,
                        )
                    )

                    x = sm.add_constant(Xt)  # Exog
                    for j in range(SimulationParameters.NUM_STOCKS.value):
                        y = Yt[j]  # Endog

                        # Regimes-weighted least squares
                        mdl = sm.WLS(y, x, weights=g, hasconst=True).fit()
                        B_s[j] = mdl.params
                        E_s[j, j] = np.var(mdl.resid)

                    V_st = (
                        B_s[:, 1:] @ St[s] @ B_s[:, 1:].T + E_s
                    )  # Stock-level covariance matrix
                    Vt[s] = (
                        V_st + V_st.conj().T / 2
                    )  # Make sure it is a Hermitian symmetric matrix.
                    Rt[s] = (
                        B_s @ np.concatenate([np.array([1]), Mt[s]]).reshape(-1, 1)
                    ).flatten()  # Stock-level returns vector

                P = mdl_hmm.transmat_
                pi_t = Gt.iloc[-1] @ P

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
