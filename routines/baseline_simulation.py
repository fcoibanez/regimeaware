import numpy as np
import statsmodels.api as sm
import pandas as pd
from regimeaware.core.opt import min_EVaR_portfolio
from regimeaware.routines import cfg
from itertools import product

# --------------------------------------------
# Simulated data
fctr_rt = pd.read_pickle(f"{cfg.fldr}/data/sim/fctr_rt.pkl")
sec_rt = pd.read_pickle(f"{cfg.fldr}/data/sim/sec_rt.pkl")

iterations = product(range(cfg.num_simulations), cfg.num_stocks, cfg.var_alphas)

for iter, n_stocks, alpha in iterations:
    print(f"Iteration: {iter}, Number of stocks: {n_stocks}, Alpha: {alpha}")

    W = np.zeros((cfg.test_periods, n_stocks))
    
    for t in range(cfg.test_periods):
        # Data available as of t
        Xt = fctr_rt[iter][:cfg.trn_periods + t]
        Yt = sec_rt[iter][:cfg.trn_periods + t, :n_stocks]

        # Single-regime benchmark
        B = np.zeros(shape=(n_stocks, Xt.shape[1] + 1))
        E = np.zeros(shape=(n_stocks, n_stocks))

        for i in range(n_stocks):
            y = Yt[:, i]  # Endog
            x = sm.add_constant(Xt)  # Exog
            mdl_ols = sm.OLS(endog=y, exog=x, hasconst=True).fit()

            B[i] = mdl_ols.params
            E[i, i] = np.var(mdl_ols.resid)

        Mt = np.mean(x, axis=0)
        St = np.cov(x, rowvar=False)

        Vt = B @ St @ B.T + E  # Stock-level covariance matrix
        Vt = Vt + Vt.conj().T / 2  # Make sure it is a Hermitian symmetric matrix.
        Rt = (B @ Mt.reshape(-1, 1)).flatten()  # Stock-level returns vector

        # Portfolio optimization
        try:
            w, _, _ = min_EVaR_portfolio(alpha=alpha, L=1, mus=[Rt], sigmas=[Vt], pi=[1])
            W[t] = w.flatten()
        except Exception as e:
            print(f"Optimization failed at t={t}: {e}")
            # W[t] = np.array([np.nan] * n_stocks)
            break

    res = pd.DataFrame(W)
    res.to_pickle(f"{cfg.fldr}/results/baseline/I{iter}_N{n_stocks}_A{alpha}.pkl")