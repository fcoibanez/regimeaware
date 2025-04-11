import numpy as np
import statsmodels.api as sm
import pandas as pd
from regimeaware.core.opt import min_EVaR_portfolio
from regimeaware.routines import cfg
from itertools import product

# TODO: Improve the erro handling
# TODO: Ignore existing files
# TODO: Add parameterization 
# TODO: multiply the filtered prob by the transition matrix

# --------------------------------------------
# Simulated data
fctr_rt = pd.read_pickle(f"{cfg.fldr}/data/sim/fctr_rt.pkl")
sec_rt = pd.read_pickle(f"{cfg.fldr}/data/sim/sec_rt.pkl")
post_probs = pd.read_pickle(f"{cfg.fldr}/data/sim/probs.pkl")
mdl_hmm = pd.read_pickle(f"{cfg.fldr}/data/sim/mdl_hmm.pkl")

iterations = product(range(cfg.num_simulations), cfg.num_stocks, cfg.var_alphas)

for iter, n_stocks, alpha in iterations:
    print(f"Iteration: {iter}, Number of stocks: {n_stocks}, Alpha: {alpha}")

    W = np.zeros((cfg.test_periods, n_stocks))
    
    for t in range(cfg.test_periods):
        # Data available as of t
        Mt = mdl_hmm.means_
        St = mdl_hmm.covars_
        
        # TODO: using the generative model, instead of the as-of-date iterative estimation
        Gt = post_probs[iter][:cfg.trn_periods + t]
        Xt = fctr_rt[iter][:cfg.trn_periods + t]
        Yt = sec_rt[iter][:cfg.trn_periods + t, :n_stocks]

        Vt = {}
        Rt = {}

        # RWLS
        for s in range(cfg.n_states):
            g = Gt[:, s]  # Posterior probabilities
            B_s = np.zeros(shape=(n_stocks, Xt.shape[1] + 1))
            E_s = np.zeros(shape=(n_stocks, n_stocks))

            for i in range(n_stocks):
                y = Yt[:, i]  # Endog
                x = sm.add_constant(Xt)  # Exog

                # Regimes-weighted least squares
                mdl = sm.WLS(y, x, weights=g, hasconst=True).fit()
                B_s[i] = mdl.params
                E_s[i, i] = np.var(mdl.resid)

            V_st = B_s[:, 1:] @ St[s] @ B_s[:, 1:].T + E_s  # Stock-level covariance matrix
            Vt[s] = V_st + V_st.conj().T / 2  # Make sure it is a Hermitian symmetric matrix.
            Rt[s] = (B_s @ np.concatenate([np.array([1]), Mt[s]]).reshape(-1, 1)).flatten()  # Stock-level returns vector

        pi_t = Gt[-1]

        # Portfolio optimization
        try:
            w, _, _ = min_EVaR_portfolio(alpha=alpha, L=1, mus=Rt, sigmas=Vt, pi=pi_t)
            W[t] = w.flatten()
        except Exception as e:
            print(f"Optimization failed at t={t}: {e}")
            # W[t] = np.array([np.nan] * n_stocks)
            break

    res = pd.DataFrame(W)
    res.to_pickle(f"{cfg.fldr}/results/model/I{iter}_N{n_stocks}_A{alpha}.pkl")
