if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import os
    import cvxpy as cvx
    from tqdm import tqdm

    NUM_SIMULATIONS = 100
    GAMMA = 10.0
    OOS_PERIODS = 1
    NUM_STOCKS = 500
    NUM_STATES = 2
    HM_RND = 123
    WDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --------------------------------------------
    # Data
    fctr_rt = pd.read_pickle(f"{WDIR}/data/ff.pkl")
    rf = fctr_rt.pop("rf")
    loadings_cov = pd.read_pickle(f'{WDIR}/data/loadings_cov.pkl')
    loadings_means = pd.read_pickle(f'{WDIR}/data/loadings_means.pkl')
    scale_by_regime = pd.read_pickle(f'{WDIR}/data/scale_by_regime.pkl')
    gen_hmm = pd.read_pickle(f'{WDIR}/data/mdl_hmm.pkl')
    stationary_prob = np.linalg.matrix_power(gen_hmm.transmat_, 99999999)[0, :]  # Limiting Markov chain distributions

    # Unconditional factor moments
    m_bar = fctr_rt.mean()
    f_bar = fctr_rt.cov()

    # Conditional factor moments
    m_s = gen_hmm.means_
    f_s = gen_hmm.covars_  # Covariance matrices are diagonal

    # --------------------------------------------
    # Returns simulation
    for iter in tqdm(range(NUM_SIMULATIONS)):
        print(iter)
        # Factor returns
        if iter == 0:
            gen_hmm.random_state = HM_RND
            np.random.seed(HM_RND)
        else:
            gen_hmm.random_state = np.random.randint(0, 1000000)

        X, Z = gen_hmm.sample(OOS_PERIODS + 1)
        G = gen_hmm.predict_proba(X)
        g_0 = G[0]  # Initial state probabilities

        # True factor loadings
        _vals = np.random.multivariate_normal(mean=loadings_means, cov=loadings_cov, size=NUM_STOCKS)
        B = pd.DataFrame(_vals, columns=loadings_means.index).T
        B.columns.name = "stock"
        B.index.names = ["factor", "state"]
        B = B.stack()

        # Stock returns
        s_t = Z[0]  # Dominant regime
        x_t = pd.Series(X[1], fctr_rt.columns)  # Factor return over the next period
        x_t.index.name = "factor"
        r_t = B.xs(s_t, level="state").mul(x_t, axis=0).groupby("stock").sum()
        e_s = [np.random.normal(loc=0, scale=scale_by_regime[x], size=NUM_STOCKS) for x in range(NUM_STATES)]
        e_t = e_s[s_t]  # Resids
        r_t += e_t  
        
        # Unconditional stock returns

        # Regime-conditional stock returns
        b_s = [B.xs(x, level='state').unstack('factor')[fctr_rt.columns].values for x in range(NUM_STATES)]
        r_s = [b_s[x] @ m_s[0].reshape(-1, 1) for x in range(NUM_STATES)]
        v_s = [b_s[x] @ f_s[x] @ b_s[x].T for x in range(NUM_STATES)]

        # RWLS portfolio
        pi_t = g_0 @ gen_hmm.transmat_
        w = cvx.Variable(NUM_STOCKS)
        constraints = [w >= 0, cvx.sum(w) == 1]

        def K(w):
            u = cvx.vstack([cvx.log(pi_t[i]) - GAMMA * r_s[i].T @ w + (GAMMA**2/2) * cvx.quad_form(w, v_s[i]) for i in range(NUM_STATES)])
            return cvx.log_sum_exp(u)
        
        objective = cvx.Minimize(K(w))
        prob1 = cvx.Problem(objective, constraints)
        prob1.solve(solver=cvx.MOSEK)
        rwls_wt = pd.Series(w.value)

        # Baseline portfolio
        b_bar = B.unstack("state").mul(stationary_prob).sum(axis=1).unstack("factor")[fctr_rt.columns]
        r_s = [b_bar.dot(m_bar).values]
        v_s = [b_bar.dot(f_bar).dot(b_bar.T).values]
        
        pi_t = [1]
        def K(w):
            u = cvx.vstack([cvx.log(pi_t[i]) - GAMMA * r_s[i].T @ w + (GAMMA**2/2) * cvx.quad_form(w, v_s[i]) for i in range(1)])
            return cvx.log_sum_exp(u)
        
        w = cvx.Variable(NUM_STOCKS)
        objective = cvx.Minimize(K(w))
        constraints = [w >= 0, cvx.sum(w) == 1]
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.MOSEK)
        baseline_wt = pd.Series(w.value)

        # Save the results
        dir_path = f"{WDIR}/results/sim/{iter}"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        pd.to_pickle(r_t, f"{dir_path}/sec_rt.pkl")
        pd.to_pickle(rwls_wt, f"{dir_path}/rwls_wt.pkl")
        pd.to_pickle(baseline_wt, f"{dir_path}/baseline_wt.pkl")