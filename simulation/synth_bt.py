if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import os

    NUM_SIMULATIONS = 10000
    OOS_PERIODS = 1
    NUM_STOCKS = 1000
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

    m_0 = fctr_rt.mean()
    v_0 = fctr_rt.cov()

    # --------------------------------------------
    # Returns simulation
    for iter in range(NUM_SIMULATIONS):
        print(iter)
        # Factor returns
        if iter == 0:
            gen_hmm.random_state = HM_RND
            np.random.seed(HM_RND)
        else:
            gen_hmm.random_state = np.random.randint(0, 1000000)

        X, Z = gen_hmm.sample(OOS_PERIODS)
        # G = gen_hmm.predict_proba(X)

        # True factor loadings
        _vals = np.random.multivariate_normal(mean=loadings_means, cov=loadings_cov, size=NUM_STOCKS)
        B = pd.DataFrame(_vals, columns=loadings_means.index).T
        B.columns.name = "stock"
        B.index.names = ["factor", "state"]
        B = B.stack()

        # Stock returns
        s_t = Z.item()  # Dominant regime
        x_t = pd.Series(X.flatten(), fctr_rt.columns)
        x_t.index.name = "factor"
        r_t = B.xs(s_t, level="state").mul(x_t, axis=0).groupby("stock").sum()
        r_t += np.random.normal(loc=0, scale=scale_by_regime[s_t], size=NUM_STOCKS)  # Resids

        b_0 = B.unstack("state").mul(stationary_prob).sum(axis=1).unstack("factor")[fctr_rt.columns]

        