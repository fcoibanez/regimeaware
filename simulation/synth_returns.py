if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from hmmlearn.hmm import GaussianHMM
    import os

    NUM_SIMULATIONS = 10000
    OOS_PERIODS = 120
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
        G = gen_hmm.predict_proba(X)
        R = np.zeros((OOS_PERIODS, NUM_STOCKS))

        # True factor loadings
        _vals = np.random.multivariate_normal(mean=loadings_means, cov=loadings_cov, size=NUM_STOCKS)
        B = pd.DataFrame(_vals, columns=loadings_means.index).T
        B.columns.name = "stock"
        B.index.names = ["factor", "state"]
        B = B.stack()

        # Stock returns
        for _t, s_t in enumerate(Z):
            x_t = pd.Series(X[_t], fctr_rt.columns)
            x_t.index.name = "factor"
            R[_t] += B.xs(s_t, level="state").mul(x_t).groupby("stock").sum()
            R[_t] += np.random.normal(loc=0, scale=scale_by_regime[s_t], size=NUM_STOCKS)  # Resids

        dir_path = f"{WDIR}/data/sim/{iter}"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        pd.to_pickle(R, f"{dir_path}/sec_rt.pkl")
        pd.to_pickle(X, f"{dir_path}/fctr_rt.pkl")
        pd.to_pickle(G, f"{dir_path}/probs.pkl")