if __name__ == "__main__":
    import tqdm
    import numpy as np
    import pandas as pd
    from hmmlearn.hmm import GaussianHMM
    from regimeaware.constants import DataConstants, SimulationParameters, Factors

    # --------------------------------------------
    # Data
    fctr_rt = pd.read_pickle(f"{DataConstants.WDIR.value}/data/ff.pkl")[
        Factors._member_names_
    ]
    loadings_cov = pd.read_pickle(
        f"{DataConstants.WDIR.value}/data/dist/loadings_cov.pkl"
    )
    loadings_means = pd.read_pickle(
        f"{DataConstants.WDIR.value}/data/dist/loadings_means.pkl"
    )
    scale_by_regime = pd.read_pickle(
        f"{DataConstants.WDIR.value}/data/dist/scale_by_regime.pkl"
    )

    # HMM params
    mdl_hmm = pd.read_pickle(f"{DataConstants.WDIR.value}/data/dist/mdl_hmm.pkl")
    param_startprob = mdl_hmm.startprob_
    param_transmat = mdl_hmm.transmat_
    param_means = mdl_hmm.means_
    param_covars = mdl_hmm.covars_

    # --------------------------------------------
    # Returns simulation
    collect_sec_rt = {}
    collect_fctr_rt = {}
    collect_probs = {}

    for iter in tqdm(range(SimulationParameters.TRIALS.value)):
        print(iter)
        # --------------------------------------------
        # Factor returns
        gen_hmm = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            min_covar=1e-3,
            tol=1e-1,
            implementation="scaling",
        )

        if iter == 0:
            gen_hmm.random_state = 123
            np.random.seed(123)

        gen_hmm.startprob_ = param_startprob
        gen_hmm.transmat_ = param_transmat
        gen_hmm.means_ = param_means
        gen_hmm.covars_ = np.array([np.diag(x) for x in param_covars])

        X, Z = gen_hmm.sample(
            SimulationParameters.IS_PERIODS.value
            + SimulationParameters.OOS_PERIODS.value
        )
        G = gen_hmm.predict_proba(X)
        R = np.zeros(
            (
                SimulationParameters.IS_PERIODS.value
                + SimulationParameters.OOS_PERIODS.value,
                SimulationParameters.NUM_STOCKS.value,
            )
        )

        # True factor loadings
        _vals = np.random.multivariate_normal(
            mean=loadings_means,
            cov=loadings_cov,
            size=SimulationParameters.NUM_STOCKS.value,
        )
        B = pd.DataFrame(_vals, columns=loadings_means.index).T
        B.columns.name = "stock"
        B.index.names = ["factor", "state"]
        B = B.stack()

        # Stock returns
        for _t, s_t in enumerate(Z):
            x_t = pd.Series(X[_t], fctr_rt.columns)
            x_t.index.name = "factor"
            R[_t] += B.xs(s_t, level="state").mul(x_t).groupby("stock").sum()
            R[_t] += np.random.normal(
                loc=0,
                scale=scale_by_regime[s_t],
                size=SimulationParameters.NUM_STOCKS.value,
            )  # Resids

        collect_sec_rt[iter] = pd.DataFrame(R)
        collect_fctr_rt[iter] = pd.DataFrame(X, columns=fctr_rt.columns)
        collect_probs[iter] = pd.DataFrame(G)

    # Cache results
    pd.to_pickle(collect_sec_rt, f"{DataConstants.WDIR.value}/data/sim/sec_rt.pkl")
    pd.to_pickle(collect_fctr_rt, f"{DataConstants.WDIR.value}/data/sim/fctr_rt.pkl")
    pd.to_pickle(collect_probs, f"{DataConstants.WDIR.value}/data/sim/probs.pkl")
