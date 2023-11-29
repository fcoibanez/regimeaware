"""Fits the HMM on real time and caches the results locally."""

if __name__ == "__main__":
    from core import cfg
    import pandas as pd
    from hmmlearn.hmm import GaussianHMM
    from tqdm import tqdm
    import numpy as np
    from sklearn import preprocessing

    data = pd.read_pickle(f'{cfg.data_fldr}/ff.pkl').sort_index()
    data = data.add(1).resample(cfg.data_freq).prod().sub(1)
    data = data[cfg.factor_set]

    rebalance_dts = pd.date_range(cfg.bt_start_dt, cfg.bt_end_dt, freq=cfg.estimation_freq)

    # Warm-up run to get initial parameters
    trn_raw = data.loc[cfg.trn_start_dt:cfg.bt_end_dt]
    scaler = preprocessing.StandardScaler(copy=True).fit(trn_raw)
    trn_std = scaler.transform(trn_raw)
    mdl = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.hm_cov,
        n_iter=cfg.hm_iter,
        algorithm=cfg.hm_algo,
        random_state=cfg.hm_rs,
        tol=cfg.hm_tol,
    )
    mdl.fit(trn_std)
    mu_t = pd.DataFrame(mdl.means_, columns=trn_raw.columns)
    mu_t = mu_t.add(scaler.mean_).mul(scaler.scale_)
    print(mu_t.mul(12))
    mu_0 = mdl.means_
    sigma_0 = mdl.covars_
    transmat_0 = mdl.transmat_
    start_p_0 = mdl.startprob_

    # Live run
    collect_emission_prob = []
    collect_mu = []
    collect_sigma = []
    collect_transmat = []
    convergence_flags = pd.Series(index=rebalance_dts)

    for as_of_dt in tqdm(rebalance_dts):
        trn_raw = data.loc[cfg.trn_start_dt:as_of_dt]
        scaler = preprocessing.StandardScaler(copy=True).fit(trn_raw)
        trn_std = scaler.transform(trn_raw)
        mdl = GaussianHMM(
            n_components=cfg.n_states,
            covariance_type=cfg.hm_cov,
            n_iter=cfg.hm_iter,
            algorithm=cfg.hm_algo,
            random_state=cfg.hm_rs,
            tol=cfg.hm_tol,
            params=cfg.hm_params_to_estimate,
            init_params=""
        )

        mdl.startprob_ = start_p_0
        mdl.means_ = mu_0
        mdl.covars_ = np.array([np.diag(x) for x in sigma_0])
        mdl.transmat_ = transmat_0

        mdl.fit(X=trn_std)
        convergence_flags[as_of_dt] = mdl.monitor_.converged

        # State-dependent moments
        mu_t = pd.DataFrame(mdl.means_, columns=trn_raw.columns)
        mu_t = mu_t.add(scaler.mean_).mul(scaler.scale_)
        mu_t.index = pd.MultiIndex.from_tuples([(as_of_dt, x) for x in mu_t.index], names=['as_of', 'state'])
        collect_mu += [mu_t]

        sigma_t = pd.concat([pd.DataFrame(x, columns=trn_raw.columns).mul(scaler.var_) for x in mdl.covars_])
        idx = pd.MultiIndex.from_tuples([(as_of_dt, y, x) for y in range(cfg.n_states) for x in trn_raw.columns], names=['as_of', 'state', 'factor'])
        sigma_t.index = idx
        sigma_t = sigma_t.reindex(idx, level='state')
        collect_sigma += [sigma_t]

        # Emission probabilities
        gamma = mdl.predict_proba(X=trn_std)
        idx = pd.MultiIndex.from_tuples([(as_of_dt, x) for x in trn_raw.index], names=['as_of', 'date'])
        emission_prob_t = pd.DataFrame(gamma, index=idx)
        collect_emission_prob += [emission_prob_t]

        # Transition matrix
        idx = pd.MultiIndex.from_tuples([(as_of_dt, x) for x in range(cfg.n_states)], names=['as_of', 'state'])
        transmat_t = pd.DataFrame(mdl.transmat_, index=idx)
        collect_transmat += [transmat_t]

        # Preparing for next iteration
        mu_0 = mdl.means_
        sigma_0 = mdl.covars_
        transmat_0 = mdl.transmat_
        start_p_0 = mdl.startprob_

    emission_prob = pd.concat(collect_emission_prob, axis=0)
    sigma = pd.concat(collect_sigma, axis=0)
    mu = pd.concat(collect_mu, axis=0)
    transmat = pd.concat(collect_transmat, axis=0)

    # Cache the results
    emission_prob.to_pickle(f"{cfg.data_fldr}/regimes/emission_prob.pkl")
    sigma.to_pickle(f"{cfg.data_fldr}/regimes/sigma.pkl")
    mu.to_pickle(f"{cfg.data_fldr}/regimes/mu.pkl")
    transmat.to_pickle(f"{cfg.data_fldr}/regimes/transmat.pkl")
