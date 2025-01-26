"""Fits the HMM on real time and caches the results locally."""

if __name__ == "__main__":
    from regimeaware.routines import cfg
    import pandas as pd
    from hmmlearn.hmm import GaussianHMM
    from tqdm import tqdm
    import numpy as np

    if cfg.estimation_freq != 'M':
        data = pd.read_pickle(f'{cfg.data_fldr}/ff_daily.pkl')[cfg.factor_set].sort_index()
        dts = pd.date_range(cfg.trn_start_dt, cfg.bt_end_dt, freq=cfg.estimation_freq)
        estimation_dts = pd.date_range(cfg.bt_start_dt - pd.DateOffset(months=1), cfg.bt_end_dt, freq=cfg.estimation_freq)
        st, ed = dts[[0, -1]]
        data = data.loc[st:ed].add(1).groupby(pd.Grouper(freq=cfg.estimation_freq)).prod().sub(1)
    else:
        data = pd.read_pickle(f'{cfg.data_fldr}/ff.pkl')[cfg.factor_set].sort_index()
        dts = pd.date_range(cfg.trn_start_dt, cfg.bt_end_dt, freq=cfg.estimation_freq)
        estimation_dts = pd.date_range(cfg.bt_start_dt, cfg.bt_end_dt, freq=cfg.estimation_freq)

    # Warm-up run to get initial parameters
    trn = data.loc[cfg.trn_start_dt:cfg.bt_end_dt]

    mdl = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.hm_cov,
        n_iter=cfg.hm_iter,
        algorithm=cfg.hm_algo,
        random_state=cfg.hm_rs,
        tol=cfg.hm_tol,
    )

    mdl.fit(trn)
    mu_0 = mdl.means_
    sigma_0 = mdl.covars_
    transmat_0 = mdl.transmat_
    start_p_0 = mdl.startprob_

    # Live run
    collect_emission_prob = []
    collect_mu = []
    collect_sigma = []
    collect_transmat = []
    convergence_flags = pd.Series(index=estimation_dts, dtype=bool)

    for as_of_dt in tqdm(estimation_dts):
        trn = data.loc[cfg.trn_start_dt:as_of_dt, cfg.factor_set]

        mdl = GaussianHMM(
            n_components=cfg.n_states,
            covariance_type=cfg.hm_cov,
            n_iter=cfg.hm_iter,
            algorithm=cfg.hm_algo,
            random_state=cfg.hm_rs,
            tol=cfg.hm_tol,
            params=cfg.hm_params_to_estimate,
            init_params=cfg.hm_init_params_to_estimate,
        )

        mdl.startprob_ = start_p_0
        mdl.means_ = mu_0

        if cfg.hm_cov == "diag":
            mdl.covars_ = np.array([np.diag(x) for x in sigma_0])
        else:
            mdl.covars_ = sigma_0
        mdl.transmat_ = transmat_0

        mdl.fit(X=trn)
        convergence_flags[as_of_dt] = mdl.monitor_.converged

        # State-dependent moments
        mu_t = pd.DataFrame(mdl.means_, columns=trn.columns)
        mu_t.index = pd.MultiIndex.from_tuples([(as_of_dt, x) for x in mu_t.index], names=['as_of', 'state'])
        collect_mu += [mu_t]

        sigma_t = pd.concat([pd.DataFrame(x, columns=trn.columns) for x in mdl.covars_])
        idx = pd.MultiIndex.from_tuples([(as_of_dt, y, x) for y in range(cfg.n_states) for x in trn.columns], names=['as_of', 'state', 'factor'])
        sigma_t.index = idx
        collect_sigma += [sigma_t]

        # Emission probabilities
        gamma = mdl.predict_proba(X=trn)
        idx = pd.MultiIndex.from_tuples([(as_of_dt, x) for x in trn.index], names=['as_of', 'date'])
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
