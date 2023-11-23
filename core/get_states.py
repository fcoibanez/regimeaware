"""Fits the HMM on real time and caches the results locally."""

if __name__ == "__main__":
    import regimeaware
    from core import cfg
    import pandas as pd
    from hmmlearn.hmm import GaussianHMM
    from tqdm import tqdm
    import itertools
    from scipy import linalg

    data = pd.read_pickle(f'{cfg.data_fldr}/ff.pkl').sort_index()
    data = data[cfg.factor_set]

    rebalance_dts = pd.date_range(cfg.bt_start_dt, cfg.bt_end_dt, freq=cfg.rebalance_freqstr)

    # Warm up
    mdl_0 = GaussianHMM(
        cfg.n_states,
        covariance_type=cfg.hm_cov,
        algorithm=cfg.hm_algo,
        random_state=cfg.hm_rs
    )

    mdl_0.fit(data.loc[cfg.trn_start_dt:])
    print(pd.DataFrame(mdl_0.means_).mul(252))

    mu_0 = mdl_0.means_
    sigma_0 = mdl_0.covars_
    p_0 = mdl_0.transmat_
    start_p_0 = mdl_0.startprob_

    collect_emission_prob = []
    collect_mu = []
    collect_sigma = []
    collect_transmat = []
    convergence_flags = pd.Series(index=rebalance_dts)

    for as_of_dt in tqdm(rebalance_dts):
    # for as_of_dt in tqdm(rebalance_dts[:10]):
        trn = data.loc[cfg.trn_start_dt:as_of_dt]
        mdl = GaussianHMM(
            n_components=cfg.n_states,
            covariance_type=cfg.hm_cov,
            n_iter=cfg.hm_iter,
            algorithm=cfg.hm_algo,
            random_state=cfg.hm_rs,
            tol=cfg.hm_tol,
            means_prior=mu_0,
            covars_prior=sigma_0,
            # transmat_prior=p_0,
            startprob_prior=start_p_0
        )
        mdl.fit(X=trn)
        convergence_flags[as_of_dt] = mdl.monitor_.converged

        # Row sorting to make it consistent across time
        perm = itertools.permutations(range(cfg.n_states))
        collect_trace = pd.Series(index=perm)
        for p in collect_trace.index:
            M0 = pd.DataFrame(mu_0).rank(axis=0)
            M1 = pd.DataFrame(mdl.means_).reindex(p).rank(axis=0)
            Md = M1.values - M0.values
            s = linalg.svd(Md, compute_uv=False)
            collect_trace[p] = s.sum()
        row_sorting = list(collect_trace.idxmin())

        # State-dependent moments
        mu_t = pd.DataFrame(mdl.means_, columns=trn.columns)
        mu_t = mu_t.reindex(row_sorting).reset_index(drop=True)
        mu_t.index = pd.MultiIndex.from_tuples([(as_of_dt, x) for x in mu_t.index], names=['as_of', 'state'])
        collect_mu += [mu_t]

        # print(pd.DataFrame(mdl.means_).mul(252))              ########################
        # print(pd.DataFrame(mu_0).mul(252))
        # print(mu_t.mul(252))

        # sigma_t = [cov[row_sorting, :][:, row_sorting] for cov in mdl.covars_]
        # sigma_t = pd.concat([pd.DataFrame(x, columns=trn.columns) for i, x in enumerate(sigma_t)])
        sigma_t = pd.concat([pd.DataFrame(x, columns=trn.columns) for i, x in enumerate(mdl.covars_)])
        idx = pd.MultiIndex.from_tuples([(as_of_dt, y, x) for y in range(cfg.n_states) for x in trn.columns], names=['as_of', 'state', 'factor'])
        sigma_t.index = idx
        sigma_t = sigma_t.reindex(idx, level='state')
        remapping = {row_sorting[i]: i for i in range(cfg.n_states)}
        sigma_t.rename(remapping, level='state', inplace=True)  # Relabel the old states to the new states
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
        p_0 = mdl.transmat_
        start_p_0 = mdl.startprob_

    emission_prob = pd.concat(collect_emission_prob, axis=0)
    sigma = pd.concat(collect_sigma, axis=0)
    mu = pd.concat(collect_mu, axis=0)
    transmat = pd.concat(collect_transmat, axis=0)

    # ------------------------------------
    import matplotlib.pyplot as plt
    i = mu['mktrf'].groupby('as_of').idxmax().apply(lambda x: x[1]).mode().item()
    mu.xs(i, level='state')['mktrf'].mul(252).plot()
    plt.show()
    print(mu.mul(252)['mktrf'].groupby('as_of').rank())
    # Reweight the differences

    dt0, dt1, dt2 = ('2011-04-29', '2011-05-31', '2011-06-30')
    M0 = mu.xs(dt0)
    M1 = mu.xs(dt1)
    M2 = mu.xs(dt2)
    U0, s0, Vh0 = linalg.svd(M0)
    U1, s1, Vh1 = linalg.svd(M1)
    U2, s2, Vh2 = linalg.svd(M2)
    '''
    Eigenvectors seem to change when the rows are reshuffled
    Singular values are pretty much the same, trace does not show difference
    '''
    sd0 = linalg.svd(M0 - M1, compute_uv=False)
    sd1 = linalg.svd(M1 - M2, compute_uv=False)
    sd2 = linalg.svd(M0 - M2, compute_uv=False)


