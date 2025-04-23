
if __name__ == "__main__":
    import statsmodels.api as sm 
    import pandas as pd
    from statsmodels.stats.weightstats import DescrStatsW
    import os
    from hmmlearn.hmm import GaussianHMM
    from datetime import datetime

    NUM_STATES = 2
    NUM_STOCKS = 100
    SIM_PERIODS = 1000
    MIN_OBS = 240
    MCAP_THRESH = 300000
    FCTRS = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
    AS_OF_DATE = datetime(2024, 12, 31)
    HM_MIN_COVAR = 1E-3
    HM_ITER = 1000
    HM_COVAR_TYPE = "full"
    HM_ALGO = "viterbi"
    HM_RND = 123
    HM_TOL = 1E-1
    HM_IMPLEMENTATION = "scaling"  # "scaling" or "log"
    WDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --------------------------------------------------------------------
    # Load the data
    fctr_rt = pd.read_pickle(f"{WDIR}/data/ff.pkl")
    rf = fctr_rt.pop("rf")
    crsp = pd.read_pickle(f'{WDIR}/data/crsp.pkl')[['ret_adj', 'shrout', 'prc']]
    rt = crsp["ret_adj"].unstack().dropna(axis=1, how='all')
    xs_rt = rt.sub(rf, axis=0)  # Excess returns

    # --------------------------------------------------------------------
    # Sample stocks
    active_flags = crsp.xs(AS_OF_DATE).all(axis=1)  # Filter 1: active stocks
    mcap = crsp['shrout'].mul(crsp['prc'].abs()).xs(AS_OF_DATE, level='date')
    mcap_flags = (mcap >= MCAP_THRESH).reindex(mcap.index).fillna(False)  # Filter 2: remove micro-caps
    length_flags = rt.reindex(active_flags.index, axis=1).count() > MIN_OBS  # Filter 3: enough observations
    sample_flags = pd.concat([active_flags, mcap_flags, length_flags], axis=1).all(axis=1)  # Put together the flags
    sample_ids = sample_flags[sample_flags].index

    # --------------------------------------------------------------------
    # Train HMM based on real data
    mdl_hmm = GaussianHMM(
        n_components=NUM_STATES,
        covariance_type=HM_COVAR_TYPE,
        random_state=HM_RND,
        min_covar=HM_MIN_COVAR,
        tol=HM_TOL,
        implementation=HM_IMPLEMENTATION
    )

    mdl_hmm.fit(fctr_rt)
    smoothed_prob = pd.DataFrame(mdl_hmm.predict_proba(fctr_rt), index=fctr_rt.index, columns=range(NUM_STATES))

    # --------------------------------------------------------------------
    # Collect regime-weighted parameters of our sample stocks
    collect_params = {}
    collect_scale = {}
    for sec_id in sample_ids:
        y = xs_rt[sec_id].dropna()
        
        state_params = []
        for s in range(NUM_STATES):
            w = smoothed_prob.loc[y.index, s]
            mdl = sm.WLS(y, sm.add_constant(fctr_rt.loc[y.index, FCTRS]), weights=w).fit(disp=0)
            b = mdl.params.copy()
            b.index = pd.MultiIndex.from_product([b.index, [s]], names=["factor", "state"])
            state_params += [b]
            collect_scale[(sec_id, s)] = mdl.scale

        collect_params[sec_id] = pd.concat(state_params)

    _midx = pd.MultiIndex.from_product([FCTRS, range(NUM_STATES)], names=["factor", "state"])
    sample_params = pd.DataFrame.from_dict(collect_params, orient="index").loc[sample_ids, _midx]
    scale_by_regime = pd.Series(collect_scale).groupby(level=1).mean()

    # Loadings join-distribution (market-cap weighted)
    wstats = DescrStatsW(sample_params, weights=mcap.reindex(sample_ids))
    loadings_means = pd.Series(wstats.mean, index=_midx, dtype=float)
    loadings_cov = pd.DataFrame(wstats.cov, index=_midx, columns=_midx, dtype=float)

    # --------------------------------------------------------------------
    # Save the results
    loadings_cov.to_pickle(f'{WDIR}/data/loadings_cov.pkl')
    loadings_means.to_pickle(f'{WDIR}/data/loadings_means.pkl')
    scale_by_regime.to_pickle(f'{WDIR}/data/scale_by_regime.pkl')
    pd.to_pickle(mdl_hmm, f'{WDIR}/data/mdl_hmm.pkl')