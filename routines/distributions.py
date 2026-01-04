"""Trains the HMM on real data and extracts conditional-distributions for parameters used in the simulations."""

if __name__ == "__main__":
    import pandas as pd
    from regimeaware.constants import (
        DataConstants,
        HMMParameters,
        Factors,
        MIN_OBS,
        MICROCAP_THRESHOLD,
    )
    import statsmodels.api as sm
    from regimeaware.core.utils import market_coverage, cleanup_ts
    from hmmlearn.hmm import GaussianHMM
    import os
    from statsmodels.stats.weightstats import DescrStatsW

    os.environ["OMP_NUM_THREADS"] = "3"

    # HMM training on real factor data
    fctr_rt = pd.read_pickle(f"{DataConstants.WDIR.value}/data/ff.pkl")
    fctr_rt = fctr_rt[[factor.name for factor in Factors]]

    mdl_hmm = GaussianHMM(
        n_components=HMMParameters.STATES.value,
        covariance_type=HMMParameters.COV.value,
        n_iter=HMMParameters.ITER.value,
        tol=HMMParameters.TOL.value,
        random_state=HMMParameters.SEED.value,
        implementation=HMMParameters.IMPLEMENTATION.value,
    )
    mdl_hmm.fit(fctr_rt)
    smoothed_prob = pd.DataFrame(
        mdl_hmm.predict_proba(fctr_rt),
        index=fctr_rt.index,
        columns=range(HMMParameters.STATES.value),
    )
    hmm_startprob = mdl_hmm.startprob_
    hmm_transmat = mdl_hmm.transmat_
    hmm_means = mdl_hmm.means_
    hmm_covars = mdl_hmm.covars_

    # Regime-weighted parameters distribution of our sample stocks
    crsp = pd.read_pickle(rf"{DataConstants.WDIR.value}/data/crsp.pkl")
    mcap = crsp["mktcap"].astype(float)
    xs_rt = crsp["excess_ret"].astype(float).dropna()

    # Excluding micro-caps
    microcap_flags = mcap.groupby("date", group_keys=False).apply(
        market_coverage, (1 - MICROCAP_THRESHOLD)
    )  # Micro-cap stocks
    xs_rt = xs_rt[~microcap_flags]
    xs_rt = xs_rt.dropna().unstack()

    # Consider only stocks with enough continous data
    xs_rt = xs_rt.apply(cleanup_ts, args=(MIN_OBS,))
    xs_rt.dropna(how="all", axis=1, inplace=True)

    collect_params = {}
    collect_scale = {}
    for sec_id in xs_rt.columns:
        y = xs_rt[sec_id].dropna()

        state_params = []
        for s in range(HMMParameters.STATES.value):
            w = smoothed_prob.loc[y.index, s]
            mdl = sm.WLS(
                y, sm.add_constant(fctr_rt.loc[y.index].astype(float)), weights=w
            ).fit(disp=0)
            b = mdl.params.copy()
            b.index = pd.MultiIndex.from_product(
                [b.index, [s]], names=["factor", "state"]
            )
            state_params += [b]
            collect_scale[(sec_id, s)] = mdl.scale

        collect_params[sec_id] = pd.concat(state_params)

    _midx = pd.MultiIndex.from_product(
        [Factors._member_names_, range(HMMParameters.STATES.value)],
        names=["factor", "state"],
    )
    sample_params = pd.DataFrame.from_dict(collect_params, orient="index").loc[
        xs_rt.columns, _midx
    ]
    scale_by_regime = pd.Series(collect_scale).groupby(level=1).mean()

    # Loadings join-distribution (market-cap weighted)
    w = mcap.reindex(xs_rt.stack().index)
    w = (w / w.groupby("date").sum()).groupby("permno").mean()
    w /= w.sum()
    wstats = DescrStatsW(sample_params, weights=w)

    loadings_means = pd.Series(wstats.mean, index=_midx)

    _temp_cov = pd.DataFrame(wstats.cov, index=_midx, columns=_midx)
    loadings_cov = pd.DataFrame(0, index=_midx, columns=_midx, dtype=float)
    for f in Factors._member_names_:
        loadings_cov.loc[f, f] = _temp_cov.loc[f, f].values

    # Cache distribution parameters
    loadings_cov.to_pickle(f"{DataConstants.WDIR.value}/data/dist/loadings_cov.pkl")
    loadings_means.to_pickle(f"{DataConstants.WDIR.value}/data/dist/loadings_means.pkl")
    scale_by_regime.to_pickle(
        f"{DataConstants.WDIR.value}/data/dist/scale_by_regime.pkl"
    )
    pd.to_pickle(mdl_hmm, f"{DataConstants.WDIR.value}/data/dist/mdl_hmm.pkl")
