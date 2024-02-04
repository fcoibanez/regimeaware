"""Fits RWLS on the historical data  and caches the results locally."""

if __name__ == "__main__":
    import pandas as pd
    from regimeaware.routines import cfg
    from regimeaware.core.methodologies import RegimeWeightedLS
    from tqdm import tqdm

    emission_prob = pd.read_pickle(f"{cfg.data_fldr}/regimes/emission_prob.pkl")
    transition_matrix = pd.read_pickle(f"{cfg.data_fldr}/regimes/transmat.pkl")

    estimation_dts = pd.date_range(cfg.bt_start_dt - pd.DateOffset(months=1), cfg.bt_end_dt, freq=cfg.estimation_freq)
    if cfg.estimation_freq != 'ME':
        rt = pd.read_pickle(f'{cfg.data_fldr}/crsp_daily.pkl')
        rt = pd.pivot_table(rt.to_frame(), index='date', columns='permno', values='ret')
        first_obs_dt = rt.apply(pd.Series.first_valid_index)
        last_obs_dt = rt.apply(pd.Series.last_valid_index)
        rt = rt.add(1).groupby(pd.Grouper(freq=cfg.estimation_freq)).prod().sub(1)
        for sec_id in rt.columns:
            rt[sec_id] = rt.loc[first_obs_dt[sec_id]:last_obs_dt[sec_id], sec_id]

        ff = pd.read_pickle(f'{cfg.data_fldr}/ff_daily.pkl').sort_index()
        ff = ff.add(1).groupby(pd.Grouper(freq=cfg.estimation_freq)).prod().sub(1)
        rf = ff['rf']
        rt = rt.sub(rf, axis=0).dropna(how='all')

    else:
        ff = pd.read_pickle(f'{cfg.data_fldr}/ff.pkl').sort_index()[cfg.factor_set]
        crsp = pd.read_pickle(f'{cfg.data_fldr}/crsp.pkl')
        rt = pd.pivot_table(crsp[['excess_ret']], index='date', columns='permno', values='excess_ret')

    rebalance_dates = pd.date_range(cfg.bt_start_dt, cfg.bt_end_dt, freq=cfg.rebalance_freq)
    collect_betas = []
    collect_var = []

    for dt in tqdm(rebalance_dates):
        as_of_dt = rt.index.asof(dt)
        n_obs = rt.loc[:dt].iloc[-cfg.obs_thresh:].count()
        length_flags = n_obs == cfg.obs_thresh
        long_ids = length_flags[length_flags].index
        active_ids = rt.xs(as_of_dt).dropna().index
        sample_ids = list(set(active_ids).intersection(long_ids))

        res_var = pd.Series(index=pd.MultiIndex.from_tuples([(dt, x) for x in sample_ids], names=['date', 'id']))

        Y = rt.loc[cfg.trn_start_dt:as_of_dt, sample_ids].copy()
        X = ff.loc[cfg.trn_start_dt:as_of_dt, cfg.factor_set].copy()
        g = emission_prob.xs(as_of_dt).copy()
        B = transition_matrix.xs(as_of_dt).copy()

        for sec_id in tqdm(Y, desc=dt.strftime('%Y %b')):
            y = Y[sec_id].dropna()
            mdl = RegimeWeightedLS(endog=y, exog=X, emission_prob=g)
            mdl.fit(add_constant=True)
            mdl.predict(transition_matrix=B, as_of=dt, horizon=cfg.forecast_horizon)

            # Forecasted betas
            res_params = mdl.cond_params.copy()
            idx = pd.MultiIndex.from_tuples([(dt, sec_id, x) for x in res_params.index], names=['date', 'id', 'factor'])
            res_params.index = idx
            collect_betas += [res_params]

            # Regression residual variance (to be used in the projection)
            res_var[(dt, sec_id)] = mdl.cond_sigma

        collect_var += [res_var]

    betas = pd.concat(collect_betas, axis=0)
    betas.to_pickle(f'{cfg.data_fldr}/exposures/forecasted_betas.pkl')

    var = pd.concat(collect_var, axis=0)
    var.to_pickle(f'{cfg.data_fldr}/exposures/var.pkl')
