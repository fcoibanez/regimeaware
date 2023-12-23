"""Fits RWLS on the historical data  and caches the results locally."""

if __name__ == "__main__":
    import pandas as pd
    from regimeaware.routines import cfg
    from tqdm import tqdm
    import statsmodels.api as sm

    ff = pd.read_pickle(f'{cfg.data_fldr}/ff.pkl').sort_index()[cfg.factor_set]
    crsp = pd.read_pickle(f'{cfg.data_fldr}/crsp.pkl')
    rt = pd.pivot_table(crsp[['excess_ret']], index='date', columns='permno', values='excess_ret')

    rebalance_dates = pd.date_range(cfg.bt_start_dt, cfg.bt_end_dt, freq=cfg.rebalance_freq)
    collect_betas = []
    collect_var = []

    for lookback in cfg.ols_windows:
        for dt in tqdm(rebalance_dates, desc=f"{lookback}-month lookback"):
            n_obs = rt.loc[:dt].iloc[-cfg.obs_thresh:].count()
            length_flags = n_obs == cfg.obs_thresh
            long_ids = length_flags[length_flags].index
            active_ids = rt.xs(dt).dropna().index
            sample_ids = list(set(active_ids).intersection(long_ids))

            res_var = pd.Series(index=pd.MultiIndex.from_tuples([(dt, x) for x in sample_ids], names=['date', 'id']))

            Y = rt.loc[cfg.trn_start_dt:dt, sample_ids].copy()
            X = ff.loc[cfg.trn_start_dt:dt].copy()
            X = sm.add_constant(X)

            for sec_id in Y:
                y = Y[sec_id].dropna()
                cmn_idx = X.join(y).dropna().iloc[-lookback:].index
                mdl = sm.OLS(y.reindex(cmn_idx), X.reindex(cmn_idx))
                mdl_fit = mdl.fit()
                f = mdl_fit.params

                # Forecasted betas
                res_params = mdl_fit.params.copy()
                idx = pd.MultiIndex.from_tuples([(dt, sec_id, x) for x in res_params.index], names=['date', 'id', 'factor'])
                res_params.index = idx
                collect_betas += [res_params]

                # Regression residual variance (to be used in the projection)
                e = mdl_fit.resid.values.reshape(-1, 1)
                res_var[(dt, sec_id)] = (e.T @ e).item() / e.shape[0]

            collect_var += [res_var]

        betas = pd.concat(collect_betas, axis=0)
        betas.to_pickle(f'{cfg.data_fldr}/exposures/ols_betas_{lookback}.pkl')

        var = pd.concat(collect_var, axis=0)
        var.to_pickle(f'{cfg.data_fldr}/exposures/ols_var_{lookback}.pkl')
