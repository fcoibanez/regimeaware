"""Fits RWLS on the historical data  and caches the results locally."""

if __name__ == "__main__":
    import pandas as pd
    from core import cfg
    import numpy as np
    from core.methodologies import RegimeWeightedLS
    from tqdm import tqdm

    emission_prob = pd.read_pickle(f"{cfg.data_fldr}/regimes/emission_prob.pkl")
    transition_matrix = pd.read_pickle(f"{cfg.data_fldr}/regimes/transmat.pkl")
    ff = pd.read_pickle(f'{cfg.data_fldr}/ff.pkl').sort_index()[cfg.factor_set]
    ff = ff.add(1).resample(cfg.data_freq).prod().sub(1).replace(0, np.nan)  # Temp!
    rt = pd.read_pickle(f'{cfg.data_fldr}/rt_sp.pkl')
    rt = rt.add(1).resample(cfg.data_freq).prod().sub(1).replace(0, np.nan)  # Temp! This has to be monthly and excess returns

    rebalance_dates = pd.date_range(cfg.bt_start_dt, cfg.bt_end_dt, freq=cfg.rebalance_freq)
    collect_params = []
    collect_cond_params = []
    collect_resid = []
    collect_var = []

    for dt in tqdm(rebalance_dates):
        n_obs = rt.loc[cfg.trn_start_dt:dt].count()
        length_flags = n_obs > cfg.obs_thresh
        long_ids = length_flags[length_flags].index
        active_ids = rt.xs(dt).dropna().index
        sample_ids = list(set(active_ids).intersection(active_ids))

        Y = rt.loc[cfg.trn_start_dt:dt, sample_ids].copy()
        X = ff.loc[cfg.trn_start_dt:dt].copy()
        g = emission_prob.xs(dt).copy()
        B = transition_matrix.xs(dt).copy()

        for sec_id in Y:
            y = Y[sec_id].dropna()
            mdl = RegimeWeightedLS(endog=y, exog=X, emission_prob=g)
            mdl.fit(add_constant=True)
            mdl.predict(transition_matrix=B, as_of=dt, horizon=cfg.forecast_horizon)

            # Betas per regime
            res_params = mdl.params.copy()
            idx = pd.MultiIndex.from_tuples([(dt, sec_id, x) for x in res_params.index], names=['date', 'id', 'factor'])
            res_params.index = idx
            collect_params += [res_params]

            # Forecasted betas
            res_cond = mdl.cond_params.copy()
            res_cond.index = idx
            collect_cond_params += [res_cond]

            # Regression residuals (to be used in the projection)
            res_resid = mdl.resid.copy()
            idx_e = pd.MultiIndex.from_tuples([(dt, sec_id, x) for x in res_resid.index], names=['date', 'id', 'obs'])
            res_resid.index = idx_e
            collect_resid += [res_resid]

            res_var = mdl.sigma.copy()
            idx_v = pd.MultiIndex.from_tuples([(dt, sec_id, x) for x in res_var.index], names=['date', 'id', 'state'])
            res_var.index = idx_v
            collect_var += [res_var]

    cond_params = pd.concat(collect_cond_params, axis=0)
    cond_params.to_pickle(f'{cfg.data_fldr}/exposures/forecasted_betas.pkl')

    params = pd.concat(collect_params, axis=0)
    params.to_pickle(f'{cfg.data_fldr}/exposures/conditional_betas.pkl')

    resid = pd.concat(collect_resid, axis=0)
    resid.to_pickle(f'{cfg.data_fldr}/exposures/residuals.pkl')

    var = pd.concat(collect_var, axis=0)
    var.to_pickle(f'{cfg.data_fldr}/exposures/var.pkl')
