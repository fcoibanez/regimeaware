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
    collect_betas = []
    collect_var = []

    for dt in tqdm(rebalance_dates):
        n_obs = rt.loc[cfg.trn_start_dt:dt].count()
        length_flags = n_obs > cfg.obs_thresh
        long_ids = length_flags[length_flags].index
        active_ids = rt.xs(dt).dropna().index
        sample_ids = list(set(active_ids).intersection(active_ids))

        res_var = pd.Series(index=pd.MultiIndex.from_tuples([(dt, x) for x in sample_ids], names=['date', 'id']))

        Y = rt.loc[cfg.trn_start_dt:dt, sample_ids].copy()
        X = ff.loc[cfg.trn_start_dt:dt].copy()
        g = emission_prob.xs(dt).copy()
        B = transition_matrix.xs(dt).copy()

        for sec_id in Y:
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
