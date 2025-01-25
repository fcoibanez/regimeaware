if __name__ == "__main__":
    import pandas as pd
    from regimeaware.routines import cfg

    rebalance_dates = pd.date_range(cfg.bt_start_dt, cfg.bt_end_dt, freq=cfg.rebalance_freq)

    # Stock must have industry mapping and price information on rebalance date
    gics_mapping = pd.read_pickle(f'{cfg.data_fldr}/gics_mapping.pkl')['sector']
    crsp = pd.read_pickle(f'{cfg.data_fldr}/crsp.pkl')[['ret', 'shrout', 'prc']]
    crsp = crsp.join(gics_mapping, how='left', on='permno')
    crsp_dates = crsp.index.get_level_values('date').unique()
    crsp_flags = crsp.all(axis=1)
    crsp_flags = crsp_flags.rename({x: crsp_dates.asof(x) for x in rebalance_dates})
    mcap = crsp['shrout'].mul(crsp['prc'].abs())
    total_mcap = mcap.groupby('date').sum()
    mcap_sorted = mcap.groupby('date', group_keys=False).apply(pd.Series.sort_values, **{'ascending': False})
    mcap_coverage = mcap_sorted.groupby('date').cumsum().div(total_mcap)
    mcap_flags = mcap_coverage < cfg.mcap_thresh
    mcap_flags = mcap_flags.reindex(mcap.index).fillna(False)
    mcap_flags = mcap_flags.rename({x: crsp_dates.asof(x) for x in rebalance_dates})

    # Stock must have factor loadings on rebalance date
    factor_loadings = pd.read_pickle(f'{cfg.data_fldr}/exposures/forecasted_betas.pkl')
    loadings_dates = crsp.index.get_level_values('date').unique()
    loadings_flags = factor_loadings.groupby(['date', 'id']).all().rename('loadings')
    loadings_flags.index.rename({'id': 'permno'}, inplace=True)
    loadings_flags = loadings_flags.rename({x: loadings_dates.asof(x) for x in rebalance_dates})

    # Tradable flags
    df = crsp_flags.to_frame('crsp').join(mcap_flags.to_frame('mcap')).join(loadings_flags.to_frame('loadings'))
    is_tradable = df.loc[rebalance_dates].fillna(False).all(axis=1)
    is_tradable.to_pickle(f'{cfg.data_fldr}/is_tradable.pkl')
