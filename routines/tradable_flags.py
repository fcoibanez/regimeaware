if __name__ == "__main__":
    import pandas as pd
    from regimeaware.routines import cfg

    gics_mapping = pd.read_pickle(f'{cfg.data_fldr}/gics_mapping.pkl')['sector']
    crsp = pd.read_pickle(f'{cfg.data_fldr}/crsp_daily.pkl')[['ret', 'shrout', 'prc']]
    crsp = crsp.join(gics_mapping, how='left', on='permno')
    factor_loadings = pd.read_pickle(f'{cfg.data_fldr}/exposures/forecasted_betas.pkl')
    loadings_flags = factor_loadings.groupby(['date', 'id']).all().rename('loadings')
    loadings_flags.index.rename({'id': 'permno'}, inplace=True)
    rebalance_dates = factor_loadings.index.get_level_values('date').unique()
    crsp_dates = crsp.index.get_level_values('date').unique()
    loadings_flags = loadings_flags.rename({x: crsp_dates.asof(x) for x in rebalance_dates})
    df = crsp.join(loadings_flags)
    is_tradable = df.dropna().all(axis=1)
    is_tradable = is_tradable.rename({crsp_dates.asof(x): x for x in rebalance_dates})
    is_tradable.to_pickle(f'{cfg.data_fldr}/is_tradable.pkl')
