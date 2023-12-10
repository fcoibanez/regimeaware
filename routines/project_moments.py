"""Projects the factor joint distribution moments onto security space."""

if __name__ == "__main__":
    import pandas as pd
    from regimeaware.core import utils
    from regimeaware.routines import cfg
    from tqdm import tqdm

    betas = pd.read_pickle(f'{cfg.data_fldr}/exposures/forecasted_betas.pkl')
    var = pd.read_pickle(f'{cfg.data_fldr}/exposures/var.pkl')
    mu = pd.read_pickle(f'{cfg.data_fldr}/regimes/mu.pkl')
    sigma = pd.read_pickle(f'{cfg.data_fldr}/regimes/sigma.pkl')
    transmat = pd.read_pickle(f'{cfg.data_fldr}/regimes/transmat.pkl')
    emission_prob = pd.read_pickle(f'{cfg.data_fldr}/regimes/emission_prob.pkl')

    rebalance_dates = pd.date_range(cfg.bt_start_dt, cfg.bt_end_dt, freq=cfg.rebalance_freq)
    collect_mu = []
    collect_sigma = []
    collect_mu_f = []
    collect_sigma_f = []

    for dt in tqdm(rebalance_dates):
        g_t = utils.looking_fwd_prob(
            transmat=transmat.xs(dt),
            current_emission_prob=emission_prob.xs(dt).xs(dt),
            horizon=cfg.forecast_horizon
        )

        theta = utils.unpack_betas(betas.xs(dt))
        sec_ids = list(theta.index)

        mu_factor = utils.expected_means(conditional_means=mu.xs(dt), probs=g_t)
        sigma_factor = utils.expected_covar(conditional_covars=sigma.xs(dt), probs=g_t)

        mu_t = utils.project_means(betas=theta, factor_means=mu_factor)
        sigma_t = utils.project_covariances(betas=theta, factor_covariance=sigma_factor, residual_variances=var.xs(dt))

        # Collect the results
        mu_t = mu_t.reindex(sec_ids)
        sigma_t = sigma_t.loc[sec_ids, sec_ids]
        idx = pd.MultiIndex.from_tuples([(dt, x) for x in sec_ids], names=['date', 'permno'])
        mu_t.index = idx
        sigma_t.index = idx
        collect_mu += [mu_t]
        collect_sigma += [sigma_t]

        mu_factor = mu_factor.reindex(cfg.factor_set)
        sigma_factor = sigma_factor.loc[cfg.factor_set, cfg.factor_set]
        idx = pd.MultiIndex.from_tuples([(dt, x) for x in cfg.factor_set], names=['date', 'factor'])
        mu_factor.index = idx
        sigma_factor.index = idx
        collect_mu_f += [mu_factor]
        collect_sigma_f += [sigma_factor]

    # Save finalized results
    res_sigma_f = pd.concat(collect_sigma_f, axis=0)
    res_mu_f = pd.concat(collect_mu_f, axis=0)
    res_sigma_f.to_pickle(f'{cfg.data_fldr}/moments/factor_covars.pkl')
    res_mu_f.to_pickle(f'{cfg.data_fldr}/moments/factor_means.pkl')

    res_sigma = pd.concat(collect_sigma, axis=0)
    res_mu = pd.concat(collect_mu, axis=0)
    res_sigma.to_pickle(f'{cfg.data_fldr}/moments/stock_covars.pkl')
    res_mu.to_pickle(f'{cfg.data_fldr}/moments/stock_means.pkl')
