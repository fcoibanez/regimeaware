if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import cvxpy as cp
    from regimeaware.routines import cfg
    from itertools import product
    from regimeaware.core import utils
    from tqdm import tqdm

    rebalance_dts = pd.date_range(start=cfg.bt_start_dt, end=cfg.bt_end_dt, freq=cfg.rebalance_freq)

    # Tradability flags & Market cap
    is_tradable = pd.read_pickle(f'{cfg.data_fldr}/is_tradable.pkl')
    mktcap = pd.read_pickle(f'{cfg.data_fldr}/mktcap.pkl')

    # Load cached factor estimates
    factor_covars = pd.read_pickle(f'{cfg.data_fldr}/moments/factor_covars.pkl')
    factor_means = pd.read_pickle(f'{cfg.data_fldr}/moments/factor_means.pkl')
    factor_loadings = pd.read_pickle(f'{cfg.data_fldr}/exposures/forecasted_betas.pkl')
    factor_variance = pd.read_pickle(f'{cfg.data_fldr}/exposures/var.pkl')

    # Backtest
    collect_wt = {}
    crsp_dts = mktcap.index.get_level_values('date').unique()
    perm = list(product(cfg.gamma_iter, rebalance_dts))
    for g, dt in tqdm(perm):
        as_of_dt = crsp_dts.asof(dt)
        loadings_t = utils.unpack_betas(factor_loadings.xs(dt))
        tradable_ids = list(is_tradable.xs(dt).index)
        mu_f = factor_means.xs(dt)[cfg.factor_set].values.reshape(-1, 1)
        mu_f_const = np.concatenate([np.array([[1]]), mu_f], axis=0)  # Adding back the constant
        Sigma_f = factor_covars.xs(dt).loc[cfg.factor_set, cfg.factor_set].values
        F = loadings_t.reindex(tradable_ids).values.T
        E = np.diag(factor_variance.xs(dt).reindex(tradable_ids))
        b = mktcap.xs(as_of_dt).reindex(tradable_ids)
        b = np.divide(b, b.sum())
        b = b.values.reshape(-1, 1)

        # Optimization problem
        gamma = cp.Parameter(nonneg=True)

        m, n = F.shape
        w = cp.Variable((n, 1))
        f = cp.Variable((m, 1))

        Sigma_f_const = np.zeros((m, m))
        Sigma_f_const[1:, 1:] = Sigma_f

        port_risk = cp.quad_form(f, Sigma_f_const) + cp.sum_squares(np.sqrt(E) @ (w - b))
        port_return = mu_f_const.T @ f

        constraints = [
            cp.sum(w) == 1,
            f == F @ (w - b),
            w >= 0
        ]

        gamma.value = g
        prob = cp.Problem(cp.Maximize(port_return - gamma * port_risk), constraints)
        prob.solve(verbose=False, solver=cp.CLARABEL)
        collect_wt[(g, dt)] = pd.Series(w.value.flatten(), index=tradable_ids)

    # Collect and save results
    wts = pd.DataFrame.from_dict(collect_wt, orient='index').fillna(0)
    wts.index.names = ['gamma', 'date']
    wts.to_pickle(f'{cfg.data_fldr}/results/sn_riskaversion_wts.pkl')
