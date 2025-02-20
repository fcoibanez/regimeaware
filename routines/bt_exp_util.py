import pandas as pd
import numpy as np
import cvxpy as cvx

def prepare_inputs(as_of_dt, probabilities, transmat, betas, means, vcv, idio_var, n_states, tradable_flags):
    collect_mu = []
    collect_sigmas = []

    tradable_ids = tradable_flags[tradable_flags].xs(as_of_dt).index

    betas_t = betas.xs(as_of_dt).loc[tradable_ids]
    pi = probabilities.xs(as_of_dt).xs(as_of_dt)[range(n_states)].values
    tmat_t = transmat.xs(as_of_dt).loc[range(n_states), range(n_states)]
    B_h = np.linalg.matrix_power(tmat_t.values, cfg.forecast_horizon)
    pi = (B_h @ pi.reshape(-1, 1)).flatten()
    factor_set = betas_t.drop('const', axis=1).columns
    means_t = means.xs(as_of_dt)[factor_set]
    vcv_t = vcv.xs(as_of_dt).loc[pd.IndexSlice[range(n_states), factor_set], factor_set]
    idio_t = idio_var.xs(as_of_dt)[tradable_ids]

    for s in range(n_states):
        b_s = betas_t.xs(s, level=1).loc[tradable_ids]

        mu_s = means_t.xs(s)
        mu_s['const'] = 1
        mu_i = b_s.dot(mu_s)
        
        V_i = b_s.drop('const', axis=1).dot(vcv_t.xs(s)).dot(b_s.drop('const', axis=1).T)
        V_i += np.diag(idio_t.xs(s))

        collect_mu += [mu_i.to_frame(s)]
        collect_sigmas += [V_i.loc[tradable_ids, tradable_ids].values]
        
    mus = pd.concat(collect_mu, axis=1).T.values
    Sigmas = np.array(collect_sigmas)

    return mus, Sigmas, pi, tradable_ids


def exp_util_portfolio(as_of_dt, probabilities, transmat, betas, means, vcv, idio_var, n_states, tradable_flags, gamma):
    mu_t, sigma_t, pi_t, sec_t = prepare_inputs(
        as_of_dt=as_of_dt,
        probabilities=probabilities,
        transmat=transmat,
        betas=betas,
        means=means,
        vcv=vcv,
        idio_var=idio_var,
        n_states=n_states,
        tradable_flags=tradable_flags
    )

    k = n_states
    n = len(sec_t)

    try:
        def K(w):
            u = cvx.vstack([cvx.log(pi_t[i])
            - gamma * mu_t[i] @ w
            + (gamma**2/2) * cvx.quad_form(w, sigma_t[i]) for i in range(k)])
            return cvx.log_sum_exp(u)

        w = cvx.Variable(n)
        objective = cvx.Minimize(K(w))
        constraints = [ w >= 0, cvx.sum(w) == 1]
        # constraints = [cvx.sum(w) == 0, w >= -1, w <= 1, cvx.norm(w,1) <= 2]
        egm_prob = cvx.Problem(objective, constraints)
        egm_prob.solve(solver=cvx.MOSEK)
        idx = pd.MultiIndex.from_tuples([(as_of_dt, x) for x in sec_t], names=('date', 'permno'))
        target_wts = pd.Series(w.value, index=idx)

        return target_wts
    except:
        print(f"Optimization failed on {as_of_dt.strftime('%Y-%m-%d')}")
        return None


if __name__ == "__main__":
    import random
    from regimeaware.routines import cfg
    import os
    import argparse

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=75, type=float)
    parser.add_argument('--shuffle', default=False)

    # Parse arguments
    args = parser.parse_args()

    gamma = args.gamma
    shuffle = args.shuffle

    rebalance_dts = pd.date_range(start=cfg.bt_start_dt, end=cfg.bt_end_dt, freq=cfg.rebalance_freq)
    rebalance_dts = list(rebalance_dts)

    if shuffle:
        random.shuffle(rebalance_dts)

    # Data
    is_tradable = pd.read_pickle(f'{cfg.data_fldr}/is_tradable.pkl')
    state_betas = pd.read_pickle(f'{cfg.data_fldr}/exposures/state_betas.pkl')
    state_means = pd.read_pickle(f'{cfg.data_fldr}/regimes/mu.pkl')
    state_vcv = pd.read_pickle(f'{cfg.data_fldr}/regimes/sigma.pkl')
    state_idio_var = pd.read_pickle(f'{cfg.data_fldr}/exposures/state_resid.pkl')
    emission_prob = pd.read_pickle(f"{cfg.data_fldr}/regimes/emission_prob.pkl")
    transition_matrix = pd.read_pickle(f"{cfg.data_fldr}/regimes/transmat.pkl")

    args = {
        'probabilities': emission_prob,
        'transmat': transition_matrix,
        'betas': state_betas,
        'means': state_means,
        'vcv': state_vcv,
        'idio_var': state_idio_var,
        'n_states': cfg.n_states,
        'tradable_flags': is_tradable,
        'gamma': gamma,
    }

    # Portfolio optimization
    avail_fldr = [x[0] for x in os.walk(fr"{cfg.data_fldr}\results")]
    fldr_name = fr"{cfg.data_fldr}\results\exp_{str(int(gamma))}"

    if fldr_name not in avail_fldr:
        os.mkdir(fldr_name)

    for dt in rebalance_dts:
        fname = f"{dt.strftime('%Y%b')}.pkl"
        if fname in os.listdir(fldr_name):
            print(f"Skipping {dt.strftime('%Y %b')}...")
            continue
        else:
            print(f"Running {dt.strftime('%Y %b')}...")
            w_t = exp_util_portfolio(as_of_dt=dt, **args)
            if w_t is not None:
                w_t = 2 * w_t / np.abs(w_t).sum()
                w_t.to_pickle(f"{fldr_name}/{fname}")

    print("EOF")
