import pandas as pd
import numpy as np
import cvxpy as cvx
from sklearn.covariance import LedoitWolf


def mvo_portfolio(gamma, sec_rt):
    n = sec_rt.shape[1]

    mu_t = sec_rt.mean().values

    try:
        cov = LedoitWolf().fit(sec_rt.dropna())
        sigma_t = cov.covariance_
    except:
        print("Shrinkage failed.")
        sigma_t = sec_rt.cov()
    
    # gamma * mu_t @ w + (gamma**2/2) * cvx.quad_form(w, sigma_t)

    try:
        def K(w):
            u = cvx.vstack([cvx.log(1)
            - gamma * mu_t @ w
            + (gamma**2/2) * cvx.quad_form(w, sigma_t)])
            return cvx.log_sum_exp(u)

        w = cvx.Variable(n)
        objective = cvx.Minimize(K(w))
        constraints = [ w >= 0, cvx.sum(w) == 1]
        # constraints = [cvx.sum(w) == 0, w >= -1, w <= 1, cvx.norm(w,1) <= 2]
        egm_prob = cvx.Problem(objective, constraints)
        egm_prob.solve(solver=cvx.MOSEK)

        return w.value
    except:
        print(f"Optimization failed")
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
    crsp = pd.read_pickle(f'{cfg.data_fldr}/crsp.pkl')
    rt = pd.pivot_table(crsp[['excess_ret']], index='date', columns='permno', values='excess_ret')

    # Portfolio optimization
    avail_fldr = [x[0] for x in os.walk(fr"{cfg.data_fldr}\results")]
    fldr_name = fr"{cfg.data_fldr}\results\mvo_{str(int(gamma))}"

    if fldr_name not in avail_fldr:
        os.mkdir(fldr_name)

    for dt in rebalance_dts:
        fname = f"{dt.strftime('%Y%b')}.pkl"
        if fname in os.listdir(fldr_name):
            print(f"Skipping {dt.strftime('%Y %b')}...")
            continue
        else:
            print(f"Running {dt.strftime('%Y %b')}...")
            tradable_ids = is_tradable[is_tradable].xs(dt).index
            sec_rt_t = rt.loc[:dt, tradable_ids]

            w_t = mvo_portfolio(gamma, sec_rt_t)
            idx = pd.MultiIndex.from_tuples([(dt, x) for x in tradable_ids], names=('date', 'permno'))
            w_t = pd.Series(w_t, index=idx)
            if w_t is not None:
                w_t = 2 * w_t / np.abs(w_t).sum()
                w_t.to_pickle(f"{fldr_name}/{fname}")

    print("EOF")
