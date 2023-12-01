import pandas as pd
from scipy.linalg import block_diag
import numpy as np


class RegimeWeightedLS:
    # sigma * I (check large scale portfolio opt paper)
    def __init__(self, exog, endog, emission_prob):
        self.endog = endog.copy()
        self.exog = exog.copy()
        self.prob = emission_prob.copy()
        self.transition_matrix = None
        self.params = None
        self.resid = None
        self.sigma = None
        self.cond_params = None
        self.cond_prob = None
        self.cond_sigma = None
        self.m = self.prob.shape[1]
        self._param_labels = list(self.exog.columns)
        self._state_labels = list(self.prob.columns)

    def fit(self, add_constant=False):
        cmn_idx = pd.concat([self.endog, self.exog], axis=1).dropna().index
        r = self.endog.reindex(cmn_idx).values.reshape(-1, 1)

        # Regressors
        Z = self.exog.reindex(cmn_idx).values
        if add_constant:
            c = np.ones((Z.shape[0], 1), dtype=float)
            Z = np.concatenate([c, Z], axis=1)
            self._param_labels = ['const'] + self._param_labels

        k = Z.shape[1]
        M = block_diag(*[Z] * self.m)
        d = block_diag(*[r] * self.m)
        G = self.prob.reindex(cmn_idx).values
        O = np.diag(G.flatten('F'))
        I = np.concatenate([np.eye(k)] * self.m, axis=1)
        betas = I @ np.linalg.inv(M.T @ O @ M) @ M.T @ O @ d
        res = pd.DataFrame(betas, index=self._param_labels, columns=self._state_labels)
        self.params = res.copy()

        self.resid = pd.DataFrame(index=cmn_idx, columns=self._state_labels)
        self.sigma = pd.Series(index=self._state_labels)
        for s in range(self.m):
            b = betas[:, s].reshape(-1, 1)
            r_hat = Z @ b
            e = r - r_hat
            self.resid[s] = pd.Series(e.flatten(), index=cmn_idx)
            self.sigma[s] = (e.T @ e).item() * 1 / r.shape[0]

    def predict(self, transition_matrix, as_of=None, horizon=1):
        betas = self.params.values
        B = transition_matrix.values
        B_h = np.linalg.matrix_power(B, horizon)  # TODO: triple-check the orientation of these matrices
        g = self.prob.asof(as_of).values.reshape(-1, 1)  # TODO: this forecast prob has to be taken to another func

        self.cond_prob = pd.Series((B_h.T @ g).flatten(), index=self._state_labels)
        self.cond_params = pd.Series((betas @ B_h.T @ g).flatten(), index=self._param_labels)
        self.cond_sigma = (self.sigma.values @ B_h.T @ g).item()


class ProjectMoments:  # Should be ConditionalJointDistribution
    def __init__(self, factor_mean, factor_cov, probabilities, residuals):
        pass


def expected_covar(covars, probs):
    w = probs.copy()
    w.index.name = "state"
    res = covars.mul(w, axis=0).groupby('factor').sum().reindex(covars.columns)
    return res


if __name__ == "__main__":
    from core import cfg
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

    for dt in tqdm(rebalance_dates):
        sec_ids = betas.xs(dt).index.get_level_values('id').unique()

        B = transmat.xs(dt).values
        B_h = np.linalg.matrix_power(B, cfg.forecast_horizon)  # TODO: triple-check the orientation of these matrices
        g = emission_prob.xs(dt).xs(dt).values.reshape(-1, 1)
        g_t = B_h.T @ g
        var_e = var.xs(dt)[sec_ids]

        theta = betas.xs(dt)
        theta = pd.pivot_table(theta.reset_index(name='value'), index='id', columns='factor', values='value')
        theta = theta[['const'] + list(sigma.columns)].reindex(sec_ids).T.values

        exp_factor_cov = expected_covar(covars=sigma.xs(dt), probs=pd.Series(g_t.flatten())).values
        exp_factor_mu = mu.xs(dt).mul(g_t.flatten(), axis=0).sum(axis=0).values.reshape(-1, 1) # TODO: all of this has to be matrix op
        exp_factor_mu = np.concatenate([np.array([[1.0]]), exp_factor_mu])
        exp_sec_mu = theta.T @ exp_factor_mu
        exp_sec_cov = theta[1:].T @ exp_factor_cov @ theta[1:] + np.diag(var_e)

        idx = pd.MultiIndex.from_tuples([(dt, x) for x in sec_ids])
        S = pd.DataFrame(exp_sec_cov, index=idx, columns=sec_ids)
        x = pd.Series(exp_sec_mu.flatten(), index=idx)

        collect_mu += [x]
        collect_sigma += [S]

    sigma = pd.concat(collect_sigma, axis=0)
    mu = pd.concat(collect_mu, axis=0)
