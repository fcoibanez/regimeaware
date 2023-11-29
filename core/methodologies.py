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
        g = self.prob.asof(as_of).values.reshape(-1, 1)

        self.cond_prob = pd.Series((B_h.T @ g).flatten(), index=self._state_labels)
        self.cond_params = pd.Series((betas @ B_h.T @ g).flatten(), index=self._param_labels)
        self.cond_sigma = (self.sigma.values @ B_h.T @ g).item()


class ProjectMoments:  # Should be ConditionalJointDistribution
    def __init__(self, factor_mean, factor_cov, probabilities, residuals):
        pass


if __name__ == "__main__":
    import os
    from datetime import datetime


