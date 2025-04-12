from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import os
from regimeaware.routines import cfg


# --------------------------------------------
# Data
fctr_rt = pd.read_pickle(f"{cfg.fldr}/ff.pkl").drop(["rf", "umd"], axis=1)
loadings_cov = pd.read_pickle(f"{cfg.fldr}/loadings_cov.pkl")
loadings_means = pd.read_pickle(f"{cfg.fldr}/loadings_means.pkl")
mcap = pd.read_pickle(f"{cfg.fldr}/mcap.pkl")
sec_rt = pd.read_pickle(f"{cfg.fldr}/sec_rt.pkl")

# --------------------------------------------
# Train HMM based on real data
mdl_hmm = GaussianHMM(
    n_components=cfg.n_states,
    covariance_type=cfg.hm_cov,
    random_state=cfg.hm_rs,
    min_covar=cfg.hm_min_covar,
    tol=cfg.hm_tol,
    implementation=cfg.hm_implementation
)

mdl_hmm.fit(fctr_rt)

param_startprob = mdl_hmm.startprob_
param_transmat = mdl_hmm.transmat_
param_means = mdl_hmm.means_
param_covars = mdl_hmm.covars_

pd.to_pickle(mdl_hmm, f"{cfg.fldr}/data/sim/mdl_hmm.pkl")

# --------------------------------------------
# Returns simulation
for iter in range(cfg.num_simulations):
    print(iter)
    # --------------------------------------------
    # Factor returns
    gen_hmm = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.hm_cov,
        min_covar=cfg.hm_min_covar,
        tol=cfg.hm_tol,
        implementation=cfg.hm_implementation
    )

    if iter == 0:
        gen_hmm.random_state = cfg.hm_rs
        np.random.seed(cfg.hm_rs)

    gen_hmm.startprob_ = param_startprob
    gen_hmm.transmat_ = param_transmat
    gen_hmm.means_ = param_means
    gen_hmm.covars_ = np.array([np.diag(x) for x in param_covars])

    X, Z = gen_hmm.sample(cfg.trn_periods + cfg.test_periods)
    G = gen_hmm.predict_proba(X)
    R = np.zeros((cfg.trn_periods + cfg.test_periods, max(cfg.num_stocks)))

    # True factor loadings
    _vals = np.random.multivariate_normal(mean=loadings_means, cov=loadings_cov, size=max(cfg.num_stocks))
    B = pd.DataFrame(_vals, columns=loadings_means.index).T
    B.columns.name = "stock"
    B.index.names = ["factor", "state"]
    B = B.stack()

    # Stock returns
    # TODO: the scale of the residual variance has to be estimated from the data
    E = np.random.normal(loc=0, scale=np.sqrt(.0025), size=R.shape)
    for _t, s_t in enumerate(Z):
        x_t = pd.Series(X[_t], fctr_rt.columns)
        x_t.index.name = "factor"
        R[_t] += B.xs(s_t, level="state").mul(x_t).groupby("stock").sum()
    R += E

    dir_path = f"{cfg.fldr}/data/sim/{iter}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    pd.to_pickle(R, f"{dir_path}/sec_rt.pkl")
    pd.to_pickle(X, f"{dir_path}/fctr_rt.pkl")
    pd.to_pickle(G, f"{dir_path}/probs.pkl")
