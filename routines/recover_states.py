"""Recover the true regime path underlying the cached simulation.

``routines/simulation.py`` draws a latent state sequence for every path but never
stores it, and it cannot be reconstructed from what was cached: inferring the most
likely path from the observed factors would produce an estimate, which is useless
as the ground truth against which other estimates are scored.

It can, however, be replayed. The draws are deterministic given the seed, so
re-running the same sequence of random calls reproduces the same states. Two
things make that reliable:

*Only three calls consume randomness* -- sampling the hidden Markov chain, drawing
each path's factor loadings, and drawing the idiosyncratic residuals each period.
The expensive projection of loadings onto factor returns in the original loop
consumes none, so it is skipped here, which is what makes this take minutes rather
than hours.

*The result is verified rather than assumed.* For a scattered subset of paths the
security returns are reconstructed in full and compared against the cached array.
A desynchronised stream corrupts everything after the point where it slips, so
agreement on the final path establishes that the stream held throughout.

The original constants are hard-coded below. They are not the values now in
``constants.py``: the cache was produced with a wider cross-section and more
trials, and the number of draws per iteration determines the whole stream, so the
replay has to match what was used at the time rather than what is configured now.
"""

import argparse

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm

from regimeaware.constants import DataConstants, Factors

# The configuration that produced data/sim/*.pkl, recovered from the shape of the
# cached arrays: 5000 paths of 720 months across 500 securities.
ORIG_TRIALS = 5000
ORIG_STOCKS = 500
ORIG_PERIODS = 720
ORIG_SEED = 123


def build_generator(params):
    """The generating model, configured exactly as in routines/simulation.py."""
    gen = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        min_covar=1e-3,
        tol=1e-1,
        implementation="scaling",
    )
    gen.startprob_ = params.startprob_
    gen.transmat_ = params.transmat_
    gen.means_ = params.means_
    gen.covars_ = np.array([np.diag(x) for x in params.covars_])
    return gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        default="0,1,100,1000,2500,3999,4998,4999",
        help="Paths to reconstruct in full and check against the cached returns.",
    )
    parser.add_argument("--trials", default=ORIG_TRIALS, type=int)
    args = parser.parse_args()

    verify = {int(x) for x in args.verify.split(",") if x}
    wdir = DataConstants.WDIR.value

    factors = pd.read_pickle(f"{wdir}/data/ff.pkl")[Factors._member_names_].columns
    loadings_cov = pd.read_pickle(f"{wdir}/data/dist/loadings_cov.pkl")
    loadings_means = pd.read_pickle(f"{wdir}/data/dist/loadings_means.pkl")
    scale_by_regime = pd.read_pickle(f"{wdir}/data/dist/scale_by_regime.pkl")
    mdl_hmm = pd.read_pickle(f"{wdir}/data/dist/mdl_hmm.pkl")

    cached_sec = pd.read_pickle(f"{wdir}/data/sim/sec_rt.pkl") if verify else {}
    cached_fctr = pd.read_pickle(f"{wdir}/data/sim/fctr_rt.pkl") if verify else {}

    collect_states = {}
    checks = {}

    for i in tqdm(range(args.trials), desc="replaying"):
        gen_hmm = build_generator(mdl_hmm)

        # simulation.py seeds only on the first iteration; every later draw
        # continues the same global stream, so the order here must be preserved
        # exactly.
        if i == 0:
            gen_hmm.random_state = ORIG_SEED
            np.random.seed(ORIG_SEED)

        X, Z = gen_hmm.sample(ORIG_PERIODS)
        gen_hmm.predict_proba(X)  # deterministic, but kept for exactness

        vals = np.random.multivariate_normal(
            mean=loadings_means, cov=loadings_cov, size=ORIG_STOCKS
        )

        if i in verify:
            B = pd.DataFrame(vals, columns=loadings_means.index).T
            B.columns.name = "stock"
            B.index.names = ["factor", "state"]
            B = B.stack()
            R = np.zeros((ORIG_PERIODS, ORIG_STOCKS))

            for t, s_t in enumerate(Z):
                x_t = pd.Series(X[t], factors)
                x_t.index.name = "factor"
                R[t] += B.xs(s_t, level="state").mul(x_t).groupby("stock").sum()
                R[t] += np.random.normal(0, scale_by_regime[s_t], ORIG_STOCKS)

            # Bit-exact agreement is the wrong bar: summing the factor
            # contributions is a floating-point reduction whose ordering depends
            # on threading, which moves the last bit. Genuine desynchronisation
            # would produce different residual draws and differences of order
            # 1e-2, so a tolerance of 1e-12 separates the two by ten orders of
            # magnitude in either direction.
            checks[i] = {
                "returns max diff": float(np.abs(R - cached_sec[i].values).max()),
                "factors max diff": float(np.abs(X - cached_fctr[i].values).max()),
            }
        else:
            # Same draws, same order; the projection consumes no randomness.
            for s_t in Z:
                np.random.normal(0, scale_by_regime[s_t], ORIG_STOCKS)

        collect_states[i] = Z.astype(np.int8)

    TOLERANCE = 1e-12

    report = pd.DataFrame(checks).T.astype(float)
    report["verified"] = report.max(axis=1) < TOLERANCE
    print()
    print(report.to_string(float_format=lambda v: f"{v:.3e}"))

    if not report["verified"].all():
        raise SystemExit(
            "replay diverged from the cached simulation; the recovered states are "
            "not trustworthy and must not be written"
        )

    states = pd.DataFrame(collect_states).T
    states.index.name = "iteration"
    states.columns.name = "period"
    states.to_pickle(f"{wdir}/data/sim/states.pkl")

    occupancy = states.stack().value_counts(normalize=True).sort_index()
    print(f"\nverified on {len(report)} paths; wrote {states.shape} to data/sim/states.pkl")
    print("unconditional state occupancy:")
    print((occupancy * 100).round(2).to_string())
