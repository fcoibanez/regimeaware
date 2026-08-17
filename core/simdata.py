"""Access to the cached Monte Carlo simulation paths.

The simulation was generated with more trials and a wider cross-section than any
single experiment consumes, so the full cache is an awkward 14 GB to hold in
memory. This module serves the subset a routine actually needs and keeps a
compact copy of it on disk, which makes running several arms in parallel
practical.
"""

import os

import pandas as pd


def load_simulation(trials, n_assets, wdir):
    """Return simulated security and factor returns for the requested subset.

    :param trials: number of simulation paths required.
    :param n_assets: size of the investable cross-section required.
    :param wdir: project working directory.
    :return: dict of security returns, dict of factor returns, both keyed by path.
    """
    cache = f"{wdir}/data/sim/compact_{trials}x{n_assets}.pkl"

    if os.path.exists(cache):
        return pd.read_pickle(cache)

    sec_rt = pd.read_pickle(f"{wdir}/data/sim/sec_rt.pkl")
    fctr_rt = pd.read_pickle(f"{wdir}/data/sim/fctr_rt.pkl")

    if trials > len(sec_rt):
        raise ValueError(
            f"Requested {trials} paths but the cache holds {len(sec_rt)}."
        )

    subset = (
        {i: sec_rt[i].iloc[:, :n_assets] for i in range(trials)},
        {i: fctr_rt[i] for i in range(trials)},
    )
    pd.to_pickle(subset, cache)

    return subset
