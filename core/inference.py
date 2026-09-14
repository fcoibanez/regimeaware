"""Statistical inference on backtested performance.

Two sources of uncertainty are at play and are kept apart deliberately.

*Estimation* uncertainty is what an investor faces: given one realised return
path of length T, how precisely is a Sharpe ratio pinned down? This is what the
Ledoit and Wolf (2008) test addresses, and with T = 60 it is severe.

*Monte Carlo* uncertainty is a property of the experiment: across many simulated
paths, how reliably does one method beat another? Because every arm is evaluated
on the identical set of paths, this can be tested pairwise, which is far more
powerful than comparing two independent means.

Reporting only the first understates what the simulation establishes; reporting
only the second overstates what an investor could verify from a single history.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from regimeaware.modules.robustsharpe import robustsharpe as rs


def paired_comparison(treatment, control, label=None):
    """Paired tests of ``treatment`` against ``control`` across simulation paths.

    Reports the mean difference alongside the median and the fraction of paths on
    which the treatment wins. These can disagree -- a method that helps on most
    paths but fails badly on a few has a positive median and a negative mean --
    and in that case neither statistic alone describes the result.

    :param treatment: Series indexed by path.
    :param control: Series indexed by path, aligned to ``treatment``.
    :return: dict of summary statistics.
    """
    treatment, control = treatment.align(control, join="inner")
    diff = (treatment - control).dropna()
    n = len(diff)

    t_stat, p_t = stats.ttest_rel(treatment.loc[diff.index], control.loc[diff.index])
    wins = int((diff > 0).sum())

    out = {
        "Mean (treatment)": treatment.loc[diff.index].mean(),
        "Mean (control)": control.loc[diff.index].mean(),
        "Delta": diff.mean(),
        "Std. Err.": diff.std(ddof=1) / np.sqrt(n),
        "t-stat": t_stat,
        "p (paired t)": p_t,
        "Median delta": diff.median(),
        "Frac. > 0": wins / n,
        "p (sign)": stats.binomtest(wins, n, 0.5).pvalue,
        "N paths": n,
    }

    # Wilcoxon is undefined when every difference is zero
    if (diff != 0).any():
        out["p (Wilcoxon)"] = stats.wilcoxon(diff)[1]
    else:
        out["p (Wilcoxon)"] = np.nan

    if label is not None:
        out = {"Comparison": label, **out}

    return out


def significance_table(metrics, base, phi, arms=None):
    """Paired tests of every metric of every arm against a reference arm.

    Answers Reviewer 2's request for the statistical significance of the reported
    measures relative to the benchmark, not only of the Sharpe ratio.

    :param metrics: dict of arm name -> DataFrame indexed by (phi, path).
    :param base: name of the reference arm.
    :param phi: risk aversion level to report.
    :return: DataFrame indexed by (arm, metric).
    """
    arms = [a for a in (arms or metrics) if a != base]
    rows = {}
    for arm in arms:
        for metric in metrics[arm].columns:
            rows[(arm, metric)] = paired_comparison(
                metrics[arm].xs(phi)[metric], metrics[base].xs(phi)[metric]
            )

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.names = ["model", "metric"]
    return out


# Ledoit and Wolf propose two implementations of their test. The asymptotic one
# uses a HAC standard error; the other studentises a circular block bootstrap.
# The bootstrap is the default here because the asymptotic version is badly
# undersized on these data: with the two strategies correlated at about 0.94 and
# only sixty observations, it rejects roughly 1% of the time under the null
# against a nominal 5%, and so fails to detect differences that are really there.
#
# The block length is one. That is not because the returns are serially
# independent -- their first-order autocorrelation is around 0.15 -- but because
# at T = 60 longer blocks leave too few distinct blocks to resample from, and the
# test becomes progressively more conservative: calibrating the size on data
# matching these portfolios gives 6.4%, 2.4%, 1.6% and 0.8% for block lengths of
# 1, 3, 6 and 10 against a nominal 5%.
BOOTSTRAP_BLOCK = 1
BOOTSTRAP_DRAWS = 499

# The resampling is seeded so that a reported rejection rate is reproducible.
# Each path draws from its own seed, derived from the path identifier: a single
# seed shared across paths would give every path the same resampling pattern and
# correlate their p-values.
BOOTSTRAP_SEED = 20261211


def path_seed(identifier):
    """Deterministic per-path seed."""
    try:
        return BOOTSTRAP_SEED + int(identifier)
    except (TypeError, ValueError):
        return BOOTSTRAP_SEED + (hash(identifier) % 10_000_019)


def ledoit_wolf_path(returns_a, returns_b, alpha=0.05, method="bootstrap",
                     block_size=BOOTSTRAP_BLOCK, draws=BOOTSTRAP_DRAWS,
                     seed=BOOTSTRAP_SEED):
    """Ledoit and Wolf (2008) test of equal Sharpe ratios on a single path.

    :param method: ``bootstrap`` for the studentised circular block bootstrap, or
        ``hac`` for the asymptotic Parzen-kernel standard error. See the note
        above on why the bootstrap is the default.
    :return: dict with the two Sharpe ratios, their difference, and the p-value.
    """
    # Align on the periods both strategies actually traded. A period missing from
    # one series (an optimisation that failed to solve, say) carries no
    # information about their relative performance and must not silently shift
    # one series against the other.
    if isinstance(returns_a, pd.Series) and isinstance(returns_b, pd.Series):
        returns_a, returns_b = returns_a.align(returns_b, join="inner")

    pair = np.column_stack(
        [np.asarray(returns_a, float), np.asarray(returns_b, float)]
    )
    pair = pair[np.isfinite(pair).all(axis=1)]

    if len(pair) < 3:
        return {k: np.nan for k in
                ("SR (control)", "SR (treatment)", "SR difference",
                 "Std. Err.", "p-value", "CI low", "CI high")}

    # Column order is (b, a) so that the reported difference is a - b
    ordered = pair[:, ::-1]
    if method == "bootstrap":
        sr, sr_diff, ci, p_val, se = rs.bootstrap_inference(
            ordered, block_size=block_size, alpha=alpha, M=draws, seed=seed
        )
    else:
        sr, sr_diff, ci, p_val, se = rs.relative_hac_inference(ordered, alpha=alpha)

    return {
        "SR (control)": sr[0],
        "SR (treatment)": sr[1],
        "SR difference": sr_diff,
        "Std. Err.": se,
        "p-value": p_val,
        "CI low": ci[0],
        "CI high": ci[1],
    }


def ledoit_wolf_batch(pairs, seeds=None, alpha=0.05, block_size=BOOTSTRAP_BLOCK,
                      draws=BOOTSTRAP_DRAWS, n_jobs=-1):
    """``ledoit_wolf_path`` over many pairs at once, in parallel.

    The bootstrap is a Python loop over resamples, so one test takes a quarter of
    a second at T = 60, and the set the manuscript reports -- five arms, three
    levels of risk aversion, a thousand paths -- takes hours in series. The tests
    are independent, so they are spread across processes.

    :param pairs: iterable of ``(returns_a, returns_b)``.
    :param seeds: one seed per pair. Defaults to the path-derived seed of each
        pair's position, which is right only when ``pairs`` is in path order.
    :return: list of ``ledoit_wolf_path`` results, in the order of ``pairs``.
    """
    pairs = list(pairs)
    if seeds is None:
        seeds = [path_seed(i) for i in range(len(pairs))]
    return Parallel(n_jobs=n_jobs)(
        delayed(ledoit_wolf_path)(
            a, b, alpha=alpha, block_size=block_size, draws=draws, seed=s
        )
        for (a, b), s in zip(pairs, seeds)
    )


def ledoit_wolf_summary(returns_a, returns_b, paths, alpha=0.05,
                        block_size=BOOTSTRAP_BLOCK, draws=BOOTSTRAP_DRAWS):
    """Ledoit and Wolf test applied path by path, then summarised.

    With T = 60 the test has little power on any single path, so the informative
    quantity is not one p-value but the distribution of them: how often a single
    five-year history would have let an investor reject equality at ``alpha``.
    That rejection frequency is reported alongside the mean Sharpe difference.

    :param returns_a: DataFrame of treatment returns indexed by (phi, path).
    :param returns_b: DataFrame of control returns, same index.
    :param paths: iterable of path identifiers to evaluate.
    :return: DataFrame with one row per path.
    """
    paths = list(paths)
    results = ledoit_wolf_batch(
        [(returns_a.loc[i].dropna(), returns_b.loc[i].dropna()) for i in paths],
        seeds=[path_seed(i) for i in paths],
        alpha=alpha, block_size=block_size, draws=draws,
    )
    rows = dict(zip(paths, results))

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "iteration"
    return out


def detection_by_horizon(returns_a, returns_b, paths, horizons, alpha=0.05,
                         block_size=BOOTSTRAP_BLOCK, draws=BOOTSTRAP_DRAWS,
                         max_histories=150):
    """How often the test detects the difference, as a function of history length.

    A test that fails to reject at five years may be reporting the absence of an
    effect or merely the shortness of the sample, and the two are worth telling
    apart. Applying the same test over progressively longer histories separates
    them: if the effect is real, the rejection rate climbs with the horizon.

    Longer histories are assembled by concatenating independent simulation paths.
    They are draws from the same process, so the concatenation has the right
    return distribution; what it does not reproduce is the continuity of a single
    regime sequence across the joins, which makes this a measurement of the
    test's power rather than a literal long backtest.

    :param horizons: multiples of one path length to evaluate.
    :return: DataFrame indexed by months, with the rejection rate.
    """
    a = [np.asarray(returns_a.loc[i].dropna(), float) for i in paths]
    b = [np.asarray(returns_b.loc[i].dropna(), float) for i in paths]

    # Every history at every horizon is an independent test, so all of them are
    # collected first and run as one parallel batch.
    jobs = []
    for k in horizons:
        n = min(max_histories, len(a) // k)
        if n < 2:
            continue
        for j in range(n):
            block = slice(j * k, (j + 1) * k)
            jobs.append((k, j, np.concatenate(a[block]), np.concatenate(b[block])))

    results = ledoit_wolf_batch(
        [(ha, hb) for _, _, ha, hb in jobs],
        seeds=[BOOTSTRAP_SEED + 1000 * k + j for k, j, _, _ in jobs],
        alpha=alpha, block_size=block_size, draws=draws,
    )

    rows = {}
    for k in horizons:
        rejected = [
            r["p-value"] < alpha
            for (kk, _, _, _), r in zip(jobs, results) if kk == k
        ]
        n = len(rejected)
        if n < 2:
            continue

        months = k * len(a[0])
        rows[months] = {
            "Years": months / 12,
            "Detection rate": float(np.mean(rejected)),
            "Histories": n,
        }

    out = pd.DataFrame(rows).T
    out.index.name = "Months"
    return out


def rejection_summary(lw_table, alpha=0.05, periods_per_year=12):
    """Collapse per-path Ledoit and Wolf results into reportable figures.

    ``Reject (favourable)`` is the share of paths on which a single realised
    history would have rejected equal Sharpe ratios *in favour of* the treatment.
    ``Reject (adverse)`` is the share rejecting against it. The remainder are
    paths on which five years of monthly data simply cannot tell the two apart.
    """
    scale = np.sqrt(periods_per_year)
    reject = lw_table["p-value"] < alpha
    better = lw_table["SR difference"] > 0

    return pd.Series(
        {
            "Mean SR difference (ann.)": lw_table["SR difference"].mean() * scale,
            "Median SR difference (ann.)": lw_table["SR difference"].median() * scale,
            "Frac. SR difference > 0": better.mean(),
            "Reject (favourable)": (reject & better).mean(),
            "Reject (adverse)": (reject & ~better).mean(),
            "Inconclusive": (~reject).mean(),
            "Median p-value": lw_table["p-value"].median(),
            "N paths": len(lw_table),
        }
    )
