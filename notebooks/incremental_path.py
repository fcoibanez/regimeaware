"""Analysis of the incremental path from Costa & Kwon (2020) to the RWLS framework.

Written as a paired script to notebooks/incremental_path.ipynb via jupytext-style
cell markers so the logic stays reviewable in version control.
"""

# %% [markdown]
# # Incremental path: from Costa & Kwon (2020) to the RWLS framework
#
# Each arm adds exactly one component to the previous one, so the contribution of
# every methodological choice is the difference between consecutive arms.
#
# | Arm | HMM emission | Loadings | Objective |
# |---|---|---|---|
# | `ck_uni` | univariate (Mkt-Rf) | hard classification, 24m | collapsed MVO |
# | `ck_multi` | multivariate (6 factors) | hard classification, 24m | collapsed MVO |
# | `rwls_mvo` | multivariate | RWLS, full sample | collapsed MVO |
# | `rwls_mixture` | multivariate | RWLS, full sample | full mixture utility |
#
# `ck_uni` reproduces the estimator of Costa & Kwon (2020); `rwls_mixture` is the
# framework proposed in this paper.
#
# All four arms estimate the HMM on the information set available at each
# decision date and share a single implementation of the projection from factor
# space to asset space, so the differences below are attributable to the three
# components being varied and to nothing else.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
from scipy.stats import entropy

from regimeaware.constants import DataConstants, SimulationParameters
from regimeaware.core.exhibits import write_tabular
from regimeaware.core.performance import path_metrics, portfolio_returns

ARMS = ["ck_uni_2s", "ck_uni", "ck_multi", "rwls_mvo", "rwls_mixture"]

LABELS = {
    "ck_uni_2s": r"Costa \& Kwon (2 states)",
    "ck_uni": "+ third state",
    "ck_multi": "+ multivariate HMM",
    "rwls_mvo": "+ RWLS",
    "rwls_mixture": "+ mixture utility",
}

# Each step changes exactly one component. The first begins from Costa and Kwon
# (2020) as published -- a univariate hidden Markov model with the number of
# states their Bayesian information criterion selects -- and moves to the three
# states used here, so that the emission-dimension step which follows varies the
# emission distribution alone rather than the state count as well.
STEPS = [
    ("ck_uni_2s", "ck_uni", "State count"),
    ("ck_uni", "ck_multi", "Emission dimension"),
    ("ck_multi", "rwls_mvo", "RWLS estimator"),
    ("rwls_mvo", "rwls_mixture", "Mixture objective"),
]

PHI_LIST = [10, 25, 50]

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.sans-serif": ["CMU Serif"]}
)

# %% [markdown]
# ## 1. Load weights and compute realised out-of-sample returns

# %%
from regimeaware.core.simdata import load_simulation

sec_rt, _ = load_simulation(
    SimulationParameters.TRIALS.value,
    SimulationParameters.NUM_STOCKS.value,
    DataConstants.WDIR.value,
)

wts = {}
for a in ARMS:
    try:
        wts[a] = pd.read_pickle(f"{DataConstants.WDIR.value}/results/{a}.pkl")
    except FileNotFoundError:
        print(f"{a:<16} not available -- skipped")

ARMS = [a for a in ARMS if a in wts]
STEPS = [s for s in STEPS if s[0] in wts and s[1] in wts]

# Paths actually completed for every arm, so all comparisons are like-for-like
common = set.intersection(
    *[set(w.index.get_level_values("iteration").unique()) for w in wts.values()]
)
common = sorted(common)
print(f"paths available for every arm: {len(common)}")

# %%
def realised_returns(weights, iterations):
    """Portfolio returns from weights decided at t and earned at t+1."""
    out = {}
    for phi in PHI_LIST:
        w_phi = weights.xs(phi, level="phi")
        for i in iterations:
            w = w_phi.xs(i, level="iteration")
            out[(phi, i)] = portfolio_returns(w, sec_rt[i])
    res = pd.DataFrame.from_dict(out, orient="index").sort_index()
    res.index.names = ["phi", "iteration"]
    return res


rets = {a: realised_returns(wts[a], common) for a in ARMS}

# %% [markdown]
# ## 2. Per-path performance metrics
#
# Computed path by path so that every statistic below carries a Monte Carlo
# distribution rather than a single point estimate.

# %%
# Metrics come from core.performance so this notebook cannot drift from the
# others: the Sharpe definition, the drift-corrected turnover and the certainty
# equivalent are all defined once, in one place.
metrics = {
    a: path_metrics(
        wts[a][wts[a].index.get_level_values("iteration").isin(common)],
        sec_rt,
        PHI_LIST,
    )
    for a in ARMS
}
print(metrics["rwls_mixture"].groupby("phi").mean().round(4).T)

# %% [markdown]
# ## 3. Main table: all arms, all risk aversion levels
#
# This is the table that replaces Table 2 in the revision.

# %%
table = pd.concat(
    {LABELS[a]: metrics[a].groupby("phi").mean() for a in ARMS},
    names=["model"],
).swaplevel().sort_index()

table = table.T[[(phi, LABELS[a]) for phi in PHI_LIST for a in ARMS]]
table.round(4)

# %% [markdown]
# ## 4. Incremental contributions
#
# The difference between consecutive arms, paired path by path. Because every arm
# sees the identical simulated return path, the pairing removes Monte Carlo noise
# and the test is on the *difference*, which is far more powerful than comparing
# two independent means.

# %%
def increment(metric, phi):
    """Paired difference for each step of the incremental path."""
    rows = []
    for lo, hi, name in STEPS:
        d = metrics[hi].xs(phi)[metric] - metrics[lo].xs(phi)[metric]
        t, p = stats.ttest_rel(metrics[hi].xs(phi)[metric], metrics[lo].xs(phi)[metric])
        _, p_w = stats.wilcoxon(d)
        rows.append(
            {
                "Step": name,
                "Delta": d.mean(),
                "Std. Err.": d.std() / np.sqrt(len(d)),
                "t-stat": t,
                "p (paired t)": p,
                "p (Wilcoxon)": p_w,
                "Frac. > 0": (d > 0).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("Step")


for phi in PHI_LIST:
    print(f"\n{'='*72}\nSharpe ratio increments, phi = {phi}\n{'='*72}")
    print(increment("Sharpe Ratio", phi).round(4).to_string())

# %%
# Table 4: the decomposition, panelled by risk aversion. Each row is a paired
# difference between consecutive arms, so the rows within a panel sum to the
# total advantage of the framework over Costa and Kwon (2020) as published.
rows = {}
for phi in PHI_LIST:
    inc = increment("Sharpe Ratio", phi)
    for step in inc.index:
        rows[(phi, step)] = {
            "Delta": inc.loc[step, "Delta"],
            "Std. Err.": inc.loc[step, "Std. Err."],
            "t-stat": inc.loc[step, "t-stat"],
            "p (paired t)": inc.loc[step, "p (paired t)"],
            "Frac. > 0": inc.loc[step, "Frac. > 0"],
        }
    rows[(phi, "Total")] = {
        "Delta": (metrics[ARMS[-1]].xs(phi)["Sharpe Ratio"].mean()
                  - metrics[ARMS[0]].xs(phi)["Sharpe Ratio"].mean()),
        "Std. Err.": np.nan,
        "t-stat": np.nan,
        "p (paired t)": np.nan,
        "Frac. > 0": np.nan,
    }

table4 = pd.DataFrame(rows).T
table4.index.names = ["phi", "step"]

write_tabular(
    table4,
    f"{DataConstants.WDIR.value}/tables/decomposition.tex",
    panel_level="phi",
    # Quantities vary across the columns here rather than down the rows, and the
    # increments are small enough relative to their standard errors that two
    # decimals would hide the precision.
    format_axis="columns",
    formats={"Delta": "num3", "Std. Err.": "num3"},
    notes=["Each row is a paired difference across the same simulated paths.",
           "Rows within a panel sum to the total by construction."],
)
print("\n" + table4.round(4).to_string())

# %% [markdown]
# ## 5. Where the gain comes from
#
# The decomposition Reviewer 3 asks for: the contribution of each component to
# the total improvement over the Costa & Kwon benchmark.

# %%
fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=True)

for ax, phi in zip(axes, PHI_LIST):
    inc = increment("Sharpe Ratio", phi)
    colors = ["0.75" if d < 0 else "0.35" for d in inc["Delta"]]
    ax.bar(range(len(inc)), inc["Delta"], color=colors, edgecolor="k", lw=0.8)
    ax.errorbar(
        range(len(inc)),
        inc["Delta"],
        yerr=1.96 * inc["Std. Err."],
        fmt="none",
        ecolor="k",
        capsize=3,
        lw=1,
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(inc)))
    # Derived from the steps actually present, so adding a step to the
    # decomposition cannot leave the axis labelled for the old one.
    ax.set_xticklabels([n.replace(" ", "\n") for n in inc.index], fontsize=7)
    ax.set_title(rf"$\varphi={phi}$")
    ax.grid(ls="--", alpha=0.5, axis="y", zorder=-25)
    ax.tick_params(bottom=False, left=False)

axes[0].set_ylabel("$\\Delta$ Sharpe Ratio")
plt.tight_layout()
plt.savefig(f"{DataConstants.WDIR.value}/img/ablation_increments.pdf", dpi=300, transparent=True)
plt.show()

# %% [markdown]
# ## 6. Distribution of outcomes across arms
#
# Stochastic dominance is what Figure 3 currently shows for two arms; here it is
# extended to the full path so the reader can see whether each step shifts the
# entire distribution or only its mean.

# %%
fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharex=True, sharey=True)

styles = {
    "ck_uni_2s": dict(ls=(0, (1, 1)), lw=1.1),
    "ck_uni": dict(ls=":", lw=1.2),
    "ck_multi": dict(ls="-.", lw=1.2),
    "rwls_mvo": dict(ls="--", lw=1.25),
    "rwls_mixture": dict(ls="-", lw=1.6),
}

for ax, phi in zip(axes, PHI_LIST):
    for a in ARMS:
        sns.kdeplot(
            data=metrics[a].xs(phi)["Sharpe Ratio"],
            cumulative=True, ax=ax, c="k", zorder=25,
            label=LABELS[a] if phi == PHI_LIST[0] else "",
            **styles[a],
        )
    ax.set_title(rf"$\varphi={phi}$")
    ax.set_xlabel("Sharpe Ratio")
    ax.grid(ls="--", alpha=0.5, zorder=-25)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1)
    ax.tick_params(bottom=False, left=False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=True, fontsize=7)
plt.tight_layout()
plt.savefig(f"{DataConstants.WDIR.value}/img/ablation_ecdf.pdf", dpi=300,
            transparent=True, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Behaviour across risk aversion
#
# Reviewer 3's fourth point: the advantage is not uniform in $\varphi$. Plotting
# every arm across the full grid of $\varphi$ shows where each component's
# contribution concentrates.

# %%
all_phis = list(SimulationParameters.RISK_AVERSION.value)

fig, ax = plt.subplots(figsize=(3.75, 3.75))
markers = {"ck_uni_2s": "D", "ck_uni": "^", "ck_multi": "v",
           "rwls_mvo": "o", "rwls_mixture": "s"}

for a in ARMS:
    sr = metrics[a]["Sharpe Ratio"].groupby("phi").mean()
    ax.plot(sr.index, sr.values, c="k", marker=markers[a], markerfacecolor="white",
            markeredgewidth=1, label=LABELS[a], **{k: v for k, v in styles[a].items() if k != "lw"})

ax.set_xlabel(r"$\varphi$")
ax.set_ylabel("Ann. Sharpe Ratio")
ax.grid(ls="--", alpha=0.5, zorder=-25)
ax.tick_params(bottom=False, left=False)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{DataConstants.WDIR.value}/img/ablation_phi.pdf", dpi=300, transparent=True)
plt.show()

# %% [markdown]
# ## 8. Significance of every metric against the Costa & Kwon benchmark
#
# Reviewer 2 asks for the statistical significance of the reported measures
# relative to the benchmark, not only the Sharpe ratio.

# %%
def significance_vs(base, phi):
    """Paired tests of every metric against a reference arm."""
    rows = {}
    for a in ARMS:
        if a == base:
            continue
        for m in metrics[a].columns:
            x, y = metrics[a].xs(phi)[m], metrics[base].xs(phi)[m]
            t, p = stats.ttest_rel(x, y)
            rows[(LABELS[a], m)] = {
                "Mean": x.mean(),
                "Delta": (x - y).mean(),
                "t-stat": t,
                "p-value": p,
            }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.names = ["model", "metric"]
    return out


for phi in PHI_LIST:
    print(f"\n{'='*72}\nAll metrics vs. Costa & Kwon, phi = {phi}\n{'='*72}")
    print(significance_vs("ck_uni", phi).round(4).to_string())

# %% [markdown]
# ## 9. Diagnosing the mixture objective
#
# The mixture step is the one result that needs care in writing up, for two
# reasons.
#
# First, it is judged here on a criterion it does not optimise. Optimising over
# the full mixture maximises expected exponential utility, not the Sharpe ratio,
# so the two metrics need not agree — and they do not.
#
# Second, its effect on the certainty equivalent has a positive median and a
# negative mean at higher risk aversion: it helps on roughly two thirds of paths
# and loses heavily on a minority. A mean-based test therefore rejects in the
# opposite direction to a sign test, and reporting either alone would misdescribe
# the result.

# %%
diag = {}
for phi in PHI_LIST:
    d_sr = metrics["rwls_mixture"].xs(phi)["Sharpe Ratio"] - metrics["rwls_mvo"].xs(phi)["Sharpe Ratio"]
    d_ce = metrics["rwls_mixture"].xs(phi)["Certainty Equivalent"] - metrics["rwls_mvo"].xs(phi)["Certainty Equivalent"]
    for name, d in [("Sharpe Ratio", d_sr), ("Certainty Equivalent", d_ce)]:
        diag[(phi, name)] = {
            "Mean": d.mean(),
            "Median": d.median(),
            "Frac. > 0": (d > 0).mean(),
            "p (sign)": stats.binomtest((d > 0).sum(), len(d), 0.5).pvalue,
            "5th pct": d.quantile(0.05),
            "95th pct": d.quantile(0.95),
        }

diag = pd.DataFrame.from_dict(diag, orient="index")
diag.index.names = ["phi", "metric"]
print(diag.round(4).to_string())

# %%
# Where the mixture objective moves the portfolio: it trades along the frontier
# rather than shifting it outward.
frontier = pd.DataFrame(
    {
        arm: metrics[arm][["Ann. Excess Return", "Ann. Std. Deviation"]]
        .groupby("phi")
        .mean()
        .stack()
        for arm in ["rwls_mvo", "rwls_mixture"]
    }
)
print("\n" + frontier.round(4).to_string())

# %% [markdown]
# ## 10. Value of knowing the regimes, and what changed since submission
#
# With the covariance error corrected, `model_oracle` and `model_estimated` are
# the same estimator differing *only* in their information set: the former is
# handed the true parameters of the data-generating process, the latter fits the
# regime model at each decision date on the observations available up to it.
#
# The gap between them is therefore interpretable, and it is the decomposition
# Reviewer 3 asks for: how much of the outperformance comes from *identifying*
# regimes as against *projecting* them into asset space. A small gap says the
# framework is robust to having to estimate its own regimes; a large one says the
# reported performance leans on regime identification that a real investor would
# not achieve.
#
# `results/model.pkl` is retained as the archived output of the original
# pipeline. It is not a comparison arm -- it carries the look-ahead in the regime
# probabilities and the inflated covariances -- but the difference against it is
# what has to be accounted for in the response to referees, since the revised
# table will not reproduce the submitted numbers.

# %%
def load_arm(name):
    """Metrics for an arm held outside the incremental path, if it has been run."""
    wts_arm = pd.read_pickle(f"{DataConstants.WDIR.value}/results/{name}.pkl")
    paths = sorted(set(wts_arm.index.get_level_values("iteration").unique()) & set(common))
    kept = wts_arm[wts_arm.index.get_level_values("iteration").isin(paths)]
    return path_metrics(kept, sec_rt, PHI_LIST), paths


panel = {}
for name in ["model_oracle", "model_estimated", "model"]:
    try:
        m, paths = load_arm(name)
        panel[name] = m["Sharpe Ratio"].groupby("phi").mean()
        print(f"{name:<18} loaded on {len(paths)} paths")
    except FileNotFoundError:
        print(f"{name:<18} not available")

panel = pd.DataFrame(panel)

if {"model_oracle", "model_estimated"} <= set(panel.columns):
    panel["value of known regimes"] = panel["model_oracle"] - panel["model_estimated"]
if "model" in panel.columns and "model_estimated" in panel.columns:
    panel["vs. submitted draft"] = panel["model_estimated"] - panel["model"]

print()
print(panel.round(4).to_string())
