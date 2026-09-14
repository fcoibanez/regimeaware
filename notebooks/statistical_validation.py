"""Statistical validation of the reported performance measures.

Paired script for notebooks/statistical_validation.ipynb.
"""

# %% [markdown]
# # Statistical validation
#
# Addresses Reviewer 2's four requests: certainty equivalent returns, the value of
# the approach against canonical naive benchmarks, breakeven transaction costs,
# and the Ledoit and Wolf (2008) test of equal Sharpe ratios — together with their
# closing point that "standard practise is also to test the statistical
# significance of the other measures reported relative to the baseline".
#
# **Two kinds of uncertainty, kept apart.** *Estimation* uncertainty is what an
# investor faces: from one realised five-year history, how precisely is a Sharpe
# ratio pinned down? That is what the Ledoit–Wolf test measures, and at $T=60$ it
# is severe. *Monte Carlo* uncertainty is a property of the experiment: across
# 1000 simulated paths, how reliably does one method beat another? Because every
# arm sees the identical paths this is testable pairwise, which is far more
# powerful. Reporting only the first understates what the simulation establishes;
# reporting only the second overstates what an investor could verify from a single
# history. Both are given below.
#
# Tables destined for the manuscript are written to `tables/` as LaTeX, so no
# number is transcribed by hand.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from regimeaware.constants import DataConstants, SimulationParameters
from regimeaware.core.exhibits import write_tabular
from regimeaware.core.inference import (
    detection_by_horizon,
    ledoit_wolf_path,
    ledoit_wolf_summary,
    paired_comparison,
    rejection_summary,
    significance_table,
)
from regimeaware.core.performance import (
    breakeven_cost,
    net_returns,
    net_sharpe_curve,
    path_metrics,
    portfolio_returns,
)
from regimeaware.core.simdata import load_simulation

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.sans-serif": ["CMU Serif"]}
)

TABLES = f"{DataConstants.WDIR.value}/tables"
PHI_LIST = [10, 25, 50]

LABELS = {
    "model_estimated": "RWLS",
    "model_oracle": "RWLS (known regimes)",
    "baseline": "Regime-agnostic",
    "rolling_ols": "Rolling OLS (36m)",
    "equalweighted": r"$1/N$",
    "global_min_var": "Min. variance",
    "ck_uni_2s": r"Costa \& Kwon",
    "ck_uni": r"Costa \& Kwon (3 states)",
}

# Six benchmarks do not fit across the text block at full width, so the table
# uses abbreviated headings and the caption spells them out. Figures keep the
# long labels, where there is room for them.
SHORT = {
    "model_estimated": "RWLS",
    "model_oracle": "RWLS (oracle)",
    "baseline": "Agnostic",
    "rolling_ols": "Roll. OLS",
    "equalweighted": r"$1/N$",
    "global_min_var": "Min. Var.",
    "ck_uni_2s": r"C\&K",
    "ck_uni": r"C\&K (3s)",
}

# The comparison set that goes into the manuscript. The three-state Costa & Kwon
# arm belongs to the decomposition and the known-regime variant to the appendix,
# not to the headline table.
MAIN_ARMS = [
    "model_estimated",
    "baseline",
    "rolling_ols",
    "equalweighted",
    "global_min_var",
    "ck_uni_2s",
]

PROPOSED = "model_estimated"

# The benchmark the submitted draft used, and so the reference for the
# significance tests Reviewer 2 asks to see alongside the performance measures.
BENCHMARK = "baseline"

# %% [markdown]
# ## 1. Load every arm

# %%
sec_rt, _ = load_simulation(
    SimulationParameters.TRIALS.value,
    SimulationParameters.NUM_STOCKS.value,
    DataConstants.WDIR.value,
)

wts = {}
for arm in LABELS:
    try:
        wts[arm] = pd.read_pickle(f"{DataConstants.WDIR.value}/results/{arm}.pkl")
    except FileNotFoundError:
        print(f"{arm:<22} not available -- skipped")

common = sorted(
    set.intersection(
        *[set(w.index.get_level_values("iteration").unique()) for w in wts.values()]
    )
)
print(f"\npaths available for every arm: {len(common)}")

metrics = {}
for arm, w in wts.items():
    keep = w[w.index.get_level_values("iteration").isin(common)]
    metrics[arm] = path_metrics(keep, sec_rt, PHI_LIST)
    print(f"{arm:<22} done")

available_main = [a for a in MAIN_ARMS if a in metrics]

# %% [markdown]
# ## 2. Table 2 — performance across benchmarks
#
# One panel per level of risk aversion, benchmarks across the columns. Certainty
# equivalent returns are included as Reviewer 2 requests, computed under the
# exponential utility the portfolios actually optimise rather than its
# mean–variance approximation.

# %%
ROWS = [
    "Ann. Excess Return",
    "Ann. Std. Deviation",
    "Skewness",
    "Kurtosis",
    "Sharpe Ratio",
    "Certainty Equivalent",
    "Max. Drawdown",
    "Value-at-Risk (95%)",
    "Expected Shortfall",
    "Portfolio Turnover",
    "Avg. Num. Constituents",
    "Effective Num. Bets",
]

panels = {}
for phi in PHI_LIST:
    for row in ROWS:
        panels[(phi, row)] = {
            SHORT[a]: metrics[a].xs(phi)[row].mean() for a in available_main
        }

table2 = pd.DataFrame(panels).T
table2.index.names = ["phi", "metric"]
table2 = table2[[SHORT[a] for a in available_main]]

write_tabular(
    table2,
    f"{TABLES}/table2_performance.tex",
    panel_level="phi",
    # Six benchmarks leave the table narrower than the text block; stretching the
    # columns to fill it keeps the numbers from bunching together.
    full_width=True,
)
print(table2.round(4).to_string())

# %% [markdown]
# ## 3. Table 3 — significance, and performance net of trading costs
#
# Reviewer 2 asks for the Ledoit–Wolf test and for the significance of the other
# measures, both relative to the benchmark. Costs are charged at the realised
# one-way spread on every unit of turnover and the test applied to the resulting
# net series, so the significance reported is significance *after* trading.
#
# Charging costs after the fact is deliberately conservative: an optimiser aware
# of trading costs would have traded less than these portfolios did, so these are
# a lower bound on what the framework delivers under frictions.

# %%
try:
    tc = pd.read_pickle(f"{DataConstants.WDIR.value}/data/dist/tcost.pkl")
    realised_one_way = tc["one_way"]
    print(f"realised one-way cost from CRSP quotes: {realised_one_way*100:.4f}%")
except FileNotFoundError:
    tc, realised_one_way = None, None
    print("run notebooks/tcost_estimation.ipynb first to obtain the spread estimate")

# %%
def net_panel(cost):
    """Gross and net Sharpe, with the Ledoit–Wolf test on the net series."""
    rows = {}
    for phi in PHI_LIST:
        b_phi = wts[BENCHMARK].xs(phi, level="phi")
        for arm in available_main:
            w_phi = wts[arm].xs(phi, level="phi")

            gross, net, lw_p = [], [], []
            for i in common:
                w_i = w_phi.xs(i, level="iteration")
                gross.append(net_sharpe_curve(w_i, sec_rt[i], 0.0)[0])
                net.append(net_sharpe_curve(w_i, sec_rt[i], cost)[0])
                if arm != BENCHMARK:
                    lw_p.append(
                        ledoit_wolf_path(
                            net_returns(w_i, sec_rt[i], cost),
                            net_returns(b_phi.xs(i, level="iteration"), sec_rt[i], cost),
                        )["p-value"]
                    )

            paired = paired_comparison(
                metrics[arm].xs(phi)["Sharpe Ratio"],
                metrics[BENCHMARK].xs(phi)["Sharpe Ratio"],
            )

            rows[(phi, LABELS[arm])] = {
                "Sharpe (gross)": np.nanmean(gross),
                "Sharpe (net)": np.nanmean(net),
                "Cost drag": np.nanmean(gross) - np.nanmean(net),
                "Delta": paired["Delta"],
                "t-stat": paired["t-stat"],
                "p (paired t)": paired["p (paired t)"],
                "LW reject 5\\%": np.mean(np.array(lw_p) < 0.05) if lw_p else np.nan,
            }

    out = pd.DataFrame(rows).T
    out.index.names = ["phi", "model"]
    return out


if realised_one_way is not None:
    table3 = net_panel(realised_one_way)

    # Reshape to metrics down the rows and models across the columns, panelled by
    # risk aversion, which is the layout the manuscript uses.
    table3_tex = table3.stack().unstack("model")
    table3_tex.index.names = ["phi", "metric"]
    table3_tex = table3_tex.reindex(
        index=pd.MultiIndex.from_product(
            [PHI_LIST, ["Sharpe (gross)", "Sharpe (net)", "Cost drag", "Delta",
                        "t-stat", "p (paired t)", "LW reject 5\\%"]],
            names=["phi", "metric"],
        ),
        columns=[LABELS[a] for a in available_main],
    )

    write_tabular(
        table3_tex,
        f"{TABLES}/table3_significance.tex",
        panel_level="phi",
        notes=[
            f"Costs charged at {realised_one_way*100:.4f}% per unit of two-sided turnover.",
            "Ledoit-Wolf tests are computed on net returns against the "
            f"{LABELS[BENCHMARK]} benchmark.",
        ],
    )
    print(table3.round(4).to_string())

# %% [markdown]
# ## 4. Ledoit and Wolf (2008), path by path
#
# At $T=60$ the test has little power on any one path, so the informative
# quantity is the *distribution* of outcomes: on what fraction of five-year
# histories would an investor have been able to reject equality. Reported
# alongside the mean Sharpe difference, so a large economic gap that a short
# sample cannot certify is visible as exactly that.

# %%
def arm_returns(arm, phi):
    """Realised returns of an arm, indexed by path."""
    w_phi = wts[arm].xs(phi, level="phi")
    return pd.DataFrame(
        {i: portfolio_returns(w_phi.xs(i, level="iteration"), sec_rt[i]) for i in common}
    ).T


lw_rows = {}
for reference in [BENCHMARK, "equalweighted", "ck_uni_2s"]:
    if reference not in wts:
        continue
    for phi in PHI_LIST:
        lw = ledoit_wolf_summary(
            arm_returns(PROPOSED, phi), arm_returns(reference, phi), common
        )
        lw_rows[(LABELS[reference], phi)] = rejection_summary(lw)

lw_table = pd.DataFrame(lw_rows).T
lw_table.index.names = ["vs.", "phi"]
print(lw_table.round(4).to_string())

# %% [markdown]
# ## 4b. How long a history the test needs
#
# The result above is easy to misread. A test that does not reject may be
# reporting that there is no difference, or merely that sixty months is too short
# to resolve one, and Reviewer 2's question -- whether the added complexity is
# worthwhile -- turns on which of the two it is.
#
# Applying the same test over progressively longer histories separates them. If
# the difference is real, the rejection rate rises with the horizon; if it is not,
# the rate stays near the nominal level however long the sample. Longer histories
# are assembled by concatenating independent paths, which measures the power of
# the test rather than constituting a literal long backtest.

# %%
horizon = detection_by_horizon(
    arm_returns(PROPOSED, 10), arm_returns(BENCHMARK, 10), common,
    horizons=[1, 2, 4, 8, 16],
)
write_tabular(
    horizon[["Years", "Detection rate"]],
    f"{TABLES}/table6_detection_horizon.tex",
    formats={"Years": "num1", "Detection rate": "pct1"},
    format_axis="columns",
    index_header="Months",
    notes=["Share of simulated histories on which the Ledoit-Wolf test rejects "
           "equality of Sharpe ratios at the 5% level."],
)
print(horizon.round(3).to_string())

# %%
fig, ax = plt.subplots(figsize=(3.75, 3.75))
ax.plot(horizon["Years"], horizon["Detection rate"], c="k", marker="s",
        markerfacecolor="white", markeredgewidth=1, lw=1.25)
ax.axhline(0.05, color="0.5", lw=0.9, ls=":")
ax.annotate("nominal size", (horizon["Years"].max(), 0.05),
            textcoords="offset points", xytext=(-4, 5), ha="right", fontsize=7,
            color="0.35")
ax.set_xscale("log")
ax.set_xticks(horizon["Years"].tolist())
ax.set_xticklabels([f"{y:.0f}" for y in horizon["Years"]])
ax.set_xlabel("Length of the evaluation history (years)")
ax.set_ylabel("Share of histories rejecting")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.set_ylim(0, 1)
ax.grid(ls="--", alpha=0.5, zorder=-25)
ax.tick_params(bottom=False, left=False)
plt.tight_layout()
plt.savefig(f"{DataConstants.WDIR.value}/img/detection_horizon.pdf", dpi=300,
            transparent=True)
plt.show()

# %% [markdown]
# ## 5. Significance of every measure
#
# The full set Reviewer 2 asks for, against the regime-agnostic benchmark and
# against the $1/N$ rule that DeMiguel, Garlappi and Uppal (2009) show is hard to
# beat. Detailed, and destined for the appendix rather than the main text.

# %%
for reference in [BENCHMARK, "equalweighted"]:
    if reference not in metrics:
        continue
    for phi in PHI_LIST:
        cols = ["Mean (treatment)", "Delta", "t-stat", "p (paired t)", "Frac. > 0"]
        tbl = significance_table(metrics, reference, phi).loc[PROPOSED, cols]
        print(f"\n{'='*78}\n{LABELS[PROPOSED]} vs {LABELS[reference]}, phi={phi}\n{'='*78}")
        print(tbl.round(4).to_string())

# %% [markdown]
# ## 6. Breakeven transaction costs
#
# Three outcomes are possible on a given path and all three are informative: a
# finite breakeven, where the proposed method trades more and costs erode its
# advantage; *never reversed*, where it trades less so costs penalise the
# benchmark harder; and *no gross lead*, where there is nothing for costs to
# erase. Turnover for both arms is shown so which case applies is visible.

# %%
rows = {}
for reference in [BENCHMARK, "rolling_ols", "equalweighted", "ck_uni_2s"]:
    if reference not in wts:
        continue
    for phi in PHI_LIST:
        w_a = wts[PROPOSED].xs(phi, level="phi")
        w_b = wts[reference].xs(phi, level="phi")
        costs = pd.Series(
            {i: breakeven_cost(w_a.xs(i, level="iteration"),
                               w_b.xs(i, level="iteration"), sec_rt[i]) for i in common}
        )
        finite = costs.replace(np.inf, np.nan).dropna()
        finite = finite[finite > 0]

        rows[(LABELS[reference], phi)] = {
            "Turnover (proposed)": metrics[PROPOSED].xs(phi)["Portfolio Turnover"].mean(),
            "Turnover (benchmark)": metrics[reference].xs(phi)["Portfolio Turnover"].mean(),
            "Median breakeven (%)": finite.median() * 100 if len(finite) else np.nan,
            "25th pct (%)": finite.quantile(0.25) * 100 if len(finite) else np.nan,
            "Frac. never reversed": (costs == np.inf).mean(),
            "Frac. no gross lead": (costs == 0).mean(),
        }

breakeven = pd.DataFrame(rows).T
breakeven.index.names = ["vs.", "phi"]
if realised_one_way is not None:
    breakeven["Multiple of realised cost"] = (
        breakeven["Median breakeven (%)"] / (realised_one_way * 100)
    )
print(breakeven.round(4).to_string())

# %% [markdown]
# ## 7. Figure — Sharpe ratio as a function of trading cost
#
# Estimates of the effective spread differ by an order of magnitude depending on
# the sample period and the weighting scheme, so rather than defend one number
# the comparison is shown across the whole range. Where a pair of lines crosses is
# that pair's breakeven, so this contains the table above and the ranking at
# realistic costs at once.

# %%
cost_grid = np.linspace(0, 0.03, 61)

curves = {}
for arm in available_main:
    for phi in PHI_LIST:
        w_phi = wts[arm].xs(phi, level="phi")
        paths = np.vstack(
            [net_sharpe_curve(w_phi.xs(i, level="iteration"), sec_rt[i], cost_grid)
             for i in common]
        )
        curves[(arm, phi)] = np.nanmean(paths, axis=0)

# %%
STYLES = {
    # The proposed framework and the benchmark it is measured against, in black.
    "model_estimated": dict(ls="-", lw=1.8, color="0.0"),
    "baseline": dict(ls="--", lw=1.3, color="0.0"),
    # Costa and Kwon solid, so it reads as a single continuous line rather than
    # one more dash pattern among several.
    "ck_uni_2s": dict(ls="-", lw=1.3, color="0.50"),
    "rolling_ols": dict(ls="-.", lw=1.2, color="0.50"),
    "equalweighted": dict(ls=(0, (1, 1.4)), lw=1.2, color="0.68"),
    "global_min_var": dict(ls=(0, (5, 2)), lw=1.1, color="0.68"),
}

fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=True)
for ax, phi in zip(axes, PHI_LIST):
    for arm in available_main:
        ax.plot(cost_grid * 100, curves[(arm, phi)],
                label=LABELS[arm] if phi == PHI_LIST[0] else "",
                **STYLES.get(arm, dict(ls="-", lw=1, color="0.0")))
    if realised_one_way is not None:
        ax.axvline(realised_one_way * 100, color="0.35", lw=1)
        ax.axvspan(tc["spread_post_dec_vw"] / 2 * 100, realised_one_way * 100,
                   color="0.85", zorder=-30)
    ax.set_title(rf"$\varphi={phi}$")
    ax.grid(ls="--", alpha=0.5, zorder=-25)
    ax.tick_params(bottom=False, left=False)

axes[0].set_ylabel("Ann. Sharpe Ratio (net)")
fig.supxlabel(r"One-way cost per unit turnover (\%)", fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.16),
           ncol=3, frameon=True, fontsize=7)
plt.tight_layout()
plt.savefig(f"{DataConstants.WDIR.value}/img/cost_sensitivity.pdf", dpi=300,
            transparent=True, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Figure — distribution of outcomes across benchmarks
#
# The stochastic dominance exhibit, extended from the two arms of the submitted
# draft to the full comparison set. Sharpe ratios use the same definition as the
# tables.

# %%
fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharex=True, sharey=True)
for ax, phi in zip(axes, PHI_LIST):
    for arm in available_main:
        sns.kdeplot(
            data=metrics[arm].xs(phi)["Sharpe Ratio"], cumulative=True, ax=ax,
            zorder=25, label=LABELS[arm] if phi == PHI_LIST[0] else "",
            **STYLES.get(arm, dict(ls="-", lw=1, color="0.0")),
        )
    ax.set_title(rf"$\varphi={phi}$")
    ax.set_xlabel("Sharpe Ratio")
    ax.grid(ls="--", alpha=0.5, zorder=-25)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1)
    ax.tick_params(bottom=False, left=False)

# The panels show a cumulative distribution, not a density; seaborn labels the
# axis "Density" regardless.
axes[0].set_ylabel("Cumulative share of paths")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.16),
           ncol=3, frameon=True, fontsize=7)
plt.tight_layout()
plt.savefig(f"{DataConstants.WDIR.value}/img/benchmark_ecdf.pdf", dpi=300,
            transparent=True, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Sensitivity to the rolling-window benchmark
#
# Reviewer 3 proposes rolling-window OLS as a tighter control. The window length
# is a free parameter of that benchmark, so the comparison is repeated across 24,
# 36 and 60 months rather than resting on one choice. Appendix material.

# %%
windows = {24: "rolling_ols_24m", 36: "rolling_ols", 60: "rolling_ols_60m"}

rows = {}
for months, arm in windows.items():
    try:
        w_arm = pd.read_pickle(f"{DataConstants.WDIR.value}/results/{arm}.pkl")
    except FileNotFoundError:
        print(f"{arm} not available -- skipped")
        continue

    keep = w_arm[w_arm.index.get_level_values("iteration").isin(common)]
    m_arm = path_metrics(keep, sec_rt, PHI_LIST)
    for phi in PHI_LIST:
        rows[(months, phi)] = paired_comparison(
            metrics[PROPOSED].xs(phi)["Sharpe Ratio"], m_arm.xs(phi)["Sharpe Ratio"]
        )

sensitivity = pd.DataFrame(rows).T[
    ["Mean (treatment)", "Mean (control)", "Delta", "Std. Err.", "t-stat",
     "p (paired t)", "Frac. > 0"]
]
sensitivity.index.names = ["window (months)", "phi"]
print(sensitivity.round(4).to_string())

# %%
if not sensitivity.empty:
    fig, ax = plt.subplots(figsize=(3.75, 3.75))
    for phi, marker in zip(PHI_LIST, ["s", "o", "^"]):
        sub = sensitivity.xs(phi, level="phi").sort_index()
        ax.errorbar(sub.index, sub["Delta"], yerr=1.96 * sub["Std. Err."],
                    marker=marker, c="k", markerfacecolor="white", markeredgewidth=1,
                    capsize=3, lw=1.2, label=rf"$\varphi={phi}$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Rolling window (months)")
    ax.set_ylabel(r"$\Delta$ Sharpe vs. rolling OLS")
    ax.set_xticks(list(windows))
    ax.grid(ls="--", alpha=0.5, zorder=-25)
    ax.tick_params(bottom=False, left=False)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{DataConstants.WDIR.value}/img/rolling_window_sensitivity.pdf",
                dpi=300, transparent=True)
    plt.show()
