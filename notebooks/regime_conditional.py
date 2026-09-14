"""Regime-conditional performance and the quality of the regime forecast.

Paired script for notebooks/regime_conditional.ipynb.
"""

# %% [markdown]
# # Regime-conditional results
#
# Reviewer 3 asks for two things this addresses. First, that "Table 2 does not
# report regime conditional metrics". Second, that it is "worth adding more
# characterizations to the forecasted regime to understand where the
# outperformance comes from".
#
# Both require knowing which regime actually governed each period. The simulation
# draws that state but the original code discarded it; it has been recovered by
# replaying the generating stream and verified against the cached returns, so the
# classifications below are the true states rather than inferred ones.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regimeaware.constants import DataConstants, SimulationParameters
from regimeaware.core.exhibits import write_tabular
from regimeaware.core.performance import portfolio_returns, sharpe_ratio
from regimeaware.core.simdata import load_simulation

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.sans-serif": ["CMU Serif"]}
)

PHI_LIST = [10, 25, 50]

# The two halves of the mechanism table share a width, and it is the full
# measure: Panel B's natural width exceeds four fifths of it, and tabular* cannot
# set a table narrower than its content -- the rules would be drawn at the
# declared width with the numbers running past them.
#
# The seam between the panels is a rule of the same weight as the outer ones, so
# the pair reads as a single table rather than two stacked under one caption.
PANEL_WIDTH = True

# State indices as fitted, ordered by market risk premium. Regime 2 is the
# short-lived state with a sharply negative momentum premium.
STATES = {0: "Bull", 1: "Reversal", 2: "Bear"}

ARMS = {
    "model_estimated": "RWLS",
    "baseline": "Regime-agnostic",
    "rolling_ols": "Rolling OLS (36m)",
    "equalweighted": r"$1/N$",
    "ck_uni": r"Costa \& Kwon",
}

PROPOSED, BENCHMARK = "model_estimated", "baseline"

# %% [markdown]
# ## 1. Load the true states and the realised returns
#
# A weight chosen at the close of period $t$ earns the return of $t+1$, so each
# period of performance is attributed to the regime governing $t+1$ -- the state
# the portfolio was exposed to, not the one it was formed in.

# %%
states = pd.read_pickle(f"{DataConstants.WDIR.value}/data/sim/states.pkl")
sec_rt, _ = load_simulation(
    SimulationParameters.TRIALS.value,
    SimulationParameters.NUM_STOCKS.value,
    DataConstants.WDIR.value,
)

wts = {}
for arm in ARMS:
    try:
        wts[arm] = pd.read_pickle(f"{DataConstants.WDIR.value}/results/{arm}.pkl")
    except FileNotFoundError:
        print(f"{arm:<18} not available -- skipped")

common = sorted(
    set.intersection(
        *[set(w.index.get_level_values("iteration").unique()) for w in wts.values()]
    )
)
print(f"paths available for every arm: {len(common)}")

# %%
def realised(arm, phi):
    """Returns indexed by (path, period), aligned to the governing regime."""
    w_phi = wts[arm].xs(phi, level="phi")
    out = {i: portfolio_returns(w_phi.xs(i, level="iteration"), sec_rt[i])
           for i in common}
    r = pd.DataFrame(out).T
    r.index.name = "iteration"
    return r


regime_of = states.loc[common]

# %% [markdown]
# ## 2. Performance within each regime
#
# Pooled across paths, because a single five-year path contains too few months of
# the short-lived Reversal state to support a per-path statistic -- roughly five
# months on average.
#
# The Sharpe ratio here is annualised **arithmetically**, unlike the unconditional
# tables, which compound. The reason is that these returns are not a time series:
# they are months drawn from a thousand different paths and grouped by the state
# that governed them, so there is no sequence to compound along. Reporting them
# with the compounding definition would be meaningless rather than merely
# different.

# %%
rows = {}
for arm in wts:
    for phi in PHI_LIST:
        r = realised(arm, phi)
        aligned = regime_of.reindex(columns=r.columns).loc[r.index]

        for s, name in STATES.items():
            mask = aligned == s
            vals = r.where(mask).stack().dropna()
            mean, vol = vals.mean() * 12, vals.std() * np.sqrt(12)
            rows[(ARMS[arm], phi, name)] = {
                "Months": len(vals),
                "Mean return": mean,
                "Volatility": vol,
                "Sharpe (arith.)": mean / vol if vol else np.nan,
                "5th pct month": vals.quantile(0.05),
                "Worst month": vals.min(),
            }

conditional = pd.DataFrame(rows).T
conditional.index.names = ["model", "phi", "regime"]
print(conditional.xs(10, level="phi").round(4).to_string())

# %% [markdown]
# ## 3. Where the outperformance is earned
#
# The advantage over the benchmark, by the regime that governed each period. This
# is the characterisation Reviewer 3 asks for: whether the framework earns its
# keep by avoiding losses in stressed states, by positioning better in calm ones,
# or evenly across both.
#
# Both the return difference and the volatility ratio are reported, because they
# do not tell the same story. Expressing each regime's contribution as a *share*
# of the total would be unstable -- the total return difference is close to zero
# at low risk aversion, so shares of it explode -- and would in any case describe
# only the return dimension, when the framework's advantage is largely a matter of
# variance.

# %%
decomp = {}
for phi in PHI_LIST:
    rp, rb = realised(PROPOSED, phi), realised(BENCHMARK, phi)
    aligned = regime_of.reindex(columns=rp.columns).loc[rp.index]
    diff = rp - rb

    for s, name in STATES.items():
        mask = aligned == s
        adv = diff.where(mask).stack().dropna()
        p_vals = rp.where(mask).stack().dropna()
        b_vals = rb.where(mask).stack().dropna()
        share = len(adv) / diff.stack().notna().sum()

        decomp[(phi, name)] = {
            "Share of months": share,
            "Return advantage (ann.)": adv.mean() * 12,
            "Contribution to total (ann.)": adv.mean() * share * 12,
            "Volatility ratio": p_vals.std() / b_vals.std(),
            "5th pct improvement": p_vals.quantile(0.05) - b_vals.quantile(0.05),
        }

decomposition = pd.DataFrame(decomp).T
decomposition.index.names = ["phi", "regime"]
print(decomposition.round(4).to_string())

# %% [markdown]
# ## 4. How good is the regime forecast?
#
# The forecast $\hat\gamma_{t+1}$ is what the optimisation actually consumes, so
# its accuracy bounds what the framework can deliver. It is compared against the
# realised state, and against a naive forecast that ignores the current state and
# always predicts the ergodic distribution -- which is the relevant benchmark,
# since beating it is the whole point of conditioning on a regime.

# %%
try:
    forecast = pd.read_pickle(f"{DataConstants.WDIR.value}/results/forecast_estimated.pkl")
except FileNotFoundError:
    forecast = None
    print("run routines/forecasts.py first")

# %%
if forecast is not None:
    fc = forecast[forecast.index.get_level_values("iteration").isin(common)]

    # The forecast made at t is a statement about the regime governing t+1
    truth = pd.Series(
        {(i, t): states.loc[i, t + 1]
         for i, t in fc.index if t + 1 in states.columns},
        name="realised",
    )
    truth.index = pd.MultiIndex.from_tuples(truth.index, names=["iteration", "period"])
    fc = fc.loc[truth.index]

    predicted = fc.values.argmax(axis=1)
    actual = truth.values

    hit = (predicted == actual).mean()
    ergodic = np.bincount(actual, minlength=3) / len(actual)

    onehot = np.zeros_like(fc.values)
    onehot[np.arange(len(actual)), actual] = 1.0
    brier = ((fc.values - onehot) ** 2).sum(axis=1).mean()
    brier_naive = ((ergodic[None, :] - onehot) ** 2).sum(axis=1).mean()

    print(f"periods scored              : {len(actual):,}")
    print(f"hit rate (modal forecast)   : {hit:.3f}")
    print(f"hit rate, always-modal state: {ergodic.max():.3f}")
    print(f"Brier score, forecast       : {brier:.4f}")
    print(f"Brier score, ergodic naive  : {brier_naive:.4f}")
    print(f"Brier skill score           : {1 - brier / brier_naive:.4f}")

# %%
if forecast is not None:
    confusion = pd.crosstab(
        pd.Series(actual, name="realised").map(STATES),
        pd.Series(predicted, name="forecast").map(STATES),
        normalize="index",
    )
    print("rows: realised state, columns: forecast state (row-normalised)")
    print(confusion.round(3).to_string())

    print("\nmean forecast probability assigned to the state that then occurred:")
    print(
        pd.Series(
            {STATES[s]: fc.values[actual == s, s].mean() for s in STATES}
        ).round(3).to_string()
    )

# %% [markdown]
# ## 5. The figure
#
# Performance by governing regime, alongside where the advantage over the
# benchmark accumulates.

# %%
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

ax = axes[0]
width, names = 0.26, list(STATES.values())
x = np.arange(len(names))
for k, arm in enumerate([PROPOSED, BENCHMARK, "equalweighted"]):
    if arm not in wts:
        continue
    vals = [conditional.loc[(ARMS[arm], 10, n), "Sharpe (arith.)"] for n in names]
    ax.bar(x + (k - 1) * width, vals, width, label=ARMS[arm],
           color=["0.3", "0.6", "0.85"][k], edgecolor="k", lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Sharpe Ratio")
ax.set_title(r"Performance by governing regime ($\varphi=10$)", fontsize=9)
ax.axhline(0, color="k", lw=0.8)
ax.grid(ls="--", alpha=0.5, axis="y", zorder=-25)
ax.tick_params(bottom=False, left=False)
ax.legend(fontsize=6)

ax = axes[1]
for k, phi in enumerate(PHI_LIST):
    vals = [decomposition.loc[(phi, n), "Volatility ratio"] for n in names]
    ax.bar(x + (k - 1) * width, vals, width, label=rf"$\varphi={phi}$",
           color=["0.3", "0.6", "0.85"][k], edgecolor="k", lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Volatility vs. benchmark")
ax.set_title("Risk reduction by regime", fontsize=9)
ax.axhline(1, color="k", lw=0.8, ls="--")
ax.axhline(0, color="k", lw=0.8)
ax.grid(ls="--", alpha=0.5, axis="y", zorder=-25)
ax.tick_params(bottom=False, left=False)
ax.legend(fontsize=6)

plt.tight_layout()
plt.savefig(f"{DataConstants.WDIR.value}/img/regime_conditional.pdf", dpi=300,
            transparent=True, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 6. Table 5 — regime-conditional performance and forecast quality
#
# Panel A reports what the framework earns in each regime against the benchmark.
# Panel B reports how well the regime is identified and predicted, which is what
# makes Panel A interpretable: the framework conditions on the regime in force
# rather than betting on the one to come, and the two panels together show why.

# %%
if forecast is not None:
    # Metrics down the rows and regimes across the columns, panelled by risk
    # aversion, matching the layout of the other tables.
    panel_a = {}
    for phi in PHI_LIST:
        for metric in ["Share of months", "Return advantage (ann.)",
                       "Volatility ratio", "5th pct improvement"]:
            panel_a[(phi, metric)] = {
                name: decomposition.loc[(phi, name), metric]
                for name in STATES.values()
            }

    table5a = pd.DataFrame(panel_a).T
    table5a.index.names = ["phi", "metric"]
    table5a = table5a[list(STATES.values())]

    write_tabular(
        table5a,
        f"{DataConstants.WDIR.value}/tables/regime_conditional.tex",
        panel_level="phi",
        full_width=PANEL_WIDTH,
        title="Panel A: Performance by the regime governing each period",
        bottom_rule="\\midrule[\\heavyrulewidth]",
        notes=["Advantage of the proposed framework over the regime-agnostic "
               "benchmark, by the regime governing each period."],
    )
    print(table5a.round(4).to_string())

# %%
if forecast is not None:
    # Panel B: how well the regime is identified, and how well it is predicted.
    diagnostics = {}
    for label, probs_file, lag in [
        ("Nowcast, known parameters", "filtered_oracle", 0),
        ("Nowcast, estimated", "filtered_estimated", 0),
        ("One-step forecast, known parameters", "forecast_oracle", 1),
        ("One-step forecast, estimated", "forecast_estimated", 1),
    ]:
        try:
            d = pd.read_pickle(f"{DataConstants.WDIR.value}/results/{probs_file}.pkl")
        except FileNotFoundError:
            continue
        tr = pd.Series({(i, t): states.loc[i, t + lag]
                        for i, t in d.index if t + lag in states.columns})
        tr.index = pd.MultiIndex.from_tuples(tr.index, names=["iteration", "period"])
        f = d.loc[tr.index]
        act = tr.values
        erg = np.bincount(act, minlength=f.shape[1]) / len(act)
        oh = np.zeros_like(f.values)
        oh[np.arange(len(act)), act] = 1.0
        brier = ((f.values - oh) ** 2).sum(axis=1).mean()
        naive = ((erg[None, :] - oh) ** 2).sum(axis=1).mean()

        diagnostics[label] = {
            "Hit rate": (f.values.argmax(axis=1) == act).mean(),
            "Hit rate, constant forecast": erg.max(),
            "Brier score": brier,
            "Brier skill score": 1 - brier / naive,
        }

    # Transposed for the paper. With the four scenarios as columns their
    # headings run to thirty characters apiece and the table cannot fit the text
    # width; as row labels they cost nothing.
    panel_b = pd.DataFrame(diagnostics).T
    panel_b.columns = ["Hit rate", "Constant forecast", "Brier score",
                       "Brier skill"]
    write_tabular(
        panel_b,
        f"{DataConstants.WDIR.value}/tables/forecast_quality.tex",
        formats={c: "num3" for c in panel_b.columns},
        format_axis="columns",
        full_width=PANEL_WIDTH,
        title="Panel B: Quality of the regime probabilities",
        top_rule=False,
        notes=["Scored against the realised regime over all simulated paths.",
               "The constant forecast always predicts the most frequent state; "
               "it is reported as the hit rate that forecast attains."],
    )
    print(panel_b.round(4).to_string())
