"""Choosing the number of market regimes.

Paired script for notebooks/state_selection.ipynb.

Supersedes the BIC-only exercise in notebooks/states_determination.ipynb, which
writes the same figure file and should not be re-run.
"""

# %% [markdown]
# # How many regimes?
#
# The submitted draft selected three states on BIC alone, averaged across three
# equal-length subperiods, and described the minimum as consistent across them.
# Two defects in that exercise had to be repaired before it could be read.
#
# The models were fitted with the library's default cap of ten EM iterations
# rather than the thousand used everywhere else in this project, and from a
# single starting point. Both understate the likelihood of the richer
# specifications, which have more parameters still to move when EM is stopped
# early and more local optima to get trapped in. Correcting them changes the
# answer: on the full sample the criteria no longer select three states.
#
# What survives, and is the substance of the exercise, is the subperiod
# structure. Within samples of about twenty years, three states is selected by
# seven of the nine criterion-subperiod combinations, and none selects more than
# five. Over the full sixty-one years every criterion selects more.
#
# The grid stops at eight. Beyond that every criterion on every sample is
# several hundred units worse than its own best, and EM convergence becomes
# unreliable, so the additional candidates carry no information.
#
# That contrast is expected rather than contradictory. The penalty an information
# criterion charges grows with $\ln T$ while the attainable gain in log-likelihood
# grows roughly in proportion to $T$, so the number of states a criterion will
# support increases with sample length. Selection over six decades therefore
# cannot separate recurring regimes from slow structural change, and the
# subperiod design is the control for exactly that.
#
# This notebook reports three criteria on four samples and lets the disagreement
# show. Note that the simulation cannot settle the question: its data-generating
# process is calibrated at three states, so the state count is three there by
# construction.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from regimeaware.constants import (
    HISTORY_END_DT,
    HISTORY_START_DT,
    DataConstants,
    Factors,
    HMMParameters,
)
from regimeaware.core.exhibits import write_tabular

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.sans-serif": ["CMU Serif"]}
)

STATES = range(1, 9)
N_SUBPERIODS = 3
N_STARTS = 10

# %% [markdown]
# ## 1. Samples
#
# The full sample is included alongside the three subperiods because it is the
# sample on which the model used throughout the paper is estimated. Reporting
# only subperiods leaves the operative choice untested.

# %%
factors = Factors._member_names_
ff = pd.read_pickle(f"{DataConstants.WDIR.value}/data/ff.pkl")
X = ff.loc[HISTORY_START_DT:HISTORY_END_DT, factors]

idxs = X.index
size = len(idxs) // N_SUBPERIODS
samples = {"Full sample": idxs}
for n in range(1, N_SUBPERIODS + 1):
    idx = idxs[(n - 1) * size: n * size if n < N_SUBPERIODS else len(idxs)]
    samples[f"{idx.min():%Y}--{idx.max():%Y}"] = idx

for label, idx in samples.items():
    print(f"{label:<14} {idx.min():%Y-%m} to {idx.max():%Y-%m}  ({len(idx)} months)")

# %% [markdown]
# ## 2. Three criteria
#
# The three criteria conventional in this literature. All are of the form
# $-2\ln\mathcal{L} + \text{penalty}$ and are minimized, differing in what they
# charge for a parameter: AIC charges $2$, BIC charges $\ln T$, and HQIC charges
# $2\ln\ln T$. Only BIC is consistent for the order of a mixture; AIC and HQIC
# are not, and both are known to select generously.
#
# The number of free parameters is $M(M-1)$ transitions, $M-1$ initial
# probabilities, and $MK$ means and $MK$ variances for the diagonal emission
# covariance used throughout.

# %%
def fit_best(sub, n):
    """Best of several random starts, by likelihood.

    Three things matter here and none was present in the exercise this replaces.

    The iteration cap has to be raised from the library default of ten, which
    stops EM long before it converges and penalizes the richer models most,
    because they have the most parameters left to move. On the full sample that
    single omission moves the gap between two and three states from 1.6 to 80.

    EM on a mixture is multimodal, so a single start compares one model's global
    optimum against another's local one. At three states the fitted likelihood
    varies across starts by more than the differences between adjacent candidate
    models.

    And the initial-state distribution is held uniform rather than estimated,
    exactly as in ``core.estimation.build_hmm``. Estimated from a single sequence
    it is not identified: it collapses onto whichever state the path begins in,
    and on some starts diverges outright. Fitting the candidates differently from
    the way the paper fits its model would not be a comparison worth reporting.
    """
    best = None
    for seed in range(N_STARTS):
        mdl = GaussianHMM(
            n_components=n,
            covariance_type=HMMParameters.COV.value,
            n_iter=HMMParameters.ITER.value,
            random_state=seed,
            min_covar=HMMParameters.MINCOV.value,
            tol=HMMParameters.TOL.value,
            implementation=HMMParameters.IMPLEMENTATION.value,
            init_params="tmc",
            params="tmc",
        )
        mdl.startprob_ = np.full(n, 1 / n)
        try:
            mdl.fit(sub)
            ll = mdl.score(sub)
        except ValueError:
            continue
        if not np.isfinite(ll):
            continue
        if best is None or ll > best[0]:
            best = (ll, mdl, seed)
    return best


rows = {}
for label, idx in samples.items():
    sub = X.reindex(idx)
    T = len(sub)
    for n in STATES:
        found = fit_best(sub, n)
        if found is None:
            print(f"  no usable fit: {label}, M={n}")
            continue
        ll, mdl, seed = found

        # Counted here rather than taken from the library, which does not know
        # that the initial distribution is held fixed: M(M-1) transitions, and a
        # mean and a variance per factor per state.
        k = n * (n - 1) + 2 * n * len(factors)

        rows[(label, n)] = {
            "AIC": -2 * ll + 2 * k,
            "BIC": -2 * ll + k * np.log(T),
            "HQIC": -2 * ll + 2 * k * np.log(np.log(T)),
            "logL": ll,
            "best seed": seed,
        }

res = pd.DataFrame(rows).T
res.index.names = ["sample", "states"]
criteria = ["BIC", "HQIC", "AIC"]

# Cached so the exhibit can be redrawn without repeating three hundred fits.
res.to_pickle(f"{DataConstants.WDIR.value}/results/state_selection.pkl")

# %% [markdown]
# ## 3. What each criterion selects

# %%
table = {}
for label in samples:
    sub = res.xs(label, level="sample")
    table[label] = {c: int(sub[c].astype(float).idxmin()) for c in criteria}
    table[label][r"$\Delta$BIC, three vs two"] = (
        float(sub["BIC"][3]) - float(sub["BIC"][2])
    )

selection = pd.DataFrame(table).T
selection.index.name = None
print(selection.round(1).to_string())

write_tabular(
    selection,
    f"{DataConstants.WDIR.value}/tables/state_selection.tex",
    formats={c: "int" for c in criteria}
    | {r"$\Delta$BIC, three vs two": "num1"},
    format_axis="columns",
    notes=["A negative entry in the final column favours three states over two.",
           "Criteria are minimized; the number reported is the minimizing M."],
)

print("\nBIC, difference from each sample's own best")
gap = res["BIC"].astype(float).unstack("states")
print(gap.sub(gap.min(axis=1), axis=0).round(1).to_string())

# %% [markdown]
# ## 4. The figure
#
# Each series is plotted as its distance above its own minimum, because the full
# One panel per sample, with all three criteria inside it, plotted at their own
# level rather than as a distance from the best model. Subtracting the minimum
# looks appealing -- it puts the selected specification at zero -- but it inflates
# the scale until the ends of the grid sit several hundred units above the
# interesting region, which then has to be truncated or plotted on an unfamiliar
# axis. At their own level the variation across $M$ is a few percent, so every
# series fits its panel and nothing has to be explained away.
#
# What the figure has to convey is where each curve bottoms out. Magnitudes are
# in the table and in the text, which is where a reader looks for them.

# %%
STYLE = {
    "BIC": dict(ls="-", lw=1.6, marker="s", ms=3.6, mfc="white", mew=1.0),
    "HQIC": dict(ls=(0, (1.4, 1.4)), lw=1.2),
    "AIC": dict(ls=(0, (5, 1.6, 1, 1.6)), lw=1.2),
}

fig, axes = plt.subplots(2, 2, figsize=(7, 4.6), sharex=True)
for ax, sample in zip(axes.ravel(), samples):
    sub = res.xs(sample, level="sample")
    for crit in criteria:
        shade = "0.15" if crit == "BIC" else "0.45"
        y = sub[crit].astype(float)
        ax.plot(list(STATES), y.values, color=shade, label=crit, **STYLE[crit])
        # Where each criterion bottoms out, which is the whole of what the
        # figure has to convey.
        ax.plot([y.idxmin()], [y.min()], marker="v", ms=5, color=shade,
                mec="none", zorder=6)

    ax.set_title(sample, fontsize=9.5)
    ax.set_xticks(list(STATES))
    ax.grid(ls="--", alpha=0.45, lw=0.5)
    ax.tick_params(bottom=False, left=False, labelsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")

for ax in axes[:, 0]:
    ax.set_ylabel("Information criterion", fontsize=9)
fig.supxlabel("Number of market regimes", fontsize=9)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.10),
           ncol=4, frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(f"{DataConstants.WDIR.value}/img/state_calibration.pdf", dpi=300,
            transparent=True, bbox_inches="tight")
plt.show()
