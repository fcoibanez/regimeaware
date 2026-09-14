"""Economic characterisation of the identified market regimes.

Paired script for notebooks/regime_characterization.ipynb.
"""

# %% [markdown]
# # What the identified regimes correspond to
#
# Reviewer 3 asks for "a time series plot highlighting or comparing how the
# forecasted regime differ from NBER recession and documented Daniel & Moskowitz
# (2016) momentum-crash dates".
#
# The regimes are estimated from factor returns alone, with no macroeconomic input
# of any kind. Whether they line up with independently dated recessions and
# momentum crashes is therefore a genuine out-of-model check rather than a
# restatement of how they were constructed.
#
# Momentum crashes are identified **empirically**, as the worst months of the UMD
# factor over the sample, rather than by transcribing a list. The episodes that
# emerge are then compared with those documented by Daniel and Moskowitz (2016).
# Selecting them from the data avoids any suggestion that the dates were chosen to
# flatter the regime classification.

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from regimeaware.core.exhibits import write_tabular
from regimeaware.constants import (
    HISTORY_END_DT,
    HISTORY_START_DT,
    DataConstants,
    Factors,
)

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.sans-serif": ["CMU Serif"]}
)

# %% [markdown]
# ## 1. Data and fitted model
#
# The sample is pinned to the period the paper reports. This matters: the cached
# factor file now extends beyond it, and `routines/distributions.py` does not
# truncate, so an unpinned re-run would silently fit a longer sample than the one
# described in the text.

# %%
mdl = pd.read_pickle(f"{DataConstants.WDIR.value}/data/dist/mdl_hmm.pkl")

factors = Factors._member_names_
ff = pd.read_pickle(f"{DataConstants.WDIR.value}/data/ff.pkl")
ff = ff.loc[HISTORY_START_DT:HISTORY_END_DT]
X = ff[factors]

print(f"sample: {X.index.min():%Y-%m} to {X.index.max():%Y-%m}  ({len(X)} months)")

probs = pd.DataFrame(
    mdl.predict_proba(X.values), index=X.index, columns=range(mdl.n_components)
)

# Order the states by their market risk premium so the labels are stable and
# economically interpretable rather than an artefact of the EM starting point.
order = np.argsort(-mdl.means_[:, factors.index("mktrf")])
umd_by_state = mdl.means_[:, factors.index("umd")]

NAMES = {}
for rank, state in enumerate(order):
    if umd_by_state[state] < -0.01:
        NAMES[state] = "Reversal"
    elif mdl.means_[state, factors.index("mktrf")] < 0:
        NAMES[state] = "Bear"
    else:
        NAMES[state] = "Bull"

probs.columns = [NAMES[c] for c in probs.columns]
probs = probs[["Bull", "Reversal", "Bear"]]
modal = probs.idxmax(axis=1)

print("\nstate labelling:", NAMES)
print("\nunconditional occupancy (%):")
print((modal.value_counts(normalize=True) * 100).round(1).to_string())

# %% [markdown]
# ## 2. Independently dated events
#
# Recession dates are read from FRED rather than transcribed, so the comparison
# cannot drift from the official chronology and needs no maintenance when the
# committee dates a new cycle. `USREC` marks the month **following** the peak
# through the trough inclusive: under the NBER convention the peak month is the
# last month of the expansion, not the first month of the contraction.
#
# Momentum crashes are the worst 1% of UMD months in the sample.

# %%
def load_usrec(index):
    """NBER-based recession indicator, aligned to the factor sample."""
    cache = f"{DataConstants.WDIR.value}/data/usrec.pkl"
    try:
        import pandas_datareader.data as web

        usrec = web.DataReader(
            "USREC", "fred", index.min() - pd.DateOffset(years=1), index.max()
        )["USREC"]
        usrec.to_pickle(cache)
    except Exception as exc:  # offline: fall back on the last retrieved copy
        if not os.path.exists(cache):
            raise
        print(f"FRED unavailable ({type(exc).__name__}), using cached USREC")
        usrec = pd.read_pickle(cache)

    usrec.index = usrec.index.to_period("M")
    return usrec.reindex(index.to_period("M")).fillna(0).astype(bool).set_axis(index)


recession = load_usrec(X.index)

crash_cutoff = ff["umd"].quantile(0.01)
crashes = ff.index[ff["umd"] <= crash_cutoff]

print(f"recession months: {recession.sum()} of {len(recession)} "
      f"({recession.mean():.1%})")
print(f"\nmomentum crashes (UMD <= {crash_cutoff:.1%}):")
print((ff.loc[crashes, "umd"] * 100).round(1).to_string())

# %% [markdown]
# ## 3. Do the regimes line up?
#
# The regimes carry no macroeconomic information, so any correspondence is
# informative about what the factor structure alone reveals.

# %%
rows = {}
for name in ["Bull", "Reversal", "Bear"]:
    p = probs[name]
    rows[name] = {
        "Mean prob., recession": p[recession].mean(),
        "Mean prob., expansion": p[~recession].mean(),
        "Ratio": p[recession].mean() / p[~recession].mean(),
        "Mean prob., crash months": p.reindex(crashes).mean(),
        "Modal share, recession": (modal[recession] == name).mean(),
        "Modal share, crash months": (modal.reindex(crashes) == name).mean(),
    }

alignment = pd.DataFrame(rows).T
print(alignment.round(3).to_string())

# %% [markdown]
# ### How much of this is mechanical?
#
# The two correspondences are not equally strong as evidence, and the difference
# should be stated rather than left for a referee to point out.
#
# The *Reversal* regime carries a large negative mean on UMD and by far the widest
# UMD dispersion of the three states. A month in which momentum loses a third of
# its value is therefore far more likely under that state than under either other,
# so classifying the worst UMD months into it is partly definitional. What is not
# definitional is that a distinct state with this signature is selected at all --
# the number of states was chosen by BIC on the joint six-factor distribution,
# with no momentum criterion imposed -- that it turns out to be short-lived and to
# act as a bridge out of stress, and that the *same* state is elevated during NBER
# recessions, which the momentum factor alone would not tell you.
#
# The *Bear* correspondence with NBER dates carries more weight, because nothing
# in the estimation sees macroeconomic data. The regimes are fitted to factor
# returns only, and the business cycle dates are assigned by a committee working
# from output, employment and income series.

# %% [markdown]
# ## 4. Does the regime lead or lag the NBER dating?
#
# The NBER dates cycles retrospectively, often with a delay of many months, while
# the regime probabilities are available in real time. Comparing the two at
# various leads shows whether the factor structure identifies stress before it is
# officially recognised.

# %%
lead_rows = {}
for lead in range(-6, 7):
    aligned = recession.shift(-lead).fillna(False).astype(bool)
    lead_rows[lead] = {
        "P(Bear) in recession": probs["Bear"][aligned].mean(),
        "P(Bear) elsewhere": probs["Bear"][~aligned].mean(),
    }

leads = pd.DataFrame(lead_rows).T
leads["Ratio"] = leads.iloc[:, 0] / leads.iloc[:, 1]
leads.index.name = "regime leads NBER by (months)"
print(leads.round(3).to_string())
print(f"\nstrongest correspondence at a lead of {leads['Ratio'].idxmax()} months")

# %% [markdown]
# ## 5. Regime persistence
#
# The transition matrix implies an expected duration of $1/(1-\pi_{ss})$ months in
# each state. Comparing that with the realised runs of the modal regime checks
# that the fitted dynamics describe the sample rather than merely fitting it.

# %%
runs = (modal != modal.shift()).cumsum()
realised = modal.groupby(runs).agg(["first", "size"])

dur = {}
for state, name in NAMES.items():
    dur[name] = {
        "Implied duration (months)": 1 / (1 - mdl.transmat_[state, state]),
        "Realised mean run": realised.loc[realised["first"] == name, "size"].mean(),
        "Realised longest run": realised.loc[realised["first"] == name, "size"].max(),
        "Number of episodes": (realised["first"] == name).sum(),
    }

print(pd.DataFrame(dur).T[
    ["Implied duration (months)", "Realised mean run", "Realised longest run",
     "Number of episodes"]
].round(2).to_string())

# %% [markdown]
# ## 6. Momentum crashes
#
# Eight months scattered across sixty years are not a time series, and drawing
# them as one flatters the evidence: on a calendar axis each occupies a fraction
# of a millimetre, and averaged in event time a smooth curve with a standard
# error band implies more structure than eight observations support. Listed with
# the probabilities assigned to them they make the point directly.
#
# The market column is the part that is not definitional. The *Reversal* state is
# characterised by a large negative mean on UMD, so assigning extreme negative
# UMD months to it is close to circular; nothing in its definition refers to the
# market factor, yet every one of these months carries a strong positive market
# return. That is the configuration the momentum-crash literature describes, a
# sharp rebound in which past losers surge.

# %%
crash_table = pd.DataFrame(
    {
        f"{d:%Y-%m}": {
            "$UMD$": ff.loc[d, "umd"],
            "$Mkt-Rf$": ff.loc[d, "mktrf"],
            "P(Bull)": probs.loc[d, "Bull"],
            "P(Reversal)": probs.loc[d, "Reversal"],
            "P(Bear)": probs.loc[d, "Bear"],
        }
        for d in crashes
    }
).T
crash_table.index.name = None

# Without the unconditional row a reader cannot tell whether a Reversal
# probability of 0.99 is remarkable.
crash_table.loc["Unconditional mean"] = {
    "$UMD$": ff["umd"].mean(),
    "$Mkt-Rf$": ff["mktrf"].mean(),
    "P(Bull)": probs["Bull"].mean(),
    "P(Reversal)": probs["Reversal"].mean(),
    "P(Bear)": probs["Bear"].mean(),
}

print(crash_table.round(4).to_string())

write_tabular(
    crash_table,
    f"{DataConstants.WDIR.value}/tables/momentum_crashes.tex",
    formats={"$UMD$": "pct2", "$Mkt-Rf$": "pct2", "P(Bull)": "num3",
             "P(Reversal)": "num3", "P(Bear)": "num3"},
    format_axis="columns",
    notes=["Months in the first percentile of the momentum factor over the "
           "estimation sample."],
)

# %% [markdown]
# ## 7. The figure
#
# The Bear probability on a nought-to-one axis with recessions shaded behind it,
# which is how this comparison is conventionally drawn. An earlier version filled
# the area under the probability, which turns a spiky monthly series into a solid
# block; a thin line leaves it legible.
#
# A six-month centred moving average is plotted. Averaging a state whose implied
# duration is 2.2 months might be expected to flatten it, and it does not, because
# the episodes arrive in clusters: the peak falls only from 1.00 to 0.97 and the
# ratio of the mean probability inside recessions to outside them moves from 2.86
# to 2.83. Every statistic quoted in the text is computed on the unsmoothed
# series, so the window affects the picture and nothing else.
#
# The sample is split across two rows. On one row it occupies a single text
# width, about nine months to a tenth of an inch, too dense to resolve episodes.
#
# The figure should be read honestly. The state is elevated through every dated
# contraction, but it also fires outside them, exceeding one half in sixty-three
# separate episodes against eighty-five recession months in total. What it
# identifies is equity market stress, of which recessions are one source among
# others.

# %%
SMOOTH_MONTHS = 6
SPLIT = pd.Timestamp("1994-06-30")


def recession_spans(mask):
    """Contiguous (start, end) spans of a boolean monthly mask."""
    out, i, m = [], 0, np.asarray(mask)
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            out.append((mask.index[i], mask.index[j]))
            i = j + 1
        else:
            i += 1
    return out


bear = probs["Bear"].rolling(SMOOTH_MONTHS, center=True).mean()
fig, axes = plt.subplots(2, 1, figsize=(7, 3.2))

for ax, (lo, hi) in zip(axes, [(bear.index[0], SPLIT), (SPLIT, bear.index[-1])]):
    for start, end in recession_spans(recession):
        if end >= lo and start <= hi:
            ax.axvspan(max(start, lo), min(end, hi), color="0.86", lw=0, zorder=0)

    window = (bear.index >= lo) & (bear.index <= hi)
    ax.plot(bear.index[window], bear[window], color="0.05", lw=0.9, zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel(r"$P(\mathrm{Bear})$", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(bottom=False, left=False, labelsize=8.5)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(f"{DataConstants.WDIR.value}/img/regime_timeline.pdf", dpi=300,
            transparent=True, bbox_inches="tight")
plt.show()
