"""
Inspection dispatch — ranked list with an inspector-capacity control.

    streamlit run app.py

Design note: the visual language borrows from the DOHMH letter-grade placard
taped in every restaurant window in New York — placard blue on cool paper,
condensed signage type, and no decoration that isn't carrying information.
"""
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import METRICS_JSON, SCORED_CSV

st.set_page_config(
    page_title="Inspection dispatch",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- tokens
INK = "#0C1D2E"
PLACARD = "#0B5EA8"
PLACARD_DEEP = "#063E70"
PAPER = "#EEF1F4"
CARD = "#FFFFFF"
RULE = "#CBD5DF"
FLAG = "#B8371F"
MUTED = "#5D7185"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo+Narrow:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.stApp { background: __PAPER__; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

html, body, [class*="css"], .stMarkdown, p, li {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  color: __INK__;
}

.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }

/* ---------- masthead ---------- */
.eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem; font-weight: 500; letter-spacing: 0.16em;
  text-transform: uppercase; color: __MUTED__;
  padding-bottom: 0.5rem;
}
.masthead {
  font-family: 'Archivo Narrow', sans-serif;
  font-weight: 700; font-size: clamp(2.4rem, 5.5vw, 3.6rem);
  line-height: 0.95; letter-spacing: -0.015em; color: __INK__;
  margin: 0;
}
.standfirst {
  font-size: 0.98rem; color: __MUTED__; max-width: 62ch;
  margin: 0.7rem 0 0 0; line-height: 1.5;
}
.hrule { border: 0; border-top: 2px solid __INK__; margin: 1.4rem 0 1.6rem 0; }

/* ---------- capacity readout ---------- */
.capwrap {
  display: flex; align-items: baseline; gap: 1.1rem;
  border-left: 5px solid __PLACARD__; padding-left: 1.1rem; margin-bottom: 0.4rem;
}
.capnum {
  font-family: 'Archivo Narrow', sans-serif; font-weight: 700;
  font-size: clamp(3rem, 9vw, 5.2rem); line-height: 0.85;
  color: __PLACARD_DEEP__; font-variant-numeric: tabular-nums;
}
.caplabel {
  font-size: 0.92rem; color: __MUTED__; line-height: 1.35; padding-bottom: 0.35rem;
}
.caplabel b { color: __INK__; font-weight: 600; }

/* ---------- stat row ---------- */
.stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 1px;
  background: __RULE__; border: 1px solid __RULE__; margin: 0.4rem 0 1.6rem 0;
}
.stat { background: __CARD__; padding: 0.95rem 1.05rem; }
.stat .k {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.63rem; font-weight: 500;
  letter-spacing: 0.13em; text-transform: uppercase; color: __MUTED__;
}
.stat .v {
  font-family: 'Archivo Narrow', sans-serif; font-weight: 700; font-size: 1.95rem;
  line-height: 1.15; color: __INK__; font-variant-numeric: tabular-nums; margin-top: 0.2rem;
}
.stat .s { font-size: 0.76rem; color: __MUTED__; margin-top: 0.1rem; }
.stat.accent .v { color: __PLACARD_DEEP__; }

/* ---------- section headings ---------- */
.sect {
  font-family: 'Archivo Narrow', sans-serif; font-weight: 700;
  font-size: 1.28rem; letter-spacing: -0.005em; color: __INK__;
  margin: 2.2rem 0 0.25rem 0;
}
.note { font-size: 0.84rem; color: __MUTED__; max-width: 66ch; line-height: 1.5; margin: 0; }

/* ---------- list ---------- */
.tbl {
  width: 100%; border-collapse: collapse; margin-top: 0.9rem;
  font-size: 0.87rem; background: __CARD__; border: 1px solid __RULE__;
}
.tbl th {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; font-weight: 500;
  letter-spacing: 0.12em; text-transform: uppercase; color: __MUTED__;
  text-align: left; padding: 0.6rem 0.75rem; border-bottom: 1px solid __RULE__;
  white-space: nowrap;
}
.tbl td { padding: 0.52rem 0.75rem; border-bottom: 1px solid #EDF1F5; vertical-align: middle; }
.tbl tr:last-child td { border-bottom: 0; }
.tbl .rk { font-family: 'IBM Plex Mono', monospace; color: __MUTED__; font-size: 0.8rem; }
.tbl .nm { font-weight: 600; }
.tbl .num {
  font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums;
  text-align: right; white-space: nowrap;
}
.bartrack { display: block; height: 5px; background: #E1E8EF; width: 88px; margin-top: 3px; }
.bar { display: block; height: 5px; background: __PLACARD__; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: __INK__; }
section[data-testid="stSidebar"] * { color: #DDE5EC; }
.sbhead {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; letter-spacing: 0.14em;
  text-transform: uppercase; color: #8FA6BA; padding-bottom: 0.35rem;
}

/* ---------- controls ---------- */
.colophon {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: __MUTED__;
  border-top: 1px solid __RULE__; margin-top: 2.6rem; padding-top: 0.9rem; line-height: 1.8;
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
"""
for k, v in {
    "__INK__": INK,
    "__PLACARD_DEEP__": PLACARD_DEEP,
    "__PLACARD__": PLACARD,
    "__PAPER__": PAPER,
    "__CARD__": CARD,
    "__RULE__": RULE,
    "__MUTED__": MUTED,
}.items():
    CSS = CSS.replace(k, v)
st.markdown(CSS, unsafe_allow_html=True)


def placard_theme():
    return {
        "config": {
            "background": CARD,
            "view": {"stroke": "transparent"},
            "font": "IBM Plex Sans",
            "axis": {
                "labelFont": "IBM Plex Mono",
                "labelFontSize": 10,
                "labelColor": MUTED,
                "titleFont": "IBM Plex Mono",
                "titleFontSize": 10,
                "titleColor": MUTED,
                "titleFontWeight": 500,
                "domainColor": RULE,
                "tickColor": RULE,
                "grid": False,
            },
            "legend": {
                "labelFont": "IBM Plex Sans",
                "labelFontSize": 11,
                "labelColor": INK,
                "titleFont": "IBM Plex Mono",
                "titleFontSize": 9,
                "titleColor": MUTED,
                "orient": "top",
                "direction": "horizontal",
            },
        }
    }


try:
    alt.theme.register("placard", enable=True)(placard_theme)
except AttributeError:  # altair < 5.5
    alt.themes.register("placard", placard_theme)
    alt.themes.enable("placard")


@st.cache_data
def load():
    scored = pd.read_csv(SCORED_CSV, parse_dates=["inspection_date"])
    metrics = json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else {}
    return scored, metrics


def fmt(n):
    return f"{n:,}"


try:
    scored, metrics = load()
except FileNotFoundError:
    st.markdown(
        "<div class='eyebrow'>No data</div>"
        "<h1 class='masthead'>Nothing to dispatch yet</h1>"
        "<p class='standfirst'>Run <code>python src/train.py</code> to score the "
        "test year, then reload this page.</p>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("<div class='sbhead'>Borough</div>", unsafe_allow_html=True)
    boros = sorted(scored.boro.dropna().astype(str).unique())
    picked = st.multiselect("Borough", boros, default=boros, label_visibility="collapsed")

    st.markdown("<div class='sbhead' style='padding-top:1.6rem'>Model</div>",
                unsafe_allow_html=True)
    st.markdown(
        "Logistic regression on establishment history"
        + (" and cuisine." if metrics.get("used_cuisine") else ", borough only.")
        + f"  \nTrained on {metrics.get('n_train', 0):,} inspections, "
        f"scored {metrics.get('n_test', 0):,}."
    )

view = scored[scored.boro.astype(str).isin(picked)] if picked else scored

# ---------------------------------------------------------------- masthead
st.markdown(
    "<div class='eyebrow'>NYC DOHMH &middot; cycle inspections &middot; scored 2026</div>"
    "<h1 class='masthead'>Inspection dispatch</h1>"
    "<p class='standfirst'>There are more establishments due than there are "
    "inspector-days to visit them. Set how many visits you can make, and this "
    "ranks which ones to make them to &mdash; using only what was on file the "
    "morning of the inspection.</p>"
    "<hr class='hrule'>",
    unsafe_allow_html=True,
)

if view.empty:
    st.markdown(
        "<div class='sect'>No establishments selected</div>"
        "<p class='note'>Pick at least one borough in the sidebar to see a list.</p>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------- control
capacity = st.slider(
    "Inspections this period",
    min_value=min(100, len(view)),
    max_value=int(len(view)),
    value=int(min(2000, len(view))),
    step=100,
    label_visibility="collapsed",
)
capacity = int(min(capacity, len(view)))

ranked = view.sort_values("risk_score", ascending=False).reset_index(drop=True)
flagged = ranked.head(capacity)

base_rate = float(ranked.failed_a.mean())
total_fail = int(ranked.failed_a.sum())
caught = int(flagged.failed_a.sum())
hit_rate = float(flagged.failed_a.mean())
threshold = float(flagged.risk_score.min())
extra = caught - int(round(capacity * base_rate))

st.markdown(
    f"<div class='capwrap'><div class='capnum'>{fmt(capacity)}</div>"
    f"<div class='caplabel'>inspections this period<br>"
    f"out of <b>{fmt(len(ranked))}</b> establishments due</div></div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='stats'>"
    f"<div class='stat accent'><div class='k'>Failures caught</div>"
    f"<div class='v'>{fmt(caught)}</div>"
    f"<div class='s'>of {fmt(total_fail)} &middot; {caught / max(total_fail,1):.1%} of all failures</div></div>"
    f"<div class='stat'><div class='k'>Hit rate</div>"
    f"<div class='v'>{hit_rate:.1%}</div>"
    f"<div class='s'>precision &middot; base rate {base_rate:.1%}</div></div>"
    f"<div class='stat'><div class='k'>Lift over random</div>"
    f"<div class='v'>{hit_rate / max(base_rate, 1e-9):.2f}&times;</div>"
    f"<div class='s'>{extra:+,} more failures found</div></div>"
    f"<div class='stat'><div class='k'>Cut at risk score</div>"
    f"<div class='v'>{threshold:.3f}</div>"
    f"<div class='s'>everything above is dispatched</div></div>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- signature
st.markdown(
    "<div class='sect'>The cut line</div>"
    "<p class='note'>Every establishment due, ordered left to right by risk "
    "score. Each bar is a slice of the queue; its height is the share of that "
    "slice that actually failed. A working model puts the tall bars on the "
    "left. The red line is where your capacity cuts the queue.</p>",
    unsafe_allow_html=True,
)

N_BANDS = 110
bands = pd.DataFrame({"pos": np.arange(len(ranked)), "failed": ranked.failed_a.to_numpy()})
bands["band"] = np.minimum(bands.pos * N_BANDS // max(len(ranked), 1), N_BANDS - 1)
strip = (
    bands.groupby("band")
    .agg(fail_rate=("failed", "mean"), start=("pos", "min"), end=("pos", "max"))
    .reset_index()
)
strip["end"] = strip["end"] + 1
strip["side"] = np.where(strip.start < capacity, "Dispatched", "Not visited")

strip_chart = (
    alt.Chart(strip)
    .mark_bar()
    .encode(
        x=alt.X("start:Q", title="establishments ranked by risk score",
                scale=alt.Scale(domain=[0, len(ranked)], nice=False),
                axis=alt.Axis(format=",d", tickCount=6)),
        x2="end:Q",
        y=alt.Y("fail_rate:Q", title="failed",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=4)),
        color=alt.Color(
            "side:N",
            title=None,
            scale=alt.Scale(domain=["Dispatched", "Not visited"],
                            range=[PLACARD_DEEP, "#C3D0DC"]),
        ),
        tooltip=[
            alt.Tooltip("start:Q", title="rank from", format=",d"),
            alt.Tooltip("end:Q", title="rank to", format=",d"),
            alt.Tooltip("fail_rate:Q", title="failed", format=".0%"),
        ],
    )
    .properties(height=170)
)
cut = (
    alt.Chart(pd.DataFrame({"x": [capacity]}))
    .mark_rule(color=FLAG, strokeWidth=2.5)
    .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0, len(ranked)], nice=False)))
)
base_strip = (
    alt.Chart(pd.DataFrame({"y": [base_rate]}))
    .mark_rule(color=MUTED, strokeDash=[3, 3], strokeWidth=1)
    .encode(y="y:Q")
)
st.altair_chart(strip_chart + base_strip + cut, width='stretch')

# ---------------------------------------------------------------- curve
st.markdown(
    "<div class='sect'>What another inspector buys you</div>"
    "<p class='note'>Coverage climbs and hit rate falls as capacity grows. That "
    "trade-off is the whole decision, and a fixed 0.5 threshold hides it.</p>",
    unsafe_allow_html=True,
)

grid = np.unique(np.linspace(1, len(ranked), 60).astype(int))
cum = ranked.failed_a.cumsum().to_numpy()
curve = pd.DataFrame(
    {
        "capacity": grid,
        "Coverage of all failures": cum[grid - 1] / max(total_fail, 1),
        "Hit rate": cum[grid - 1] / grid,
    }
).melt("capacity", var_name="measure", value_name="value")

line = (
    alt.Chart(curve)
    .mark_line(strokeWidth=2.2)
    .encode(
        x=alt.X("capacity:Q", title="inspections performed",
                scale=alt.Scale(domain=[0, len(ranked)], nice=False),
                axis=alt.Axis(format=",d", tickCount=6)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5)),
        color=alt.Color("measure:N", title=None,
                        scale=alt.Scale(
                            domain=["Coverage of all failures", "Hit rate"],
                            range=[PLACARD, FLAG])),
        tooltip=[
            alt.Tooltip("capacity:Q", title="inspections", format=",d"),
            alt.Tooltip("measure:N", title=""),
            alt.Tooltip("value:Q", title="value", format=".1%"),
        ],
    )
    .properties(height=250)
)
base_line = (
    alt.Chart(pd.DataFrame({"y": [base_rate]}))
    .mark_rule(color=MUTED, strokeDash=[3, 3], strokeWidth=1)
    .encode(y="y:Q")
)
here = (
    alt.Chart(pd.DataFrame({"x": [capacity]}))
    .mark_rule(color=INK, strokeWidth=1.5, strokeDash=[2, 2])
    .encode(x="x:Q")
)
st.altair_chart(line + base_line + here, width='stretch')
st.markdown(
    f"<p class='note'>The dashed horizontal line is the base rate "
    f"({base_rate:.1%}) &mdash; where hit rate lands if you pick at random.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- list
SHOW = 25
st.markdown(
    f"<div class='sect'>Dispatch list</div>"
    f"<p class='note'>First {min(SHOW, capacity)} of {fmt(capacity)}. "
    f"Download below for the full list.</p>",
    unsafe_allow_html=True,
)

rows = []
for r in flagged.head(SHOW).itertuples():
    pfr = "&mdash;" if pd.isna(r.prior_fail_rate) else f"{r.prior_fail_rate:.0%}"
    pms = "&mdash;" if pd.isna(r.prior_mean_score) else f"{r.prior_mean_score:.0f}"
    width = max(1, int(round(float(r.risk_score) * 88)))
    name = str(r.dba).title() if pd.notna(r.dba) else "&mdash;"
    rows.append(
        f"<tr><td class='rk'>{r.rank}</td>"
        f"<td class='nm'>{name}</td>"
        f"<td>{r.boro}</td>"
        f"<td class='num'>{r.prior_n}</td>"
        f"<td class='num'>{pfr}</td>"
        f"<td class='num'>{pms}</td>"
        f"<td class='num'>{r.risk_score:.3f}"
        f"<span class='bartrack'><span class='bar' style='width:{width}px'></span></span>"
        f"</td></tr>"
    )

st.markdown(
    "<table class='tbl'><thead><tr>"
    "<th>Rank</th><th>Establishment</th><th>Borough</th>"
    "<th style='text-align:right'>Prior visits</th>"
    "<th style='text-align:right'>Prior fail rate</th>"
    "<th style='text-align:right'>Prior mean score</th>"
    "<th style='text-align:right'>Risk score</th>"
    "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
    unsafe_allow_html=True,
)

st.download_button(
    f"Download all {fmt(capacity)}",
    flagged[["rank", "camis", "dba", "boro", "zipcode", "risk_score",
             "prior_n", "prior_fail_rate", "prior_mean_score"]].to_csv(index=False),
    file_name=f"dispatch_top{capacity}.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------- colophon
auc = metrics.get("auc_with_cuisine")
auc_nc = metrics.get("auc_without_cuisine")
st.markdown(
    "<div class='colophon'>NYC Open Data &middot; DOHMH restaurant inspection "
    "results (43nn-pn8j)<br>"
    + (f"Test AUC {auc:.3f} &middot; base rate {base_rate:.1%} &middot; "
       f"ranks a queue, does not judge an establishment" if auc else "")
    + "</div>",
    unsafe_allow_html=True,
)

with st.expander("What this model can and cannot do"):
    if auc:
        cuisine_cost = abs(auc - auc_nc) if auc_nc else None
        cost_txt = (
            f"Removing it cost {cuisine_cost:.3f} AUC."
            if cuisine_cost
            else "The cost was measured before dropping it."
        )
        st.markdown(
            f"""
**It ranks, it doesn't judge.** Test AUC is {auc:.3f} — useful for ordering a
queue of thousands, not for deciding any single establishment's fate.

**History is thin.** The published dataset keeps a rolling three-year window,
so most establishments have only one or two prior inspections to learn from.

**The target is drifting.** Failure rates rise across the period
(0.365 → 0.380 → 0.405), so a model trained on earlier years under-predicts
later ones. It needs retraining on a schedule, not once.

**Cuisine was dropped on purpose.** Cuisine type carried the largest
coefficients in the first version — larger than any inspection-history
feature. Ranking enforcement partly by what food a restaurant serves is a
policy decision, not a modeling detail. {cost_txt}

**Inspecting changes the data.** Visiting the flagged establishments writes
their next history rows, which changes future predictions. A real deployment
needs a randomized holdout to stay honest about whether the model still works.
            """
        )
    else:
        st.markdown("Run `python src/train.py` to populate metrics.")
