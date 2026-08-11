"""
Every decision that could reasonably have gone the other way lives here,
so it is visible in one place instead of buried in the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

RAW_CSV = DATA / "inspections_raw.csv"
MODEL_CSV = DATA / "model_df.csv"
SCORED_CSV = DATA / "scored_test.csv"
METRICS_JSON = DATA / "metrics.json"
LOGIT_PKL = MODELS / "logit.joblib"

# --- source ---------------------------------------------------------------
SODA_URL = "https://data.cityofnewyork.us/resource/43nn-pn8j.csv"
PAGE_SIZE = 50_000

# --- population -----------------------------------------------------------
# Routine unannounced inspections only. Re-inspections are selected on the
# outcome of the inspection before them; pre-permit visits have no history.
INSPECTION_TYPE = "Cycle Inspection / Initial Inspection"

# The dataset keeps a rolling ~3-year window. Everything before 2022 is a
# scattered handful of rows (~2,800 across 12 years vs 288K in 2022+).
MIN_YEAR = 2022

# --- label ----------------------------------------------------------------
# DOHMH's own A-grade cutoff. On Cycle/Initial inspections the grade column
# is unusable as a label: passing earns an A on the spot, failing goes
# ungraded to a re-inspection, so ~40% nulls ARE the failures. Score is not
# missing-not-at-random the way grade is, so the label is built from score.
A_GRADE_MAX_SCORE = 13

# --- time split -----------------------------------------------------------
BURN_IN_END = 2023      # 2022-2023 exist only to give later rows a history
MODEL_START_YEAR = 2024  # first year a row can be trained or tested on
TRAIN_END = 2025         # train on 2024-2025
TEST_YEAR = 2026         # score 2026

# An establishment needs at least one prior inspection for history features
# to mean anything.
MIN_PRIOR_N = 1

# --- features -------------------------------------------------------------
NUM_FEATURES = [
    "prior_n",
    "prior_fail_rate",
    "prior_mean_score",
    "prior_max_score",
    "prior_mean_viol",
    "days_since_last",
]
CAT_FEATURES = ["boro", "cuisine"]

# Cuisine is the strongest single block of coefficients in the model, which
# means the model is partly learning "this kind of restaurant fails." In an
# enforcement setting that is a targeting decision, not just a feature.
# train.py fits the model both ways and reports what dropping it costs.
USE_CUISINE = False
