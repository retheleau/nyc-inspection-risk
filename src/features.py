"""
Turn the raw violation-level CSV into one row per inspection with history
features computed only from inspections that happened BEFORE it.

    python src/features.py

The leakage rule for this project: a feature is allowed only if DOHMH could
have computed it the morning of the inspection. Everything named prior_* is
shifted one inspection back within the establishment before any aggregation.
"""
import pandas as pd

from config import (
    A_GRADE_MAX_SCORE,
    INSPECTION_TYPE,
    MIN_PRIOR_N,
    MIN_YEAR,
    MODEL_CSV,
    MODEL_START_YEAR,
    RAW_CSV,
    TEST_YEAR,
)


def load_raw(path=RAW_CSV):
    raw = pd.read_csv(path, low_memory=False)
    raw["inspection_date"] = pd.to_datetime(raw.inspection_date, errors="coerce")
    raw = raw[raw.inspection_date.notna()]
    return raw[raw.inspection_date.dt.year >= MIN_YEAR]


def to_inspection_level(raw):
    """Collapse violation rows to one row per (establishment, inspection date)."""
    cyc = raw[raw.inspection_type == INSPECTION_TYPE]

    insp = (
        cyc.groupby(["camis", "inspection_date"])
        .agg(
            dba=("dba", "first"),
            boro=("boro", "first"),
            cuisine=("cuisine_description", "first"),
            zipcode=("zipcode", "first"),
            action=("action", "first"),
            score=("score", "first"),
            n_violations=("violation_code", "count"),
            n_critical=("critical_flag", lambda s: (s == "Critical").sum()),
        )
        .reset_index()
        .sort_values(["camis", "inspection_date"])
        .reset_index(drop=True)
    )

    # A scored inspection is the unit of analysis; a row with no score has no
    # label. On this population that drops ~0 rows, but the rule is explicit.
    insp = insp[insp.score.notna()].reset_index(drop=True)

    # DOHMH writes "0" for records with no borough assigned.
    insp = insp[insp.boro.astype(str) != "0"].reset_index(drop=True)

    insp["year"] = insp.inspection_date.dt.year
    insp["failed_a"] = (insp.score > A_GRADE_MAX_SCORE).astype(int)
    return insp


def add_history_features(insp):
    """Expanding history per establishment, shifted so the current row is excluded."""
    insp = insp.sort_values(["camis", "inspection_date"]).reset_index(drop=True)
    g = insp.groupby("camis")

    insp["prior_n"] = g.cumcount()
    insp["prior_fail_rate"] = g.failed_a.transform(lambda s: s.shift(1).expanding().mean())
    insp["prior_mean_score"] = g.score.transform(lambda s: s.shift(1).expanding().mean())
    insp["prior_max_score"] = g.score.transform(lambda s: s.shift(1).expanding().max())
    insp["prior_mean_viol"] = g.n_violations.transform(lambda s: s.shift(1).expanding().mean())

    # Known the morning of the inspection: how long since the last visit.
    insp["days_since_last"] = g.inspection_date.diff().dt.days
    return insp


def build_model_frame(insp):
    """Keep only rows that are both modelable and inside the train/test window."""
    keep = (insp.year >= MODEL_START_YEAR) & (insp.year <= TEST_YEAR)
    model_df = insp[keep & (insp.prior_n >= MIN_PRIOR_N)].copy()
    return model_df.reset_index(drop=True)


def main():
    raw = load_raw()
    insp = to_inspection_level(raw)
    insp = add_history_features(insp)
    model_df = build_model_frame(insp)
    model_df.to_csv(MODEL_CSV, index=False)

    print(f"inspections (all years, {INSPECTION_TYPE}): {len(insp):,}")
    print(f"establishments: {insp.camis.nunique():,}\n")
    print("fail rate by year (all inspections):")
    print(
        insp.groupby("year")
        .agg(inspections=("camis", "size"), fail_rate=("failed_a", "mean"))
        .round(3)
        .to_string()
    )
    print(f"\nmodeling rows (prior_n >= {MIN_PRIOR_N}): {len(model_df):,}")
    print(f"establishments: {model_df.camis.nunique():,}")
    print(f"\nsaved -> {MODEL_CSV}")


if __name__ == "__main__":
    main()
