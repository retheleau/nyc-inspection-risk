"""
Train and evaluate on a time-based split, then save the scored 2026 rows for
the app.

    python src/train.py

Order matters here: two baselines run before any model, so "did the model
earn its place" has an answer instead of a vibe.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CAT_FEATURES,
    LOGIT_PKL,
    METRICS_JSON,
    MODEL_CSV,
    NUM_FEATURES,
    SCORED_CSV,
    TEST_YEAR,
    TRAIN_END,
    USE_CUISINE,
)

CAPACITIES = [500, 1000, 2000, 3000, 5000]


def load_split(path=MODEL_CSV):
    df = pd.read_csv(path, parse_dates=["inspection_date"])
    train = df[df.year <= TRAIN_END].copy()
    test = df[df.year == TEST_YEAR].copy()
    return train, test


def make_pipeline(clf, cat_features):
    steps = [
        (
            "num",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                ]
            ),
            NUM_FEATURES,
        )
    ]
    if cat_features:
        steps.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "oh",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=50,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                cat_features,
            )
        )
    return Pipeline([("pre", ColumnTransformer(steps)), ("clf", clf)])


def score_row(name, y, pred, prob=None):
    row = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, prob) if prob is not None else None,
    }
    auc = f" auc={row['auc']:.3f}" if row["auc"] is not None else ""
    print(
        f"{name:28s} acc={row['accuracy']:.3f} "
        f"prec={row['precision']:.3f} rec={row['recall']:.3f}{auc}"
    )
    return row


def capacity_table(prob, y, capacities=CAPACITIES):
    """
    What actually gets used. The city does not inspect everyone with p > 0.5;
    it has N inspector-days and wants the N riskiest establishments.
    """
    y = np.asarray(y)
    base = y.mean()
    rows = []
    for cap in capacities:
        if cap > len(y):
            continue
        thr = float(np.quantile(prob, 1 - cap / len(y)))
        flag = prob >= thr
        caught = int(y[flag].sum())
        prec = float(y[flag].mean())
        rows.append(
            {
                "capacity": cap,
                "threshold": thr,
                "caught": caught,
                "total_failures": int(y.sum()),
                "catch_rate": caught / y.sum(),
                "precision": prec,
                "lift": prec / base,
            }
        )
    return rows


def main():
    train, test = load_split()
    y_tr, y_te = train.failed_a, test.failed_a
    print(f"train {len(train):,} ({TRAIN_END} and earlier)  "
          f"test {len(test):,} ({TEST_YEAR})")
    print(f"test base rate: {y_te.mean():.3f}\n")

    # ---------- baselines, before any model ----------
    print("--- baselines ---")
    rows = [score_row("always predict pass", y_te, np.zeros(len(test), dtype=int))]
    prior = test.prior_fail_rate.fillna(train.failed_a.mean())
    rows.append(
        score_row("own prior fail rate", y_te, (prior > 0.5).astype(int), prior)
    )

    # ---------- models ----------
    cat = CAT_FEATURES if USE_CUISINE else [c for c in CAT_FEATURES if c != "cuisine"]
    cols = NUM_FEATURES + cat
    print(f"\n--- models (categoricals: {cat or 'none'}) ---")

    logit = make_pipeline(LogisticRegression(max_iter=2000), cat)
    logit.fit(train[cols], y_tr)
    p_lr = logit.predict_proba(test[cols])[:, 1]
    rows.append(score_row("logistic regression", y_te, (p_lr > 0.5).astype(int), p_lr))

    hgb = make_pipeline(HistGradientBoostingClassifier(random_state=0), cat)
    hgb.fit(train[cols], y_tr)
    p_gb = hgb.predict_proba(test[cols])[:, 1]
    rows.append(score_row("gradient boosting", y_te, (p_gb > 0.5).astype(int), p_gb))

    # ---------- what cuisine is worth ----------
    auc_full = roc_auc_score(y_te, p_lr)
    if USE_CUISINE:
        # Cuisine is the largest coefficient block. Before shipping a model that
        # targets restaurants partly by cuisine type, price the alternative.
        no_cuisine_cols = NUM_FEATURES + ["boro"]
        logit_nc = make_pipeline(LogisticRegression(max_iter=2000), ["boro"])
        logit_nc.fit(train[no_cuisine_cols], y_tr)
        p_nc = logit_nc.predict_proba(test[no_cuisine_cols])[:, 1]
        auc_nc = roc_auc_score(y_te, p_nc)
        print(
            f"\nlogistic without cuisine   auc={auc_nc:.3f} "
            f"(cuisine is worth {auc_full - auc_nc:+.3f} AUC)"
        )
    else:
        # The shipped model already excludes cuisine; comparing it to itself
        # would print a meaningless +0.000.
        auc_nc = auc_full
        print("\ncuisine excluded from the shipped model (USE_CUISINE = False)")

    # ---------- coefficients ----------
    feat = logit.named_steps["pre"].get_feature_names_out()
    coef = pd.Series(logit.named_steps["clf"].coef_[0], index=feat)
    print("\n--- history coefficients (log-odds, standardized) ---")
    print(coef[[f for f in feat if f.startswith("num__")]].round(3).to_string())
    print("\n--- largest categorical coefficients ---")
    print(
        coef[[f for f in feat if f.startswith("cat__")]]
        .sort_values(key=abs, ascending=False)
        .head(8)
        .round(3)
        .to_string()
    )

    # ---------- capacity, not 0.5 ----------
    caps = capacity_table(p_lr, y_te)
    print("\n--- ranked inspection list under capacity ---")
    for c in caps:
        print(
            f"inspect top {c['capacity']:5d}: thr={c['threshold']:.3f}  "
            f"catches {c['caught']}/{c['total_failures']} "
            f"({c['catch_rate']:.1%})  precision={c['precision']:.3f}  "
            f"lift={c['lift']:.2f}x"
        )

    # ---------- artifacts ----------
    scored = test[
        ["camis", "dba", "boro", "cuisine", "zipcode", "inspection_date",
         "prior_n", "prior_fail_rate", "prior_mean_score", "failed_a"]
    ].copy()
    scored["risk_score"] = p_lr
    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    scored.to_csv(SCORED_CSV, index=False)

    joblib.dump(logit, LOGIT_PKL)
    METRICS_JSON.write_text(
        json.dumps(
            {
                "n_train": len(train),
                "n_test": len(test),
                "test_base_rate": float(y_te.mean()),
                "used_cuisine": USE_CUISINE,
                "auc_with_cuisine": float(auc_full),
                "auc_without_cuisine": float(auc_nc),
                "models": rows,
                "capacity": caps,
            },
            indent=2,
        )
    )
    print(f"\nsaved -> {SCORED_CSV}\nsaved -> {LOGIT_PKL}\nsaved -> {METRICS_JSON}")


if __name__ == "__main__":
    main()
