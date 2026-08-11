# README skeleton — write the prose yourself, delete this file after

The numbers below are from your actual run. The section prompts are the
questions an interviewer will ask about this project. Answer each in two or
three sentences in your own voice, save it as `README.md`, delete this file.

Do not let me write this part. The README is the artifact that proves the
reasoning was yours, and it's the thing you'll be reading back off your own
screen when someone asks "walk me through a project."

---

## Title + one-line pitch
> Prompt: What does this do, in one sentence, for someone who will not read
> further? (Hint: it's not "predicts restaurant inspections." It's about who
> gets inspected first when there aren't enough inspectors.)

## The question
> Prompt: Why is this the useful question rather than the obvious one? What
> decision does the output feed?

## The data
- NYC Open Data, DOHMH Restaurant Inspection Results, dataset `43nn-pn8j`
- 291,245 raw violation rows pulled via the SODA API
- One row per **violation**, collapsed to one row per **inspection**
  (`camis` + `inspection_date`)
- Pre-2022 dropped: ~2,800 scattered rows across 12 years vs 288K in 2022+

## The three decisions worth defending
> Prompt: These are your best interview material. Write each as "I found X,
> so I did Y."

1. **The grade column is unusable as a label.** On Cycle/Initial inspections
   only `A` and `N` appear — no `B` or `C`. Passing earns an A on the spot;
   failing goes ungraded to a re-inspection. So the ~40% nulls *are* the
   failures. Missing-not-at-random, conditioned on the outcome.
   → Label built from score instead: `failed_a = score > 13`, using DOHMH's
   own A-grade cutoff (max score among A grades in the data is exactly 13.0).

2. **Population restricted to Cycle Inspection / Initial Inspection**
   (7,738 of 15,006 in the 2023 sample). Re-inspections are selected on the
   outcome of the inspection before them. Pre-permit visits have no history.

3. **No feature DOHMH couldn't have computed that morning.** Everything named
   `prior_*` is shifted one inspection back within the establishment before
   any aggregation. `days_since_last` is allowed because the scheduled date is
   known in advance.

## Split
| | years | rows |
|---|---|---|
| burn-in (history only) | 2022–2023 | — |
| train | 2024–2025 | 15,620 |
| test | 2026 | 7,285 |

22,905 modeling rows across 16,657 establishments after requiring `prior_n >= 1`.
Time-based, not random — a random split would let the model see an
establishment's future when predicting its past.

## Results
Test-set (2026) performance:

| model | acc | prec | rec | AUC |
|---|---|---|---|---|
| always predict pass | 0.595 | 0.000 | 0.000 | — |
| own prior fail rate (baseline) | 0.652 | 0.578 | 0.519 | 0.645 |
| logistic regression | 0.654 | 0.633 | 0.351 | **0.691** |
| gradient boosting | 0.651 | 0.609 | 0.391 | 0.680 |

> Prompt: Two things to say about this table. (a) Logistic beat gradient
> boosting — why is that plausible rather than embarrassing? (b) Logistic has
> a *worse* recall than the baseline at the 0.5 threshold despite a better
> AUC. What does that tell you about 0.5 as a threshold?

## Capacity, not 0.5
The real output. Test base rate is 40.5%.

| inspect top | threshold | failures caught | catch rate | precision | lift |
|---|---|---|---|---|---|
| 500 | 0.648 | 355 / 2,953 | 12.0% | 0.710 | 1.75x |
| 1,000 | 0.571 | 676 / 2,953 | 22.9% | 0.676 | 1.67x |
| 2,000 | 0.464 | 1,217 / 2,953 | 41.2% | 0.609 | 1.50x |
| 3,000 | 0.383 | 1,697 / 2,953 | 57.5% | 0.566 | 1.40x |

> Prompt: State the operational payoff in plain English. At 2,000 inspections,
> how many more failures does this find than sending inspectors at random?

## Limitations
> Prompt: Write these as things you'd say before an interviewer has to ask.

- Fail rate drifts upward across the period (0.365 → 0.380 → 0.405); a model
  trained on earlier years systematically under-predicts later ones.
- `prior_n` has median 1 and max 4 — the published dataset is a rolling
  ~3-year window, so establishment history is thin by construction.
- **Cuisine.** The largest coefficients in the model are cuisine dummies, not
  history features. Ranking enforcement partly by cuisine type is a policy
  decision, not a modeling detail. `train.py` reports what dropping cuisine
  costs in AUC; put that number here and say what you'd recommend.
- Feedback loop: inspecting the flagged establishments changes their future
  history, which changes future predictions. A real deployment needs a
  randomized holdout.

## Running it
```bash
pip install -r requirements.txt
python src/pull_data.py    # ~5 min, hits the SODA API
python src/features.py
python src/train.py
streamlit run app.py
```

## Repo layout
```
src/config.py     every decision that could have gone the other way
src/pull_data.py  paged SODA API pull -> data/inspections_raw.csv
src/features.py   collapse to inspection level, label, history features
src/train.py      baselines -> models -> capacity curve -> artifacts
app.py            ranked list with the inspector-capacity slider
```
