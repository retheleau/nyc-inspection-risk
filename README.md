# NYC Restaurant Inspection Failure Prediction

Live app: https://nyc-inspection-risk.streamlit.app/

The New York City Department of Health and Mental Hygiene
has more restaurants due for inspections than inspectors available to complete the work.
With only the knowledge on record before inspections occur, can we predict which 
restaurants are most likely to fail? Then steer the inspectors to the restaurants 
forecast to fail, placing limited labor where it helps the most people.


## Result

Test set is every Cycle Initial inspection in 2026 — 7,280 inspections, 2,951 of
which failed. **The base rate is 40.5%.** That number comes first because
predicting "pass" for every restaurant in the city scores 59.5% accuracy and
catches nothing.


| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Always predict pass | 0.595 | 0.000 | 0.000 | — |
| Own prior fail rate (baseline) | 0.652 | 0.578 | 0.519 | 0.645 |
| **Logistic regression (shipped)** | 0.642 | 0.612 | 0.319 | **0.681** |
| Gradient boosting | 0.648 | 0.606 | 0.377 | 0.673 |



AUC measures how well the model *orders* the list — 0.5 is a coin flip, 1.0 is
perfect. The bar to beat was 0.645: the establishment's own prior failure rate,
which is essentially "did they fail last time." The shipped model reaches 0.681.
That is a real gain but a modest one, and most of the signal is still prior
failure.

Recall appears worse than the baseline (0.319 vs 0.519), which is a cutoff artifact, not a paradox. Accuracy, precision and recall are all recorded at the default 0.5 threshold; AUC records how well the model ranks across every possible threshold. The model ranks better; its default operating point is just too conservative, which is why the threshold belongs in the next section.

### What it looks like in practice

The output is not a probability column, it is a ranked queue. The threshold is
set by how many inspections the department can run, not by 0.5.





| Inspections available | Failures caught | Hit rate | Lift over base rate |
|---|---|---|---|
| 500 | 337 of 2,951 | 67.4% | 1.66× |
| 1,000 | 641 of 2,951 | 64.1% | 1.58× |
| 2,000 | 1,209 of 2,951 | 60.5% | 1.49× |
| 3,000 | 1,675 of 2,951 | 55.8% | 1.38× |
| 5,000 | 2,406 of 2,951 | 48.1% | 1.19× |


If DOHMH can run 500 inspections this period, the model's picks fail at 67%
against a 40% base rate. Inspect more and you catch more of the total failures
but the hit rate drops — at 5,000 you find 82% of all failures at a 48% hit
rate. That trade-off belongs to the operations team, and the app lets them
move it.

## Data

[DOHMH New York City Restaurant Inspection Results](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j),
NYC Open Data, pulled through the Socrata API — 291,245 rows.

The raw feed is violation-level, not inspection-level. An inspection that cites
four violations produces four rows, with the establishment, date, score and
grade repeated on each. The first thing the pipeline does is collapse it to one
row per establishment-inspection, keyed on `camis` and `inspection_date`.
Without that step, restaurants with more violations would be weighted more
heavily purely as an artifact of the file format.

Three filters, applied in that order:

- **Pre-2022 rows dropped.** The dataset holds a rolling three-year window.
  About 2,800 rows are scattered across 2010–2021 against 288,000 from 2022
  onward — stragglers, not history.
- **Cycle Inspection / Initial Inspection only.** Re-inspections exist only
  because an establishment already failed, so including them puts the outcome
  into the sample selection. Pre-permit inspections are brand-new
  establishments with no inspection history to build features from.
- **Records with borough written as `0` removed.** 57 inspections with no
  borough assigned.

What remains: **45,817 inspections**, of which **22,879** are usable for
modelling once each row is required to have at least one prior inspection
behind it.

## Label

The obvious target is the letter grade. It doesn't work.

On Cycle Initial inspections the grade column holds 26,870 A's, 19,249 blanks,
376 N's and a single C. No B's at all — in a city where B and C placards hang
in windows everywhere.

The blanks aren't clerical gaps. Graded inspections average 9.4 points;
ungraded ones average 28.8. DOHMH awards an A on the spot when an
establishment passes; anything worse goes to re-inspection and is graded
there. So the missing grades *are* the failures — missing not at random, on
exactly the outcome being predicted. Using grade as the label and dropping the
nulls would delete every failure and train a model on a dataset where everyone
passed. It would score beautifully and be worthless.

Score is complete where grade is 41% missing, so the label is score-based:

`failed_a = score > 13`

13 is DOHMH's own A threshold, confirmed from the data — the maximum score
among A-graded inspections is exactly 13.0.


