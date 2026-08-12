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


