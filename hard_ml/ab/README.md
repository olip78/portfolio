# Advanced A/B Testing

*Final training project for the Advanced A/B Testing module of the Hard ML specialisation.*

## Task

Implement a web service for evaluating A/B experiments. The synthetic dataset contains eight weeks of purchase history for approximately 10 million users. In the final week, 1,000 experiments were conducted with the goal of increasing average revenue per user. For each experiment, the service must determine whether the observed effect is statistically significant.

## Approach

The solution combines CUPED, using predicted user-level sales as a covariate, with post-stratification.

## Contents

- [`notebooks/data_analysis.ipynb`](./notebooks/data_analysis.ipynb): exploratory analysis, selection of strata and predictive features, minimum detectable effect estimation, and validation of the CUPED and post-stratification procedures on historical data.
- [`notebooks/modelling.ipynb`](./notebooks/modelling.ipynb): development of the user-level sales prediction model.
- [`app/solution.py`](./app/solution.py): Flask service that processes one experiment request and tests whether the observed effect is statistically significant.

The exercise treats experiment requests independently and therefore does not apply a multiple-testing correction.
