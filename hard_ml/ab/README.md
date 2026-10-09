# Final project for Advanced A/B Testing Module Hard ML Specialisation

## Problem statement:

"As a final project, a web service needs to be implemented to evaluate A/B experiments. There is synthetic data on users (10e6) and their purchases for 8 weeks. In the last week a series of experiments (1000) were conducted, the aim of which was to increase the average revenue per person. The task of the service is to determine if there is an effect or not for an experiment.”

## Solution approach:

CUPED with sales predictions for each user as a covariate + post stratification


## Content:

The notebook [notebook/data_analysis.ipynb](./notebooks/data_analysis.ipynb) contains: 
- EDA, that helped to identify strata as well as features for sales prediction model 
- MDE estimation, validation of CUPED/post stratification approaches as well as statistical test on historical data

In the notebook [notebook/modelling.ipynb](./notebooks/modelling.ipynb) a sales prediction model was developed.

[app/solution.py](./app/solution.py) is a flask service, handling a single AAB experiment request and checking, whether there was a statistically significant effect or not. No multiple testing correction was used because experiments could be handle only queued.