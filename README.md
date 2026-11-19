# Data Science Portfolio

## Current personal projects

- [Forecasting and Optimization for Industrial Energy Management](https://github.com/olip78/industrial-energy-optimization)
  <br><sub>End-to-end decision system for an industrial consumer with flexible load, PV generation, battery storage, and access to the German day-ahead and intraday markets.</sub>
  <br><sub>Point and probabilistic forecasting, deterministic and stochastic MILP, model-predictive control, out-of-sample economic replay.</sub>
  <br><sub>Deployed ML service simulating real-time industrial operation on historical data: fresh forecasts, day-ahead planning, hourly MPC and a data/model monitoring dashboard synchronized with Europe/Berlin operating hours.</sub>
  <br><sub>Developed in collaboration with OpenAI Codex.</sub>
  <br>
  <br><sub>Python, CatBoost, SciPy/HiGHS, FastAPI, MLflow, Docker Compose, GitLab CI/CD; blue-green application deployment.</sub>
  <br><sub>[Live ML service](http://81.26.188.18:8080) · [Technical report](https://github.com/olip78/industrial-energy-optimization/blob/main/docs/project_report.pdf) · [Service source and CI/CD](https://gitlab.com/olip78/energy-optimization)</sub>

- [Retail Pricing: Forecasts, Experiments and Learning](https://github.com/olip78/retail-pricing)
  <br><sub>Compared CatBoost with two models that separate baseline demand from price response for 11 juice products. Evaluated forecast accuracy, response stability, and how the models behave when used to optimize prices.</sub>
  <br><sub>[Part I report](https://github.com/olip78/retail-pricing/blob/main/docs/part_1_historical_data.md)</sub>

## Selected commercial ML case studies

High-level descriptions of commercial projects whose source code and client data are proprietary.

- [Browser fingerprinting and identity resolution](./case_studies/browser-fingerprinting.md)
  <br><sub>End-to-end identity-resolution system combining representation learning, candidate retrieval, pairwise matching, calibrated decisioning, and an evolving Identity Graph</sub>
- [LLM-assisted matching and classification for multilingual laboratory catalogues](./case_studies/laboratory-catalogue-matching.md)
  <br><sub>Structured catalogue matching, LLM-based enrichment, graph-constrained candidate selection, ML reranking, and neural classification</sub>

## Selected personal projects (2023-2024)

Earlier personal projects in forecasting, deep learning, and MLOps.

- [End-to-end ML service for multivariate short-term taxi-demand forecasting](https://github.com/olip78/ynyt)
  <br><sub>Fourier harmonics, spatio-temporal transformers, PyTorch, FastAPI, MLflow, Docker, Docker Swarm, GitLab CI/CD, scheduled inference</sub>
- [Enefit: day-ahead forecasting of prosumer energy consumption and PV generation](https://github.com/olip78/Enefit)
  <br><sub>Multivariate time series, encoder-decoder spatio-temporal transformer, PyTorch</sub>
- [VAE for anomaly detection in the IEEE-CIS Fraud Detection challenge](https://github.com/olip78/DGM-for-anomaly-detection)
  <br><sub>Variational autoencoder, anomaly detection, PyTorch</sub>

## Advanced applied-ML training projects (2022-2023)

Selected projects from five two-week assignments completed between November 2022 and May 2023 as part of advanced ML training and subsequently refactored.

- [Advanced A/B testing](./hard_ml/ab)
  <br><sub>CUPED, post-stratification</sub>
- [Ranking and matching](./hard_ml/ranking)
  <br><sub>KNRM, FAISS, Flask</sub>
- [Uplift modelling](./hard_ml/uplift)
  <br><sub>Feature store, CausalML, flow-based application</sub>
- [Dynamic pricing](./hard_ml/dynamic_pricing)
  <br><sub>Bayesian optimization, contextual bandits</sub>

## Selected ML engineering training projects (2023)

- [Dynamic-pricing optimization](./ml_simulator/pricing/)
  <br><sub>Mixed-integer linear programming for revenue and margin optimization, including business constraints on product-group prices</sub>
- [Demand-forecasting service](./ml_simulator/demand_forecast/)
  <br><sub>Quantile regression, bootstrap uncertainty estimation, ClearML training and inference pipelines, and FastAPI</sub>

[Selected professional development and certifications](./self_study.md)
