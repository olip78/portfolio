# Data Science Portfolio

## Commercial ML case studies

- [Matching multilingual laboratory catalogues](./case_studies/laboratory-catalogue-matching.md)
  <br><sub>As the sole data scientist and developer, I built a tool that helps laboratory experts map local test names to a common catalogue. It uses LLM-assisted translation, retrieves candidates with compatible test attributes, and reranks them with CatBoost. The correct match appeared in the first five suggestions in about 94.5% of evaluated cases. Integration into the client’s core platform was planned as a later step.</sub>

- [Browser fingerprinting and identity resolution](./case_studies/browser-fingerprinting.md)
  <br><sub>As the sole data scientist, I designed a system to recognise returning browsers when signals change and reliable identity labels are scarce. In a sequential offline evaluation, the pairwise model reached 97.2% precision at 86.3% recall, with 7.1% of events left uncertain. A production component based partly on my research implementation was integrated for the first client.</sub>

## Current personal projects

- [Forecasting and optimization for industrial energy management](https://github.com/olip78/industrial-energy-optimization)
  <br><sub>When should an industrial site run flexible loads or charge its battery? The service forecasts solar generation and electricity prices, plans a day ahead, and revises the schedule hourly. Across 261 held-out days, this strategy reduced simulated electricity costs by 6.4% against a rule-based schedule. The dashboard replays historical data on a Berlin-time clock. Developed in collaboration with OpenAI Codex.</sub>
  <br><sub>[Live dashboard](http://81.26.188.18:8080) · [Technical report](https://github.com/olip78/industrial-energy-optimization/blob/main/docs/project_report.pdf) · [Service source and CI/CD](https://gitlab.com/olip78/energy-optimization)</sub>

- [Retail pricing, demand forecasts and price response](https://github.com/olip78/retail-pricing)
  <br><sub>I compared three demand models for 11 juice products. CatBoost gave the most accurate sales forecasts, yet in one optimization check it recommended a price increase of at least 40% in 94.6% of decisions. The completed historical study examines why these recommendations need a controlled price experiment. Developed in collaboration with OpenAI Codex.</sub>
  <br><sub>[Part I report](https://github.com/olip78/retail-pricing/blob/main/docs/part_1_historical_data.md)</sub>

## Selected personal projects (2023-2024)

Earlier personal projects in forecasting, deep learning, and MLOps.

- [End-to-end ML service for multivariate short-term taxi-demand forecasting](https://github.com/olip78/ynyt)
  <br><sub>Ridge baseline with Fourier harmonics (2020); spatio-temporal transformer (2023); PyTorch, FastAPI, MLflow, Docker, Docker Swarm, GitLab CI/CD, scheduled inference</sub>
  <br>[Live dashboard](http://81.26.188.18:8082)
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
