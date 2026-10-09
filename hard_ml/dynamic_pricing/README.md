# Dynamic Pricing

*Final training project for the Dynamic Pricing module of the Hard ML specialisation.*

## Task

Build a dynamic-pricing service for 1,000 products with the goal of maximising profit. The service is evaluated with a simulator covering 29 days of sales and purchasing activity.

For a single SKU, the objective can be written as:

$$\sum_{u}\left(\sum_{d}S_{u,d}\cdot\left(P_{d}(\theta) - C_{d} \right) + B\cdot I[\sum_{d}S_{u,d}\geq S\_{plan}]\right) \to \max_{p(\cdot)}$$

where $u$ denotes a user, $d$ a day, $P_d(\theta)$ the product price, $S_{u,d}$ the quantity purchased, $C_d$ the product cost, $S_{plan}$ the sales target, and $B$ the retrospective bonus.

The simulation allows new customers and assumes no cannibalisation between products.

## Data

The synthetic data includes:

- historical transactions;
- sales targets and retrospective bonuses;
- product costs;
- promotional campaigns;
- competitors' prices.

## Approaches

### Demand modelling and optimisation

The first approach estimates a product demand function, $D_{SKU}(p, \theta)$, from historical or experimental data. For each pricing period, the price is selected by solving:

$$p\cdot D_{SKU}(p, \theta)\to \max_{p}$$

Products were first [clustered by their sales time series](./notebooks/EDA.ipynb). A separate gradient-boosting demand model was then fitted for each cluster, and the optimal weekly price for each SKU was selected using [Bayesian optimisation](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html).

### Contextual bandits

The second approach uses a contextual-bandit model to select a pricing strategy from product and customer features. The implementation is based on the [LaunchpadAI `space-bandits` project](https://github.com/LaunchpadAI/space-bandits).

The reward function combines unit profit with an additional component intended to encourage fulfilment of the sales target:

$$P_{d} - C_{d} + \dfrac{B}{S\_{plan} - S\_{cum\_d}}\cdot I[b_{0} < r_{d} < b_{1}]$$

where $S_{cum_d}$ is cumulative sales, $r_d$ is the current sales pace, and $b_0$ and $b_1$ are its lower and upper bounds.

## Contents

- [`notebooks/EDA.ipynb`](./notebooks/EDA.ipynb): SKU clustering and context-feature selection.
- [`notebooks/price_modelling.ipynb`](./notebooks/price_modelling.ipynb): demand modelling and price optimisation.
- [`client.py`](./client.py): contextual-bandit service client.
