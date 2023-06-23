# Final project of Dynamic pricing module of Hard ML specialization

## Task
Build a dynamic pricing service for 1,000 products aimed at profit maximisation. The service is tested on a sales simulator, that models 29 days sales / buying activity. 

For a single SKU (product) the task can be formulated as the following optimization problem:

$$\sum_{u}\left(\sum_{d}S_{u,d}\cdot\left(P_{d}(\theta) - C_{d} \right) + B\cdot I[\sum_{d}S_{u,d}\geq S\_{plan}]\right) \to \max_{p(\cdot)}$$

Where:

$u$, $d$ - user, day

$P_{d}(\theta)$ - price of the SKU on the day $d$

$\theta$ - pricing factors

$S_{u,d}$ - amount of the product, bought by the user $u$ on the day $d$

$C_{d}$ - the product cost on the day $d$ 

$S\_{plan}$, $B$ - sales plan and back bonus

None: there may be new customers; there is no cannibalization.

## Data
Synthetic data:

- Historical transactions
- Sales plan / back bonuses
- Cost prices
- Promo actions
- Some competitors prices

## Solution
The main aim of the project was to experiment with dynamic pricing approaches that I know of:
- Price modelling over demand 
- Using an RL model 
### Price modelling over demand 
##### Approach:
On the first step, using historical or experimental data, one tries to fit a product demand function as a regression on price and some other factors: $D_{SKU}(p, \theta)$, after that for each time period the optimal price can be determined by solving the profit (or some other objective) optimization problem: $p\cdot D_{SKU}(p, \theta)\to \max_{p}$
##### Solution:
At the preliminary stage, I tried [to cluster](./notebooks/EDA.ipynb) all products (SKU) time series. Then on the main step I fitted a gradient boosting model for each cluster, after that for each SKU / week determined the optimal price by using [bayesian optimization](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html).


### RL model: contextual bandits
##### Approach:
By this approach an RL algorithm tries to learn to determine the optimal price from available price strategies, based on particular context (product and user features) and a cost function.
##### Solution:
I used LaunchpadAI [implementation](https://github.com/LaunchpadAI/space-bandits) of contextual bandits. Context vector is discussed [here](./notebooks/EDA.ipynb). By setting reward function, besides the profit component, I tried to stimulate fulfillment of the sales plan:

$$P_{d} - C_{d} + \dfrac{B}{S\_{plan} - S\_{cum\_d}}\cdot I[b_{0} < r_{d} < b_{1}]$$

where 

$P_{d}$, $C_{d}$ price and cost of the product on day $d$

$S\_{cum\_d}=\sum_{d-1}S_{d}$

$r_{d}$ - sales dynamic: 
$$r_{d} = \dfrac{29}{(d-1)\cdot S\_{plan}}\cdot S\_{cum\_d}  $$
$b_{0}$, $b_{1}$ - low and upper bounds, $b_{1}=1.01$, $b_{0}$ - a kind of sigmoid function on $d$

## Content

- SKU time series clustering, context vector / feature selection: [/notebooks/EDA.ipynb](./notebooks/EDA.ipynb)
- Price modelling: [/notebooks/price_modelling.ipynb](./notebooks/price_modelling.ipynb)
- Contextual bandits service: [dynamic_pricing/client.py](./client.py)
