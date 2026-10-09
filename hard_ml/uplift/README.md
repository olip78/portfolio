# Uplift Modelling for Campaign Targeting

*Final training project for the Uplift Modelling module of the Hard ML specialisation.*

## Business setting

A retail chain ran a marketing campaign in which randomly selected customers received a personalised discount. The task is to select the audience for a future campaign so that its expected total profit is maximised, taking into account product revenue, the discount, communication costs, and the presence of a competitor.

The synthetic dataset contains 250 days of sales history, basic customer attributes, and information about previous campaigns.

## Objective

The optimisation problem is formulated as:

$$|T|\cdot \left( \dfrac{1}{|T|}\sum_{u \in T}(r\cdot S_{u}^{30} - d\cdot I[S_{u}^{7} > 0]-s) - \dfrac{1}{|C|}\sum_{u \in C} S_{u}^{30}\right) \to \max_{T}$$

where:

- $T$ and $C$ are the target and control groups;
- $S_{u}^{30}$ and $S_{u}^{7}$ are customer $u$'s purchases during the 30 and 7 days after the campaign starts;
- $r$ is product revenue, with $r=28$;
- $s$ is the communication cost, with $s=1$;
- $d$ is the discount value, with $d=40$.

## Solution

- [Feature store](./uplift-campaign/upcampaign/datalib)
- [Uplift modelling](./notebooks/learning.ipynb)
- [Campaign application](./uplift-campaign/upcampaign)

The flow-based application uses the campaign [configuration](./configs/basic_campaign.json) to:

- filter out inactive customers;
- select the target group using uplift-model scores and a threshold chosen during training;
- create two control groups representing offer and no-offer treatments.

The output is a CSV file containing customer-level group and treatment labels. The campaign is started with [`uplift-campaign/run.py`](./uplift-campaign/run.py).

| Parameter | Description |
|---|---|
| `--run-id` | Run identifier |
| `--config` | Path to the campaign configuration |
| `--system-config` | Path to the database configuration |
| `--date-to` | Activity cutoff date |
| `-o`, `--output` | Path to the output file |
