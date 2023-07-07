# Final project of Uplift modelling module of Hard ML specialization

## Legend
A chain store "Cunning Fox" sells a certain product in a number of cities. Six months ago "Cunning Fox” ran a marketing campaign. A number of customers were sent letters with a personalised offer - "Get a 40-coin discount on the first purchase of one gram of the product". The discount is valid for a week." The former marketer of Sly Fox didn't take long to think about it and chose the audience for the communication at random.

You are faced with the task of selecting of a target group of the "Cunning Fox" customers, so that the total profit from a campaign with such an offer is as large as possible. The "Cunning Fox" has a competitor.

Synthetic data: Information about sales for the last 250 days, basic customer information and the history of the launched campaigns.

## Problem specification
The problem can be formulated in the following way:
$$|T|\cdot \left( \dfrac{1}{|T|}\sum_{u \in T}(r\cdot S_{u}^{30} - d\cdot I[S_{u}^{7} > 0]-s) - \dfrac{1}{|C|}\sum_{u \in C}\cdot S_{u}^{30}\right) \to \max_{T}$$

where:

$T$, $C$ - target and control groups,

$S_{u}^{30}$, $S_{u}^{7}$  - purchases of the customer $u$ during 30/7 days after campaign start,

$r$ - revenue of product, $r=28$,

$s$ - cost of mailing, $s=1$,

$d$ - value of discount, $d=40$.

## Solution content 

- Feature store: [/upcampaign/datalib]/upcampaign/datalib
- Uplift modeling: [/notebooks/learning.ipynb](./notebooks/learning.ipynb)
- Flow-based Application for an Uplift-Campaign: [/upcampaign]/upcampaign

## Application Uplift-Campaign

The application is based on apporach and materials of one of the lectures of the course. The application, according to campaign [configuration](./configs/basic_campaign.json):
- filters not active customers out,
- determines Target group, selected according uplift model scores and selected by training threshold
- selects two Control groups of customers: offer/no offer

The application output is a customer csv table with group (target/control) and treatment (offer/no offer) labels.

The campaign is run by [run.py](./uplift-campaign/run.py) file with the following arguments:

Parameter                 |Description   |	
| :------------------------ |:-------------|
| --run-id 	       |run id |
| --config | path to campaign config |
| --system-config | path to db config |
| --date-to | threshold for active customers|
|-o --output| path to the output file|

