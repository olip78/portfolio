SELECT 
  Id AS review_id,
  toDateTime(Time) as dt,
  Score as rating,
  multiIf(Score=1, 'negative', Score=5, 'positive', 'neutral') as sentiment,
  Text as review
FROM simulator.flyingfood_reviews
ORDER BY review_id
