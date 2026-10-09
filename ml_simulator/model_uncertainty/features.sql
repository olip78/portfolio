WITH base_query AS (
  SELECT
    product_name,
    monday,
    MAX(price) as max_price,
    COUNT() as y
  FROM
    (
     SELECT
    price,
    product_name,
    toDate(toDate(dt) - (toDayOfWeek(dt) - 1)) as monday
FROM
    default.data_sales_train
) format_query
  GROUP BY
    product_name,
    monday
  ORDER BY
    product_name,
    monday
),

lagged_values AS (
SELECT
    product_name,
    monday,
    max_price,
    y,
  FIRST_VALUE(y) OVER(PARTITION BY product_name order by monday rows between 1 preceding and 1 preceding) as y_lag_1,
  FIRST_VALUE(y) OVER(PARTITION BY product_name order by monday rows between 2 preceding and 2 preceding) as y_lag_2,
  FIRST_VALUE(y) OVER(PARTITION BY product_name order by monday rows between 3 preceding and 3 preceding) as y_lag_3,
  FIRST_VALUE(y) OVER(PARTITION BY product_name order by monday rows between 4 preceding and 4 preceding) as y_lag_4,
  FIRST_VALUE(y) OVER(PARTITION BY product_name order by monday rows between 5 preceding and 5 preceding) as y_lag_5,
  FIRST_VALUE(y) OVER(PARTITION BY product_name order by monday rows between 6 preceding and 6 preceding) as y_lag_6,
  MAX(y) OVER(PARTITION BY product_name order by monday rows between 3 preceding and 1 preceding) as y_max_3,
  MAX(y) OVER(PARTITION BY product_name order by monday rows between 6 preceding and 1 preceding) as y_max_6
FROM
  base_query
    ),
    
lagged_values_with_alls AS (
SELECT *,
  LEAST(y_lag_1, y_lag_2, y_lag_3)  as y_min_3,
  LEAST(y_lag_1, y_lag_2, y_lag_3, y_lag_4, y_lag_5, y_lag_6)  as y_min_6,
  (y_lag_1 + y_lag_2 + y_lag_3) / 3 as y_avg_3,
  (y_lag_1 + y_lag_2 + y_lag_3 + y_lag_4 + y_lag_5 + y_lag_6) / 6  as y_avg_6,
  SUM(y_lag_1) OVER(PARTITION BY monday) as y_all_lag_1,
  SUM(y_lag_2) OVER(PARTITION BY monday) as y_all_lag_2,
  SUM(y_lag_3) OVER(PARTITION BY monday) as y_all_lag_3,
  SUM(y_lag_4) OVER(PARTITION BY monday) as y_all_lag_4,
  SUM(y_lag_5) OVER(PARTITION BY monday) as y_all_lag_5,
  SUM(y_lag_6) OVER(PARTITION BY monday) as y_all_lag_6
FROM lagged_values
    ),

lagged_values_result AS (
SELECT *,
  LEAST(y_all_lag_1, y_all_lag_2, y_all_lag_3)  as y_all_min_3,
  LEAST(y_all_lag_1, y_all_lag_2, y_all_lag_3, y_all_lag_4, y_all_lag_5, y_all_lag_6)  as y_all_min_6,
  greatest(y_all_lag_1, y_all_lag_2, y_all_lag_3)  as y_all_max_3,
  greatest(y_all_lag_1, y_all_lag_2, y_all_lag_3, y_all_lag_4, y_all_lag_5, y_all_lag_6)  as y_all_max_6,
  (y_all_lag_1 + y_all_lag_2 + y_all_lag_3) / 3 as y_all_avg_3,
  (y_all_lag_1 + y_all_lag_2 + y_all_lag_3 + y_all_lag_4 + y_all_lag_5 + y_all_lag_6) / 6  as y_all_avg_6
FROM lagged_values_with_alls
    )
    
SELECT *
FROM lagged_values_result
ORDER BY
    product_name,
    monday
