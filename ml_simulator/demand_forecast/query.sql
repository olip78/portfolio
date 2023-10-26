WITH demand_orders AS (
    SELECT 
        default.demand_orders.* EXCEPT(timestamp, order_id),
        formatDateTime(default.demand_orders.timestamp, '%Y-%m-%d') as day,
        b.* EXCEPT(order_id, status)
    FROM default.demand_orders
    LEFT OUTER JOIN default.demand_orders_status b ON default.demand_orders.order_id = b.order_id
),

item_id_list AS (
    SELECT 
      sku_id, 
      FIRST_VALUE(sku) as sku
    FROM demand_orders
    GROUP BY sku_id
),
day_list AS (
    SELECT DISTINCT day AS day 
    FROM demand_orders
),

product AS (
SELECT *
FROM day_list
CROSS JOIN item_id_list),

demand_orders_grouped AS(
  SELECT 
    day, sku_id,
    FIRST_VALUE(price) as price,
    SUM(qty) as qty
  FROM demand_orders
  WHERE status_id in (1,3,4,5,6)
  GROUP BY day, sku_id
),

pre_final AS(
SELECT
  product.*,
  b.* EXCEPT (sku_id, day)
FROM product
LEFT OUTER JOIN demand_orders_grouped b ON product.sku_id = b.sku_id AND product.day = b.day
ORDER BY sku_id, day
settings join_use_nulls=1
)

SELECT * EXCEPT(qty, price),
  ifNull(qty, 0) qty,
  any(price) OVER (PARTITION BY sku_id
      ORDER BY day ASC
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED following) as price
FROM pre_final
