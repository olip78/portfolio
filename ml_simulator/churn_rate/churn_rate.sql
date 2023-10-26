WITH churn_tasks AS (
  SELECT 
    task_id, 
    MAX(task_steps) as task_steps, 
    FIRST_VALUE(task_level) as task_level
  FROM default.churn_tasks
  GROUP BY task_id
),

churn_submits AS (
  SELECT *
  FROM default.churn_submits
),

churn_submits_with_tasks AS (
  SELECT 
    churn_submits.*, 
    churn_tasks.* EXCEPT(task_id)
  FROM churn_submits
  LEFT OUTER JOIN churn_tasks ON churn_submits.task_id = churn_tasks.task_id
),

churn_submits_with_tasks_mod AS (
  SELECT 
    * EXCEPT(submit_id, submit, task_steps, timestamp),
    CAST(timestamp AS DATE) AS day,
    CAST(timestamp AS DATE) AS last_day,
    submit / task_steps AS submit
  FROM churn_submits_with_tasks
),

base_query AS (
    SELECT 
        user_id, 
        day,
        SUM(task_level) as max_task_level,
        FIRST_VALUE(last_day) as last_day,
        anyOrNull(day) OVER (PARTITION BY user_id
          ORDER BY day ASC
          ROWS BETWEEN 1 PRECEDING AND  1 PRECEDING) as last_day_tmp,
        COUNT() as n_submits,
        COUNT(DISTINCT task_id) as n_tasks,
        COUNT((CASE WHEN is_solved = True THEN 1 END)) as n_solved,
        CASE 
          WHEN n_submits >= 0 THEN 1
          ELSE 0
        END AS activday
    FROM churn_submits_with_tasks_mod
    GROUP BY user_id, day
    ORDER BY user_id, day
),

base_query_with_diff AS( 
SELECT * EXCEPT(day_diff),
    MAX(day_diff) OVER (PARTITION BY user_id
          ORDER BY day ASC
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW ) as max_off_period
 FROM (SELECT * EXCEPT(last_day_tmp),
  day - last_day_tmp as day_diff 
FROM base_query)
),

user_id_list AS (
    SELECT DISTINCT user_id AS user_id 
    FROM base_query
),
day_list AS (
    SELECT DISTINCT day AS day 
    FROM base_query
),

product AS (
SELECT *
FROM user_id_list
CROSS JOIN day_list),

main_grid AS (

SELECT
  product.*,
  b.* EXCEPT (user_id, day)
FROM product
LEFT OUTER JOIN base_query_with_diff b ON product.user_id = b.user_id AND product.day = b.day
ORDER BY user_id, day
settings join_use_nulls=1
),

main_grid_with_filled_last_day AS (
SELECT * EXCEPT (n_submits, n_tasks, n_solved, last_day, max_task_level), 
  ifNull(n_submits, 0) n_submits,
  ifNull(n_tasks, 0) n_tasks,
  ifNull(n_solved, 0) n_solved,
  ifNull(max_task_level, 0) max_task_level,
  MAX(last_day) OVER (PARTITION BY user_id
      ORDER BY day ASC
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW ) as last_day  
FROM main_grid
),

pre_final AS (

SELECT * EXCEPT (last_day, max_off_period),
  day - last_day as days_offline,
  CASE 
        WHEN days_offline >= ifNull(max_off_period, 0) THEN days_offline 
        ELSE max_off_period
  END AS max_off_period,
  SUM(n_submits) OVER w14 AS sum_submits_14d,
  SUM(n_solved) OVER w14 AS sum_success_14d,
  SUM(n_solved) OVER (PARTITION BY user_id
          ORDER BY
            day ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as solved_total,
  SUM(n_submits) OVER (PARTITION BY user_id
          ORDER BY
            day ASC Rows BETWEEN 1 following AND 14 following) AS sum_target
from main_grid_with_filled_last_day
WINDOW
    w14 AS (PARTITION BY user_id
          ORDER BY
            day ASC Rows BETWEEN 13 PRECEDING AND CURRENT ROW)   
),

final AS (
SELECT * EXCEPT (sum_submits_14d, sum_target, n_submits, n_tasks, n_solved, activday),
  sum_submits_14d / 14 as avg_submits_14d,
  CASE WHEN sum_submits_14d = 0
       THEN 0
       ELSE sum_success_14d / sum_submits_14d
  END AS success_rate_14d,
  CASE WHEN sum_target > 0
       THEN 0
       ELSE 1
  END AS target_14d,
  SUM(activday) OVER (PARTITION BY user_id
          ORDER BY
            day ASC ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as active_7,
  SUM(activday) OVER (PARTITION BY user_id
          ORDER BY
            day ASC ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) as active_14,
  SUM(activday) OVER (PARTITION BY user_id
          ORDER BY
            day ASC ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) as active_21
  
FROM pre_final
)

SELECT
    day,
    user_id,
    (- 0.19159356*max_task_level + 0.05115319*max_off_period + 0.13604974*sum_success_14d - 
     0.08571104 * solved_total - 0.8485789*avg_submits_14d - 0.37046575*active_7 - 
     0.37046575 * active_14) AS predict_14d 
FROM (
    SELECT * EXCEPT (active_7, active_14, active_21),
      ifNull(active_7, 0) active_7,
      ifNull(active_7, 0) active_14,
      ifNull(active_7, 0) active_21
    FROM final
    ORDER BY user_id, day
    ) features
WHERE day = '2022-11-19'
