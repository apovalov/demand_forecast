WITH main AS (
    WITH
    calendar AS (
        SELECT DISTINCT toDate(timestamp) AS date
        FROM default.demand_orders
    ),
    sku_list AS (
        SELECT DISTINCT sku_id, sku, price
        FROM default.demand_orders
    ),
    sales_data AS (
        SELECT
            toDate(o.timestamp) AS date,
            o.sku_id,
            sumIf(o.qty, dos.status_id IN (1, 3, 4, 5, 6)) AS qty
        FROM default.demand_orders o
        JOIN default.demand_orders_status dos ON o.order_id = dos.order_id
        GROUP BY date, o.sku_id
    )

    SELECT
        formatDateTime(c.date, '%Y-%m-%d') AS day,
        s.sku_id,
        s.sku,
        s.price,
        COALESCE(sd.qty, 0) AS qty
    FROM calendar c
    CROSS JOIN sku_list s
    LEFT JOIN sales_data sd ON c.date = sd.date AND s.sku_id = sd.sku_id
    ORDER BY s.sku_id, c.date
)
SELECT
  day,
  s.sku_id as sku_id,
  s.sku as sku,
  s.price as price,
  qty
FROM
  main;
