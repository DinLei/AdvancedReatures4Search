create table if not exists ai_tmp.nlp_products_time_series_dl (
    product_id      bigint,
    category_id     int,
    ai_date         string,
    view_count      decimal(11, 5),
    wish_count      decimal(11, 5),
    cart_count      decimal(11, 5),
    order_count     decimal(11, 5)
)
row format delimited
fields terminated by ',';


SELECT
product_id,
category_id,
dctime AS ai_date,
sum(view_count) AS click_uv,
sum(wish_count) AS wish_uv,
sum(cart_count) AS cart_uv,
sum(order_count) AS order_num
FROM ai.nlp_category_action_static_daily_nc
GROUP BY dctime, category_id, product_id
ORDER BY product_id, category_id, dctime