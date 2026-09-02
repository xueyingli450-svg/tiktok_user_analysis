-- ========================================================
-- basic_metrics.sql
-- 业务目标: 统计抖音商城宏观大盘流量与用户活跃基础指标
-- ========================================================

SELECT
    -- 1. 独立访客总数 (Unique Visitor / UV): 按人头去重统计
    COUNT(DISTINCT user_id) AS total_unique_visitors,

    -- 2. 覆盖商品总数: 去重统计有多少个不同的商品
    COUNT(DISTINCT item_id) AS total_unique_items,

    -- 3. 覆盖商品类目总数: 去重统计涉及多少个品类
    COUNT(DISTINCT item_category) AS total_unique_categories,

    -- 4. 平台总交互行为数 (Total Behaviors): 包含浏览/收藏/加购/购买的总和
    COUNT(*) AS total_interactions,

    -- 5. 页面浏览总次数 (Page View / PV): behavior_type = 1 的总次数
    SUM(CASE WHEN behavior_type = 1 THEN 1 ELSE 0 END) AS total_page_views,

    -- 6. 最终下单购买总次数 (Total Purchases): behavior_type = 4 的总次数
    SUM(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS total_purchases,

    -- 7. 人均浏览频次 (Average Page Views per Unique Visitor): PV / UV
    ROUND(
        CAST(SUM(CASE WHEN behavior_type = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(DISTINCT user_id),
        2
    ) AS avg_page_views_per_user

FROM user_behavior;