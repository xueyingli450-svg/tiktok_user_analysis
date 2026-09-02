-- ========================================================
-- time_distribution.sql
-- 业务目标: 统计 24 小时活跃时段分布与每日日活 (DAU) 趋势
-- ========================================================

-- --------------------------------------------------------
-- 查询 1: 统计 24 小时各时段的用户活跃度与下单量 (0点 ~ 23点)
-- --------------------------------------------------------
SELECT
    hour,                                                       -- 一天中的第几点 (0 到 23)
    COUNT(*) AS total_interactions,                             -- 该小时内的总互动量 (浏览/收藏/加购/购买)
    COUNT(DISTINCT user_id) AS active_unique_visitors,          -- 该小时内活跃的独立访客人数 (Unique Visitor)
    SUM(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS total_purchases -- 该小时内的购买下单总次数
FROM user_behavior
GROUP BY hour
ORDER BY hour ASC;

-- --------------------------------------------------------
-- 查询 2: 统计每日日活跃用户数 (Daily Active User / DAU) 与每日成交量
-- --------------------------------------------------------
SELECT
    date,                                                       -- 具体日期 (YYYY-MM-DD)
    COUNT(DISTINCT user_id) AS daily_active_users,              -- 每日独立活跃用户数 (DAU)
    COUNT(*) AS daily_total_interactions,                       -- 每日总交互次数 (包含浏览/收藏/加购)
    SUM(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS daily_purchases -- 每日成交下单总次数
FROM user_behavior
GROUP BY date
ORDER BY date ASC;