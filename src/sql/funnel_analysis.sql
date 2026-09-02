-- ========================================================
-- funnel_analysis.sql
-- 业务目标: 统计全链路转化漏斗 (浏览 -> 收藏/加购 -> 购买) 各环节人数与行为量
-- ========================================================

SELECT
    -- 1. 行为总量维度 (Total Behavior Counts)
    SUM(CASE WHEN behavior_type = 1 THEN 1 ELSE 0 END) AS total_page_views,      -- 总浏览次数 (PV)
    SUM(CASE WHEN behavior_type = 2 THEN 1 ELSE 0 END) AS total_favorites,       -- 总收藏次数 (Fav)
    SUM(CASE WHEN behavior_type = 3 THEN 1 ELSE 0 END) AS total_cart_additions,  -- 总加购次数 (Cart)
    SUM(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS total_purchases,       -- 总购买次数 (Buy)

    -- 2. 独立用户人数维度 (Unique User Counts per Stage)
    -- 第一层: 浏览过商品的独立用户数 (Page View Users)
    COUNT(DISTINCT CASE WHEN behavior_type = 1 THEN user_id END) AS pv_unique_users,

    -- 第二层: 产生过购买意向的独立用户数 (收藏或加购过的人数 / Intent Users)
    COUNT(DISTINCT CASE WHEN behavior_type IN (2, 3) THEN user_id END) AS interest_unique_users,

    -- 第三层: 最终完成下单购买的独立用户数 (Buyer Users)
    COUNT(DISTINCT CASE WHEN behavior_type = 4 THEN user_id END) AS buy_unique_users

FROM user_behavior;