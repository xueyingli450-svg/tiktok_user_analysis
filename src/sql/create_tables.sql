-- 1. 每次建表前先删除旧表（保证幂等性与干净入库）
DROP TABLE IF EXISTS user_behavior;

-- 2. 创建用户行为主表（新增星期特征）
CREATE TABLE user_behavior (
    user_id         INTEGER NOT NULL,   -- 用户唯一标识
    item_id         INTEGER NOT NULL,   -- 商品唯一标识
    item_category   INTEGER NOT NULL,   -- 商品类目标识
    behavior_type   INTEGER NOT NULL,   -- 行为类型: 1=pv, 2=fav, 3=cart, 4=buy
    time            TEXT NOT NULL,      -- 行为时间: YYYY-MM-DD HH
    date            TEXT NOT NULL,      -- 日期: YYYY-MM-DD
    hour            INTEGER NOT NULL,   -- 小时: 0~23
    day_of_week     INTEGER NOT NULL,   -- 星期几数字: 0=周一, 6=周日
    weekday_name    TEXT NOT NULL       -- 星期几中文: 周一 ~ 周日
);

-- 3. 创建核心查询索引（新增热力图专用索引）
CREATE INDEX IF NOT EXISTS idx_user_time ON user_behavior (user_id, time);
CREATE INDEX IF NOT EXISTS idx_behavior_type ON user_behavior (behavior_type);
CREATE INDEX IF NOT EXISTS idx_date_hour ON user_behavior (date, hour);
CREATE INDEX IF NOT EXISTS idx_weekday_hour ON user_behavior (day_of_week, hour);
CREATE INDEX IF NOT EXISTS idx_item_category ON user_behavior (item_id, item_category);