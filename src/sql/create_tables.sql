-- 1. 如果存在旧表先删除，确保每次运行都是全新干净的表（防重复）
DROP TABLE IF EXISTS user_behavior;

-- 2. 创建用户行为主表
CREATE TABLE user_behavior (
    user_id         INTEGER NOT NULL,   -- 用户唯一标识
    item_id         INTEGER NOT NULL,   -- 商品唯一标识
    item_category   INTEGER NOT NULL,   -- 商品类目标识
    behavior_type   INTEGER NOT NULL,   -- 行为类型: 1=pv, 2=fav, 3=cart, 4=buy
    time            TEXT NOT NULL,      -- 行为时间: YYYY-MM-DD HH
    date            TEXT NOT NULL,      -- 日期: YYYY-MM-DD
    hour            INTEGER NOT NULL    -- 小时: 0~23
);

-- 3. 创建 4 个核心索引
CREATE INDEX IF NOT EXISTS idx_user_time ON user_behavior (user_id, time);
CREATE INDEX IF NOT EXISTS idx_behavior_type ON user_behavior (behavior_type);
CREATE INDEX IF NOT EXISTS idx_date_hour ON user_behavior (date, hour);
CREATE INDEX IF NOT EXISTS idx_item_category ON user_behavior (item_id, item_category);