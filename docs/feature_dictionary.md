# 抖音商城用户行为分析 · 特征工程字典

- **数据源**: `data/processed/user_feature_matrix.parquet`
- **样本量**: 9,746 名独立用户
- **特征总数**: 34 个衍生特征 (不含 user_id 主键)
- **更新日期**: 第二周

---

## 一、 用户基础行为与 RFM 价值分群特征 (8 个)

| 字段名称 | 字段类型 | 计算逻辑与定义说明 |
| :--- | :--- | :--- |
| `frequency` | int64 | 用户在观测周期内的总交互行为次数 (浏览+收藏+加购+购买) |
| `pv_count` | int64 | 用户的商品浏览 (PV) 总次数 |
| `fav_count` | int64 | 用户的商品收藏 (Fav) 总次数 |
| `cart_count` | int64 | 用户的商品加购物车 (Cart) 总次数 |
| `buy_count` | int64 | 用户的商品最终购买下单 (Buy) 总次数 |
| `recency_days` | float64 | 用户最后一次活跃时间距离数据截止时间的天数间隔 (R 值，越小越近期) |
| `monetary_score` | int64 | 用户加权偏好总分 (购买×4 + 加购×3 + 收藏×2 + 浏览×1) |
| `user_segment` | string | RFM 四象限聚类结果：高价值用户、潜力用户、流失预警用户、沉睡用户 |

---

## 二、 用户时序、流转与业务偏好特征 (8 个)

| 字段名称 | 字段类型 | 计算逻辑与定义说明 |
| :--- | :--- | :--- |
| `avg_time_interval_min` | float64 | 用户相邻两次点击操作的平均时间间隔 (分钟)，衡量决策节奏 |
| `std_time_interval_min` | float64 | 用户相邻点击时间间隔的标准差，衡量活跃时间的离散度 |
| `active_days` | int64 | 用户在观测周期内活跃的不同自然天数 (活跃天数) |
| `night_owl_score` | float64 | 晚间黄金时段 (20:00-23:00) 行为数占该用户总行为数的比例 |
| `weekend_ratio` | float64 | 周末 (周六与周日) 行为数占该用户总行为数的比例 |
| `comparison_intensity` | float64 | 高频比价强度得分 (总交互次数 / 独立浏览商品数) |
| `pv_to_cart_rate` | float64 | 浏览到加购的流转转化率 (加购次数 / 浏览次数+1) |
| `cart_to_buy_rate` | float64 | 加购到购买的流转转化率 (购买次数 / 加购次数+1) |

---

## 三、 SVD 潜在语义降维隐式特征 (16 个)

| 字段名称 | 字段类型 | 计算逻辑与定义说明 |
| :--- | :--- | :--- |
| `svd_dim_0` ~ `svd_dim_15` | float32 | 基于“用户-商品”加权交互稀疏矩阵，通过 TruncatedSVD 截断奇异值分解提取出的 16 维用户隐式偏好向量 (User Latent Embedding) |

---

## 四、 商品类目目标编码特征 (2 个)

| 字段名称 | 字段类型 | 计算逻辑与定义说明 |
| :--- | :--- | :--- |
| `user_avg_category_conversion` | float64 | 用户所接触的所有商品类目的历史平均购买转化率均值 (Target Encoding 平滑均值) |
| `user_max_category_conversion` | float64 | 用户所接触的类目中历史购买转化率最高的值 |