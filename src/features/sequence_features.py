"""用户时序、转化与业务偏好特征提取模块

提取：
1. 相邻点击时间差 (平均决策思考间隔);
2. 浏览到加购、加购到购买转化率;
3. 晚间黄金时段活跃得分与周末活跃占比;
4. 重复比价强度.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def extract_sequence_features() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parquet_path = (
        project_root / "data" / "processed" / "user_behavior_clean.parquet"
    )
    output_seq_path = (
        project_root / "data" / "processed" / "user_sequence_features.parquet"
    )

    print("开始提取用户时序与行为流转特征...")
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H")

    # 1. 按用户和时间排序，计算相邻点击的时间差 (分钟)
    print("正在计算用户相邻操作时间间隔...")
    df = df.sort_values(by=["user_id", "datetime"])
    df["prev_time"] = df.groupby("user_id")["datetime"].shift(1)
    df["time_diff_min"] = (
        df["datetime"] - df["prev_time"]
    ).dt.total_seconds() / 60.0

    # 2. 用户级时序统计
    user_time_stats = (
        df.groupby("user_id")
        .agg(
            avg_time_interval_min=("time_diff_min", "mean"),
            std_time_interval_min=("time_diff_min", "std"),
            active_days=("date", "nunique"),
        )
        .reset_index()
    )
    user_time_stats["avg_time_interval_min"] = user_time_stats[
        "avg_time_interval_min"
    ].fillna(0)
    user_time_stats["std_time_interval_min"] = user_time_stats[
        "std_time_interval_min"
    ].fillna(0)

    # 3. 提取时段画像：晚间活跃得分 (20-23点占比) 与周末活跃占比
    print("正在提取用户时段画像特征...")
    df["is_night"] = df["hour"].between(20, 23).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    user_persona = (
        df.groupby("user_id")
        .agg(
            total_actions=("behavior_type", "count"),
            night_actions=("is_night", "sum"),
            weekend_actions=("is_weekend", "sum"),
            unique_items=("item_id", "nunique"),
        )
        .reset_index()
    )

    user_persona["night_owl_score"] = (
        user_persona["night_actions"] / user_persona["total_actions"]
    ).round(4)
    user_persona["weekend_ratio"] = (
        user_persona["weekend_actions"] / user_persona["total_actions"]
    ).round(4)

    # 比价强度：总交互数 / 独立商品数 (值越大说明反复看相同商品的频次越高)
    user_persona["comparison_intensity"] = (
        user_persona["total_actions"] / user_persona["unique_items"]
    ).round(2)

    # 4. 提取行为流转转化率 (PV -> Cart, Cart -> Buy)
    print("正在计算行为流转转化率...")
    user_behavior_counts = (
        df.groupby("user_id")
        .agg(
            pv_count=("behavior_type", lambda x: (x == 1).sum()),
            cart_count=("behavior_type", lambda x: (x == 3).sum()),
            buy_count=("behavior_type", lambda x: (x == 4).sum()),
        )
        .reset_index()
    )

    user_behavior_counts["pv_to_cart_rate"] = (
        user_behavior_counts["cart_count"]
        / (user_behavior_counts["pv_count"] + 1)
    ).round(4)
    user_behavior_counts["cart_to_buy_rate"] = (
        user_behavior_counts["buy_count"]
        / (user_behavior_counts["cart_count"] + 1)
    ).round(4)

    # 5. 合并全部时序与业务特征
    print("正在合并特征表...")
    features = user_time_stats.merge(
        user_persona[
            [
                "user_id",
                "night_owl_score",
                "weekend_ratio",
                "comparison_intensity",
            ]
        ],
        on="user_id",
        how="left",
    ).merge(
        user_behavior_counts[
            ["user_id", "pv_to_cart_rate", "cart_to_buy_rate"]
        ],
        on="user_id",
        how="left",
    )

    features.to_parquet(output_seq_path, index=False, engine="pyarrow")
    print(f"时序与偏好特征表已保存至: {output_seq_path.name}")
    print(f"共提取 {len(features):,} 名用户的 {len(features.columns)-1} 个衍生特征。")


if __name__ == "__main__":
    extract_sequence_features()