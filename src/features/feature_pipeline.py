"""特征工程总装流水线模块

整合周一至周四的所有特征：
1. RFM 用户价值分群特征
2. 时序与业务偏好特征
3. SVD 16 维隐式语义特征
4. 高维类目 Target Encoding (目标编码) 特征
并输出为标准的特征矩阵大宽表 (user_feature_matrix.parquet)。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_feature_pipeline() -> None:
    project_root = Path(__file__).resolve().parents[2]
    clean_data_path = (
        project_root / "data" / "processed" / "user_behavior_clean.parquet"
    )
    rfm_path = project_root / "data" / "processed" / "user_rfm_features.parquet"
    seq_path = (
        project_root / "data" / "processed" / "user_sequence_features.parquet"
    )
    svd_path = project_root / "data" / "processed" / "user_svd_features.parquet"
    output_matrix_path = (
        project_root / "data" / "processed" / "user_feature_matrix.parquet"
    )

    print("开始整合全维度特征大宽表...")

    # 1. 计算类目级 Target Encoding (各类目的平均购买转化率)
    print("正在计算商品类目的 Target Encoding 目标编码...")
    df_raw = pd.read_parquet(clean_data_path)

    # 统计每个类目的总互动数和购买数
    category_stats = (
        df_raw.groupby("item_category")
        .agg(
            total_actions=("behavior_type", "count"),
            buy_actions=("behavior_type", lambda x: (x == 4).sum()),
        )
        .reset_index()
    )

    # 计算平滑转化率 (添加平滑项防止小样本类目产生极端值)
    global_mean_buy = (df_raw["behavior_type"] == 4).mean()
    smoothing = 10
    category_stats["category_target_encoded"] = (
        category_stats["buy_actions"] + smoothing * global_mean_buy
    ) / (category_stats["total_actions"] + smoothing)

    # 将类目转化率映射回交互记录，计算每个用户接触类目的平均转化率
    df_with_cat = df_raw.merge(
        category_stats[["item_category", "category_target_encoded"]],
        on="item_category",
        how="left",
    )
    user_cat_features = (
        df_with_cat.groupby("user_id")["category_target_encoded"]
        .agg(
            user_avg_category_conversion="mean",
            user_max_category_conversion="max",
        )
        .reset_index()
    )

    # 2. 读取已生成的各模块特征表
    print("正在加载 RFM、时序与 SVD 特征表...")
    df_rfm = pd.read_parquet(rfm_path)
    df_seq = pd.read_parquet(seq_path)
    df_svd = pd.read_parquet(svd_path)

    # 3. 关联并融合成特征大宽表
    print("正在合并特征大宽表...")
    feature_matrix = df_rfm.merge(df_seq, on="user_id", how="inner")
    feature_matrix = feature_matrix.merge(df_svd, on="user_id", how="inner")
    feature_matrix = feature_matrix.merge(
        user_cat_features, on="user_id", how="left"
    )

    # 4. 剔除辅助列，只保留纯特征列
    drop_cols = ["last_time"]  # 剔除具体时间戳
    feature_matrix = feature_matrix.drop(
        columns=[c for c in drop_cols if c in feature_matrix.columns]
    )

    # 5. 填补缺失值并导出
    feature_matrix = feature_matrix.fillna(0)
    feature_matrix.to_parquet(
        output_matrix_path, index=False, engine="pyarrow"
    )

    print("\n特征大宽表构建完成:")
    print(f"- 文件保存路径: {output_matrix_path.name}")
    print(f"- 覆盖独立用户数: {len(feature_matrix):,} 人")
    print(
        f"- 总特征维度数: {len(feature_matrix.columns) - 1} 个 (不含 user_id)"
    )
    print("\n特征字段列表:")
    print(list(feature_matrix.columns))


if __name__ == "__main__":
    build_feature_pipeline()