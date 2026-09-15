"""建模数据集构建与样本切分模块

功能：
1. 划分特征观测期与未来 7 天预测期，提取二分类购买标签 (1/0);
2. 关联 34 维用户特征大宽表;
3. 采用分层抽样 (Stratified Split) 按 7:1:2 划分训练集、验证集与测试集;
4. 输出标准训练/验证/测试数据集 Parquet 文件。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_modeling_datasets() -> None:
    project_root = Path(__file__).resolve().parents[2]
    clean_data_path = (
        project_root / "data" / "processed" / "user_behavior_clean.parquet"
    )
    feature_matrix_path = (
        project_root / "data" / "processed" / "user_feature_matrix.parquet"
    )
    processed_dir = project_root / "data" / "processed"

    print("开始构建第三阶段算法建模数据集与标签...")

    # 1. 加载行为日志，划分时间窗口以提取未来 7 天的真实购买标签
    print("正在加载行为日志以划分时间窗口...")
    df_raw = pd.read_parquet(clean_data_path)

    # 确定标签预测窗口：以最后 7 天 (2025-12-12 到 2025-12-18) 作为预测目标
    split_date_str = "2025-12-12"
    print(f"预测目标窗口划分时间点: {split_date_str}")

    future_data = df_raw[df_raw["date"] >= split_date_str]

    # 提取在未来 7 天产生过购买行为 (behavior_type == 4) 的用户 ID 集合
    buyers_in_future = set(
        future_data[future_data["behavior_type"] == 4]["user_id"].unique()
    )
    print(f"未来 7 天产生真实购买行为的用户数: {len(buyers_in_future):,} 人")

    # 2. 读取特征大宽表并打上二分类标签
    print("正在加载特征大宽表并匹配标签...")
    feature_df = pd.read_parquet(feature_matrix_path)

    # 打标：后 7 天买过为 1，未买为 0
    feature_df["target_label"] = (
        feature_df["user_id"].isin(buyers_in_future).astype(int)
    )

    # 统计正负样本分布
    total_users = len(feature_df)
    pos_count = feature_df["target_label"].sum()
    neg_count = total_users - pos_count
    pos_ratio = (pos_count / total_users) * 100

    print("\n样本集标签分布统计:")
    print(f"- 正样本 (Label=1, 未来7天有购买): {pos_count:,} 人 ({pos_ratio:.2f}%)")
    print(
        f"- 负样本 (Label=0, 未来7天无购买): {neg_count:,} 人 ({100 - pos_ratio:.2f}%)"
    )
    print(f"- 类别不平衡比 (负/正): {neg_count / pos_count:.2f} : 1")

    # 3. 准备特征矩阵 X 与标签 y
    # 剔除 user_id 与文本列 user_segment
    exclude_cols = ["user_id", "user_segment", "target_label"]
    feature_cols = [c for c in feature_df.columns if c not in exclude_cols]

    X = feature_df[feature_cols]
    y = feature_df["target_label"]
    user_ids = feature_df["user_id"]

    print(f"\n用于模型输入的有效特征维度: {len(feature_cols)} 个")

    # 4. 执行分层抽样切分 (Train 70% / Val 10% / Test 20%)
    print("正在执行分层抽样切分 (70% 训练集 / 10% 验证集 / 20% 测试集)...")
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, user_ids, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp,
        y_temp,
        id_temp,
        test_size=(2 / 3),
        random_state=42,
        stratify=y_temp,
    )

    # 5. 组合并输出 Parquet 文件
    train_df = pd.concat([id_train, X_train, y_train], axis=1)
    val_df = pd.concat([id_val, X_val, y_val], axis=1)
    test_df = pd.concat([id_test, X_test, y_test], axis=1)

    train_path = processed_dir / "train_data.parquet"
    val_path = processed_dir / "val_data.parquet"
    test_path = processed_dir / "test_data.parquet"

    train_df.to_parquet(train_path, index=False, engine="pyarrow")
    val_df.to_parquet(val_path, index=False, engine="pyarrow")
    test_df.to_parquet(test_path, index=False, engine="pyarrow")

    print("\n数据集切分与导出完成:")
    print(
        f"- 训练集 (train_data.parquet): {len(train_df):,} 行 (正样本: {y_train.sum():,})"
    )
    print(
        f"- 验证集 (val_data.parquet)  : {len(val_df):,} 行 (正样本: {y_val.sum():,})"
    )
    print(
        f"- 测试集 (test_data.parquet) : {len(test_df):,} 行 (正样本: {y_test.sum():,})"
    )
    print("================ 样本构建任务全部完成 ================")


if __name__ == "__main__":
    build_modeling_datasets()