"""用户-商品交互矩阵 SVD 降维特征提取模块

构建用户与商品的加权交互稀疏矩阵，通过截断奇异值分解 (TruncatedSVD)，
为每位用户提取 16 维隐式语义偏好向量 (User Latent Embedding)。
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def extract_svd_features() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parquet_path = (
        project_root / "data" / "processed" / "user_behavior_clean.parquet"
    )
    output_svd_path = (
        project_root / "data" / "processed" / "user_svd_features.parquet"
    )

    print("开始构建用户-商品交互矩阵并提取 SVD 隐式特征...")
    df = pd.read_parquet(parquet_path)

    # 1. 给不同行为赋予权重 (购买4分, 加购3分, 收藏2分, 浏览1分)
    weight_map = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    df["action_weight"] = df["behavior_type"].map(weight_map)

    # 2. 按用户和商品聚合总权重
    print("正在聚合用户-商品交互权重...")
    user_item_weights = (
        df.groupby(["user_id", "item_id"])["action_weight"]
        .sum()
        .reset_index()
    )

    # 3. 将 user_id 和 item_id 映射为连续索引，便于构建稀疏矩阵
    unique_users = user_item_weights["user_id"].unique()
    unique_items = user_item_weights["item_id"].unique()

    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    item_to_idx = {iid: i for i, iid in enumerate(unique_items)}

    row_indices = user_item_weights["user_id"].map(user_to_idx)
    col_indices = user_item_weights["item_id"].map(item_to_idx)
    values = user_item_weights["action_weight"].values

    # 4. 构建 CSR 压缩稀疏矩阵 (避免大内存占用)
    print(
        f"构建稀疏矩阵: {len(unique_users):,} 用户 × {len(unique_items):,} 商品..."
    )
    interaction_matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(unique_users), len(unique_items)),
        dtype=np.float32,
    )

    # 5. 执行 TruncatedSVD 降维分解 (提取 16 维隐式向量)
    print("正在执行 SVD 奇异值分解 (降维至 16 维)...")
    n_components = 16
    svd = TruncatedSVD(
        n_components=n_components, algorithm="randomized", random_state=42
    )
    user_embeddings = svd.fit_transform(interaction_matrix)

    explained_variance_ratio = svd.explained_variance_ratio_.sum()
    print(f"SVD 16 维累计解释方差比例: {explained_variance_ratio * 100:.2f}%")

    # 6. 整理为 DataFrame 并导出
    svd_column_names = [f"svd_dim_{i}" for i in range(n_components)]
    svd_df = pd.DataFrame(user_embeddings, columns=svd_column_names)
    svd_df.insert(0, "user_id", unique_users)

    # 保留 4 位小数以压缩体积
    svd_df[svd_column_names] = svd_df[svd_column_names].round(4)

    svd_df.to_parquet(output_svd_path, index=False, engine="pyarrow")
    print(f"SVD 隐式特征表已保存至: {output_svd_path.name}")
    print(
        f"共提取 {len(svd_df):,} 名用户的 {n_components} 维隐式特征向量。"
    )


if __name__ == "__main__":
    extract_svd_features()