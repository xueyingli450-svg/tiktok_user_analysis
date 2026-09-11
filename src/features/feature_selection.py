"""特征分析与相关性检验模块

计算特征大宽表中各特征之间的皮尔逊相关系数，
排查多重共线性，并生成特征相关性热力图 (fig9)。
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils.db_connector import get_project_root

# 配置绘图字体与风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(
    style="white",
    font="SimHei",
    rc={"font.sans-serif": ["SimHei", "Microsoft YaHei"]},
)


def analyze_feature_correlation() -> None:
    project_root = get_project_root()
    matrix_path = (
        project_root / "data" / "processed" / "user_feature_matrix.parquet"
    )
    figures_dir = project_root / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("开始分析特征矩阵相关性与特征质量...")
    df = pd.read_parquet(matrix_path)

    # 1. 选取核心业务特征（排除 user_id 和文本类别列，挑选最具业务解释性的特征绘制热力图）
    core_features = [
        "frequency",
        "pv_count",
        "fav_count",
        "cart_count",
        "buy_count",
        "recency_days",
        "monetary_score",
        "avg_time_interval_min",
        "active_days",
        "night_owl_score",
        "weekend_ratio",
        "comparison_intensity",
        "pv_to_cart_rate",
        "cart_to_buy_rate",
        "user_avg_category_conversion",
        "svd_dim_0",
        "svd_dim_1",
    ]

    # 过滤存在于表中的列
    selected_cols = [c for c in core_features if c in df.columns]
    df_core = df[selected_cols]

    # 2. 计算皮尔逊相关系数矩阵
    print("正在计算皮尔逊相关系数矩阵...")
    corr_matrix = df_core.corr()

    # 3. 绘制相关性热力图 (fig9)
    print("正在生成特征相关性热力图...")
    plt.figure(figsize=(14, 11))

    # 生成上三角掩码，避免重复显示对称区域
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "皮尔逊相关系数 (r)"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
    )

    plt.title(
        "抖音商城核心衍生特征相关性矩阵热力图",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    fig9_path = figures_dir / "fig9_feature_correlation.png"
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    print(f"图表已保存至: {fig9_path.name}")

    # 4. 打印高相关特征对 (相关系数 > 0.85 的潜在冗余特征)
    print("\n高相关性特征排查 (r > 0.85):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feat_a = corr_matrix.columns[i]
            feat_b = corr_matrix.columns[j]
            r_val = corr_matrix.iloc[i, j]
            if abs(r_val) > 0.85:
                high_corr_pairs.append((feat_a, feat_b, r_val))
                print(f"- {feat_a} 与 {feat_b}: r = {r_val:.4f}")

    if not high_corr_pairs:
        print("- 未发现严重共线性冗余特征对，特征质量良好。")


if __name__ == "__main__":
    analyze_feature_correlation()