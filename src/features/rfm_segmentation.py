"""RFM 用户价值分群模块

计算用户的 R（最近交互天数间隔）、F（交互频次）、M（行为加权得分），
并将用户划分为四类群体，输出特征表与分布图。
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 配置绘图字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_rfm_features() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parquet_path = (
        project_root / "data" / "processed" / "user_behavior_clean.parquet"
    )
    output_rfm_path = (
        project_root / "data" / "processed" / "user_rfm_features.parquet"
    )
    figures_dir = project_root / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("开始计算 RFM 用户分群特征...")
    df = pd.read_parquet(parquet_path)

    # 确保时间列为 datetime 格式
    df["datetime"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H")
    max_datetime = df["datetime"].max()

    # 1. 计算每个用户的 R、F、各行为次数
    print("正在聚合用户基础行为数据...")
    user_summary = (
        df.groupby("user_id")
        .agg(
            last_time=("datetime", "max"),
            frequency=("behavior_type", "count"),
            pv_count=("behavior_type", lambda x: (x == 1).sum()),
            fav_count=("behavior_type", lambda x: (x == 2).sum()),
            cart_count=("behavior_type", lambda x: (x == 3).sum()),
            buy_count=("behavior_type", lambda x: (x == 4).sum()),
        )
        .reset_index()
    )

    # 2. 计算 R (最近活跃时间距离数据最大时间的天数差)
    user_summary["recency_days"] = (
        max_datetime - user_summary["last_time"]
    ).dt.total_seconds() / (24 * 3600)

    # 3. 计算 M (加权偏好得分：购买4分，加购3分，收藏2分，浏览1分)
    user_summary["monetary_score"] = (
        user_summary["buy_count"] * 4
        + user_summary["cart_count"] * 3
        + user_summary["fav_count"] * 2
        + user_summary["pv_count"] * 1
    )

    # 4. 根据中位数划分 R 和 F 得分 (R越大代表越久没来，得分越低)
    r_median = user_summary["recency_days"].median()
    f_median = user_summary["frequency"].median()

    def get_rfm_segment(row):
        r_score = 1 if row["recency_days"] <= r_median else 0  # 1代表近期活跃
        f_score = 1 if row["frequency"] >= f_median else 0  # 1代表高频

        if r_score == 1 and f_score == 1:
            return "高价值用户"
        elif r_score == 1 and f_score == 0:
            return "潜力用户"
        elif r_score == 0 and f_score == 1:
            return "流失预警用户"
        else:
            return "沉睡用户"

    user_summary["user_segment"] = user_summary.apply(
        get_rfm_segment, axis=1
    )

    # 5. 打印各分群统计结果
    segment_counts = user_summary["user_segment"].value_counts()
    print("\n用户分群统计结果:")
    for segment, count in segment_counts.items():
        pct = (count / len(user_summary)) * 100
        print(f"- {segment}: {count:,} 人 ({pct:.2f}%)")

    # 6. 保存特征表
    user_summary.to_parquet(output_rfm_path, index=False, engine="pyarrow")
    print(f"\nRFM 特征表已保存至: {output_rfm_path.name}")

    # 7. 绘制 RFM 分布散点图 (fig8)
    plt.figure(figsize=(10, 6))
    palette = {
        "高价值用户": "#d62728",
        "潜力用户": "#2ca02c",
        "流失预警用户": "#ff7f0e",
        "沉睡用户": "#7f7f7f",
    }
    sns.scatterplot(
        data=user_summary,
        x="recency_days",
        y="frequency",
        hue="user_segment",
        palette=palette,
        alpha=0.6,
        s=30,
    )
    plt.axvline(
        x=r_median, color="black", linestyle="--", alpha=0.5, label="R 中位数线"
    )
    plt.axhline(
        y=f_median, color="black", linestyle=":", alpha=0.5, label="F 中位数线"
    )
    plt.yscale("log")  # F 频次跨度大，采用对数坐标轴便于观察
    plt.title("RFM 用户价值四象限分群分布图", fontsize=14, pad=15)
    plt.xlabel("最近活跃间隔天数 (Recency / 天)", fontsize=11)
    plt.ylabel("总交互频次 (Frequency / 对数刻度)", fontsize=11)
    plt.legend(title="用户群体分类", loc="upper right")
    plt.tight_layout()

    fig8_path = figures_dir / "fig8_rfm_user_clusters.png"
    plt.savefig(fig8_path, dpi=300)
    plt.close()
    print(f"图表已保存至: {fig8_path.name}")


if __name__ == "__main__":
    calculate_rfm_features()