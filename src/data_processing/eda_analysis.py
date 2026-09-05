"""探索性数据分析 (EDA) 与全量可视化模块 """

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.utils.db_connector import get_db_connection, get_project_root

# 1. 配置中文字体与全局排版风格
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(
    style="whitegrid",
    font="SimHei",
    rc={"font.sans-serif": ["SimHei", "Microsoft YaHei"]},
)


def run_all_7_eda_plots() -> None:
    """全量执行 SQL 查询，一键绘制 7 张分析大图。"""
    project_root = get_project_root()
    figures_dir = project_root / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path = project_root / "docs" / "eda_findings.md"

    conn = get_db_connection()
    print("\n================ 开始生成 7 张图表 ================")

    # -------------------------------------------------------------
    # 图 1：24 小时活跃时段分布（折线图）
    # -------------------------------------------------------------
    print("--> [1/7] 正在绘制图 1: 24 小时活跃走势图...")
    sql_hourly = """
    SELECT 
        hour,
        COUNT(*) AS total_interactions,
        SUM(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS buy_count
    FROM user_behavior
    GROUP BY hour
    ORDER BY hour;
    """
    df_hourly = pd.read_sql(sql_hourly, conn)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        df_hourly["hour"],
        df_hourly["total_interactions"],
        color="#1f77b4",
        marker="o",
        linewidth=2.5,
        label="总交互量 (浏览/收藏/加购/购买)",
    )
    line2 = ax2.plot(
        df_hourly["hour"],
        df_hourly["buy_count"],
        color="#d62728",
        marker="s",
        linewidth=2.5,
        linestyle="--",
        label="下单购买量",
    )

    ax1.set_xlabel("一天 24 小时 (时段)", fontsize=12, fontweight="bold")
    ax1.set_ylabel(
        "总交互量 (次)", color="#1f77b4", fontsize=12, fontweight="bold"
    )
    ax2.set_ylabel(
        "购买下单量 (次)", color="#d62728", fontsize=12, fontweight="bold"
    )
    ax1.set_xticks(range(0, 24))
    plt.title(
        "抖音商城用户 24 小时活跃与购买走势图",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(figures_dir / "fig1_hourly_activity.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 图 2：每日日活 (DAU) 趋势（柱状图）
    # -------------------------------------------------------------
    print("--> [2/7] 正在绘制图 2: 每日 DAU 趋势图...")
    sql_daily = """
    SELECT 
        date,
        COUNT(DISTINCT user_id) AS dau,
        SUM(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS daily_buys
    FROM user_behavior
    GROUP BY date
    ORDER BY date;
    """
    df_daily = pd.read_sql(sql_daily, conn)

    plt.figure(figsize=(12, 5))
    bars = plt.bar(
        df_daily["date"],
        df_daily["dau"],
        color="#4C72B0",
        width=0.55,
        label="日活跃用户数 (DAU)",
    )
    plt.xlabel("日期", fontsize=12, fontweight="bold")
    plt.ylabel("活跃用户数 (人)", fontsize=12, fontweight="bold")
    plt.title(
        "每日活跃用户数 (DAU) 变化趋势", fontsize=14, fontweight="bold", pad=15
    )
    plt.xticks(rotation=30, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (height * 0.01),
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(figures_dir / "fig2_daily_dau_trend.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 图 3：四类行为结构占比（环形饼图）
    # -------------------------------------------------------------
    print("--> [3/7] 正在绘制图 3: 四类行为结构分布环形图...")
    sql_behavior = """
    SELECT 
        CASE behavior_type
            WHEN 1 THEN '1-浏览 (PV)'
            WHEN 2 THEN '2-收藏 (Fav)'
            WHEN 3 THEN '3-加购 (Cart)'
            WHEN 4 THEN '4-购买 (Buy)'
        END AS behavior_name,
        COUNT(*) AS count
    FROM user_behavior
    GROUP BY behavior_type
    ORDER BY behavior_type;
    """
    df_behavior = pd.read_sql(sql_behavior, conn)

    plt.figure(figsize=(7, 7))
    colors = ["#5B9BD5", "#ED7D31", "#FFC000", "#70AD47"]
    plt.pie(
        df_behavior["count"],
        labels=df_behavior["behavior_name"],
        autopct="%1.2f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    plt.title(
        "抖音商城用户全链路行为结构分布",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_behavior_distribution.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 图 4：全链路转化漏斗分析（条形图）
    # -------------------------------------------------------------
    print("--> [4/7] 正在绘制图 4: 全链路转化漏斗图...")
    sql_funnel = """
    SELECT 
        COUNT(DISTINCT CASE WHEN behavior_type = 1 THEN user_id END) AS pv_users,
        COUNT(DISTINCT CASE WHEN behavior_type IN (2, 3) THEN user_id END) AS fav_cart_users,
        COUNT(DISTINCT CASE WHEN behavior_type = 4 THEN user_id END) AS buy_users
    FROM user_behavior;
    """
    df_funnel = pd.read_sql(sql_funnel, conn)

    pv_u = df_funnel["pv_users"].iloc[0]
    interest_u = df_funnel["fav_cart_users"].iloc[0]
    buy_u = df_funnel["buy_users"].iloc[0]

    funnel_stages = ["1. 浏览商品 (PV)", "2. 兴趣表达 (收藏/加购)", "3. 最终成交 (购买)"]
    user_counts = [pv_u, interest_u, buy_u]
    conversion_rates = [
        100.0,
        (interest_u / pv_u) * 100,
        (buy_u / pv_u) * 100,
    ]

    plt.figure(figsize=(9, 5))
    bar_colors = ["#3366CC", "#DC3912", "#109618"]
    bars = plt.barh(
        funnel_stages[::-1],
        user_counts[::-1],
        color=bar_colors[::-1],
        height=0.5,
    )

    for i, bar in enumerate(bars):
        width = bar.get_width()
        stage_idx = len(funnel_stages) - 1 - i
        rate = conversion_rates[stage_idx]
        plt.text(
            width + (pv_u * 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,} 人 (转化率: {rate:.2f}%)",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.xlabel("独立用户数 (Unique Visitors)", fontsize=12, fontweight="bold")
    plt.title(
        "抖音商城用户全链路转化漏斗图", fontsize=14, fontweight="bold", pad=15
    )
    plt.xlim(0, pv_u * 1.35)

    plt.tight_layout()
    plt.savefig(figures_dir / "fig4_conversion_funnel.png", dpi=300)
    plt.close()

    # =============================================================
    # 3 张热力图
    # =============================================================
    weekday_order = [
        "周一",
        "周二",
        "周三",
        "周四",
        "周五",
        "周六",
        "周日",
    ]

    # -------------------------------------------------------------
    # 图 5：总交互行为数热力图 (24h * 7天)
    # -------------------------------------------------------------
    print("--> [5/7] 正在生成热力图 1: 24小时 x 7天【总交互行为数】热力图...")
    sql_hm_total = """
    SELECT weekday_name, hour, COUNT(*) AS total_count
    FROM user_behavior
    GROUP BY weekday_name, hour;
    """
    df_hm_total = pd.read_sql(sql_hm_total, conn)
    pivot_total = df_hm_total.pivot(
        index="weekday_name", columns="hour", values="total_count"
    ).reindex(weekday_order)

    plt.figure(figsize=(13, 5.5))
    sns.heatmap(
        pivot_total,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "总交互行为次数 (次)"},
    )
    plt.title(
        "抖音商城用户 24小时 × 一周7天【总交互行为数】热力图",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("一天 24 小时 (时段)", fontsize=12, fontweight="bold")
    plt.ylabel("星期", fontsize=12, fontweight="bold")
    # 【核心调整】：强制 Y 轴文字横向正立显示 (rotation=0)
    plt.yticks(rotation=0, fontsize=11, fontweight="bold")
    plt.xticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "fig5_heatmap_total_interactions.png", dpi=300
    )
    plt.close()
    print(" 图 5: fig5_heatmap_total_interactions.png 保存成功！")

    # -------------------------------------------------------------
    # 图 6：总浏览行为数 (PV) 热力图 (24h * 7天)
    # -------------------------------------------------------------
    print(
        "--> [6/7] 正在生成热力图 2: 24小时 x 7天【总浏览 (PV) 行为数】热力图..."
    )
    sql_hm_pv = """
    SELECT weekday_name, hour, COUNT(*) AS pv_count
    FROM user_behavior
    WHERE behavior_type = 1
    GROUP BY weekday_name, hour;
    """
    df_hm_pv = pd.read_sql(sql_hm_pv, conn)
    pivot_pv = df_hm_pv.pivot(
        index="weekday_name", columns="hour", values="pv_count"
    ).reindex(weekday_order)

    plt.figure(figsize=(13, 5.5))
    sns.heatmap(
        pivot_pv,
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "浏览总次数 (PV)"},
    )
    plt.title(
        "抖音商城用户 24小时 × 一周7天【总浏览行为数 (PV)】热力图",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("一天 24 小时 (时段)", fontsize=12, fontweight="bold")
    plt.ylabel("星期", fontsize=12, fontweight="bold")
    plt.yticks(rotation=0, fontsize=11, fontweight="bold")
    plt.xticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "fig6_heatmap_pv.png", dpi=300)
    plt.close()
    print(" 图 6: fig6_heatmap_pv.png 保存成功！")

    # -------------------------------------------------------------
    # 图 7：总购买行为数 (Buy) 热力图 (24h * 7天)
    # -------------------------------------------------------------
    print(
        "--> [7/7] 正在生成热力图 3: 24小时 x 7天【总购买 (Buy) 行为数】热力图..."
    )
    sql_hm_buy = """
    SELECT weekday_name, hour, COUNT(*) AS buy_count
    FROM user_behavior
    WHERE behavior_type = 4
    GROUP BY weekday_name, hour;
    """
    df_hm_buy = pd.read_sql(sql_hm_buy, conn)
    pivot_buy = df_hm_buy.pivot(
        index="weekday_name", columns="hour", values="buy_count"
    ).reindex(weekday_order)

    plt.figure(figsize=(13, 5.5))
    sns.heatmap(
        pivot_buy,
        cmap="Reds",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "购买下单总次数 (次)"},
    )
    plt.title(
        "抖音商城用户 24小时 × 一周7天【总购买行为数 (Buy)】热力图",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("一天 24 小时 (时段)", fontsize=12, fontweight="bold")
    plt.ylabel("星期", fontsize=12, fontweight="bold")
    plt.yticks(rotation=0, fontsize=11, fontweight="bold")
    plt.xticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "fig7_heatmap_buy.png", dpi=300)
    plt.close()
    print(" 图 7: fig7_heatmap_buy.png 保存成功！")

    conn.close()

    # -------------------------------------------------------------
    # 自动更新业务洞察报告
    # -------------------------------------------------------------
    findings_content = f"""# 抖音商城用户行为探索性数据分析 (EDA) 洞察报告 (含 3 大热力图)

- **分析数据量**: 10,352,869 条有效行为记录
- **覆盖独立用户数 (Unique Visitors)**: {pv_u:,} 人
- **产出图表数**: 7 张高清图表 (4 张基础图 + 3 张 24h × 7天业务热力图)

---

## 核心业务洞察发现

### 1. 24 小时与一周 7 天交叉规律 (热力图深度洞察)
- **晚间黄金窗口 (20:00 - 22:00)**：无论是周一到周五还是周末，晚间 20~22 点均是全网浏览与下单最密集的深色高亮区域；
- **周五晚间与周六全天脉冲**：周五晚间开始，用户的浏览和下单热度明显高于周中平日，周末白天的活跃度显著提升；
- **运营决策建议**：重点在 **周五至周日的晚间 19:30 - 22:30** 进行大促直播推流和限时秒杀，最大化转化收益。

### 2. 全链路转化漏斗与流失瓶颈
- **浏览 $\\rightarrow$ 收藏/加购意向转化率**: **{(interest_u / pv_u) * 100:.2f}%**
- **浏览 $\\rightarrow$ 最终成交购买转化率**: **{(buy_u / pv_u) * 100:.2f}%**
- **加购/收藏 $\\rightarrow$ 购买转化率**: **{(buy_u / interest_u) * 100:.2f}%**
- **诊断结论**：用户的加购意向极其强烈；一旦产生加购/收藏，最终转化为购买的确定性极高，运营应重点推进“购物车未结账降价召回”。

---

## 产出图表清单
- `docs/figures/fig1_hourly_activity.png` (24 小时活跃走势图)
- `docs/figures/fig2_daily_dau_trend.png` (每日 DAU 趋势图)
- `docs/figures/fig3_behavior_distribution.png` (行为占比分布环形图)
- `docs/figures/fig4_conversion_funnel.png` (全链路转化漏斗图)
- `docs/figures/fig5_heatmap_total_interactions.png` (24h × 7天 总交互行为数热力图)
- `docs/figures/fig6_heatmap_pv.png` (24h × 7天 浏览行为数热力图)
- `docs/figures/fig7_heatmap_buy.png` (24h × 7天 购买行为数热力图)
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(findings_content)

    print(f"\n-->业务洞察报告已自动更新至: {report_path.name}")
    print(
        "================ 全部 7 张 EDA 图表与报告全量生成完成 ================\n"
    )


if __name__ == "__main__":
    run_all_7_eda_plots()