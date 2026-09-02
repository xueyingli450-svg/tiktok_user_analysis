#除去空值（dropna）
#异常行为过滤（isin）
#时间有效性校验
#四元组去重（drop_duplicates)
#IQR 异常爬虫排查

from pathlib import Path
import pandas as pd

def clean_user_behavior_data() -> None:
    """读取原始用户行为数据，执行清洗、时间过滤与 IQR 异常排查。"""
    # 1. 相对路径定位
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = (
        project_root / "data" / "raw" / "user_behavior_processed.csv"
    )
    processed_dir = project_root / "data" / "processed"
    docs_dir = project_root / "docs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    output_parquet_path = processed_dir / "user_behavior_clean.parquet"
    report_md_path = docs_dir / "data_quality_report.md"

    print("================ 开始执行核心数据清洗 ================")
    print(f"--> 读取原始数据: {raw_data_path}")

    # 2. 读取原始数据
    df_raw = pd.read_csv(raw_data_path)
    total_raw_rows = len(df_raw)
    print(f"--> 原始数据加载完成，总行数: {total_raw_rows:,} 行")

    # 3. 工序 1：剔除缺失值 (NaN)
    df_step1 = df_raw.dropna().copy()
    null_dropped = total_raw_rows - len(df_step1)
    print(f"--> [工序1] 缺失值清洗完成，剔除缺失行: {null_dropped:,} 行")

    # 4. 工序 2：行为类型校验（严格限制为 1, 2, 3, 4）
    df_step1["behavior_type"] = pd.to_numeric(
        df_step1["behavior_type"], errors="coerce"
    )
    valid_behavior_mask = df_step1["behavior_type"].isin([1, 2, 3, 4])
    df_step2 = df_step1[valid_behavior_mask].copy()
    df_step2["behavior_type"] = df_step2["behavior_type"].astype("int8")
    invalid_behavior_dropped = len(df_step1) - len(df_step2)
    print(
        f"--> [工序2] 行为校验完成，剔除非法行为: {invalid_behavior_dropped:,} 行"
    )

    # 5. 工序 3：时间格式校验与合法性过滤
    df_step2["datetime"] = pd.to_datetime(
        df_step2["time"], format="%Y-%m-%d %H", errors="coerce"
    )
    df_step3 = df_step2.dropna(subset=["datetime"]).copy()
    df_step3["date"] = df_step3["datetime"].dt.date.astype(str)
    df_step3["hour"] = df_step3["datetime"].dt.hour.astype("int8")
    invalid_time_dropped = len(df_step2) - len(df_step3)
    print(
        f"--> [工序3] 时间解析完成，剔除异常时间: {invalid_time_dropped:,} 行"
    )

    # 6. 工序 4：业务四元组去重 (user_id, item_id, behavior_type, time)
    df_step4 = df_step3.drop_duplicates(
        subset=["user_id", "item_id", "behavior_type", "time"]
    ).copy()
    duplicate_dropped = len(df_step3) - len(df_step4)
    print(
        f"--> [工序4] 四元组去重完成，剔除重复记录: {duplicate_dropped:,} 行"
    )

    # 7. 工序 5：IQR 异常检测（识别并剔除极高频异常爬虫账号）
    print("--> [工序5] 正在执行 IQR 异常检测 (排查疑似爬虫账号)...")
    user_counts = df_step4["user_id"].value_counts()
    q1 = user_counts.quantile(0.25)
    q3 = user_counts.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr  # 设定极端异常阈值

    abnormal_users = user_counts[user_counts > upper_bound].index
    df_clean = df_step4[~df_step4["user_id"].isin(abnormal_users)].copy()
    bot_dropped = len(df_step4) - len(df_clean)
    print(
        f"    └─ IQR 阈值: {upper_bound:.1f} 次 | 识别异常用户: {len(abnormal_users):,} 个 | 剔除异常记录: {bot_dropped:,} 行"
    )

    # 8. 类型优化并导出为 Parquet 格式
    df_clean["user_id"] = df_clean["user_id"].astype("int32")
    df_clean["item_id"] = df_clean["item_id"].astype("int32")
    df_clean["item_category"] = df_clean["item_category"].astype("int32")

    print(f"\n--> 正在导出标准化 Parquet 数据至: {output_parquet_path}")
    df_clean.to_parquet(output_parquet_path, index=False, engine="pyarrow")
    print("-->Parquet 数据导出成功！")

    # 9. 自动生成《数据清洗质量报告》
    final_clean_rows = len(df_clean)
    total_dropped = total_raw_rows - final_clean_rows
    retention_rate = (final_clean_rows / total_raw_rows) * 100

    report_content = f"""# 抖音商城用户行为数据清洗与质量报告

- **报告生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **处理状态**:完成全流程清洗与 Parquet 转换

---

## 一、 数据清洗统计指标

| 统计项 | 数据量 (行) | 占比说明 |
| :--- | :--- | :--- |
| **原始总数据量** | {total_raw_rows:,} 行 | 100.00% |
| **清洗后有效数据量** | {final_clean_rows:,} 行 | **{retention_rate:.2f}% (数据留存率)** |
| **清洗过滤脏数据总量** | {total_dropped:,} 行 | **{100 - retention_rate:.2f}% (总剔除率)** |
| ├ 缺失值记录 (NaN) | {null_dropped:,} 行 | - |
| ├ 非法行为类型 (非1~4) | {invalid_behavior_dropped:,} 行 | - |
| ├ 非法时间格式记录 | {invalid_time_dropped:,} 行 | - |
| ├ 业务四元组重复记录 | {duplicate_dropped:,} 行 | 消除用户重复误触与网络重发 |
| └ **IQR 极端爬虫行为记录** | **{bot_dropped:,} 行** | 剔除 **{len(abnormal_users):,}** 个极端账号 (阈值: >{upper_bound:.1f} 次) |

---

## 二、 基础业务维度概况

- **有效独立用户数 (UV)**: {df_clean['user_id'].nunique():,} 人
- **有效独立商品数**: {df_clean['item_id'].nunique():,} 件
- **有效商品类目数**: {df_clean['item_category'].nunique():,} 类
- **核心业务日期范围**: {df_clean['date'].min()} 至 {df_clean['date'].max()}
- **行为类型分布 (PV / Fav / Cart / Buy)**:
  - **1 - 浏览 (pv)**: {(df_clean['behavior_type'] == 1).sum():,} 次 ({((df_clean['behavior_type'] == 1).sum() / final_clean_rows) * 100:.2f}%)
  - **2 - 收藏 (fav)**: {(df_clean['behavior_type'] == 2).sum():,} 次 ({((df_clean['behavior_type'] == 2).sum() / final_clean_rows) * 100:.2f}%)
  - **3 - 加购 (cart)**: {(df_clean['behavior_type'] == 3).sum():,} 次 ({((df_clean['behavior_type'] == 3).sum() / final_clean_rows) * 100:.2f}%)
  - **4 - 购买 (buy)**: {(df_clean['behavior_type'] == 4).sum():,} 次 ({((df_clean['behavior_type'] == 4).sum() / final_clean_rows) * 100:.2f}%)

---

## 三、 产出文件说明
1. **标准化数据文件**: `data/processed/user_behavior_clean.parquet`
2. **数据质量报告**: `docs/data_quality_report.md`
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"-->数据质量报告已自动保存至: {report_md_path}")
    print("\n================ 数据清洗任务圆满完成 ================")


if __name__ == "__main__":
    clean_user_behavior_data()