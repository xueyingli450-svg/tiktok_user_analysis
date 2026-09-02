"""数据入库模块：

读取清洗后的 Parquet 数据，自动创建 SQLite 数据库表与索引，
并高效批量导入 500 多万条数据。
"""

import time
from pathlib import Path
import pandas as pd
from src.utils.db_connector import (
    execute_sql_file,
    get_db_connection,
    get_project_root,
)


def load_parquet_to_sqlite() -> None:
    """将清洗后的 Parquet 数据全量导入 SQLite 数据库并建立索引。"""
    project_root = get_project_root()
    parquet_path = (
        project_root / "data" / "processed" / "user_behavior_clean.parquet"
    )
    sql_schema_path = project_root / "src" / "sql" / ("create_tables.sql")

    print("================ 开始执行 SQLite 数据库批量入库 ================")
    start_time = time.time()

    # 1. 第一步：执行建表与索引的 SQL 脚本
    print(f"--> 步骤 1: 执行 DDL 建表脚本: {sql_schema_path.name}")
    execute_sql_file(sql_schema_path)

    # 2. 第二步：从 Parquet 文件中高速读取数据
    print(f"\n--> 步骤 2: 读取标准 Parquet 数据: {parquet_path.name}")
    df_clean = pd.read_parquet(parquet_path)

    # 【核心修复】：精准对齐 SQLite 数据表的 7 个字段（剔除辅助列 datetime）
    target_columns = [
        "user_id",
        "item_id",
        "item_category",
        "behavior_type",
        "time",
        "date",
        "hour",
    ]
    df_clean = df_clean[target_columns]
    total_records = len(df_clean)
    print(f" 成功筛选入库字段: {total_records:,} 行")

    # 3. 第三步：批量写入 SQLite 数据库
    print("\n--> 步骤 3: 正在批量写入 SQLite 数据库 (data/ecommerce.db)...")
    conn = get_db_connection()

    # SQLite 高性能写入加速配置
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")

    # 每次写入 20 万行
    chunk_size = 200_000
    df_clean.to_sql(
        name="user_behavior",
        con=conn,
        if_exists="append",
        index=False,
        chunksize=chunk_size,
    )
    conn.commit()

    # 4. 第四步：SQL 校验数据库中的实际行数
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM user_behavior;")
    db_count = cursor.fetchone()[0]
    conn.close()

    elapsed_time = time.time() - start_time
    print(f"\n================ SQLite 入库校验报告 ================")
    print(f"Parquet 源数据行数 : {total_records:,} 行")
    print(f"SQLite 数据库实际行数: {db_count:,} 行")
    print(
        f"数据入库一致性校验   : {'完全一致 (100% 成功)' if total_records == db_count else '行数不一致'}"
    )
    print(f"总耗时              : {elapsed_time:.2f} 秒")
    print(f"数据库文件路径      : data/ecommerce.db")
    print("=====================================================")


if __name__ == "__main__":
    load_parquet_to_sqlite()