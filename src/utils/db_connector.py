"""SQLite 数据库交互工具模块

提供数据库连接获取、SQL 脚本文件执行及带索引优化的统一管理。
"""

from pathlib import Path
import sqlite3

def get_project_root() -> Path:
    """动态获取项目根目录路径。"""
    return Path(__file__).resolve().parents[2]

def get_db_path() -> Path:
    """获取 SQLite 数据库文件的标准相对路径 (data/ecommerce.db)。"""
    return get_project_root() / "data" / "ecommerce.db"

def get_db_connection() -> sqlite3.Connection:
    """获取 SQLite 数据库连接对象。
    Returns:
        sqlite3.Connection: 数据库连接对象。
    """
    db_path = get_db_path()
    # 确保 data 目录存在
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn

def execute_sql_file(sql_file_path: Path) -> None:
    """读取并执行指定的外部 .sql 脚本文件。
    Args:
        sql_file_path: 目标 .sql 文件的 Path 路径对象。
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    print(f"--> 正在读取并执行 SQL 脚本: {sql_file_path.name}")
    with open(sql_file_path, "r", encoding="utf-8") as f:
        sql_script = f.read()

    # executescript 允许一次性执行包含多条 SQL（建表 + 4个索引）的整篇脚本
    cursor.executescript(sql_script)
    conn.commit()
    conn.close()
    print(f"--> SQL 脚本 [{sql_file_path.name}] 执行完成，表结构与索引创建成功！")