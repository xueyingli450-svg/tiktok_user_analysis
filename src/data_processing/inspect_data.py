"""初步数据验证脚本：通过使用相对路径加载原始数据前十行，验证环境与数据字段"""

from pathlib import Path
import pandas as pd

def inspect_data_sample():
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = project_root / "data" / "raw" / "user_behavior_processed.csv"
    print(f"--> 项目根目录定位成功: {project_root}")
    print(f"-->准备读取数据文件: {raw_data_path}")

    column_names = [
        "user_id",
        "item_id",
        "category_id",
        "behavior_type",
        "timestamp",
    ]

    df_sample = pd.read_csv(raw_data_path, names=column_names, nrows=10)

    print("\n========== 数据前 10 行样例 ==========")
    print(df_sample)

    print("\n========== 字段类型 (Dtypes) ==========")
    print(df_sample.dtypes)

if __name__ == "__main__":
    inspect_data_sample()

