# 抖音商城用户行为数据分析与算法建模系统

本项目基于抖音商城千万级用户全链路行为日志（浏览、收藏、加购、购买），构建从**大数据工程清洗、SQLite 关系型数仓搭建、全链路转化漏斗分析、RFM 用户价值分群**到**机器学习与深度学习转化预测（DIN / LightGBM）**的全流程实战闭环。

---

## 技术栈与工具链

- **核心语言**: Python 3.12+
- **数据工程**: Pandas, PyArrow (Apache Parquet 列式存储), NumPy
- **数据库与数仓**: SQLite 3 (带 B-Tree 复合索引优化)
- **可视化与分析**: Matplotlib, Seaborn, SQL
- **工程规范**: 遵循字节跳动 Python 编码规范（PEP 8、相对路径解耦、模块化设计、Conventional Commits）

---

## 项目目录结构

```text
tiktok_user_analysis/
├── data/                       # 数据层
│   ├── raw/                    # 原始行为日志 (只读原料)
│   ├── processed/              # 清洗后标准 Parquet 数据
│   └── ecommerce.db            # SQLite 关系型数据库 (已建立 4 大核心索引)
├── src/                        # 代码层
│   ├── data_processing/        # 数据清洗、质检与入库脚本
│   │   ├── 01_inspect_data.py  # 数据格式与相对路径探测
│   │   ├── 02_clean_data.py    # 1225万行全流程清洗、IQR爬虫排查与Parquet导出
│   │   └── 03_load_to_sqlite.py# 幂等性批量入库脚本 (518万行数据安全灌库)
│   ├── sql/                    # 独立 SQL 脚本库
│   │   └── 01_create_tables.sql# DDL 表结构定义与复合索引优化
│   └── utils/                  # 通用工具模块
│       └── db_connector.py     # SQLite 数据库统一交互连接器
├── docs/                       # 文档与结果层
│   ├── data_quality_report.md  # 全自动生成的数据清洗质量报告
│   └── figures/                # 可视化图表产出目录
├── requirements.txt            # 环境依赖清单
├── .gitignore                  # Git 忽略配置 (保护大数据不被误传)
└── README.md                   # 项目主说明文档