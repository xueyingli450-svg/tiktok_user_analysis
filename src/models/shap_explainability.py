"""模型可解释性与特征归因分析模块

使用 SHAP (Shapley Additive Explanations) 算法，
对树模型 (LightGBM) 进行全局特征重要性解释与边际贡献度分析，
并生成 SHAP 蜂窝摘要图 (fig11)。
"""

from pathlib import Path
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from src.utils.db_connector import get_project_root

# 配置绘图字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def run_shap_analysis() -> None:
    project_root = get_project_root()
    train_path = project_root / "data" / "processed" / "train_data.parquet"
    test_path = project_root / "data" / "processed" / "test_data.parquet"
    figures_dir = project_root / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("开始加载数据并拟合解释模型...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    exclude_cols = ["user_id", "target_label"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["target_label"]
    X_test = test_df[feature_cols]

    # 1. 训练 LightGBM 模型
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=20,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    # 2. 计算 SHAP 解释值 (使用 TreeExplainer)
    print("正在计算测试集样本的 SHAP 贡献值 (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)

    # 抽取 500 个测试样本进行快速且准确的全局特征解释
    sample_test = X_test.sample(n=min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample_test)

    # 处理二分类输出结构 (提取正类预测的 SHAP 值)
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_vals_pos = shap_values[:, :, 1]
    else:
        shap_vals_pos = shap_values

    # 3. 统计各特征的平均绝对 SHAP 贡献度并排序
    mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
    importance_df = pd.DataFrame(
        {"特征名称": feature_cols, "平均绝对贡献度 (Mean |SHAP|)": mean_abs_shap}
    ).sort_values(by="平均绝对贡献度 (Mean |SHAP|)", ascending=False)

    print("\n影响用户未来购买决策的前 10 大核心驱动特征:")
    print(importance_df.head(10).to_string(index=False))

    # 4. 绘制并保存 SHAP 蜂窝图 (fig11)
    print("\n正在生成 SHAP 全局特征贡献度蜂窝摘要图...")
    plt.figure(figsize=(11, 7))

    shap.summary_plot(
        shap_vals_pos,
        sample_test,
        feature_names=feature_cols,
        max_display=15,
        show=False,
    )

    plt.title(
        "抖音商城购买预测模型 · Top 15 特征 SHAP 归因贡献度图",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("SHAP 贡献值 (对下单概率的正负向影响)", fontsize=11)
    plt.tight_layout()

    fig11_path = figures_dir / "fig11_shap_summary.png"
    plt.savefig(fig11_path, dpi=300)
    plt.close()

    print(f"SHAP 归因分析图已保存至: {fig11_path.name}")
    print("================ SHAP 可解释性分析全部完成 ================")


if __name__ == "__main__":
    run_shap_analysis()