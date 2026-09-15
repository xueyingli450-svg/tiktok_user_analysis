"""传统分类基线模型训练与多指标对比模块

训练三种经典分类器：
1. 逻辑回归 (Logistic Regression，线性基线)
2. XGBoost (梯度提升树基准)
3. LightGBM (高效树模型基准)
在独立测试集上评估 AUC、准确率、精确率、召回率与 F1 分数，并绘制 ROC 曲线 (fig10)。
"""

from pathlib import Path
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.utils.db_connector import get_project_root

# 配置绘图字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def train_and_evaluate_baselines() -> None:
    project_root = get_project_root()
    train_path = project_root / "data" / "processed" / "train_data.parquet"
    test_path = project_root / "data" / "processed" / "test_data.parquet"
    figures_dir = project_root / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("开始加载训练集与测试集数据...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 分离特征矩阵 X 与标签 y
    exclude_cols = ["user_id", "target_label"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["target_label"]

    X_test = test_df[feature_cols]
    y_test = test_df["target_label"]

    print(f"训练样本数: {len(X_train):,} | 测试样本数: {len(X_test):,}")
    print(f"输入特征维度: {len(feature_cols)} 维")

    # 对逻辑回归进行特征标准化处理 (树模型不需要)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. 初始化三种经典分类模型
    models = {
        "逻辑回归 (Logistic Regression)": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            random_state=42,
            verbosity=-1,
        ),
    }

    results = []
    roc_data = {}

    print("\n================ 开始多模型训练与评估 ================")

    for name, model in models.items():
        print(f"正在训练模型: {name}...")

        # 逻辑回归使用标准化特征，树模型使用原始特征
        if "逻辑回归" in name:
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

        # 计算 5 大核心评估指标
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append(
            {
                "模型名称": name,
                "AUC (核心指标)": round(auc, 4),
                "准确率 (Accuracy)": round(acc, 4),
                "精确率 (Precision)": round(prec, 4),
                "召回率 (Recall)": round(rec, 4),
                "F1 分数": round(f1, 4),
            }
        )

        # 记录 ROC 曲线数据
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_data[name] = (fpr, tpr, auc)

    # 2. 打印综合对比表格
    results_df = pd.DataFrame(results)
    print("\n测试集多模型性能评估对比表:")
    print(results_df.to_string(index=False))

    # 3. 绘制 ROC 评估曲线对比图 (fig10)
    print("\n正在生成多模型 ROC 评估对比曲线...")
    plt.figure(figsize=(9, 7))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
        plt.plot(
            fpr,
            tpr,
            color=color,
            lw=2.2,
            label=f"{name} (AUC = {auc_val:.4f})",
        )

    # 绘制对角基准线 (随机猜测线)
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("假正率 (False Positive Rate / FPR)", fontsize=11)
    plt.ylabel("真正率 (True Positive Rate / TPR)", fontsize=11)
    plt.title(
        "抖音商城用户未来购买预测 · 多模型 ROC 对比曲线",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    fig10_path = figures_dir / "fig10_model_roc_curves.png"
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    print(f"ROC 曲线图已保存至: {fig10_path.name}")
    print("================ 基准模型评估全部完成 ================")


if __name__ == "__main__":
    train_and_evaluate_baselines()