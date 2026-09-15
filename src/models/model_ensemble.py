"""多模型 Stacking 集成融合模块

构建由 LightGBM、XGBoost 与多层感知机 (MLP 神经网络) 组成的基学习器池，
以逻辑回归为元学习器执行 5 折交叉验证 Stacking 概率融合，
并在独立测试集上评估集成模型的最终性能。
"""

from pathlib import Path
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.utils.db_connector import get_project_root


def train_stacking_ensemble() -> None:
    project_root = get_project_root()
    train_path = project_root / "data" / "processed" / "train_data.parquet"
    test_path = project_root / "data" / "processed" / "test_data.parquet"

    print("开始加载建模数据集...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    exclude_cols = ["user_id", "target_label"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["target_label"]

    X_test = test_df[feature_cols]
    y_test = test_df["target_label"]

    # 标准化特征 (供神经网络与元学习器使用)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. 定义三位初级基学习器 (Base Learners)
    base_learners = [
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=20,
                random_state=42,
                verbosity=-1,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=4,
                random_state=42,
                eval_metric="logloss",
            ),
        ),
        (
            "mlp_nn",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=300,
                alpha=0.01,
                random_state=42,
            ),
        ),
    ]

    # 2. 构建 5 折交叉验证 Stacking 分类器 (以逻辑回归为元学习器)
    print("\n================ 开始训练 Stacking 融合模型 (5 折交叉验证) ================")
    stacking_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(random_state=42),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    stacking_model.fit(X_train_scaled, y_train)

    # 3. 测试集预测与评估
    print("正在测试集上评估 Stacking 融合性能...")
    y_pred_proba = stacking_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = stacking_model.predict(X_test_scaled)

    stacking_auc = roc_auc_score(y_test, y_pred_proba)
    stacking_acc = accuracy_score(y_test, y_pred)
    stacking_prec = precision_score(y_test, y_pred, zero_division=0)
    stacking_rec = recall_score(y_test, y_pred, zero_division=0)
    stacking_f1 = f1_score(y_test, y_pred, zero_division=0)

    # 4. 单模型与 Stacking 融合对比
    print("\n================ 模型融合最终性能对比表 ================")
    comparison_data = [
        {"模型方案": "逻辑回归 (单模型基线)", "测试集 AUC": 0.8529, "召回率 (Recall)": 0.8987},
        {"模型方案": "XGBoost (单模型)", "测试集 AUC": 0.8750, "召回率 (Recall)": 0.9443},
        {"模型方案": "LightGBM (调优单模型)", "测试集 AUC": 0.8756, "召回率 (Recall)": 0.9482},
        {
            "模型方案": "Stacking 融合模型 (LGBM+XGB+MLP)",
            "测试集 AUC": round(stacking_auc, 4),
            "召回率 (Recall)": round(stacking_rec, 4),
        },
    ]

    print(pd.DataFrame(comparison_data).to_string(index=False))
    print(f"\nStacking 融合模型详细指标:")
    print(f"- 准确率 (Accuracy): {stacking_acc:.4f}")
    print(f"- 精确率 (Precision): {stacking_prec:.4f}")
    print(f"- 召回率 (Recall)  : {stacking_rec:.4f}")
    print(f"- F1 分数 (F1-Score): {stacking_f1:.4f}")
    print("=====================================================")


if __name__ == "__main__":
    train_stacking_ensemble()