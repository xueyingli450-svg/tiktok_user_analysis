"""Optuna 自动化超参数调优模块

使用贝叶斯优化搜索算法，以验证集 (val_data) AUC 最大化为目标，
对 LightGBM 模型的深度、叶子数、学习率及正则化参数进行 30 轮自动调优，
并在测试集上验证调优前后的性能增益。
"""

from pathlib import Path
from lightgbm import LGBMClassifier
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.utils.db_connector import get_project_root

# 压制 optuna 过于冗长的日志输出
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_lightgbm_hyperparameters() -> None:
    project_root = get_project_root()
    train_path = project_root / "data" / "processed" / "train_data.parquet"
    val_path = project_root / "data" / "processed" / "val_data.parquet"
    test_path = project_root / "data" / "processed" / "test_data.parquet"

    print("开始加载训练集、验证集与测试集...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    exclude_cols = ["user_id", "target_label"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["target_label"]

    X_val = val_df[feature_cols]
    y_val = val_df["target_label"]

    X_test = test_df[feature_cols]
    y_test = test_df["target_label"]

    print(
        f"样本量: 训练集 {len(X_train):,} | 验证集 {len(X_val):,} | 测试集 {len(X_test):,}"
    )

    # 1. 定义 Optuna 目标优化函数 (在验证集上评估 AUC)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 10, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10.0, log=True
            ),
            "random_state": 42,
            "verbosity": -1,
        }

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # 在验证集上预测概率并计算 AUC
        val_preds = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_preds)
        return val_auc

    # 2. 启动贝叶斯超参数搜索
    print("\n================ 启动 Optuna 贝叶斯自动调优 (30 轮搜索) ================")
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\n调参完成!")
    print(f"验证集最优 AUC 评分: {study.best_value:.4f}")
    print("\n搜索到的最优超参数组合:")
    for k, v in study.best_params.items():
        print(f"- {k}: {v}")

    # 3. 使用最优参数重新训练并在测试集上最终评估
    print("\n正在使用最优参数在测试集上验证最终性能...")
    best_model = LGBMClassifier(
        **study.best_params, random_state=42, verbosity=-1
    )
    best_model.fit(X_train, y_train)

    test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred = best_model.predict(X_test)

    tuned_auc = roc_auc_score(y_test, test_pred_proba)
    tuned_acc = accuracy_score(y_test, test_pred)
    tuned_prec = precision_score(y_test, test_pred, zero_division=0)
    tuned_rec = recall_score(y_test, test_pred, zero_division=0)
    tuned_f1 = f1_score(y_test, test_pred, zero_division=0)

    # 4. 对比调优前后结果
    baseline_auc = 0.8758  # 昨天基准默认参数的 AUC

    print("\n================ 调优前后性能对比报告 ================")
    print(
        f"测试集基准 LightGBM AUC : {baseline_auc:.4f} (默认参数 / 未调优)"
    )
    print(
        f"测试集调优后 LightGBM AUC: {tuned_auc:.4f} (Optuna 搜索最优参数)"
    )
    print(
        f"AUC 绝对增益提升        : +{tuned_auc - baseline_auc:.4f} ({((tuned_auc - baseline_auc)/baseline_auc)*100:+.2f}%)"
    )
    print(f"调优后准确率 (Accuracy) : {tuned_acc:.4f}")
    print(f"调优后精确率 (Precision): {tuned_prec:.4f}")
    print(f"调优后召回率 (Recall)   : {tuned_rec:.4f}")
    print(f"调优后 F1 分数          : {tuned_f1:.4f}")
    print("=====================================================")


if __name__ == "__main__":
    tune_lightgbm_hyperparameters()