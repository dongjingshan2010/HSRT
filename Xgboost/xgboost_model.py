import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import torch
from tqdm import tqdm
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


# ==========================================
# 数据加载函数 - 使用 data_loader.py 中的函数
# ==========================================

def load_data_with_data_loader(healthy_files, stroke_files, feature_dim=32, batch_size_processing=500,
                               use_easyensemble=True):
    """使用 data_loader.py 中的函数加载数据（优化内存使用版本）"""
    from data_loader import load_and_process_data_by_year

    print("使用 data_loader.py 加载数据...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        train_loader, val_loader, test_loader, train_dataset, processor = load_and_process_data_by_year(
            healthy_files=healthy_files,
            stroke_files=stroke_files,
            feature_dim=feature_dim,
            batch_size=64,
            K=10
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("GPU内存不足，尝试在CPU上处理数据...")
            device = torch.device('cpu')
            import data_loader
            data_loader.device = device
            train_loader, val_loader, test_loader, train_dataset, processor = load_and_process_data_by_year(
                healthy_files=healthy_files,
                stroke_files=stroke_files,
                feature_dim=feature_dim,
                batch_size=64,
                K=10
            )
        else:
            raise e

    print("\n数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 个样本")
    print(f"  验证集: {len(val_loader.dataset)} 个样本")
    print(f"  测试集: {len(test_loader.dataset)} 个样本")

    # 提取验证数据
    print("分批提取验证数据...")
    val_features_list, val_labels_list = [], []
    for batch_idx, (features, labels) in enumerate(val_loader):
        val_features_list.append(features.cpu().numpy())
        val_labels_list.append(labels.cpu().numpy())
        if batch_idx % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    val_features = np.vstack(val_features_list)
    val_labels = np.hstack(val_labels_list)
    val_features_flat = val_features.reshape(val_features.shape[0], -1)

    # 提取测试数据
    print("分批提取测试数据...")
    test_features_list, test_labels_list = [], []
    for batch_idx, (features, labels) in enumerate(test_loader):
        test_features_list.append(features.cpu().numpy())
        test_labels_list.append(labels.cpu().numpy())
        if batch_idx % 10 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    test_features = np.vstack(test_features_list)
    test_labels = np.hstack(test_labels_list)
    test_features_flat = test_features.reshape(test_features.shape[0], -1)

    # 生成字段名称
    field_names = []
    num_fields = train_dataset.positive_embeddings.shape[1] if hasattr(train_dataset, 'positive_embeddings') else 0
    feature_dim = train_dataset.positive_embeddings.shape[2] if hasattr(train_dataset,
                                                                        'positive_embeddings') else feature_dim

    if hasattr(processor, 'field_names') and processor.field_names:
        field_names = processor.field_names
        print(f"从处理器获取字段名: {len(field_names)} 个字段")
    else:
        field_names = [f"field_{i}" for i in range(num_fields)]
        print(f"生成默认字段名: {len(field_names)} 个字段")

    feature_names = []
    for field_idx, field_name in enumerate(field_names):
        for dim_idx in range(feature_dim):
            feature_names.append(f"{field_name}_dim{dim_idx}")
    print(f"特征维度总数: {len(feature_names)} (字段数: {num_fields}, 每字段维度: {feature_dim})")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    result = {
        'X_val': val_features_flat,
        'y_val': val_labels,
        'X_test': test_features_flat,
        'y_test': test_labels,
        'feature_names': feature_names,
        'field_names': field_names,
        'num_fields': num_fields,
        'feature_dim': feature_dim,
        'processor': processor,
        'train_dataset': train_dataset
    }

    if not use_easyensemble:
        # 提取全部训练数据（非EasyEnsemble模式）
        print("分批提取训练数据...")
        positive_embeddings = train_dataset.positive_embeddings
        negative_embeddings = train_dataset.negative_embeddings
        train_features_list, train_labels_list = [], []

        num_positive = len(positive_embeddings)
        for i in range(0, num_positive, batch_size_processing):
            end = min(i + batch_size_processing, num_positive)
            batch = positive_embeddings[i:end]
            batch_flat = batch.cpu().numpy().reshape(batch.shape[0], -1)
            train_features_list.append(batch_flat)
            train_labels_list.append(np.ones(len(batch)))
            del batch, batch_flat
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        num_negative = len(negative_embeddings)
        for i in range(0, num_negative, batch_size_processing):
            end = min(i + batch_size_processing, num_negative)
            batch = negative_embeddings[i:end]
            batch_flat = batch.cpu().numpy().reshape(batch.shape[0], -1)
            train_features_list.append(batch_flat)
            train_labels_list.append(np.zeros(len(batch)))
            del batch, batch_flat
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        train_features_flat = np.vstack(train_features_list)
        train_labels = np.hstack(train_labels_list)
        result['X_train'] = train_features_flat
        result['y_train'] = train_labels
        print(f"训练集形状: {train_features_flat.shape}, 标签: {train_labels.shape}")
    else:
        print("使用EasyEnsemble方式，不提取全部训练数据")

    print(f"验证集形状: {val_features_flat.shape}, 标签: {val_labels.shape}")
    print(f"测试集形状: {test_features_flat.shape}, 标签: {test_labels.shape}")

    return result


def preprocess_data(df, drop_columns=None):
    """预处理数据 - 保留此函数用于非嵌入数据"""
    df_processed = df.copy()
    if drop_columns:
        existing_drop_cols = [col for col in drop_columns if col in df_processed.columns]
        df_processed = df_processed.drop(columns=existing_drop_cols)

    if 'label' in df_processed.columns:
        y = df_processed['label'].values
        X = df_processed.drop(columns=['label'])
    else:
        raise ValueError("无label列")

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    for col in categorical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')

    for col in numerical_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders, X.columns.tolist()


def balance_data_undersample(X, y):
    """欠采样"""
    if isinstance(X, np.ndarray):
        df_X = pd.DataFrame(X)
    else:
        df_X = X.copy()
    df_y = pd.Series(y, name='label')
    idx_0 = df_y[df_y == 0].index
    idx_1 = df_y[df_y == 1].index
    min_len = min(len(idx_0), len(idx_1))

    idx_0_selected = np.random.choice(idx_0, min_len, replace=False)
    idx_1_selected = np.random.choice(idx_1, min_len, replace=False)

    balanced_indices = np.concatenate([idx_0_selected, idx_1_selected])
    np.random.shuffle(balanced_indices)

    y_balanced = y[balanced_indices].astype(int)
    return X[balanced_indices], y_balanced


def compute_binary_metrics(y_true, y_pred, y_proba=None):
    """计算二分类各项指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    g_mean = math.sqrt(sensitivity * specificity) if sensitivity > 0 and specificity > 0 else 0

    auc = 0.5
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            pass

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'balanced_accuracy': balanced_acc,
        'g_mean': g_mean,
        'auc': auc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def train_single_model(train_features, train_labels, val_features, val_labels,
                       params, early_stopping_rounds=20):
    """
    训练单个 XGBoost 模型，使用 early_stopping 并保存验证集 loss 最小的模型
    返回：训练好的模型（最佳迭代），以及验证集上的各项指标
    """
    # 复制参数，避免修改原字典
    model_params = params.copy()
    # 设置 n_estimators 足够大，早停会提前停止
    if 'n_estimators' not in model_params or model_params['n_estimators'] < 1000:
        model_params['n_estimators'] = 1000

    try:
        # 尝试适配新版 XGBoost API (>= 1.6.0)
        # 较新版本中 early_stopping_rounds 必须在模型初始化时传入
        model_params_new = model_params.copy()
        model_params_new['early_stopping_rounds'] = early_stopping_rounds
        model = XGBClassifier(**model_params_new)
        model.fit(
            train_features, train_labels,
            eval_set=[(val_features, val_labels)],
            verbose=False
        )
    except TypeError:
        # 如果捕获到 TypeError，说明是较老版本的 XGBoost
        # 退回到在 fit 方法中传入 early_stopping_rounds 的方式
        model = XGBClassifier(**model_params)
        model.fit(
            train_features, train_labels,
            eval_set=[(val_features, val_labels)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )

    # 验证集预测
    val_proba = model.predict_proba(val_features)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    metrics = compute_binary_metrics(val_labels, val_pred, val_proba)

    return model, metrics


def compute_field_importance(models, field_names, feature_names):
    """
    计算字段级别的平均重要性（对模型列表中的每个模型归一化后平均）
    返回：字典 {field_name: avg_importance}
    """
    all_field_importances = []
    for model in models:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            field_importance = {}
            for field_name in field_names:
                field_features = [f for f in feature_names if f.startswith(field_name)]
                if field_features:
                    indices = [feature_names.index(f) for f in field_features if f in feature_names]
                    if indices:
                        field_importance[field_name] = sum(importances[idx] for idx in indices)
            total = sum(field_importance.values())
            if total > 0:
                field_importance = {k: v / total for k, v in field_importance.items()}
            all_field_importances.append(field_importance)

    if not all_field_importances:
        return {}

    avg_field_importance = {}
    for field_name in field_names:
        values = [imp.get(field_name, 0) for imp in all_field_importances]
        avg_field_importance[field_name] = np.mean(values)

    return avg_field_importance


def plot_field_importance(importance_dict, top_n=20, save_path=None, title_suffix=""):
    """根据字段重要性字典绘制柱状图"""
    df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    })
    df = df.sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(8, 16))
    plt.barh(range(len(df)), df['importance'].values)
    plt.yticks(range(len(df)), df['feature'].values, fontsize=24)
    plt.xlabel('Importance Score', fontsize=24)
    plt.title(f'Field Importance Ranking (Top {top_n}) {title_suffix}', fontsize=28, pad=15)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return df


def plot_feature_importance_ensemble(models, field_names, feature_names, top_n=20, save_path=None):
    """绘制集成模型的特征重要性图（字段级别）- 保留旧接口，内部调用新函数"""
    imp_dict = compute_field_importance(models, field_names, feature_names)
    if not imp_dict:
        print("没有可用的特征重要性数据")
        return None
    return plot_field_importance(imp_dict, top_n, save_path)


def plot_roc_curve_ensemble(models, X_test, y_test, save_path=None):
    """绘制集成模型的ROC曲线（等权重平均概率）"""
    all_probas = [model.predict_proba(X_test)[:, 1] for model in models]
    avg_proba = np.mean(all_probas, axis=0)
    fpr, tpr, _ = roc_curve(y_test, avg_proba)
    auc_score = roc_auc_score(y_test, avg_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (EasyEnsemble集成，{len(models)}个模型)')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return auc_score


def test_ensemble_models(models, model_metrics_list, X_test, y_test,
                         ensemble_method='max_voting', selection_metric='f1'):
    """
    使用多种集成方法评估模型

    参数:
        models: 模型列表
        model_metrics_list: 每个模型在验证集上的指标字典列表
        X_test, y_test: 测试数据
        ensemble_method: 集成方法
            - 'selection_metric_weighted': 基于 selection_metric 加权平均概率
            - 'sensitivity_weighted': 基于敏感度加权平均概率
            - 'accuracy_weighted': 基于准确率加权平均概率
            - 'equal_weighted': 等权重平均概率
            - 'max_voting': 最大投票法
        selection_metric: 用于加权或记录的指标名称
    """
    n_models = len(models)
    if n_models == 0:
        print("没有模型可供集成")
        return None

    # 提取各项指标
    accuracies = [m['accuracy'] for m in model_metrics_list]
    sensitivities = [m['sensitivity'] for m in model_metrics_list]
    specificities = [m['specificity'] for m in model_metrics_list]
    precisions = [m['precision'] for m in model_metrics_list]   # 新增精确率列表
    f1s = [m['f1'] for m in model_metrics_list]
    balanced_accs = [m['balanced_accuracy'] for m in model_metrics_list]
    g_means = [m['g_mean'] for m in model_metrics_list]
    aucs = [m['auc'] for m in model_metrics_list]

    # 确定权重
    if ensemble_method == 'selection_metric_weighted':
        if selection_metric == 'accuracy':
            weights = np.array(accuracies)
        elif selection_metric == 'sensitivity':
            weights = np.array(sensitivities)
        elif selection_metric == 'specificity':
            weights = np.array(specificities)
        elif selection_metric == 'precision':
            weights = np.array(precisions)   # 支持精确率加权
        elif selection_metric == 'f1':
            weights = np.array(f1s)
        elif selection_metric == 'balanced_accuracy':
            weights = np.array(balanced_accs)
        elif selection_metric == 'g_mean':
            weights = np.array(g_means)
        elif selection_metric == 'auc':
            weights = np.array(aucs)
        else:
            weights = np.ones(n_models)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(n_models) / n_models
        print(f"\n集成方法: 基于验证集 {selection_metric} 加权平均")
    elif ensemble_method == 'sensitivity_weighted':
        weights = np.array(sensitivities)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_models) / n_models
        print("\n集成方法: 基于验证集中风敏感度加权平均")
    elif ensemble_method == 'accuracy_weighted':
        weights = np.array(accuracies)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_models) / n_models
        print("\n集成方法: 基于验证集准确率加权平均")
    elif ensemble_method == 'equal_weighted':
        weights = np.ones(n_models) / n_models
        print("\n集成方法: 等权重平均")
    elif ensemble_method == 'max_voting':
        weights = None  # 投票法不需要权重，但在平票时使用平均概率
        print("\n集成方法: 最大投票法（平票时使用等权重概率平均）")
    else:
        weights = np.ones(n_models) / n_models
        print(f"\n未知集成方法 {ensemble_method}，使用等权重平均")

    # 获取所有模型在测试集上的预测概率和预测类别
    all_proba = []
    all_pred = []
    for model in models:
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        all_proba.append(proba)
        all_pred.append(pred)
    all_proba = np.array(all_proba)  # shape: (n_models, n_samples)
    all_pred = np.array(all_pred)

    # 集成预测
    if ensemble_method == 'max_voting':
        final_pred = []
        for i in range(len(y_test)):
            votes = all_pred[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            max_count = np.max(counts)
            candidates = unique[counts == max_count]
            if len(candidates) == 1:
                final_pred.append(candidates[0])
            else:
                # 平票：使用平均概率（等权重）
                avg_prob = np.mean(all_proba[:, i])
                final_pred.append(1 if avg_prob >= 0.5 else 0)
        final_pred = np.array(final_pred)
        final_proba = np.mean(all_proba, axis=0)  # 用于计算AUC
    else:
        # 加权平均概率
        final_proba = np.average(all_proba, axis=0, weights=weights)
        final_pred = (final_proba >= 0.5).astype(int)

    # 计算集成模型的指标
    metrics = compute_binary_metrics(y_test, final_pred, final_proba)

    print(f"\n=== EasyEnsemble集成模型测试结果（{ensemble_method}）===")
    print(f"测试集大小: {len(y_test)}")
    print(f"集成准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")          # 新增精确率输出
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"中风敏感度: {metrics['sensitivity']:.4f}")
    print(f"特异性: {metrics['specificity']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"平衡准确率: {metrics['balanced_accuracy']:.4f}")
    print(f"G-mean: {metrics['g_mean']:.4f}")

    print("\n各模型验证集性能及权重:")
    for i in range(n_models):
        w = weights[i] if weights is not None else 1/n_models
        print(f"  模型 {i+1}: 准确率={accuracies[i]:.4f}, 敏感度={sensitivities[i]:.4f}, "
              f"精确率={precisions[i]:.4f}, F1={f1s[i]:.4f}, 权重={w:.4f}")  # 新增精确率

    # 混淆矩阵
    cm = confusion_matrix(y_test, final_pred)
    print("\n混淆矩阵:")
    print(f"         预测健康   预测中风")
    print(f"实际健康  {cm[0, 0]:>8}  {cm[0, 1]:>8}")
    print(f"实际中风  {cm[1, 0]:>8}  {cm[1, 1]:>8}")

    return metrics


def run_single_experiment(run_id, seed):
    """执行一次完整的训练和评估流程，返回集成方法的测试指标字典 和 字段重要性字典"""
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print("=" * 80)
    print(f"运行实验 {run_id+1}/3，随机种子: {seed}")
    print("=" * 80)

    # 定义文件路径字典
    healthy_files = {
        2000: '../healthy16.csv',
        2017: '../healthy17.csv',
        2018: '../healthy18.csv'
    }
    stroke_files = {
        2007: '../Stroke16.csv',
        2017: '../Stroke17.csv',
        2018: '../Stroke18.csv'
    }

    print(f"健康数据文件: {len(healthy_files)} 个")
    print(f"中风数据文件: {len(stroke_files)} 个")

    if not healthy_files or not stroke_files:
        print("错误：找不到数据文件。")
        return None, None

    try:
        data_dict = load_data_with_data_loader(
            healthy_files=healthy_files,
            stroke_files=stroke_files,
            feature_dim=32,
            use_easyensemble=True
        )
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        feature_names = data_dict['feature_names']
        field_names = data_dict['field_names']
        train_dataset = data_dict['train_dataset']
        print(f"\n数据加载成功: 训练集使用EasyEnsemble ({train_dataset.K}个子集)")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None

    # 平衡验证集和测试集（用于公平评估）
    val_class_counts = np.bincount(y_val.astype(int))
    test_class_counts = np.bincount(y_test.astype(int))
    print(f"\n原始验证集分布: 0:{val_class_counts[0]}, 1:{val_class_counts[1]}")
    print(f"原始测试集分布: 0:{test_class_counts[0]}, 1:{test_class_counts[1]}")

    if val_class_counts[0] != val_class_counts[1]:
        X_val_bal, y_val_bal = balance_data_undersample(X_val, y_val)
    else:
        X_val_bal, y_val_bal = X_val, y_val

    if test_class_counts[0] != test_class_counts[1]:
        X_test_bal, y_test_bal = balance_data_undersample(X_test, y_test)
    else:
        X_test_bal, y_test_bal = X_test, y_test

    print(f"平衡后验证集: {X_val_bal.shape}, 分布: {np.bincount(y_val_bal.astype(int))}")
    print(f"平衡后测试集: {X_test_bal.shape}, 分布: {np.bincount(y_test_bal.astype(int))}")

    # 确保数据格式
    X_val_bal = np.ascontiguousarray(X_val_bal.astype(np.float32))
    y_val_bal = y_val_bal.astype(np.int32)
    X_test_bal = np.ascontiguousarray(X_test_bal.astype(np.float32))
    y_test_bal = y_test_bal.astype(np.int32)

    # XGBoost 参数（启用 early_stopping，早停基于 logloss 最小化）
    params = {
        'objective': 'binary:logistic',
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'random_state': seed,           # 使用传入的种子
        'n_estimators': 1000,          # 足够大，早停会提前停止
        'n_jobs': -1,                   # 使用所有CPU核心
        'use_label_encoder': False,
        'eval_metric': 'logloss',       # 早停监视的指标，选择 logloss 以保存 loss 最小的模型
        'tree_method': 'hist',          # 直方图方法，内存友好
        'importance_type': 'gain',
    }

    # 训练多个子模型
    n_models = min(10, train_dataset.K)
    models = []
    model_metrics_list = []

    for subset_idx in range(n_models):
        print(f"\n{'='*50}")
        print(f"训练第 {subset_idx+1}/{n_models} 个模型")
        print('='*50)

        # 切换到当前子集
        if subset_idx > 0:
            train_dataset.next_subset()

        # 从当前子集提取训练数据
        X_train_list, y_train_list = [], []
        num_samples = len(train_dataset)
        batch_size = 64
        for i in tqdm(range(0, num_samples, batch_size), desc="提取训练数据"):
            end = min(i + batch_size, num_samples)
            batch_X, batch_y = [], []
            for j in range(i, end):
                feat, lab = train_dataset[j]
                batch_X.append(feat.cpu().numpy().reshape(1, -1))
                batch_y.append(lab.cpu().numpy())
            if batch_X:
                X_train_list.append(np.vstack(batch_X))
                y_train_list.append(np.hstack(batch_y))
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)
        X_train = np.ascontiguousarray(X_train.astype(np.float32))
        y_train = y_train.astype(np.int32)

        print(f"当前子集大小: {X_train.shape}, 分布: {np.bincount(y_train)}")

        # 训练模型（early_stopping）
        model, val_metrics = train_single_model(
            X_train, y_train,
            X_val_bal, y_val_bal,
            params,
            early_stopping_rounds=20
        )
        models.append(model)
        model_metrics_list.append(val_metrics)

        print(f"模型 {subset_idx+1} 验证集性能:")
        for k, v in val_metrics.items():
            if k not in ['tp','tn','fp','fn']:
                print(f"  {k}: {v:.4f}")

    # 创建本次运行的输出目录
    output_dir = f'./xgboost_easyensemble_output_run_{run_id+1}'
    os.makedirs(output_dir, exist_ok=True)

    # 保存所有模型及指标
    for i, model in enumerate(models):
        joblib.dump(model, os.path.join(output_dir, f'xgboost_model_{i+1}.pkl'))

    # 保存指标列表（转换为普通浮点数以便JSON序列化）
    serializable_metrics = []
    for m in model_metrics_list:
        serializable_metrics.append({k: float(v) if isinstance(v, (np.floating, float)) else v
                                      for k, v in m.items() if k not in ['tp','tn','fp','fn']})
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(serializable_metrics, f, indent=2)

    # 计算字段重要性并保存CSV（不绘图）
    print("\n计算字段重要性...")
    field_importance_dict = compute_field_importance(models, field_names, feature_names)
    if field_importance_dict:
        imp_df = pd.DataFrame({
            'feature': list(field_importance_dict.keys()),
            'importance': list(field_importance_dict.values())
        }).sort_values('importance', ascending=False)
        imp_df.to_csv(os.path.join(output_dir, 'field_importance.csv'),
                      index=False, encoding='utf-8-sig')
    else:
        imp_df = None

    # 绘制ROC曲线（等权重平均）- 保留，因为是每次运行都需要的图
    plot_roc_curve_ensemble(models, X_test_bal, y_test_bal,
                            save_path=os.path.join(output_dir, 'roc_curve.png'))

    # 测试不同的集成方法
    ensemble_methods = [
        'equal_weighted',
        'accuracy_weighted',
        'sensitivity_weighted',
        ('selection_metric_weighted', 'f1'),  # 使用F1加权
        'max_voting'
    ]
    results = {}
    for method_item in ensemble_methods:
        if isinstance(method_item, tuple):
            method, sel = method_item
            metrics = test_ensemble_models(
                models, model_metrics_list, X_test_bal, y_test_bal,
                ensemble_method=method,
                selection_metric=sel
            )
        else:
            metrics = test_ensemble_models(
                models, model_metrics_list, X_test_bal, y_test_bal,
                ensemble_method=method_item,
                selection_metric=None
            )
        results[str(method_item)] = metrics

    # 保存集成测试结果
    with open(os.path.join(output_dir, 'ensemble_test_results.json'), 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items() if kk not in ['tp','tn','fp','fn']}
                   for k, v in results.items()}, f, indent=2)

    # 打印本次运行的汇总
    print("\n" + "="*60)
    print(f"实验 {run_id+1} 结果汇总")
    print("="*60)
    for method, metrics in results.items():
        print(f"\n集成方法: {method}")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  中风敏感度: {metrics['sensitivity']:.4f}")
        print(f"  特异度: {metrics['specificity']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")

    return results, field_importance_dict


def main():
    n_runs = 3
    base_seed = 42
    all_results = []
    all_field_importance_dicts = []  # 存储每次运行的字段重要性字典

    for run_id in range(n_runs):
        seed = base_seed + run_id
        results, field_imp_dict = run_single_experiment(run_id, seed)
        if results is not None:
            all_results.append(results)
            if field_imp_dict:
                all_field_importance_dicts.append(field_imp_dict)

    if not all_results:
        print("所有实验均失败！")
        return

    # 计算平均字段重要性
    if all_field_importance_dicts:
        # 获取所有字段名（假设每次运行字段相同）
        field_names = list(all_field_importance_dicts[0].keys())
        avg_importance = {}
        for f in field_names:
            values = [d.get(f, 0) for d in all_field_importance_dicts]
            avg_importance[f] = np.mean(values)

        # 绘制平均重要性图
        output_dir = './xgboost_easyensemble_output'
        os.makedirs(output_dir, exist_ok=True)
        plot_field_importance(avg_importance, top_n=15,
                              save_path=os.path.join(output_dir, 'field_importance_avg.png'))

        # ,title_suffix = "(Average over runs)"

        # 保存平均重要性 CSV
        avg_df = pd.DataFrame({
            'feature': list(avg_importance.keys()),
            'importance': list(avg_importance.values())
        }).sort_values('importance', ascending=False)
        avg_df.to_csv(os.path.join(output_dir, 'field_importance_avg.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"\n平均字段重要性已保存至 {output_dir}/field_importance_avg.csv")

    # 计算各集成方法下各指标的均值和标准差
    print("\n" + "="*80)
    print("多次实验结果汇总 (平均值 ± 标准差)")
    print("="*80)

    method_names = list(all_results[0].keys())
    for method in method_names:
        print(f"\n集成方法: {method}")
        acc_list = []
        prec_list = []
        sens_list = []
        spec_list = []
        f1_list = []
        auc_list = []

        for run_res in all_results:
            m = run_res[method]
            acc_list.append(m['accuracy'])
            prec_list.append(m['precision'])
            sens_list.append(m['sensitivity'])
            spec_list.append(m['specificity'])
            f1_list.append(m['f1'])
            auc_list.append(m['auc'])

        acc_mean = np.mean(acc_list)
        acc_std = np.std(acc_list)
        prec_mean = np.mean(prec_list)
        prec_std = np.std(prec_list)
        sens_mean = np.mean(sens_list)
        sens_std = np.std(sens_list)
        spec_mean = np.mean(spec_list)
        spec_std = np.std(spec_list)
        f1_mean = np.mean(f1_list)
        f1_std = np.std(f1_list)
        auc_mean = np.mean(auc_list)
        auc_std = np.std(auc_list)

        print(f"  准确率: {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"  精确率: {prec_mean:.4f} ± {prec_std:.4f}")
        print(f"  中风敏感度: {sens_mean:.4f} ± {sens_std:.4f}")
        print(f"  特异度: {spec_mean:.4f} ± {spec_std:.4f}")
        print(f"  F1: {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")


if __name__ == "__main__":
    main()