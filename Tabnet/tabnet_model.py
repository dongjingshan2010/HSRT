import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import joblib
import os
import json

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


# ==========================================
# 数据加载函数 - 使用 data_loader.py 中的函数
# ==========================================

def load_data_with_easyensemble(healthy_files, stroke_files, feature_dim=64, num_models=5):
    """
    使用EasyEnsemble方式加载数据

    Args:
        healthy_files: 健康数据文件路径字典
        stroke_files: 中风数据文件路径字典
        feature_dim: 特征维度
        num_models: 要训练的模型数量（EasyEnsemble子集数）

    Returns:
        包含训练、验证、测试数据的字典
    """
    # 导入 data_loader 模块
    from data_loader import load_and_process_data_by_year

    print(f"使用EasyEnsemble方式加载数据，将训练 {num_models} 个模型")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 使用 data_loader 中的函数加载数据
        train_loader, val_loader, test_loader, train_dataset, processor = load_and_process_data_by_year(
            healthy_files=healthy_files,
            stroke_files=stroke_files,
            feature_dim=feature_dim,
            batch_size=64,  # DataLoader的批次大小
            K=10  # EasyEnsemble的子集数量
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"GPU内存不足，尝试在CPU上处理数据...")
            # 强制使用CPU
            device = torch.device('cpu')
            # 重新导入并设置
            import data_loader
            data_loader.device = device

            # 重新加载数据
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
    print("提取验证数据...")
    val_features_list = []
    val_labels_list = []

    for batch_idx, batch in enumerate(val_loader):
        features, labels = batch
        val_features_list.append(features.cpu().numpy())
        val_labels_list.append(labels.cpu().numpy())

    val_features = np.vstack(val_features_list)
    val_labels = np.hstack(val_labels_list)

    # 获取特征维度信息
    if len(val_features.shape) == 3:
        # 三维特征：[样本数, 字段数, 特征维度]
        num_samples, num_fields, feat_dim = val_features.shape
        val_features_flat = val_features.reshape(num_samples, -1)
        total_features = num_fields * feat_dim
    else:
        # 已经是二维特征
        val_features_flat = val_features
        total_features = val_features_flat.shape[1]

    # 提取测试数据
    print("提取测试数据...")
    test_features_list = []
    test_labels_list = []

    for batch_idx, batch in enumerate(test_loader):
        features, labels = batch
        test_features_list.append(features.cpu().numpy())
        test_labels_list.append(labels.cpu().numpy())

    test_features = np.vstack(test_features_list)
    test_labels = np.hstack(test_labels_list)

    if len(test_features.shape) == 3:
        test_features_flat = test_features.reshape(test_features.shape[0], -1)
    else:
        test_features_flat = test_features

    # 生成字段名称
    field_names = []
    if hasattr(train_dataset, 'positive_embeddings'):
        num_fields = train_dataset.positive_embeddings.shape[1]
        feature_dim = train_dataset.positive_embeddings.shape[2]
    else:
        num_fields = total_features // feature_dim
        feature_dim = feature_dim

    # 从处理器获取字段名
    if hasattr(processor, 'field_names') and processor.field_names:
        field_names = processor.field_names
        print(f"从处理器获取字段名: {len(field_names)} 个字段")
    else:
        field_names = [f"field_{i}" for i in range(num_fields)]
        print(f"生成默认字段名: {len(field_names)} 个字段")

    # 为每个字段的每个特征维度生成名称
    feature_names = []
    for field_idx, field_name in enumerate(field_names):
        for dim_idx in range(feature_dim):
            feature_names.append(f"{field_name}_dim{dim_idx}")

    print(f"特征维度总数: {len(feature_names)} (字段数: {num_fields}, 每字段维度: {feature_dim})")

    # 清理GPU内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'X_val': val_features_flat,
        'y_val': val_labels,
        'X_test': test_features_flat,
        'y_test': test_labels,
        'feature_names': feature_names,
        'field_names': field_names,
        'num_fields': num_fields,
        'feature_dim': feature_dim,
        'train_dataset': train_dataset,  # EasyEnsemble训练数据集
        'processor': processor,
        'num_models': num_models
    }


def extract_subset_data(train_dataset, subset_idx):
    """
    从EasyEnsemble训练数据集中提取指定子集的数据

    Args:
        train_dataset: EasyEnsembleTrainDataset对象
        subset_idx: 子集索引（0-based）

    Returns:
        X_subset: 特征数据
        y_subset: 标签数据
    """
    # 确保切换到正确的子集
    if subset_idx > 0:
        # 切换到目标子集
        for _ in range(subset_idx):
            train_dataset.next_subset()

    # 获取当前子集的数据
    X_list = []
    y_list = []

    # 使用DataLoader来批量提取数据，提高效率
    batch_size = 64
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    for batch_features, batch_labels in dataloader:
        # 将特征展平
        batch_features_np = batch_features.cpu().numpy()
        if len(batch_features_np.shape) == 3:
            batch_features_flat = batch_features_np.reshape(batch_features_np.shape[0], -1)
        else:
            batch_features_flat = batch_features_np

        X_list.append(batch_features_flat)
        y_list.append(batch_labels.cpu().numpy())

    if X_list:
        X_subset = np.vstack(X_list)
        y_subset = np.hstack(y_list)
    else:
        # 备用方法：直接提取
        X_subset = []
        y_subset = []
        for i in range(len(train_dataset)):
            features, labels = train_dataset[i]
            features_flat = features.cpu().numpy().reshape(1, -1)
            X_subset.append(features_flat)
            y_subset.append(labels.cpu().numpy())

        X_subset = np.vstack(X_subset)
        y_subset = np.hstack(y_subset)

    return X_subset, y_subset


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
    return X[balanced_indices], y[balanced_indices]


def evaluate_ensemble(models, X_test, y_test):
    """
    评估集成模型

    Args:
        models: 模型列表
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        评估指标字典
    """
    # 集成预测：取所有模型预测概率的平均值
    all_probas = []
    for model in models:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        all_probas.append(y_pred_proba)

    # 平均概率
    avg_proba = np.mean(all_probas, axis=0)
    y_pred = (avg_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_test, avg_proba)

    print(f"\n=== EasyEnsemble集成评估结果 ({len(models)}个模型) ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"特异度: {specificity:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"精确率: {precision:.4f}")

    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'specificity': specificity, 'auc': auc
    }


def plot_roc_curve_ensemble(models, X_test, y_test, save_path=None):
    """绘制集成模型的ROC曲线"""
    from sklearn.metrics import roc_curve

    # 集成预测
    all_probas = []
    for model in models:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        all_probas.append(y_pred_proba)

    # 平均概率
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

    return auc_score


# ==========================================
# 字段级别特征重要性计算与绘图函数
# ==========================================

def compute_field_importance(models, field_names, feature_names):
    """
    计算字段级别的平均重要性（对每个模型归一化后平均）
    Args:
        models: 训练好的 TabNet 模型列表
        field_names: 字段名称列表
        feature_names: 所有特征的名称列表（字段名_dim*）
    Returns:
        字典 {field_name: avg_importance}
    """
    all_field_importances = []
    for model in models:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            field_importance = {}
            for field_name in field_names:
                # 找出属于当前字段的所有特征索引
                field_features = [f for f in feature_names if f.startswith(field_name)]
                if field_features:
                    indices = [feature_names.index(f) for f in field_features if f in feature_names]
                    if indices:
                        # 将该字段下所有维度的特征重要性求和，作为字段的重要性
                        field_importance[field_name] = sum(importances[idx] for idx in indices)
            # 归一化
            total = sum(field_importance.values())
            if total > 0:
                field_importance = {k: v / total for k, v in field_importance.items()}
            all_field_importances.append(field_importance)

    if not all_field_importances:
        return {}

    # 计算每个字段的平均重要性（各模型平均）
    avg_field_importance = {}
    for field_name in field_names:
        values = [imp.get(field_name, 0) for imp in all_field_importances]
        avg_field_importance[field_name] = np.mean(values)

    return avg_field_importance


def plot_field_importance(importance_dict, top_n=20, save_path=None, title_suffix=""):
    """根据字段重要性字典绘制水平条形图"""
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
        print(f"字段重要性图已保存至: {save_path}")

    plt.show()
    plt.close()
    return df


def plot_feature_importance_ensemble(models, field_names, feature_names, top_n=20, save_path=None):
    """保留原接口，内部调用新函数（用于单次运行）"""
    imp_dict = compute_field_importance(models, field_names, feature_names)
    if not imp_dict:
        print("没有可用的特征重要性数据")
        return None
    return plot_field_importance(imp_dict, top_n, save_path)


# ==========================================
# 单次实验函数
# ==========================================

def run_single_experiment(run_id, seed):
    """执行一次完整的训练和评估流程，返回集成评估指标和字段重要性字典"""
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
        2000: '../healthy16.csv',  # 假设这是00-16年健康数据
        2017: '../healthy17.csv',
        2018: '../healthy18.csv'
    }
    stroke_files = {
        2007: '../Stroke16.csv',  # 中风数据从2007年开始
        2017: '../Stroke17.csv',
        2018: '../Stroke18.csv'
    }

    print(f"健康数据文件: {len(healthy_files)} 个")
    print(f"中风数据文件: {len(stroke_files)} 个")

    if not healthy_files or not stroke_files:
        print("错误：找不到数据文件。")
        return None, None

    # 设置EasyEnsemble参数
    num_models = 3  # 训练10个模型

    try:
        # 使用EasyEnsemble方式加载数据
        data_dict = load_data_with_easyensemble(
            healthy_files=healthy_files,
            stroke_files=stroke_files,
            feature_dim=32,
            num_models=num_models
        )

        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        feature_names = data_dict['feature_names']
        field_names = data_dict['field_names']
        train_dataset = data_dict['train_dataset']

        print(f"\n数据加载成功:")
        print(f"  验证集: {X_val.shape}, 标签分布: {np.bincount(y_val.astype(int))}")
        print(f"  测试集: {X_test.shape}, 标签分布: {np.bincount(y_test.astype(int))}")

    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # 检查验证集和测试集是否平衡
    print("\n检查数据集平衡性...")
    val_class_counts = np.bincount(y_val.astype(int))
    test_class_counts = np.bincount(y_test.astype(int))

    print(f"验证集类别分布: 0:{val_class_counts[0]}, 1:{val_class_counts[1]}")
    print(f"测试集类别分布: 0:{test_class_counts[0]}, 1:{test_class_counts[1]}")

    # 验证集和测试集进行平衡采样（用于公平评估）
    if val_class_counts[0] != val_class_counts[1]:
        print("验证集不平衡，进行平衡采样...")
        X_val_bal, y_val_bal = balance_data_undersample(X_val, y_val)
    else:
        X_val_bal, y_val_bal = X_val, y_val
        print("验证集已平衡，无需采样")

    if test_class_counts[0] != test_class_counts[1]:
        print("测试集不平衡，进行平衡采样...")
        X_test_bal, y_test_bal = balance_data_undersample(X_test, y_test)
    else:
        X_test_bal, y_test_bal = X_test, y_test
        print("测试集已平衡，无需采样")

    print(f"\n最终数据集形状:")
    print(f"  验证集: {X_val_bal.shape}, 标签分布: {np.bincount(y_val_bal.astype(int))}")
    print(f"  测试集: {X_test_bal.shape}, 标签分布: {np.bincount(y_test_bal.astype(int))}")

    # TabNet 参数配置
    tabnet_init_params = {
        'n_d': 8,  # 决策层宽度
        'n_a': 8,  # 注意力层宽度
        'n_steps': 3,  # 决策步骤数
        'gamma': 1.3,  # 正则化参数
        'lambda_sparse': 1e-3,  # 稀疏性正则化
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': {'lr': 2e-2, 'weight_decay': 1e-5},
        'mask_type': 'entmax',  # 注意力掩码类型
        'n_shared': 2,  # 共享层数
        'n_independent': 2,  # 独立层数
        'momentum': 0.02,  # 批归一化动量
        'clip_value': None,  # 梯度裁剪
        'verbose': 0,  # 减少输出
        'device_name': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': seed,  # 部分版本支持seed参数，传入以防万一
    }

    # TabNet 训练参数
    tabnet_fit_params = {
        'max_epochs': 50,  # 减少最大训练轮次
        'patience': 5,  # 早停耐心值
        'batch_size': 64,  # 批次大小
        'virtual_batch_size': 64,  # 虚拟批次大小
    }

    # 使用EasyEnsemble训练多个TabNet模型
    print(f"\n使用EasyEnsemble训练TabNet模型 (训练{num_models}个模型)...")

    models = []

    for model_idx in range(num_models):
        print(f"\n训练第 {model_idx + 1}/{num_models} 个模型...")

        # 提取当前子集的数据
        print("提取当前子集数据...")
        X_train_subset, y_train_subset = extract_subset_data(train_dataset, model_idx)

        print(f"当前子集形状: {X_train_subset.shape}, 标签分布: {np.bincount(y_train_subset.astype(int))}")

        # 确保数据格式正确
        X_train_subset = np.ascontiguousarray(X_train_subset.astype(np.float32))
        y_train_subset = y_train_subset.astype(np.int32)

        # 创建并训练TabNet模型
        model = TabNetClassifier(**tabnet_init_params)

        try:
            model.fit(
                X_train=X_train_subset,
                y_train=y_train_subset,
                eval_set=[(X_val_bal, y_val_bal)],
                eval_name=['val'],
                eval_metric=['logloss'],  # 改为 logloss，确保保存验证集损失最小的模型
                max_epochs=tabnet_fit_params['max_epochs'],
                patience=tabnet_fit_params['patience'],
                batch_size=tabnet_fit_params['batch_size'],
                virtual_batch_size=tabnet_fit_params['virtual_batch_size'],
                num_workers=0,
                drop_last=False
            )

            models.append(model)
            print(f"第 {model_idx + 1} 个模型训练完成")

        except Exception as e:
            print(f"训练第 {model_idx + 1} 个模型时出错: {e}")
            # 尝试使用更小的批次大小
            print("尝试使用更小的批次大小...")
            try:
                model.fit(
                    X_train=X_train_subset,
                    y_train=y_train_subset,
                    eval_set=[(X_val_bal, y_val_bal)],
                    eval_name=['val'],
                    eval_metric=['logloss'],
                    max_epochs=30,
                    patience=5,
                    batch_size=64,
                    virtual_batch_size=32,
                    num_workers=0,
                    drop_last=False
                )

                models.append(model)
                print(f"第 {model_idx + 1} 个模型（小批次）训练完成")

            except Exception as e2:
                print(f"使用小批次也失败: {e2}")
                continue

    print(f"\nEasyEnsemble训练完成，共训练 {len(models)} 个模型")

    if not models:
        print("错误：没有训练出任何模型")
        return None, None

    # 评估集成模型
    metrics = evaluate_ensemble(models, X_test_bal, y_test_bal)

    # 创建本次运行的输出目录
    output_dir = f'./tabnet_easyensemble_output_run_{run_id+1}'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制ROC曲线（每次运行单独保存）
    auc_score = plot_roc_curve_ensemble(
        models,
        X_test_bal,
        y_test_bal,
        save_path=os.path.join(output_dir, 'roc_curve.png')
    )
    plt.show()
    plt.close()

    # 计算字段重要性（不绘图，仅返回字典）
    print("\n计算字段重要性...")
    field_importance_dict = compute_field_importance(models, field_names, feature_names)

    # 保存字段重要性CSV（单次运行）
    if field_importance_dict:
        imp_df = pd.DataFrame({
            'feature': list(field_importance_dict.keys()),
            'importance': list(field_importance_dict.values())
        }).sort_values('importance', ascending=False)
        imp_df.to_csv(os.path.join(output_dir, 'field_importance.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"字段重要性已保存至 {output_dir}/field_importance.csv")

    # 保存模型
    for i, model in enumerate(models):
        model.save_model(os.path.join(output_dir, f'tabnet_model_{i + 1}.zip'))

    # 保存模型参数
    joblib.dump({
        'init_params': tabnet_init_params,
        'fit_params': tabnet_fit_params,
        'num_models': len(models)
    }, os.path.join(output_dir, 'tabnet_params.pkl'))

    # 保存特征名
    with open(os.path.join(output_dir, 'feature_names.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'field_names.json'), 'w', encoding='utf-8') as f:
        json.dump(field_names, f, ensure_ascii=False, indent=2)

    # 保存评估结果
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n=== 本次运行评估结果 ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"\n本次运行结果已保存至 {output_dir}/")

    return metrics, field_importance_dict


# ==========================================
# 主函数：多次实验并汇总
# ==========================================

def main():
    n_runs = 3
    base_seed = 42
    all_metrics = []               # 每次运行的指标字典
    all_field_importance_dicts = [] # 每次运行的字段重要性字典

    for run_id in range(n_runs):
        seed = base_seed + run_id
        metrics, field_imp_dict = run_single_experiment(run_id, seed)
        if metrics is not None:
            all_metrics.append(metrics)
        if field_imp_dict:
            all_field_importance_dicts.append(field_imp_dict)

    if not all_metrics:
        print("所有实验均失败！")
        return

    # 计算平均字段重要性并绘图
    if all_field_importance_dicts:
        # 获取所有字段名（假设每次运行字段相同）
        field_names = list(all_field_importance_dicts[0].keys())
        avg_importance = {}
        for f in field_names:
            values = [d.get(f, 0) for d in all_field_importance_dicts]
            avg_importance[f] = np.mean(values)

        # 创建平均结果输出目录
        avg_output_dir = './tabnet_easyensemble_output'
        os.makedirs(avg_output_dir, exist_ok=True)

        # 绘制平均字段重要性图
        plot_field_importance(
            avg_importance,
            top_n=15,
            save_path=os.path.join(avg_output_dir, 'field_importance_avg.png')
        )

        # 保存平均重要性CSV
        avg_df = pd.DataFrame({
            'feature': list(avg_importance.keys()),
            'importance': list(avg_importance.values())
        }).sort_values('importance', ascending=False)
        avg_df.to_csv(os.path.join(avg_output_dir, 'field_importance_avg.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"\n平均字段重要性已保存至 {avg_output_dir}/field_importance_avg.csv")

    # 计算各指标的均值和标准差
    print("\n" + "="*80)
    print("多次实验结果汇总 (平均值 ± 标准差)")
    print("="*80)

    # 提取各指标列表
    acc_list = [m['accuracy'] for m in all_metrics]
    prec_list = [m['precision'] for m in all_metrics]
    recall_list = [m['recall'] for m in all_metrics]
    spec_list = [m['specificity'] for m in all_metrics]
    f1_list = [m['f1'] for m in all_metrics]
    auc_list = [m['auc'] for m in all_metrics]

    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    prec_mean, prec_std = np.mean(prec_list), np.std(prec_list)
    recall_mean, recall_std = np.mean(recall_list), np.std(recall_list)
    spec_mean, spec_std = np.mean(spec_list), np.std(spec_list)
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)

    print("\n集成方法: 等权重平均")
    print(f"  准确率: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  精确率: {prec_mean:.4f} ± {prec_std:.4f}")
    print(f"  召回率 (敏感度): {recall_mean:.4f} ± {recall_std:.4f}")
    print(f"  特异度: {spec_mean:.4f} ± {spec_std:.4f}")
    print(f"  F1: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")

    # 将汇总结果保存到文件
    summary = {
        'accuracy': {'mean': float(acc_mean), 'std': float(acc_std)},
        'precision': {'mean': float(prec_mean), 'std': float(prec_std)},
        'recall': {'mean': float(recall_mean), 'std': float(recall_std)},
        'specificity': {'mean': float(spec_mean), 'std': float(spec_std)},
        'f1': {'mean': float(f1_mean), 'std': float(f1_std)},
        'auc': {'mean': float(auc_mean), 'std': float(auc_std)},
    }
    with open(os.path.join(avg_output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总结果已保存至 {avg_output_dir}/summary.json")


if __name__ == "__main__":
    main()