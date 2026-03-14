"""
单次训练验证测试版本 - 支持EasyEnsemble多模型训练，带学习率调度器，支持多种评估指标选择最佳模型
"""

import json
import warnings
from GradientAnalyzer import GradientImportanceAnalyzer, StrategicGradientImportanceAnalyzer, AttentionAnalyzer
from data_loader import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    StepLR,
    MultiStepLR,
    ReduceLROnPlateau,
    CyclicLR,
    OneCycleLR,
    LambdaLR
)
from collections import Counter
import gc
import os
import math
from sklearn.metrics import roc_auc_score  # 新增导入AUC计算

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')

from HealthDataTransformer5 import HealthDataTransformeruniLSTM

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 新增：固定顺序LSTM模型（消融实验用）
# ==========================================
class HealthDataTransformeruniLSTM_FixedOrder(nn.Module):
    """
    消融模型：直接将特征按固定顺序输入LSTM，不使用Order-Agnostic Attention。
    输入形状: (batch_size, num_fields, feature_dim)
    输出: (batch_size, num_classes)
    """
    def __init__(self, num_fields, feature_dim, hidden_dim=512, num_layers=5, num_heads=4, num_classes=2):
        super(HealthDataTransformeruniLSTM_FixedOrder, self).__init__()
        self.num_fields = num_fields
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 可选：对输入特征进行线性变换（如果原始模型有这一层）
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # LSTM层：输入维度 hidden_dim，隐藏层维度 hidden_dim
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if num_layers > 1 else 0
        )

        # 分类头：取LSTM最后时间步的隐藏状态
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, num_fields, feature_dim)
        batch_size = x.size(0)

        # 线性投影
        x_proj = self.input_proj(x)  # (batch, num_fields, hidden_dim)

        # LSTM处理固定顺序序列
        lstm_out, (h_n, c_n) = self.lstm(x_proj)  # lstm_out: (batch, num_fields, hidden_dim)

        # 取最后一个时间步的隐藏状态（也可以是所有时间步的平均，这里取最后一个）
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # 分类
        logits = self.classifier(last_hidden)
        return logits

# ==========================================
# 学习率调度器工厂函数
# ==========================================
def create_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """
    创建学习率调度器

    参数:
        optimizer: 优化器
        scheduler_type: 调度器类型，可选值:
            - 'cosine': 余弦退火
            - 'cosine_warm_restarts': 带热重启的余弦退火
            - 'exponential': 指数衰减
            - 'step': 步长衰减
            - 'multi_step': 多步长衰减
            - 'plateau': 基于验证损失的衰减
            - 'cyclic': 循环学习率
            - 'one_cycle': 单周期学习率
            - 'warmup_cosine': 预热+余弦退火
            - 'warmup_linear': 预热+线性衰减
            - 'custom': 自定义lambda函数
        **kwargs: 调度器参数

    返回:
        学习率调度器
    """
    if scheduler_type == 'cosine':
        # 余弦退火
        T_max = kwargs.get('T_max', 50)  # 半周期数
        eta_min = kwargs.get('eta_min', 1e-6)  # 最小学习率
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_type == 'cosine_warm_restarts':
        # 带热重启的余弦退火
        T_0 = kwargs.get('T_0', 10)  # 第一次重启的周期数
        T_mult = kwargs.get('T_mult', 2)  # 重启周期倍增因子
        eta_min = kwargs.get('eta_min', 1e-6)
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    elif scheduler_type == 'exponential':
        # 指数衰减
        gamma = kwargs.get('gamma', 0.95)  # 衰减因子
        return ExponentialLR(optimizer, gamma=gamma)

    elif scheduler_type == 'step':
        # 步长衰减
        step_size = kwargs.get('step_size', 10)  # 每多少epoch衰减一次
        gamma = kwargs.get('gamma', 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == 'multi_step':
        # 多步长衰减
        milestones = kwargs.get('milestones', [20, 40, 60])  # 衰减点
        gamma = kwargs.get('gamma', 0.5)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_type == 'plateau':
        # 基于验证损失的衰减
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.5)
        patience = kwargs.get('patience', 5)
        threshold = kwargs.get('threshold', 1e-4)
        min_lr = kwargs.get('min_lr', 1e-6)
        return ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
            threshold=threshold, min_lr=min_lr, verbose=True
        )

    elif scheduler_type == 'cyclic':
        # 循环学习率
        base_lr = kwargs.get('base_lr', 1e-5)  # 基础学习率
        max_lr = kwargs.get('max_lr', 1e-3)  # 最大学习率
        step_size_up = kwargs.get('step_size_up', 2000)  # 上升步数
        step_size_down = kwargs.get('step_size_down', 2000)  # 下降步数
        mode = kwargs.get('mode', 'triangular')
        return CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr,
            step_size_up=step_size_up, step_size_down=step_size_down,
            mode=mode, cycle_momentum=False
        )

    elif scheduler_type == 'one_cycle':
        # 单周期学习率
        max_lr = kwargs.get('max_lr', 1e-3)
        total_steps = kwargs.get('total_steps', 100)
        pct_start = kwargs.get('pct_start', 0.3)  # 预热阶段比例
        anneal_strategy = kwargs.get('anneal_strategy', 'cos')
        return OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=pct_start, anneal_strategy=anneal_strategy
        )

    elif scheduler_type == 'warmup_cosine':
        # 预热+余弦退火（自定义）
        def warmup_cosine_decay(epoch):
            warmup_epochs = kwargs.get('warmup_epochs', 5)
            total_epochs = kwargs.get('total_epochs', 50)

            if epoch < warmup_epochs:
                # 线性预热
                return (epoch + 1) / warmup_epochs
            else:
                # 余弦退火
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=warmup_cosine_decay)

    elif scheduler_type == 'warmup_linear':
        # 预热+线性衰减（自定义）
        def warmup_linear_decay(epoch):
            warmup_epochs = kwargs.get('warmup_epochs', 5)
            total_epochs = kwargs.get('total_epochs', 50)

            if epoch < warmup_epochs:
                # 线性预热
                return (epoch + 1) / warmup_epochs
            else:
                # 线性衰减
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return max(0.0, 1.0 - progress)

        return LambdaLR(optimizer, lr_lambda=warmup_linear_decay)

    elif scheduler_type == 'custom':
        # 自定义lambda函数
        lr_lambda = kwargs.get('lr_lambda', lambda epoch: 0.95 ** epoch)
        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        # 默认不使用调度器
        return None


# ==========================================
# 训练和评估函数 - 支持EasyEnsemble多模型训练
# ==========================================

def calculate_class_metrics(predictions, labels, num_classes=2):
    """
    计算各类别的准确率、敏感度（召回率）
    """
    class_accuracy = []
    class_sensitivity = []  # 召回率/敏感度

    for i in range(num_classes):
        class_mask = (labels == i)

        if class_mask.sum() > 0:
            class_correct = ((predictions == i) & (labels == i)).sum()
            class_total = class_mask.sum()
            class_acc = 100 * class_correct / class_total if class_total > 0 else 0

            tp = ((predictions == i) & (labels == i)).sum()
            fn = ((predictions != i) & (labels == i)).sum()
            class_sens = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0

            class_accuracy.append(class_acc)
            class_sensitivity.append(class_sens)
        else:
            class_accuracy.append(0)
            class_sensitivity.append(0)

    return class_accuracy, class_sensitivity


def calculate_confusion_matrix(predictions, labels, num_classes=2):
    """
    计算混淆矩阵及其相关指标
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(predictions)):
        true_label = labels[i]
        pred_label = predictions[i]
        cm[true_label, pred_label] += 1

    metrics = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2

        metrics[f'class_{i}_sensitivity'] = sensitivity
        metrics[f'class_{i}_specificity'] = specificity
        metrics[f'class_{i}_precision'] = precision
        metrics[f'class_{i}_f1_score'] = f1_score
        metrics[f'class_{i}_balanced_accuracy'] = balanced_accuracy

    return cm, metrics


def calculate_combined_metrics(val_class_metrics, selection_metric, alpha=0.7):
    """
    根据选择指标计算相应的分数

    参数:
        val_class_metrics: 验证集类别指标字典
        selection_metric: 选择指标类型
        alpha: 加权分数中敏感度的权重（仅当selection_metric='weighted_score'时有效）

    返回:
        metric_value: 指标值
        metric_name: 指标名称
    """
    if selection_metric == 'accuracy':
        metric_value = 0.0  # 将在外部计算
        metric_name = "准确率"
    elif selection_metric == 'sensitivity_stroke':
        metric_value = val_class_metrics['class_1_sensitivity'] * 100
        metric_name = "中风机别敏感度"
    elif selection_metric == 'specificity_stroke':
        metric_value = val_class_metrics['class_1_specificity'] * 100
        metric_name = "中风机别特异性"
    elif selection_metric == 'f1_score':
        metric_value = val_class_metrics['class_1_f1_score'] * 100
        metric_name = "F1分数"
    elif selection_metric == 'balanced_accuracy':
        metric_value = val_class_metrics['class_1_balanced_accuracy'] * 100
        metric_name = "平衡准确率"
    elif selection_metric == 'g_mean':
        # 几何平均数（敏感度和特异性的几何平均）
        sensitivity = val_class_metrics['class_1_sensitivity']
        specificity = val_class_metrics['class_1_specificity']
        g_mean = math.sqrt(sensitivity * specificity) if sensitivity > 0 and specificity > 0 else 0
        metric_value = g_mean * 100
        metric_name = "几何平均数(G-mean)"
    elif selection_metric == 'weighted_score':
        # 加权分数（alpha * 敏感度 + (1-alpha) * 准确率）
        sensitivity = val_class_metrics['class_1_sensitivity']
        specificity = val_class_metrics['class_1_specificity']
        weighted_score = alpha * sensitivity + (1 - alpha) * specificity
        metric_value = weighted_score * 100
        metric_name = f"加权分数(α={alpha})"
    elif selection_metric == 'custom_metric':
        # 自定义指标：F1分数和平衡准确率的加权平均
        f1_score = val_class_metrics['class_1_f1_score']
        balanced_accuracy = val_class_metrics['class_1_balanced_accuracy']
        custom_metric = 0.7 * f1_score + 0.3 * balanced_accuracy
        metric_value = custom_metric * 100
        metric_name = "自定义指标(F1+平衡准确率)"
    # 新增AUC选项（实际值需在外部传入，此处仅返回占位，由调用处覆盖）
    elif selection_metric == 'auc':
        metric_value = 0.0
        metric_name = "AUC"
    else:
        # 默认使用中风机别敏感度
        metric_value = val_class_metrics['class_1_sensitivity'] * 100
        metric_name = "中风机别敏感度"

    return metric_value, metric_name


def train_single_model(model, train_dataset, train_loader, val_loader,
                       num_epochs=50, learning_rate=0.001,
                       patience=5, save_path='./output/best_model.pth',
                       model_idx=0, total_models=1,
                       scheduler_config=None,
                       selection_metric='sensitivity_stroke',
                       alpha=0.7):
    """
    训练单个模型，支持以不同指标保存最佳模型

    参数:
        scheduler_config: 调度器配置字典
        selection_metric: 选择最佳模型的指标，可选值:
            - 'accuracy': 准确率
            - 'sensitivity_stroke': 中风机别敏感度（推荐）
            - 'specificity_stroke': 中风机别特异性
            - 'f1_score': F1分数
            - 'balanced_accuracy': 平衡准确率
            - 'g_mean': 几何平均数(G-mean)
            - 'weighted_score': 加权分数（alpha * 敏感度 + (1-alpha) * 特异性）
            - 'custom_metric': 自定义指标（F1分数和平衡准确率的加权平均）
            - 'auc': AUC值（新增）
        alpha: 加权分数中敏感度的权重（仅当selection_metric='weighted_score'时有效）
    """
    model = model.to(device)
    print(f"训练第 {model_idx + 1}/{total_models} 个模型, 设备: {next(model.parameters()).device}")
    print(f"选择指标: {selection_metric}")

    if model_idx > 0:
        train_dataset.next_subset()

    print("计算类别权重...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels)

    class_counts = torch.bincount(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)

    class_weights = total_samples / (num_classes * class_counts.float())

    print(f"类别分布: {class_counts.tolist()}")
    print(f"类别权重: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建学习率调度器
    scheduler = None
    if scheduler_config:
        scheduler_type = scheduler_config.get('type', 'cosine')
        scheduler_params = scheduler_config.copy()
        scheduler_params.pop('type', None)

        # 为特定调度器设置默认参数
        if scheduler_type == 'cosine':
            scheduler_params.setdefault('T_max', num_epochs)
            scheduler_params.setdefault('eta_min', 1e-6)
        elif scheduler_type == 'warmup_cosine' or scheduler_type == 'warmup_linear':
            scheduler_params.setdefault('total_epochs', num_epochs)
            scheduler_params.setdefault('warmup_epochs', max(3, num_epochs // 10))

        scheduler = create_scheduler(optimizer, scheduler_type, **scheduler_params)
        print(f"使用学习率调度器: {scheduler_type}, 参数: {scheduler_params}")
    else:
        print("未使用学习率调度器")

    # 初始化最佳指标
    best_metric = 0.0
    best_val_acc = 0.0
    best_val_sensitivity_stroke = 0.0
    best_val_specificity_stroke = 0.0
    best_val_auc = 0.0  # 新增记录最佳AUC
    best_epoch = 0
    patience_counter = 0

    train_metrics_history = {
        'loss': [], 'accuracy': [],
        'sensitivity_stroke': [], 'specificity_stroke': [],
        'learning_rate': []  # 记录学习率变化
    }
    val_metrics_history = {
        'loss': [], 'accuracy': [],
        'sensitivity_stroke': [], 'specificity_stroke': [],
        'auc': []  # 新增记录验证集AUC
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        all_train_preds = []
        all_train_labels = []

        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(batch_labels.cpu().numpy())

            del outputs, predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_accuracy = 100 * train_correct / train_total

        train_cm, train_class_metrics = calculate_confusion_matrix(
            np.array(all_train_preds),
            np.array(all_train_labels)
        )

        train_sensitivity_stroke = train_class_metrics['class_1_sensitivity'] * 100  # 中风机别敏感度
        train_specificity_stroke = train_class_metrics['class_1_specificity'] * 100  # 中风机别特异性

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        train_metrics_history['learning_rate'].append(current_lr)

        train_metrics_history['loss'].append(train_loss / len(train_loader))
        train_metrics_history['accuracy'].append(train_accuracy)
        train_metrics_history['sensitivity_stroke'].append(train_sensitivity_stroke)
        train_metrics_history['specificity_stroke'].append(train_specificity_stroke)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        all_val_preds = []
        all_val_labels = []
        all_val_probs = []  # 新增收集概率用于AUC计算

        with torch.no_grad():
            for batch_embeddings, batch_labels in val_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(batch_labels.cpu().numpy())
                all_val_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率

                del outputs, predicted, probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        val_accuracy = 100 * val_correct / val_total

        val_cm, val_class_metrics = calculate_confusion_matrix(
            np.array(all_val_preds),
            np.array(all_val_labels)
        )

        val_sensitivity_stroke = val_class_metrics['class_1_sensitivity'] * 100  # 中风机别敏感度
        val_specificity_stroke = val_class_metrics['class_1_specificity'] * 100  # 中风机别特异性
        val_auc = roc_auc_score(all_val_labels, all_val_probs)  # 计算AUC

        # 计算选择指标
        if selection_metric == 'auc':
            metric_value = val_auc * 100
            metric_name = "AUC"
        elif selection_metric == 'accuracy':
            metric_value = val_accuracy
            metric_name = "准确率"
        else:
            metric_value, metric_name = calculate_combined_metrics(
                val_class_metrics,
                selection_metric,
                alpha
            )

        val_metrics_history['loss'].append(val_loss / len(val_loader))
        val_metrics_history['accuracy'].append(val_accuracy)
        val_metrics_history['sensitivity_stroke'].append(val_sensitivity_stroke)
        val_metrics_history['specificity_stroke'].append(val_specificity_stroke)
        val_metrics_history['auc'].append(val_auc * 100)

        print(f'\n模型 {model_idx + 1}, Epoch [{epoch + 1}/{num_epochs}], '
              f'LR: {current_lr:.6f}, '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Val Acc: {val_accuracy:.2f}%, '
              f'Val Sens(中风): {val_sensitivity_stroke:.2f}%, '
              f'Val AUC: {val_auc * 100:.2f}%, '
              f'{metric_name}: {metric_value:.2f}%')

        # 更新学习率调度器
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                # ReduceLROnPlateau需要验证损失
                scheduler.step(val_loss / len(val_loader))
            else:
                # 其他调度器按照epoch更新
                scheduler.step()

        # 根据选择指标判断是否保存模型
        if metric_value > best_metric:
            best_metric = metric_value
            best_val_acc = val_accuracy
            best_val_sensitivity_stroke = val_sensitivity_stroke
            best_val_specificity_stroke = val_specificity_stroke
            best_val_auc = val_auc * 100
            best_epoch = epoch + 1
            patience_counter = 0

            model_save_path = save_path.replace('.pth', f'_model{model_idx + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_accuracy': val_accuracy,
                'val_sensitivity_stroke': val_sensitivity_stroke,
                'val_specificity_stroke': val_specificity_stroke,
                'val_auc': val_auc * 100,  # 保存AUC值
                f'val_{selection_metric}': metric_value,
                'train_accuracy': train_accuracy,
                'train_sensitivity_stroke': train_sensitivity_stroke,
                'class_weights': class_weights,
                'model_idx': model_idx,
                'selection_metric': selection_metric,
                'scheduler_config': scheduler_config,
                'current_lr': current_lr,
            }, model_save_path, _use_new_zipfile_serialization=True)
            print(f"第 {model_idx + 1} 个模型已保存! "
                  f"{metric_name}: {metric_value:.2f}%, "
                  f"验证准确率: {val_accuracy:.2f}%, "
                  f"验证AUC: {val_auc * 100:.2f}%, "
                  f"中风类别敏感度: {val_sensitivity_stroke:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"第 {model_idx + 1} 个模型早停触发! 在 epoch {epoch + 1} 停止训练")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n第 {model_idx + 1} 个模型训练完成! ")
    print(f"最佳{metric_name}: {best_metric:.2f}% (epoch {best_epoch})")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最佳验证AUC: {best_val_auc:.2f}%")
    print(f"最佳中风类别敏感度: {best_val_sensitivity_stroke:.2f}%")
    print(f"最佳中风类别特异性: {best_val_specificity_stroke:.2f}%")

    return best_val_acc, model, train_metrics_history, val_metrics_history


def train_easyensemble_models(model_class, train_dataset, train_loader, val_loader,
                              num_models=5, num_epochs=50, learning_rate=0.001,
                              patience=5, save_dir='./output',
                              scheduler_config=None,
                              selection_metric='sensitivity_stroke',
                              alpha=0.7):
    """
    训练EasyEnsemble多个模型，支持以不同指标保存最佳模型

    参数:
        selection_metric: 选择最佳模型的指标
        alpha: 加权分数中敏感度的权重（仅当selection_metric='weighted_score'时有效）
    """
    os.makedirs(save_dir, exist_ok=True)

    models = []
    model_accuracies = []
    model_selection_metrics = []  # 存储每个模型的选择指标值
    model_sensitivities_stroke = []  # 存储每个模型的最佳中风机别敏感度
    model_aucs = []  # 新增存储每个模型的最佳AUC
    all_train_histories = []
    all_val_histories = []

    num_fields = train_dataset.positive_embeddings.shape[1] if hasattr(train_dataset, 'positive_embeddings') else 0
    feature_dim = train_dataset.positive_embeddings.shape[2] if hasattr(train_dataset, 'positive_embeddings') else 32

    print(f"训练 {num_models} 个EasyEnsemble模型...")
    print(f"每个模型输入字段数: {num_fields}")
    print(f"字段特征维度: {feature_dim}")
    print(f"选择指标: {selection_metric}")

    if scheduler_config:
        print(f"学习率调度器配置: {scheduler_config}")

    for i in range(min(num_models, train_dataset.K)):
        print(f"\n{'=' * 60}")
        print(f"开始训练第 {i + 1}/{min(num_models, train_dataset.K)} 个模型")
        print(f"{'=' * 60}")

        model = model_class(
            num_fields=num_fields,
            feature_dim=feature_dim,
            hidden_dim=512,
            num_layers=5,
            num_heads=4,
            num_classes=2
        )

        best_acc, trained_model, train_history, val_history = train_single_model(
            model=model,
            train_dataset=train_dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            patience=patience,
            save_path=os.path.join(save_dir, 'best_health_transformer_model.pth'),
            model_idx=i,
            total_models=min(num_models, train_dataset.K),
            scheduler_config=scheduler_config,
            selection_metric=selection_metric,
            alpha=alpha
        )

        models.append(trained_model)
        model_accuracies.append(best_acc)
        all_train_histories.append(train_history)
        all_val_histories.append(val_history)

        model_path = os.path.join(save_dir, f'best_health_transformer_model_model{i + 1}.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'val_sensitivity_stroke' in checkpoint:
                model_sensitivities_stroke.append(checkpoint['val_sensitivity_stroke'])
            else:
                model_sensitivities_stroke.append(0.0)

            if 'val_auc' in checkpoint:
                model_aucs.append(checkpoint['val_auc'])
            else:
                model_aucs.append(0.0)

            # 获取选择指标值
            if f'val_{selection_metric}' in checkpoint:
                model_selection_metrics.append(checkpoint[f'val_{selection_metric}'])
            elif selection_metric == 'sensitivity_stroke' and 'val_sensitivity_stroke' in checkpoint:
                model_selection_metrics.append(checkpoint['val_sensitivity_stroke'])
            elif selection_metric == 'accuracy' and 'val_accuracy' in checkpoint:
                model_selection_metrics.append(checkpoint['val_accuracy'])
            elif selection_metric == 'auc' and 'val_auc' in checkpoint:
                model_selection_metrics.append(checkpoint['val_auc'])
            else:
                model_selection_metrics.append(0.0)
        else:
            model_sensitivities_stroke.append(0.0)
            model_aucs.append(0.0)
            model_selection_metrics.append(0.0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble_info = {
        'num_models': len(models),
        'model_accuracies': model_accuracies,
        'model_sensitivities_stroke': model_sensitivities_stroke,
        'model_aucs': model_aucs,
        'model_selection_metrics': model_selection_metrics,
        'selection_metric': selection_metric,
        'avg_accuracy': np.mean(model_accuracies),
        'avg_sensitivity_stroke': np.mean(model_sensitivities_stroke),
        'avg_auc': np.mean(model_aucs),
        'avg_selection_metric': np.mean(model_selection_metrics),
        'max_accuracy': np.max(model_accuracies),
        'max_sensitivity_stroke': np.max(model_sensitivities_stroke),
        'max_auc': np.max(model_aucs),
        'max_selection_metric': np.max(model_selection_metrics),
        'min_accuracy': np.min(model_accuracies),
        'min_sensitivity_stroke': np.min(model_sensitivities_stroke),
        'min_auc': np.min(model_aucs),
        'min_selection_metric': np.min(model_selection_metrics),
        'scheduler_config': scheduler_config,
        'alpha': alpha if selection_metric == 'weighted_score' else None
    }

    with open(os.path.join(save_dir, 'ensemble_info.json'), 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                   for k, v in ensemble_info.items()}, f, indent=2)

    training_history = {
        'train_histories': all_train_histories,
        'val_histories': all_val_histories,
        'model_accuracies': model_accuracies,
        'model_sensitivities_stroke': model_sensitivities_stroke,
        'model_aucs': model_aucs,
        'model_selection_metrics': model_selection_metrics,
        'selection_metric': selection_metric
    }
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        serializable_history = {}
        for key, value in training_history.items():
            if key in ['train_histories', 'val_histories']:
                serializable_history[key] = []
                for hist in value:
                    serializable_hist = {}
                    for metric_name, metric_values in hist.items():
                        serializable_hist[metric_name] = [float(v) for v in metric_values]
                    serializable_history[key].append(serializable_hist)
            elif key in ['model_accuracies', 'model_sensitivities_stroke', 'model_aucs', 'model_selection_metrics']:
                serializable_history[key] = [float(v) for v in value]
            else:
                serializable_history[key] = value
        json.dump(serializable_history, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EasyEnsemble训练完成!")
    print(f"{'=' * 60}")
    print(f"训练模型数量: {len(models)}")
    print(f"选择指标: {selection_metric}")
    print(f"模型验证准确率: {model_accuracies}")
    print(f"模型中风类别敏感度: {model_sensitivities_stroke}")
    print(f"模型验证AUC: {model_aucs}")
    print(f"模型选择指标值: {model_selection_metrics}")
    print(f"平均验证准确率: {np.mean(model_accuracies):.2f}%")
    print(f"平均中风类别敏感度: {np.mean(model_sensitivities_stroke):.2f}%")
    print(f"平均验证AUC: {np.mean(model_aucs):.2f}%")
    print(f"平均选择指标值: {np.mean(model_selection_metrics):.2f}%")
    print(f"最佳模型准确率: {np.max(model_accuracies):.2f}%")
    print(f"最佳模型中风机别敏感度: {np.max(model_sensitivities_stroke):.2f}%")
    print(f"最佳模型验证AUC: {np.max(model_aucs):.2f}%")
    print(f"最佳模型选择指标值: {np.max(model_selection_metrics):.2f}%")

    if scheduler_config:
        print(f"使用的学习率调度器: {scheduler_config.get('type', 'None')}")

    return models, model_accuracies, model_sensitivities_stroke, model_selection_metrics


def test_ensemble_models(models, test_loader, model_dir='./output',
                         ensemble_method='selection_metric_weighted',
                         selection_metric='sensitivity_stroke'):
    """
    在测试集上评估EasyEnsemble集成模型

    参数:
        ensemble_method: 集成方法，可选值:
            - 'selection_metric_weighted': 使用选择指标加权平均（默认）
            - 'sensitivity_weighted': 中风类别敏感度加权平均
            - 'accuracy_weighted': 准确率加权平均
            - 'equal_weighted': 等权重平均
            - 'max_voting': 最大投票法
        selection_metric: 选择指标（仅当ensemble_method='selection_metric_weighted'时有效）
    """
    if not models:
        print("没有可用的模型进行测试")
        return None

    print(f"\n{'=' * 60}")
    print(f"在测试集上评估EasyEnsemble集成模型 ({len(models)}个模型)")
    print(f"集成方法: {ensemble_method}")
    if ensemble_method == 'selection_metric_weighted':
        print(f"选择指标: {selection_metric}")
    print(f"{'=' * 60}")

    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_models_predictions = []
    all_models_probabilities = []

    model_weights_list = []
    model_sensitivities_stroke = []
    model_accuracies = []
    model_aucs = []  # 新增存储每个模型的最佳AUC
    model_selection_metrics = []

    for model_idx, model in enumerate(models):
        print(f"加载第 {model_idx + 1} 个模型...")

        model_path = os.path.join(model_dir, f'best_health_transformer_model_model{model_idx + 1}.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])

            # 获取各种指标
            if 'val_sensitivity_stroke' in checkpoint:
                sensitivity_stroke = checkpoint['val_sensitivity_stroke']
                model_sensitivities_stroke.append(sensitivity_stroke)
            else:
                sensitivity_stroke = 0.0
                model_sensitivities_stroke.append(sensitivity_stroke)

            if 'val_accuracy' in checkpoint:
                accuracy = checkpoint['val_accuracy']
                model_accuracies.append(accuracy)
            else:
                accuracy = 0.0
                model_accuracies.append(accuracy)

            if 'val_auc' in checkpoint:
                auc_val = checkpoint['val_auc']
                model_aucs.append(auc_val)
            else:
                auc_val = 0.0
                model_aucs.append(auc_val)

            # 获取选择指标
            selection_metric_value = 0.0
            if f'val_{selection_metric}' in checkpoint:
                selection_metric_value = checkpoint[f'val_{selection_metric}']
            elif selection_metric == 'sensitivity_stroke' and 'val_sensitivity_stroke' in checkpoint:
                selection_metric_value = checkpoint['val_sensitivity_stroke']
            elif selection_metric == 'accuracy' and 'val_accuracy' in checkpoint:
                selection_metric_value = checkpoint['val_accuracy']
            elif selection_metric == 'auc' and 'val_auc' in checkpoint:
                selection_metric_value = checkpoint['val_auc']

            model_selection_metrics.append(selection_metric_value)

            print(f"  已加载最佳模型 (验证准确率: {accuracy:.2f}%, "
                  f"中风类别敏感度: {sensitivity_stroke:.2f}%, "
                  f"验证AUC: {auc_val:.2f}%, "
                  f"{selection_metric}: {selection_metric_value:.2f}%)")
        else:
            sensitivity_stroke = 0.0
            accuracy = 0.0
            auc_val = 0.0
            selection_metric_value = 0.0
            model_sensitivities_stroke.append(sensitivity_stroke)
            model_accuracies.append(accuracy)
            model_aucs.append(auc_val)
            model_selection_metrics.append(selection_metric_value)
            print(f"  未找到保存的模型，使用当前模型状态")

        model.eval()

    # 根据集成方法计算权重
    if ensemble_method == 'selection_metric_weighted':
        model_weights = np.array(model_selection_metrics) / np.sum(model_selection_metrics) if np.sum(
            model_selection_metrics) > 0 else np.ones(len(models)) / len(models)
        print(f"\n模型权重（基于验证集{selection_metric}）：")
    elif ensemble_method == 'sensitivity_weighted':

        T = 0.5  # 温度参数，T<1 放大差异，T>1 缩小差异
        exp_scores = np.exp(np.array(model_sensitivities_stroke) / T)
        model_weights = exp_scores / np.sum(exp_scores)

        # model_weights = np.array(model_sensitivities_stroke) / np.sum(model_sensitivities_stroke) if np.sum(
        #     model_sensitivities_stroke) > 0 else np.ones(len(models)) / len(models)
        print(f"\n模型权重（基于验证集中风类别敏感度）：")
    elif ensemble_method == 'accuracy_weighted':
        model_weights = np.array(model_accuracies) / np.sum(model_accuracies) if np.sum(
            model_accuracies) > 0 else np.ones(len(models)) / len(models)
        print(f"\n模型权重（基于验证集准确率）：")
    elif ensemble_method == 'equal_weighted':
        model_weights = np.ones(len(models)) / len(models)
        print(f"\n模型权重（等权重）：")
    elif ensemble_method == 'max_voting':
        model_weights = np.ones(len(models))  # 最大投票法不需要权重，但为了统一代码，我们使用等权重
        print(f"\n集成方法：最大投票法")
    else:
        model_weights = np.array(model_sensitivities_stroke) / np.sum(model_sensitivities_stroke) if np.sum(
            model_sensitivities_stroke) > 0 else np.ones(len(models)) / len(models)
        print(f"\n模型权重（基于验证集中风类别敏感度，默认）：")

    for i, weight in enumerate(model_weights):
        print(f"  模型 {i + 1}: 权重={weight:.4f}, "
              f"准确率={model_accuracies[i]:.2f}%, "
              f"敏感度={model_sensitivities_stroke[i]:.2f}%, "
              f"AUC={model_aucs[i]:.2f}%")

    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            batch_model_predictions = []
            batch_model_probabilities = []

            for model in models:
                outputs = model(batch_embeddings)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                batch_model_predictions.append(predicted.cpu().numpy())
                batch_model_probabilities.append(probabilities.cpu().numpy()[:, 1])

            batch_model_probabilities_np = np.array(batch_model_probabilities)

            if ensemble_method == 'max_voting':
                # 最大投票法
                batch_model_predictions_np = np.array(batch_model_predictions)
                ensemble_predictions = []
                for j in range(batch_model_predictions_np.shape[1]):
                    votes = batch_model_predictions_np[:, j]
                    # 统计每个类别的票数
                    unique, counts = np.unique(votes, return_counts=True)
                    # 选择票数最多的类别，如果平票则选择概率平均较高的类别
                    max_votes = np.max(counts)
                    candidates = unique[counts == max_votes]
                    if len(candidates) == 1:
                        ensemble_predictions.append(candidates[0])
                    else:
                        # 平票时使用加权概率平均
                        avg_probs = np.average(batch_model_probabilities_np[:, j], weights=model_weights)
                        ensemble_predictions.append(1 if avg_probs >= 0.5 else 0)
                ensemble_predictions = np.array(ensemble_predictions)
                weighted_avg_probabilities = np.average(batch_model_probabilities_np, axis=0, weights=model_weights)
            else:
                # 加权平均法
                weighted_avg_probabilities = np.average(
                    batch_model_probabilities_np,
                    axis=0,
                    weights=model_weights
                )
                ensemble_predictions = (weighted_avg_probabilities >= 0.5).astype(int)

            all_predictions.extend(ensemble_predictions)
            all_labels.extend(batch_labels.cpu().numpy())
            all_probabilities.extend(weighted_avg_probabilities)

            if len(all_models_predictions) == 0:
                for i in range(len(models)):
                    all_models_predictions.append([])
                    all_models_probabilities.append([])
            for i, (pred, prob) in enumerate(zip(batch_model_predictions, batch_model_probabilities)):
                all_models_predictions[i].extend(pred)
                all_models_probabilities[i].extend(prob)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    test_accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions)

    tn, fp, fn, tp = cm.ravel()

    stroke_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    stroke_precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\n=== EasyEnsemble集成模型测试结果（{ensemble_method}）===")
    print(f"测试集大小: {len(all_labels)}")
    print(f"集成模型测试准确率: {test_accuracy:.2f}%")

    print(f"\n=== 各模型测试性能 ===")
    for i, (model_preds, model_probs) in enumerate(zip(all_models_predictions, all_models_probabilities)):
        model_acc = 100 * (np.array(model_preds) == all_labels).sum() / len(all_labels)

        model_cm = confusion_matrix(all_labels, np.array(model_preds))
        if model_cm.size == 4:
            model_tn, model_fp, model_fn, model_tp = model_cm.ravel()
            model_stroke_sens = 100 * model_tp / (model_tp + model_fn) if (model_tp + model_fn) > 0 else 0
        else:
            model_stroke_sens = 0

        print(f"模型 {i + 1}: 准确率={model_acc:.2f}%, "
              f"中风类别敏感度={model_stroke_sens:.2f}%, "
              f"权重={model_weights[i]:.4f}")

    print(f"\n=== 各类别性能 ===")

    stroke_mask = (all_labels == 1)
    if stroke_mask.sum() > 0:
        stroke_accuracy = 100 * (all_predictions[stroke_mask] == all_labels[stroke_mask]).sum() / stroke_mask.sum()
        print(f"中风类别: 准确率={stroke_accuracy:.2f}%, "
              f"敏感度/召回率={stroke_sensitivity * 100:.2f}%, "
              f"精确率={stroke_precision * 100:.2f}% (样本数: {stroke_mask.sum()})")

    auc_score = 0.0
    try:
        auc_score = roc_auc_score(all_labels, all_probabilities)
        print(f"AUC: {auc_score:.4f}")
    except Exception as e:
        print(f"AUC计算失败: {e}")

    print("\n混淆矩阵:")
    print(f"         预测健康   预测中风")
    print(f"实际健康  {cm[0, 0]:>8}  {cm[0, 1]:>8}")
    print(f"实际中风  {cm[1, 0]:>8}  {cm[1, 1]:>8}")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = stroke_sensitivity
    f1_score = 2 * stroke_precision * recall / (stroke_precision + recall) if (stroke_precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    g_mean = math.sqrt(recall * specificity) if recall > 0 and specificity > 0 else 0
    balanced_accuracy = (recall + specificity) / 2

    print(f"\n详细指标:")
    print(f"精确率 (Precision): {stroke_precision:.4f}")
    print(f"召回率/敏感度 (Recall/Sensitivity): {recall:.4f}")
    print(f"特异性 (Specificity): {specificity:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    print(f"几何平均数(G-mean): {g_mean:.4f}")
    print(f"平衡准确率 (Balanced Accuracy): {balanced_accuracy:.4f}")

    metrics = {
        'accuracy': test_accuracy,
        'auc': auc_score,
        'precision': stroke_precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'g_mean': g_mean,
        'balanced_accuracy': balanced_accuracy,
        'ensemble_size': len(models),
        'ensemble_method': ensemble_method,
        'selection_metric': selection_metric if ensemble_method == 'selection_metric_weighted' else None,
        'model_weights': model_weights.tolist(),
        'stroke_sensitivity': stroke_sensitivity,
        'stroke_accuracy': stroke_accuracy if 'stroke_accuracy' in locals() else 0
    }

    return metrics


# ==========================================
# 分析函数
# ==========================================

def merge_field_importances(importance_list, method='weighted_average', weights=None):
    """
    合并多种字段重要性计算方法的结果
    """
    if not importance_list:
        return {}

    if weights is None:
        weights = [1.0] * len(importance_list)

    if len(weights) != len(importance_list):
        weights = [1.0] * len(importance_list)

    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    all_fields = set()
    for imp_dict in importance_list:
        all_fields.update(imp_dict.keys())

    merged_importance = {}

    if method == 'weighted_average':
        for field in all_fields:
            weighted_sum = 0.0
            for i, imp_dict in enumerate(importance_list):
                if field in imp_dict:
                    weighted_sum += imp_dict[field] * weights[i]
            merged_importance[field] = weighted_sum

    elif method == 'rank_average':
        field_ranks = {field: [] for field in all_fields}

        for imp_dict in importance_list:
            sorted_fields = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
            for rank, (field, _) in enumerate(sorted_fields):
                if field in field_ranks:
                    field_ranks[field].append(rank + 1)

        for field, ranks in field_ranks.items():
            if ranks:
                avg_rank = sum(ranks) / len(ranks)
                merged_importance[field] = 1.0 / avg_rank

    elif method == 'geometric_mean':
        for field in all_fields:
            product = 1.0
            count = 0
            for i, imp_dict in enumerate(importance_list):
                if field in imp_dict and imp_dict[field] > 0:
                    product *= imp_dict[field] ** weights[i]
                    count += 1
            if count > 0:
                merged_importance[field] = product
            else:
                merged_importance[field] = 0.0

    elif method == 'max_pooling':
        for field in all_fields:
            max_importance = 0.0
            for imp_dict in importance_list:
                if field in imp_dict and imp_dict[field] > max_importance:
                    max_importance = imp_dict[field]
            merged_importance[field] = max_importance

    else:
        return merge_field_importances(importance_list, 'weighted_average', weights)

    total = sum(merged_importance.values())
    if total > 0:
        for field in merged_importance:
            merged_importance[field] /= total

    return merged_importance


def print_comparison_report(importance_list, method_names, field_names, top_k=10):
    """
    打印不同重要性计算方法的对比报告
    """
    print("\n" + "=" * 80)
    print("不同重要性计算方法对比")
    print("=" * 80)

    top_fields_per_method = []
    for i, (imp_dict, method_name) in enumerate(zip(importance_list, method_names)):
        if imp_dict:
            sorted_fields = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            top_fields_per_method.append((method_name, sorted_fields))

    for method_name, top_fields in top_fields_per_method:
        print(f"\n{method_name}方法 - Top {top_k} 字段:")
        print("-" * 50)
        for j, (field, importance) in enumerate(top_fields, 1):
            print(f"  {j:2d}. {field:<25} {importance:.4f}")

    print(f"\n一致性分析 (各方法Top {top_k} 字段的交集):")
    print("-" * 50)

    top_sets = []
    for method_name, top_fields in top_fields_per_method:
        top_set = set(field for field, _ in top_fields)
        top_sets.append(top_set)
        print(f"  {method_name}: {len(top_set)} 个字段")

    common_fields = set.intersection(*top_sets) if top_sets else set()
    print(f"\n共同重要的字段 ({len(common_fields)} 个):")
    for field in sorted(common_fields):
        print(f"  - {field}")

    print(f"\n字段综合排名 (基于各方法排名的平均值):")
    print("-" * 50)

    field_avg_ranks = {}
    for field in field_names:
        ranks = []
        for imp_dict in importance_list:
            if field in imp_dict:
                sorted_fields = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
                for rank, (f, _) in enumerate(sorted_fields):
                    if f == field:
                        ranks.append(rank + 1)
                        break
        if ranks:
            field_avg_ranks[field] = sum(ranks) / len(ranks)

    sorted_avg_ranks = sorted(field_avg_ranks.items(), key=lambda x: x[1])[:top_k]
    for i, (field, avg_rank) in enumerate(sorted_avg_ranks, 1):
        print(f"  {i:2d}. {field:<25} 平均排名: {avg_rank:.1f}")


def analyze_ensemble_attention(models, test_loader, field_names, num_samples=100,
                               save_path='attention_analysis.png'):
    """
    分析EasyEnsemble集成模型的注意力机制，识别关键字段
    """
    print("\n" + "=" * 80)
    print("开始分析EasyEnsemble集成模型的注意力机制...")
    print("=" * 80)

    all_field_importances = []
    all_attention_maps = []

    for i, model in enumerate(models):
        print(f"\n分析第 {i + 1}/{len(models)} 个模型的注意力...")

        attention_analyzer = AttentionAnalyzer(model, field_names)

        attention_maps = attention_analyzer.analyze_attention(test_loader, num_samples=num_samples // len(models))

        if not attention_maps:
            print(f"第 {i + 1} 个模型未能收集到注意力权重，可能模型结构不支持")
            continue

        print(f"第 {i + 1} 个模型成功收集了 {len(attention_maps)} 层的注意力数据")

        print("计算字段重要性...")

        field_importance_max = attention_analyzer.get_field_importance(
            attention_maps, method='max_attention', head_aggregation='full'
        )

        field_importance_in_degree = attention_analyzer.get_field_importance(
            attention_maps, method='in_degree', head_aggregation='full'
        )

        importance_methods = [field_importance_max, field_importance_in_degree]

        merged_field_importance = merge_field_importances(
            importance_methods,
            method='weighted_average',
            weights=[0.5, 0.5]
        )

        all_field_importances.append(merged_field_importance)
        all_attention_maps.append(attention_maps)

        if merged_field_importance:
            top_fields = sorted(merged_field_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"第 {i + 1} 个模型Top 5字段:")
            for rank, (field, importance) in enumerate(top_fields, 1):
                print(f"  {rank}. {field}: {importance:.4f}")

    if not all_field_importances:
        print("未能收集到任何模型的注意力数据")
        return None, None

    print(f"\n集成 {len(all_field_importances)} 个模型的字段重要性...")
    ensemble_field_importance = merge_field_importances(
        all_field_importances,
        method='weighted_average',
        weights=[1.0] * len(all_field_importances)
    )

    print("\n=== EasyEnsemble集成模型的字段重要性分析 ===")
    top_fields = sorted(ensemble_field_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"Top {len(top_fields)} 关键影响字段:")
    for i, (field, importance) in enumerate(top_fields, 1):
        print(f"  {i:2d}. {field:<25} {importance:.4f}")

    if all_attention_maps and save_path:
        print("生成可视化图表...")
        try:
            attention_analyzer = AttentionAnalyzer(models[0], field_names)
            attention_analyzer.visualize_attention(
                all_attention_maps[0],
                ensemble_field_importance,
                save_path=save_path
            )
        except Exception as e:
            print(f"可视化生成失败: {e}")

    return ensemble_field_importance, all_attention_maps


# ==========================================
# 模型保存和加载
# ==========================================

def save_ensemble_models(models, save_dir='./output'):
    """
    保存EasyEnsemble所有模型
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, model in enumerate(models):
        model_path = os.path.join(save_dir, f'ensemble_model_{i + 1}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_idx': i,
            'model_architecture': model.__class__.__name__
        }, model_path)
        print(f"模型 {i + 1} 已保存为 '{model_path}'")

    print(f"总共保存了 {len(models)} 个模型到 '{save_dir}'")


def load_ensemble_models(model_class, model_dir='./output', model_args=None):
    """
    加载EasyEnsemble所有模型
    """
    if model_args is None:
        model_args = {}

    models = []
    model_files = [f for f in os.listdir(model_dir) if f.startswith('ensemble_model_') and f.endswith('.pth')]

    if not model_files:
        model_files = [f for f in os.listdir(model_dir) if
                       f.startswith('best_health_transformer_model_model') and f.endswith('.pth')]

    if not model_files:
        print(f"在 '{model_dir}' 中没有找到模型文件")
        return []

    print(f"找到 {len(model_files)} 个模型文件，正在加载...")

    for model_file in sorted(model_files):
        try:
            model_path = os.path.join(model_dir, model_file)
            checkpoint = torch.load(model_path, map_location='cpu')

            model = model_class(**model_args)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)

            models.append(model)
            print(f"已加载模型: {model_file}")
        except Exception as e:
            print(f"加载模型 {model_file} 时出错: {e}")

    print(f"成功加载了 {len(models)} 个模型")
    return models


# ==========================================
# 可视化学习率变化
# ==========================================

def plot_learning_rate_schedule(train_histories, save_path='./output/learning_rate_schedule.png'):
    """
    可视化学习率变化
    """
    if not train_histories or 'learning_rate' not in train_histories[0]:
        print("没有学习率数据可可视化")
        return

    plt.figure(figsize=(12, 8))

    for i, history in enumerate(train_histories):
        if 'learning_rate' in history:
            lr_values = history['learning_rate']
            epochs = list(range(1, len(lr_values) + 1))
            plt.plot(epochs, lr_values, marker='o', markersize=4, linewidth=2, label=f'Model {i + 1}')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.title('Learning Rate Schedule Across Models', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习率变化图已保存至: {save_path}")

    plt.show()


# ==========================================
# 主函数 - EasyEnsemble多模型训练（使用消融模型）
# ==========================================

if __name__ == "__main__":
    feature_dim = 32
    if torch.cuda.is_available():
        print(f"GPU内存使用情况:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"  已缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    healthy_files = {
        2000: 'healthy16.csv',
        2017: 'healthy17.csv',
        2018: 'healthy18.csv'
    }

    stroke_files = {
        2007: 'Stroke16.csv',
        2017: 'Stroke17.csv',
        2018: 'Stroke18.csv'
    }

    print("=" * 80)
    print("开始按年份加载和处理数据")
    print("=" * 80)

    train_loader, val_loader, test_loader, train_dataset, processor = load_and_process_data_by_year(
        healthy_files=healthy_files,
        stroke_files=stroke_files,
        feature_dim=feature_dim,
        batch_size=32,
        K=10
    )

    num_fields = len(processor.field_names)

    print("\n" + "=" * 80)
    print("开始训练EasyEnsemble多个模型（消融实验：固定顺序LSTM）")
    print("=" * 80)

    # 选择学习率调度器配置
    scheduler_config = {
        'type': 'warmup_cosine',  # 可选: 'cosine', 'warmup_cosine', 'warmup_linear', 'exponential', 'step', 'plateau'等
        'warmup_epochs': 5,  # 预热epoch数
        'total_epochs': 50,  # 总epoch数
        'eta_min': 1e-6  # 最小学习率
    }

    # 选择模型保存指标配置
    selection_metric = 'auc'  # 修改为AUC指标
    # 可选值:
    # - 'accuracy': 准确率
    # - 'sensitivity_stroke': 中风类别敏感度（推荐）
    # - 'specificity_stroke': 中风类别特异性
    # - 'f1_score': F1分数
    # - 'balanced_accuracy': 平衡准确率
    # - 'g_mean': 几何平均数(G-mean)
    # - 'weighted_score': 加权分数（alpha * 敏感度 + (1-alpha) * 特异性）
    # - 'custom_metric': 自定义指标（F1分数和平衡准确率的加权平均）
    # - 'auc': AUC值（新增）

    alpha = 0.5  # 加权分数中敏感度的权重（仅当selection_metric='weighted_score'时有效）

    # 选择集成方法
    ensemble_method = 'max_voting'  # 集成方法
    # 可选值:
    # - 'selection_metric_weighted': 使用选择指标加权平均（默认）
    # - 'sensitivity_weighted': 中风类别敏感度加权平均
    # - 'accuracy_weighted': 准确率加权平均
    # - 'equal_weighted': 等权重平均
    # - 'max_voting': 最大投票法

    # 使用消融模型：固定顺序LSTM
    models, model_accuracies, model_sensitivities_stroke, model_selection_metrics = train_easyensemble_models(
        model_class=HealthDataTransformeruniLSTM_FixedOrder,  # 修改为消融模型
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        num_models=10,
        num_epochs=50,
        learning_rate=0.0002,
        patience=9,
        save_dir='./output',
        scheduler_config=scheduler_config,
        selection_metric=selection_metric,
        alpha=alpha
    )

    save_ensemble_models(models, save_dir='./output')

    print("\n" + "=" * 80)
    print("在测试集上评估EasyEnsemble集成模型")
    print("=" * 80)

    test_metrics = test_ensemble_models(
        models,
        test_loader,
        './output',
        ensemble_method=ensemble_method,
        selection_metric=selection_metric
    )

    print("\n" + "=" * 80)
    print("分析EasyEnsemble集成模型的注意力机制")
    print("=" * 80)

    ensemble_field_importance, attention_maps = analyze_ensemble_attention(
        models=models,
        test_loader=test_loader,
        field_names=processor.field_names,
        num_samples=min(200, len(test_loader.dataset)),
        save_path='./output/ensemble_attention_analysis.png'
    )

    if ensemble_field_importance:
        importance_df = pd.DataFrame(list(ensemble_field_importance.items()), columns=['Field', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv('./output/ensemble_field_importance.csv', index=False, encoding='utf-8-sig')
        print(f"集成模型字段重要性已保存至: ensemble_field_importance.csv")

        top_fields = importance_df.head(10)
        print(f"\nTop 10关键影响字段:")
        for i, row in top_fields.iterrows():
            print(f"  {i + 1}. {row['Field']}: {row['Importance']:.4f}")

    print("\n" + "=" * 80)
    print("最终结果汇总")
    print("=" * 80)

    if test_metrics:
        print(f"选择指标: {selection_metric}")
        print(f"集成方法: {test_metrics['ensemble_method']}")
        print(f"集成模型测试集准确率: {test_metrics['accuracy']:.2f}%")
        print(f"集成模型测试集AUC: {test_metrics['auc']:.4f}")
        print(f"集成模型测试集精确率: {test_metrics['precision']:.4f}")
        print(f"集成模型测试集召回率/中风类别敏感度: {test_metrics['recall']:.4f}")
        print(f"集成模型测试集特异性: {test_metrics['specificity']:.4f}")
        print(f"集成模型测试集F1分数: {test_metrics['f1_score']:.4f}")
        print(f"集成模型测试集几何平均数(G-mean): {test_metrics['g_mean']:.4f}")
        print(f"集成模型测试集平衡准确率: {test_metrics['balanced_accuracy']:.4f}")
        print(f"集成模型测试集中风类别敏感度: {test_metrics['stroke_sensitivity']:.4f}")
        print(f"集成模型数量: {test_metrics['ensemble_size']}")

        print(f"\n模型权重:")
        for i, weight in enumerate(test_metrics['model_weights']):
            print(f"  模型 {i + 1}: 权重={weight:.4f}")

    results_dict = {
        'ensemble_size': len(models),
        'model_accuracies': [float(acc) for acc in model_accuracies],
        'model_sensitivities_stroke': [float(sens) for sens in model_sensitivities_stroke],
        'model_selection_metrics': [float(metric) for metric in model_selection_metrics],
        'selection_metric': selection_metric,
        'avg_model_accuracy': float(np.mean(model_accuracies)) if model_accuracies else 0,
        'avg_model_sensitivity_stroke': float(np.mean(model_sensitivities_stroke)) if model_sensitivities_stroke else 0,
        'avg_model_selection_metric': float(np.mean(model_selection_metrics)) if model_selection_metrics else 0,
        'max_model_accuracy': float(np.max(model_accuracies)) if model_accuracies else 0,
        'max_model_sensitivity_stroke': float(np.max(model_sensitivities_stroke)) if model_sensitivities_stroke else 0,
        'max_model_selection_metric': float(np.max(model_selection_metrics)) if model_selection_metrics else 0,
        'scheduler_config': scheduler_config,
        'alpha': alpha if selection_metric == 'weighted_score' else None
    }

    if test_metrics:
        results_dict.update({
            'test_accuracy': float(test_metrics['accuracy']),
            'test_auc': float(test_metrics['auc']),
            'test_precision': float(test_metrics['precision']),
            'test_recall': float(test_metrics['recall']),
            'test_f1_score': float(test_metrics['f1_score']),
            'test_specificity': float(test_metrics['specificity']),
            'test_g_mean': float(test_metrics['g_mean']),
            'test_balanced_accuracy': float(test_metrics['balanced_accuracy']),
            'test_stroke_sensitivity': float(test_metrics['stroke_sensitivity']),
            'ensemble_method': test_metrics['ensemble_method'],
            'model_weights': test_metrics['model_weights']
        })

    with open('./output/ensemble_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    results_df = pd.DataFrame([results_dict])
    results_df.to_csv('./output/ensemble_training_results.csv', index=False, encoding='utf-8-sig')
    print("训练结果已保存至: ensemble_training_results.csv")

    # 可视化学习率变化
    try:
        # 加载训练历史
        with open('./output/training_history.json', 'r') as f:
            training_history = json.load(f)

        if 'train_histories' in training_history:
            plot_learning_rate_schedule(
                training_history['train_histories'],
                save_path='./output/learning_rate_schedule.png'
            )
    except Exception as e:
        print(f"无法可视化学习率变化: {e}")

    print("\nEasyEnsemble多模型训练和分析完成!")