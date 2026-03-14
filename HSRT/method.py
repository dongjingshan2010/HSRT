"""
单次训练验证测试版本 - 支持EasyEnsemble和难负样本挖掘
"""

from GradientAnalyzer import GradientImportanceAnalyzer, StrategicGradientImportanceAnalyzer, AttentionAnalyzer
from data_loader import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
import gc
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')


from HealthDataTransformer2 import HealthDataTransformeruniLSTM


# ==========================================
# 训练和评估函数 - 支持EasyEnsemble和难负样本挖掘
# ==========================================

def train_model_with_easyensemble_and_hardnegatives(model, train_dataset, train_loader, val_loader,
                                                    num_epochs=50, learning_rate=0.001,
                                                    patience=5, save_path='./output/best_model.pth',
                                                    hard_negative_interval=5):
    """
    训练模型 - 支持EasyEnsemble和难负样本挖掘

    Args:
        model: 模型
        train_dataset: EasyEnsemble训练数据集
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 总训练轮数
        learning_rate: 学习率
        patience: 早停耐心值
        save_path: 模型保存路径
        hard_negative_interval: 难负样本挖掘间隔（轮次）
    """
    # 将模型移动到设备
    model = model.to(device)
    print(f"模型所在设备: {next(model.parameters()).device}")

    # 计算类别权重
    print("计算类别权重...")

    # 统计训练集中的类别分布
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels)

    # 计算类别分布
    class_counts = torch.bincount(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)

    # 计算权重：总样本数 / (类别数 * 类别样本数)
    class_weights = total_samples / (num_classes * class_counts.float())

    print(f"类别分布: {class_counts.tolist()}")
    print(f"类别权重: {class_weights.tolist()}")

    # 定义损失函数和优化器 - 使用类别权重
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化变量
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        # 添加每个类别的统计
        class_correct = [0, 0]
        class_total = [0, 0]

        # 在epoch开始时切换到下一个阴性子集
        if epoch > 0:
            train_dataset.next_subset()

        # 执行难负样本挖掘
        if epoch > 0 and epoch % hard_negative_interval == 0:
            print(f"\n=== 第 {epoch} 轮，执行难负样本挖掘 ===")
            hard_negatives = train_dataset.update_hard_negatives(model, device, num_samples=2000, top_percent=0.1)
            print(f"已识别 {len(hard_negatives)} 个难负样本")
            print(f"当前难负样本池大小: {len(train_dataset.hard_negative_pool)}")

        for batch_embeddings, batch_labels in train_loader:
            # 数据已经在GPU上，不需要移动

            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)

            # 反向传播
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            # 统计每个类别的准确率
            for i in range(2):
                mask = (batch_labels == i)
                class_correct[i] += (predicted[mask] == batch_labels[mask]).sum().item()
                class_total[i] += mask.sum().item()

            # 清理中间变量
            del outputs, predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        # 添加每个类别的验证统计
        val_class_correct = [0, 0]
        val_class_total = [0, 0]

        with torch.no_grad():
            for batch_embeddings, batch_labels in val_loader:
                # 数据已经在GPU上，不需要移动

                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

                # 统计每个类别的验证准确率
                for i in range(2):
                    mask = (batch_labels == i)
                    val_class_correct[i] += (predicted[mask] == batch_labels[mask]).sum().item()
                    val_class_total[i] += mask.sum().item()

                # 清理中间变量
                del outputs, predicted
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 计算准确率
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total

        # 打印统计信息
        print(f'\nEpoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Val Acc: {val_accuracy:.2f}%')

        # 打印每个类别的准确率
        for i in range(2):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f'  Class {i} Train Acc: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')
            if val_class_total[i] > 0:
                val_class_acc = 100 * val_class_correct[i] / val_class_total[i]
                print(f'  Class {i} Val Acc: {val_class_acc:.2f}% ({val_class_correct[i]}/{val_class_total[i]})')

        # 检查是否是最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            patience_counter = 0

            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'train_accuracy': train_accuracy,
                'class_weights': class_weights  # 保存类别权重以便后续使用
            }, save_path)
            print(f"最佳模型已保存! 验证准确率: {val_accuracy:.2f}%")
        else:
            patience_counter += 1

        # 早停检查
        if patience_counter >= patience:
            print(f"早停触发! 在 epoch {epoch + 1} 停止训练")
            break

        # 每个epoch结束后清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}% (epoch {best_epoch})")
    return best_val_acc


from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


def test_model(model, test_loader, model_path='./output/best_model.pth'):
    """
    在测试集上评估模型 - 返回多个评估指标
    """
    # 加载最佳模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载最佳模型 (验证准确率: {checkpoint['val_accuracy']:.2f}%)")
    else:
        print("未找到保存的模型，使用当前模型参数")

    model.eval()

    test_correct = 0
    test_total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # 存储预测结果用于详细分析
    all_predictions = []
    all_labels = []
    all_probabilities = []  # 新增：存储预测概率用于AUC计算

    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            # 确保数据在正确的设备上
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)

            test_loss += loss.item()

            # 获取预测类别
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()

            # 获取预测概率（使用softmax）
            probabilities = torch.softmax(outputs, dim=1)

            # 保存预测结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())  # 保存概率值

    test_accuracy = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)

    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    print(f"\n=== 测试集评估结果 ===")
    print(f"测试集大小: {test_total}")
    print(f"测试损失: {avg_test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.2f}%")

    # 计算各类别的准确率
    for class_label in [0, 1]:
        class_mask = (all_labels == class_label)
        if class_mask.sum() > 0:
            class_accuracy = 100 * (all_predictions[class_mask] == all_labels[class_mask]).sum() / class_mask.sum()
            class_name = "健康" if class_label == 0 else "中风"
            print(f"{class_name}类别准确率: {class_accuracy:.2f}% (样本数: {class_mask.sum()})")

    # 计算AUC指标
    auc_score = 0.0
    try:
        # 使用正类（类别1）的概率计算AUC
        auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
        print(f"AUC: {auc_score:.4f}")
    except Exception as e:
        print(f"AUC计算失败: {e}")

    # 计算其他分类指标
    print(f"\n=== 详细分类指标 ===")

    # 混淆矩阵
    precision = recall = f1_score = specificity = 0.0
    try:
        cm = confusion_matrix(all_labels, all_predictions)
        print("混淆矩阵:")
        print(f"         预测健康   预测中风")
        print(f"实际健康  {cm[0, 0]:>8}  {cm[0, 1]:>8}")
        print(f"实际中风  {cm[1, 0]:>8}  {cm[1, 1]:>8}")

        # 计算精确率、召回率、F1分数
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"\n详细指标:")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1_score:.4f}")
        print(f"特异性 (Specificity): {specificity:.4f}")
    except Exception as e:
        print(f"计算详细指标时出错: {e}")

    # 返回所有评估指标
    metrics = {
        'accuracy': test_accuracy,
        'auc': auc_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'test_loss': avg_test_loss
    }

    return metrics


# ==========================================
# 分析函数
# ==========================================

def merge_field_importances(importance_list, method='weighted_average', weights=None):
    """
    合并多种字段重要性计算方法的结果

    Args:
        importance_list: 多个重要性字典的列表
        method: 合并方法 ('weighted_average', 'rank_average', 'geometric_mean', 'max_pooling')
        weights: 各方法的权重列表

    Returns:
        合并后的字段重要性字典
    """
    if not importance_list:
        return {}

    # 默认权重
    if weights is None:
        weights = [1.0] * len(importance_list)

    # 确保权重和重要性列表长度一致
    if len(weights) != len(importance_list):
        weights = [1.0] * len(importance_list)

    # 归一化权重
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    all_fields = set()
    for imp_dict in importance_list:
        all_fields.update(imp_dict.keys())

    merged_importance = {}

    if method == 'weighted_average':
        # 加权平均
        for field in all_fields:
            weighted_sum = 0.0
            for i, imp_dict in enumerate(importance_list):
                if field in imp_dict:
                    weighted_sum += imp_dict[field] * weights[i]
            merged_importance[field] = weighted_sum

    elif method == 'rank_average':
        # 基于排名的平均
        field_ranks = {field: [] for field in all_fields}

        for imp_dict in importance_list:
            # 按重要性排序
            sorted_fields = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
            for rank, (field, _) in enumerate(sorted_fields):
                if field in field_ranks:
                    field_ranks[field].append(rank + 1)  # 排名从1开始

        for field, ranks in field_ranks.items():
            if ranks:
                # 计算平均排名（越小越好），然后转换为重要性分数（越大越好）
                avg_rank = sum(ranks) / len(ranks)
                merged_importance[field] = 1.0 / avg_rank  # 排名越靠前，重要性越高

    elif method == 'geometric_mean':
        # 几何平均
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
        # 取各方法中的最大值
        for field in all_fields:
            max_importance = 0.0
            for imp_dict in importance_list:
                if field in imp_dict and imp_dict[field] > max_importance:
                    max_importance = imp_dict[field]
            merged_importance[field] = max_importance

    else:
        # 默认使用加权平均
        return merge_field_importances(importance_list, 'weighted_average', weights)

    # 归一化最终结果
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

    # 为每个方法计算Top-K字段
    top_fields_per_method = []
    for i, (imp_dict, method_name) in enumerate(zip(importance_list, method_names)):
        if imp_dict:
            sorted_fields = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            top_fields_per_method.append((method_name, sorted_fields))

    # 打印各方法的Top字段
    for method_name, top_fields in top_fields_per_method:
        print(f"\n{method_name}方法 - Top {top_k} 字段:")
        print("-" * 50)
        for j, (field, importance) in enumerate(top_fields, 1):
            print(f"  {j:2d}. {field:<25} {importance:.4f}")

    # 计算一致性分析
    print(f"\n一致性分析 (各方法Top {top_k} 字段的交集):")
    print("-" * 50)

    # 获取各方法的Top字段集合
    top_sets = []
    for method_name, top_fields in top_fields_per_method:
        top_set = set(field for field, _ in top_fields)
        top_sets.append(top_set)
        print(f"  {method_name}: {len(top_set)} 个字段")

    # 计算交集
    common_fields = set.intersection(*top_sets) if top_sets else set()
    print(f"\n共同重要的字段 ({len(common_fields)} 个):")
    for field in sorted(common_fields):
        print(f"  - {field}")

    # 计算各字段在不同方法中的平均排名
    print(f"\n字段综合排名 (基于各方法排名的平均值):")
    print("-" * 50)

    field_avg_ranks = {}
    for field in field_names:
        ranks = []
        for imp_dict in importance_list:
            if field in imp_dict:
                # 计算该字段在该方法中的排名
                sorted_fields = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
                for rank, (f, _) in enumerate(sorted_fields):
                    if f == field:
                        ranks.append(rank + 1)
                        break
        if ranks:
            field_avg_ranks[field] = sum(ranks) / len(ranks)

    # 按平均排名排序（排名越小越好）
    sorted_avg_ranks = sorted(field_avg_ranks.items(), key=lambda x: x[1])[:top_k]
    for i, (field, avg_rank) in enumerate(sorted_avg_ranks, 1):
        print(f"  {i:2d}. {field:<25} 平均排名: {avg_rank:.1f}")


def analyze_model_attention(model, test_loader, field_names, train_loader=None, num_samples=100,
                            save_path='attention_analysis.png', merge_method='weighted_average',
                            weights=None, include_gradient_importance=False):
    """
    分析模型的注意力机制，识别关键字段 - 增强版，包含梯度重要性分析
    """
    print("\n" + "=" * 80)
    print("开始分析Transformer注意力机制...")
    print("=" * 80)

    # 创建注意力分析器
    attention_analyzer = AttentionAnalyzer(model, field_names)

    # 分析注意力权重
    print("收集注意力权重...")
    attention_maps = attention_analyzer.analyze_attention(test_loader, num_samples=num_samples)

    if not attention_maps:
        print("未能收集到注意力权重，可能模型结构不支持")
        return None, None

    print(f"成功收集了 {len(attention_maps)} 层的注意力数据")

    # 使用多种方法计算字段重要性
    print("计算字段重要性...")

    # 方法1: 基于最大关注度
    field_importance_max = attention_analyzer.get_field_importance(
        attention_maps, method='max_attention', head_aggregation='full'
    )

    # 方法2: 基于入度
    field_importance_in_degree = attention_analyzer.get_field_importance(
        attention_maps, method='in_degree', head_aggregation='full'
    )

    # 方法3: 基于方差
    field_importance_variance = attention_analyzer.get_field_importance_variance(attention_maps)

    # 方法4: 基于中心性
    field_importance_centrality = attention_analyzer.get_field_importance(
        attention_maps, method='centrality', head_aggregation='mean'
    )

    # 方法5: 基于训练梯度（如果提供训练数据）
    field_importance_gradient = {}
    if include_gradient_importance and train_loader is not None:
        print("计算基于训练梯度的字段重要性...")
        gradient_analyzer = GradientImportanceAnalyzer(model, field_names)
        field_importance_gradient = gradient_analyzer.compute_gradient_importance_alternative(
            train_loader, num_batches=30
        )

        # 可视化梯度重要性
        gradient_analyzer.visualize_gradient_importance(
            field_importance_gradient,
            save_path=save_path.replace('.png', '_gradient.png') if save_path else None
        )

        # 打印梯度重要性报告
        gradient_analyzer.print_gradient_importance_report(field_importance_gradient)

    # 合并多种重要性计算方法
    importance_methods = [field_importance_max, field_importance_in_degree,
                          field_importance_variance, field_importance_centrality]
    method_names = ['最大关注度', '入度', '方差', '中心性']

    if field_importance_gradient:
        importance_methods.append(field_importance_gradient)
        method_names.append('训练梯度')

    merged_field_importance = merge_field_importances(
        importance_methods,
        method=merge_method,
        weights=weights
    )

    # 打印详细报告
    print("\n=== 合并后的字段重要性分析 ===")
    attention_analyzer.print_field_importance_report(merged_field_importance, attention_maps)

    # 打印各方法对比
    print_comparison_report(
        importance_methods,
        method_names,
        field_names
    )

    # 可视化注意力
    print("生成可视化图表...")
    attention_analyzer.visualize_attention(attention_maps, merged_field_importance, save_path=save_path)

    return merged_field_importance, attention_maps


def analyze_model_attention_with_strategic_gradients(model, train_loader, test_loader, field_names,
                                                     total_epochs=30, num_samples=100,
                                                     save_path='attention_analysis.png'):
    """包含策略性梯度重要性分析的完整方法"""

    # 原有的注意力分析
    attention_analyzer = AttentionAnalyzer(model, field_names)
    attention_maps = attention_analyzer.analyze_attention(test_loader, num_samples=num_samples)

    # 策略性梯度重要性分析
    print("\n" + "=" * 60)
    print("策略性梯度重要性分析")
    print("=" * 60)

    strategic_analyzer = StrategicGradientImportanceAnalyzer(model, field_names)

    # 分析梯度演化（可选）
    gradient_evolution = strategic_analyzer.analyze_gradient_evolution(
        train_loader, total_epochs, sample_epochs=5
    )

    # 使用动态策略计算梯度重要性
    field_importance_gradient = strategic_analyzer.compute_strategic_gradient_importance(
        train_loader, total_epochs, strategy='dynamic'
    )

    # 原有的注意力重要性方法
    field_importance_max = attention_analyzer.get_field_importance(
        attention_maps, method='max_attention', head_aggregation='full'
    )
    field_importance_in_degree = attention_analyzer.get_field_importance(
        attention_maps, method='in_degree', head_aggregation='full'
    )

    # 合并所有重要性方法
    importance_methods = [
        field_importance_max,
        field_importance_in_degree,
        field_importance_gradient
    ]
    method_names = ['最大关注度', '入度', '动态梯度']

    merged_field_importance = merge_field_importances(
        importance_methods,
        method='weighted_average',
        weights=[0.3, 0.3, 0.4]  # 给梯度方法更高权重
    )

    # 后续的可视化和报告...
    return merged_field_importance, attention_maps


# ==========================================
# 模型保存和加载
# ==========================================

def save_model(model, filepath):
    """保存模型，包含设备信息"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__
    }, filepath)
    print(f"模型已保存为 '{filepath}'")


def load_model(filepath, model_class, **model_args):
    """加载模型"""
    checkpoint = torch.load(filepath, map_location=device)
    model = model_class(**model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"模型已从 '{filepath}' 加载")
    return model


def compare_field_importance_results(basic_importance, strategic_importance, field_names, top_k=10):
    """
    比较两种分析方法的字段重要性结果
    """
    print("\n" + "=" * 80)
    print("字段重要性分析方法比较")
    print("=" * 80)

    if not basic_importance or not strategic_importance:
        print("缺少一种或多种重要性分析结果")
        return

    # 获取两种方法的Top-K字段
    basic_top = sorted(basic_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
    strategic_top = sorted(strategic_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # 创建比较表格
    basic_dict = {field: score for field, score in basic_top}
    strategic_dict = {field: score for field, score in strategic_top}

    print(f"\n两种方法Top-{top_k}字段对比:")
    print("-" * 100)
    print(f"{'排名':<4} {'基本方法字段':<25} {'分数':<10} {'策略方法字段':<25} {'分数':<10} {'是否相同':<8}")
    print("-" * 100)

    for i in range(top_k):
        basic_field = basic_top[i][0] if i < len(basic_top) else ""
        basic_score = f"{basic_top[i][1]:.4f}" if i < len(basic_top) else ""
        strategic_field = strategic_top[i][0] if i < len(strategic_top) else ""
        strategic_score = f"{strategic_top[i][1]:.4f}" if i < len(strategic_top) else ""
        same_field = "✓" if basic_field == strategic_field else ""

        print(
            f"{i + 1:<4} {basic_field:<25} {basic_score:<10} {strategic_field:<25} {strategic_score:<10} {same_field:<8}")

    # 计算相似度指标
    basic_set = set([field for field, _ in basic_top])
    strategic_set = set([field for field, _ in strategic_top])

    intersection = basic_set.intersection(strategic_set)
    union = basic_set.union(strategic_set)

    jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0

    print(f"\n相似度分析:")
    print(f"  Jaccard相似度 (Top-{top_k}): {jaccard_similarity:.2f}")
    print(f"  共同重要字段 ({len(intersection)} 个): {', '.join(sorted(intersection))}")

    # 识别只在一种方法中重要的字段
    only_basic = basic_set - strategic_set
    only_strategic = strategic_set - basic_set

    if only_basic:
        print(f"\n只在基本方法中重要的字段 ({len(only_basic)} 个):")
        for field in sorted(only_basic):
            print(f"  - {field}: {basic_importance[field]:.4f}")

    if only_strategic:
        print(f"\n只在策略方法中重要的字段 ({len(only_strategic)} 个):")
        for field in sorted(only_strategic):
            print(f"  - {field}: {strategic_importance[field]:.4f}")

    # 计算相关性
    if basic_importance and strategic_importance:
        common_fields = set(basic_importance.keys()).intersection(set(strategic_importance.keys()))
        if len(common_fields) >= 2:
            basic_scores = [basic_importance[field] for field in common_fields]
            strategic_scores = [strategic_importance[field] for field in common_fields]

            try:
                correlation = np.corrcoef(basic_scores, strategic_scores)[0, 1]
                print(f"\n相关性分析:")
                print(f"  Pearson相关系数: {correlation:.3f}")
                print(f"  分析字段数: {len(common_fields)}")
            except:
                print("\n无法计算相关性系数")

    print("=" * 80)


# ==========================================
# 主函数 - 支持按年份划分数据和EasyEnsemble
# ==========================================

if __name__ == "__main__":
    # 检查GPU使用情况
    feature_dim = 32  # 降低特征维度以减少内存使用
    if torch.cuda.is_available():
        print(f"GPU内存使用情况:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"  已缓存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

    # 定义数据文件路径
    healthy_files = {
        2000: 'healthy16.csv',  # 假设这是00-16年健康数据
        2017: 'healthy17.csv',
        2018: 'healthy18.csv'
    }

    stroke_files = {
        2007: 'Stroke16.csv',  # 中风数据从2007年开始
        2017: 'Stroke17.csv',
        2018: 'Stroke18.csv'
    }

    # 加载和处理数据
    print("=" * 80)
    print("开始按年份加载和处理数据")
    print("=" * 80)

    train_loader, val_loader, test_loader, train_dataset, processor = load_and_process_data_by_year(
        healthy_files=healthy_files,
        stroke_files=stroke_files,
        feature_dim=feature_dim,
        batch_size=16,
        K=50  # 将阴性样本划分为10个子集
    )

    # 创建模型
    print("\n" + "=" * 80)
    print("创建模型")
    print("=" * 80)

    num_fields = len(processor.field_names)

    model = HealthDataTransformeruniLSTM(
        num_fields=num_fields,
        feature_dim=feature_dim,
        hidden_dim=512,
        num_layers=5,
        num_heads=4,
        num_classes=2
    ).to(device)

    print(f"模型架构: Attention + LSTM + Transformer")
    print(f"输入字段数: {num_fields}")
    print(f"字段特征维度: {feature_dim}")
    print(f"Hidden Dim: 128")
    print(f"Transformer Layers: 2")
    print(f"Attention Heads: 4")
    print(f"使用设备: {device}")

    # 训练模型（使用EasyEnsemble和难负样本挖掘）
    print("\n" + "=" * 80)
    print("开始训练模型（使用EasyEnsemble和难负样本挖掘）")
    print("=" * 80)

    best_val_acc = train_model_with_easyensemble_and_hardnegatives(
        model=model,
        train_dataset=train_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=0.0001,
        patience=12,
        save_path='./output/best_health_transformer_model.pth',
        hard_negative_interval=5  # 每5个轮次进行一次难负样本挖掘
    )

    # 在测试集上评估最佳模型
    print("\n" + "=" * 80)
    print("在测试集上评估模型")
    print("=" * 80)

    test_metrics = test_model(model, test_loader, './output/best_health_transformer_model.pth')

    # 分析注意力机制和关键字段
    print("\n" + "=" * 80)
    print("分析模型注意力机制")
    print("=" * 80)

    # 加载最佳模型
    checkpoint = torch.load('./output/best_health_transformer_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载最佳模型 (验证准确率: {checkpoint['val_accuracy']:.2f}%)")

    # 方法1: 基本注意力分析
    print("\n方法1: 基本注意力分析 (analyze_model_attention)...")
    basic_field_importance, basic_attention_maps = analyze_model_attention(
        model=model,
        test_loader=test_loader,
        field_names=processor.field_names,
        train_loader=train_loader,
        num_samples=min(200, len(test_loader.dataset)),
        save_path='./output/attention_analysis_basic.png',
        include_gradient_importance=False
    )

    # 方法2: 策略梯度注意力分析
    print("\n方法2: 策略梯度注意力分析 (analyze_model_attention_with_strategic_gradients)...")
    strategic_field_importance, strategic_attention_maps = analyze_model_attention_with_strategic_gradients(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        field_names=processor.field_names,
        total_epochs=30,
        num_samples=min(200, len(test_loader.dataset)),
        save_path='./output/attention_analysis_strategic.png'
    )

    # 比较两种方法的结果
    if basic_field_importance and strategic_field_importance:
        print("\n" + "=" * 80)
        print("比较两种重要性分析方法的结果")
        print("=" * 80)

        compare_field_importance_results(
            basic_field_importance,
            strategic_field_importance,
            processor.field_names,
            top_k=15
        )

        # 合并两种方法的结果
        combined_field_importance = merge_field_importances(
            [basic_field_importance, strategic_field_importance],
            method='weighted_average',
            weights=[0.5, 0.5]
        )

        # 打印合并后的Top字段
        top_combined = sorted(combined_field_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n合并后的Top 10关键影响字段:")
        for i, (field, importance) in enumerate(top_combined, 1):
            print(f"  {i}. {field}: {importance:.4f}")

        # 保存字段重要性结果
        for method_name, importance_dict in [("basic", basic_field_importance),
                                             ("strategic", strategic_field_importance),
                                             ("combined", combined_field_importance)]:
            importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Field', 'Importance'])
            importance_df = importance_df.sort_values('Importance', ascending=False)
            importance_df.to_csv(f'./output/field_importance_{method_name}.csv',
                                 index=False, encoding='utf-8-sig')
            print(f"字段重要性 ({method_name}) 已保存至: field_importance_{method_name}.csv")
    else:
        print("\n警告: 一种或两种重要性分析方法返回了空结果")

    # 保存模型
    save_model(model, './output/health_transformer_model_final.pth')

    # 保存最终结果
    print("\n" + "=" * 80)
    print("最终结果汇总")
    print("=" * 80)

    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试集准确率: {test_metrics['accuracy']:.2f}%")
    print(f"测试集AUC: {test_metrics['auc']:.4f}")
    print(f"测试集精确率: {test_metrics['precision']:.4f}")
    print(f"测试集召回率: {test_metrics['recall']:.4f}")
    print(f"测试集F1分数: {test_metrics['f1_score']:.4f}")
    print(f"测试集特异性: {test_metrics['specificity']:.4f}")

    # 保存结果到文件
    results_df = pd.DataFrame([{
        'val_accuracy': best_val_acc,
        'test_accuracy': test_metrics['accuracy'],
        'test_auc': test_metrics['auc'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1_score': test_metrics['f1_score'],
        'test_specificity': test_metrics['specificity'],
        'test_loss': test_metrics['test_loss']
    }])

    results_df.to_csv('./output/training_results.csv', index=False, encoding='utf-8-sig')
    print("训练结果已保存至: training_results.csv")

    print("\n训练和分析完成!")