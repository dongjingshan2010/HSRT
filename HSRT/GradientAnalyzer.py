import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter
import warnings
import gc
import os
import matplotlib.pyplot as plt


class GradientImportanceAnalyzer:
    """基于训练梯度的字段重要性分析器"""

    def __init__(self, model, field_names, feature_dim=64):
        self.model = model
        self.field_names = field_names
        self.feature_dim = feature_dim
        self.gradient_accumulator = {field: 0.0 for field in field_names}
        self.gradient_count = {field: 0 for field in field_names}

    def register_hooks(self):
        """为每个字段的嵌入层注册梯度钩子"""
        self.hooks = []

        # 为每个字段的线性映射层注册钩子
        for field_name in self.field_names:
            if field_name in self.model.processor.field_embeddings:
                layer = self.model.processor.field_embeddings[field_name]

                def make_hook(field):
                    def hook(module, grad_input, grad_output):
                        # grad_output[0] 的形状: [batch_size, feature_dim]
                        if grad_output[0] is not None:
                            # 计算梯度的L2范数
                            gradient_norm = grad_output[0].norm(dim=1).mean().item()
                            self.gradient_accumulator[field] += gradient_norm
                            self.gradient_count[field] += 1

                    return hook

                hook = layer.register_full_backward_hook(make_hook(field_name))
                self.hooks.append(hook)

        print(f"已为 {len(self.hooks)} 个字段注册梯度钩子")

    def remove_hooks(self):
        """移除所有梯度钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("已移除所有梯度钩子")

    def compute_gradient_importance(self, dataloader, num_batches=100):
        """计算基于梯度的字段重要性"""
        print("开始计算基于梯度的字段重要性...")

        # 重置累加器
        self.gradient_accumulator = {field: 0.0 for field in self.field_names}
        self.gradient_count = {field: 0 for field in self.field_names}

        # 注册梯度钩子
        self.register_hooks()

        # 设置模型为训练模式以计算梯度
        self.model.train()

        batch_count = 0
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (batch_embeddings, batch_labels) in enumerate(dataloader):
            if batch_count >= num_batches:
                break

            # 清零梯度
            self.model.zero_grad()

            # 前向传播
            outputs = self.model(batch_embeddings)
            loss = criterion(outputs, batch_labels)

            # 反向传播
            loss.backward()

            batch_count += 1

            if batch_count % 20 == 0:
                print(f"已处理 {batch_count}/{num_batches} 批次")

        # 移除钩子
        self.remove_hooks()

        # 计算平均梯度
        field_gradient_importance = {}
        for field_name in self.field_names:
            if self.gradient_count[field_name] > 0:
                avg_gradient = self.gradient_accumulator[field_name] / self.gradient_count[field_name]
                field_gradient_importance[field_name] = avg_gradient
            else:
                field_gradient_importance[field_name] = 0.0

        # 归一化重要性分数
        total_gradient = sum(field_gradient_importance.values())
        if total_gradient > 0:
            for field_name in field_gradient_importance:
                field_gradient_importance[field_name] /= total_gradient

        print("基于梯度的字段重要性计算完成")
        return field_gradient_importance

    def compute_gradient_importance_alternative(self, dataloader, num_batches=50):
        """备选的梯度重要性计算方法 - 直接分析输入嵌入的梯度"""
        print("开始计算备选梯度重要性...")

        field_gradient_norms = {field: [] for field in self.field_names}

        self.model.train()
        batch_count = 0
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (batch_embeddings, batch_labels) in enumerate(dataloader):
            if batch_count >= num_batches:
                break

            # 确保需要梯度
            batch_embeddings.requires_grad_(True)

            # 清零梯度
            self.model.zero_grad()

            # 前向传播
            outputs = self.model(batch_embeddings)
            loss = criterion(outputs, batch_labels)

            # 反向传播
            loss.backward()

            # 计算每个字段的梯度范数
            # batch_embeddings形状: [batch_size, num_fields, feature_dim]
            if batch_embeddings.grad is not None:
                gradients = batch_embeddings.grad  # [batch_size, num_fields, feature_dim]

                # 计算每个字段的梯度L2范数 (沿特征维度)
                field_grad_norms = gradients.norm(dim=2)  # [batch_size, num_fields]

                # 对批次求平均
                batch_field_importance = field_grad_norms.mean(dim=0)  # [num_fields]

                # 存储每个字段的梯度范数
                for field_idx, field_name in enumerate(self.field_names):
                    if field_idx < len(batch_field_importance):
                        field_gradient_norms[field_name].append(batch_field_importance[field_idx].item())

            batch_count += 1

            if batch_count % 10 == 0:
                print(f"已处理 {batch_count}/{num_batches} 批次")

        # 计算每个字段的平均梯度重要性
        field_gradient_importance = {}
        for field_name in self.field_names:
            if field_gradient_norms[field_name]:
                avg_gradient = np.mean(field_gradient_norms[field_name])
                field_gradient_importance[field_name] = avg_gradient
            else:
                field_gradient_importance[field_name] = 0.0

        # 归一化
        total_gradient = sum(field_gradient_importance.values())
        if total_gradient > 0:
            for field_name in field_gradient_importance:
                field_gradient_importance[field_name] /= total_gradient

        print("备选梯度重要性计算完成")
        return field_gradient_importance

    def visualize_gradient_importance(self, gradient_importance, save_path=None):
        """可视化基于梯度的字段重要性"""
        if not gradient_importance:
            print("没有可用的梯度重要性数据")
            return

        # 按重要性排序
        sorted_importance = sorted(gradient_importance.items(), key=lambda x: x[1], reverse=True)
        top_fields = sorted_importance[:15]  # 显示前15个

        field_names = [field[0] for field in top_fields]
        importance_scores = [field[1] for field in top_fields]

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(field_names)), importance_scores)
        plt.yticks(range(len(field_names)), field_names, fontsize=10)
        plt.xlabel('梯度重要性分数')
        plt.title('基于训练梯度的字段重要性排名 (Top 15)')

        # 在条形上添加数值
        for i, (bar, score) in enumerate(zip(bars, importance_scores)):
            plt.text(score + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{score:.4f}', ha='left', va='center', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"梯度重要性图已保存至: {save_path}")

        plt.show()

    def print_gradient_importance_report(self, gradient_importance):
        """打印梯度重要性报告"""
        if not gradient_importance:
            print("没有可用的梯度重要性数据")
            return

        print("\n" + "=" * 80)
        print("基于训练梯度的字段重要性分析报告")
        print("=" * 80)

        # 按重要性排序
        sorted_fields = sorted(gradient_importance.items(), key=lambda x: x[1], reverse=True)

        print(f"\n总字段数: {len(sorted_fields)}")
        print("\nTop 20 梯度重要字段:")
        print("-" * 60)
        print(f"{'排名':<4} {'字段名':<30} {'梯度重要性':<12} {'百分比':<10}")
        print("-" * 60)

        for i, (field_name, importance) in enumerate(sorted_fields[:20], 1):
            percentage = importance * 100
            print(f"{i:<4} {field_name:<30} {importance:.6f}     {percentage:.2f}%")

        # 统计信息
        top_10_importance = sum(importance for _, importance in sorted_fields[:10])
        top_20_importance = sum(importance for _, importance in sorted_fields[:20])

        print(f"\n统计信息:")
        print(f"前10个字段累计梯度重要性: {top_10_importance:.3f} ({top_10_importance * 100:.1f}%)")
        print(f"前20个字段累计梯度重要性: {top_20_importance:.3f} ({top_20_importance * 100:.1f}%)")

        # 识别关键字段（重要性超过平均值的2倍）
        mean_importance = np.mean([imp for _, imp in sorted_fields])
        key_fields = [(field, imp) for field, imp in sorted_fields if imp > 2 * mean_importance]

        print(f"\n关键梯度字段 (重要性 > {2 * mean_importance:.4f}):")
        for field, imp in key_fields:
            print(f"  - {field}: {imp:.4f}")

        print("=" * 80)


class StrategicGradientImportanceAnalyzer:
    """基于策略性训练阶段选择的梯度重要性分析器"""

    def __init__(self, model, field_names):
        self.model = model
        self.field_names = field_names
        self.gradient_history = {field: [] for field in field_names}
        self.epoch_gradients = {}  # 按epoch存储梯度

    def compute_strategic_gradient_importance(self, train_loader, total_epochs,
                                              strategy='mid_training', target_epoch=None):
        """
        基于策略选择训练阶段的梯度重要性分析

        Args:
            train_loader: 训练数据加载器
            total_epochs: 总训练轮数
            strategy: 梯度选择策略
                - 'early': 早期训练 (epoch 1-3)
                - 'mid': 中期训练 (epoch 总轮数的30%-70%)
                - 'late': 后期训练 (最后几个epoch)
                - 'dynamic': 动态选择梯度最大的epoch
                - 'multiple': 多个阶段平均
            target_epoch: 指定特定epoch（如果为None则自动选择）
        """
        print(f"使用策略 '{strategy}' 计算梯度重要性...")

        # 根据策略确定目标epoch
        if target_epoch is not None:
            target_epochs = [target_epoch]
        elif strategy == 'early':
            target_epochs = [1, 2, 3]  # 早期训练
        elif strategy == 'mid':
            mid_start = max(1, int(total_epochs * 0.3))
            mid_end = min(total_epochs, int(total_epochs * 0.7))
            target_epochs = list(range(mid_start, mid_end + 1))
        elif strategy == 'late':
            target_epochs = list(range(max(1, total_epochs - 2), total_epochs + 1))
        elif strategy == 'multiple':
            # 选择多个代表性阶段
            target_epochs = [
                1,  # 早期
                max(2, total_epochs // 3),  # 前中期
                max(3, total_epochs * 2 // 3),  # 后中期
                total_epochs  # 最后
            ]
        else:  # 'dynamic' 动态选择
            target_epochs = list(range(1, total_epochs + 1))

        print(f"目标训练阶段: {target_epochs}")

        # 收集目标epoch的梯度
        epoch_gradients = {}
        for epoch in target_epochs:
            print(f"收集第 {epoch} 轮的梯度...")
            gradients = self._compute_epoch_gradients(train_loader, epoch, total_epochs)
            epoch_gradients[epoch] = gradients

        # 根据策略合并梯度
        if strategy == 'dynamic':
            # 动态选择：找到梯度变化最大的epoch
            field_importance = self._select_dynamic_epoch(epoch_gradients)
        elif strategy == 'multiple':
            # 多阶段平均
            field_importance = self._average_multiple_epochs(epoch_gradients)
        else:
            # 单阶段：使用第一个目标epoch
            field_importance = epoch_gradients[target_epochs[0]]

        return field_importance

    def _compute_epoch_gradients(self, train_loader, target_epoch, total_epochs):
        """计算特定训练轮数的梯度重要性"""
        # 模拟训练到目标轮数（在实际应用中，应该从检查点加载对应轮数的模型）
        # 这里我们假设模型已经训练到目标轮数

        self.model.train()
        gradient_accumulator = {field: [] for field in self.field_names}

        # 使用少量批次计算梯度（避免计算开销过大）
        num_batches = min(20, len(train_loader))
        batch_count = 0

        for batch_idx, (batch_embeddings, batch_labels) in enumerate(train_loader):
            if batch_count >= num_batches:
                break

            batch_embeddings.requires_grad_(True)
            self.model.zero_grad()

            # 前向传播
            outputs = self.model(batch_embeddings)
            loss = nn.CrossEntropyLoss()(outputs, batch_labels)

            # 反向传播
            loss.backward()

            # 分析梯度
            if batch_embeddings.grad is not None:
                gradients = batch_embeddings.grad.detach()

                # 计算每个字段的梯度重要性
                # 使用多种度量方法
                abs_gradients = gradients.abs()
                mean_abs = abs_gradients.mean(dim=[0, 2])  # [num_fields]

                # 存储每个字段的梯度
                for field_idx, field_name in enumerate(self.field_names):
                    if field_idx < len(mean_abs):
                        gradient_accumulator[field_name].append(mean_abs[field_idx].item())

            batch_count += 1

        # 计算平均梯度重要性
        field_importance = {}
        for field_name in self.field_names:
            if gradient_accumulator[field_name]:
                avg_gradient = np.mean(gradient_accumulator[field_name])
                field_importance[field_name] = avg_gradient
            else:
                field_importance[field_name] = 0.0

        # 归一化
        total = sum(field_importance.values())
        if total > 0:
            for field_name in field_importance:
                field_importance[field_name] /= total

        print(f"第 {target_epoch} 轮梯度计算完成，平均梯度强度: {np.mean(list(field_importance.values())):.6f}")

        return field_importance

    def _select_dynamic_epoch(self, epoch_gradients):
        """动态选择信息量最大的训练阶段"""
        print("动态选择信息量最大的训练阶段...")

        # 计算每个epoch的梯度统计量
        epoch_stats = {}
        for epoch, gradients in epoch_gradients.items():
            grad_values = list(gradients.values())
            epoch_stats[epoch] = {
                'mean': np.mean(grad_values),
                'std': np.std(grad_values),
                'max': np.max(grad_values),
                'entropy': self._compute_entropy(gradients)
            }

        # 选择标准：梯度强度适中且区分度高的阶段
        best_epoch = None
        best_score = -1

        for epoch, stats in epoch_stats.items():
            # 综合评分：考虑均值、标准差和熵
            score = (
                    stats['mean'] * 0.4 +  # 梯度强度
                    stats['std'] * 0.3 +  # 梯度区分度
                    stats['entropy'] * 0.3  # 信息量
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch

        print(f"选择第 {best_epoch} 轮作为代表性阶段 (评分: {best_score:.4f})")
        return epoch_gradients[best_epoch]

    def _average_multiple_epochs(self, epoch_gradients):
        """合并多个训练阶段的梯度重要性"""
        print("合并多个训练阶段的梯度重要性...")

        # 收集所有字段在所有epoch的重要性分数
        field_scores = {field: [] for field in self.field_names}

        for epoch, gradients in epoch_gradients.items():
            for field_name, importance in gradients.items():
                field_scores[field_name].append(importance)

        # 计算加权平均（后期epoch权重较低，因为梯度可能太小）
        field_importance = {}
        total_epochs = len(epoch_gradients)

        for field_name, scores in field_scores.items():
            if scores:
                # 给不同epoch分配权重（前期和中期权重较高）
                weights = []
                epochs = list(epoch_gradients.keys())
                for i, epoch in enumerate(epochs):
                    # 前期和中期权重为1，后期权重递减
                    if epoch <= total_epochs * 0.7:
                        weight = 1.0
                    else:
                        # 后期权重线性递减到0.3
                        progress = (epoch - total_epochs * 0.7) / (total_epochs * 0.3)
                        weight = 1.0 - 0.7 * min(1.0, progress)
                    weights.append(weight)

                # 计算加权平均
                weighted_avg = np.average(scores, weights=weights)
                field_importance[field_name] = weighted_avg
            else:
                field_importance[field_name] = 0.0

        # 归一化
        total = sum(field_importance.values())
        if total > 0:
            for field_name in field_importance:
                field_importance[field_name] /= total

        return field_importance

    def _compute_entropy(self, gradient_dict):
        """计算梯度分布的熵（衡量信息量）"""
        values = np.array(list(gradient_dict.values()))
        if values.sum() == 0:
            return 0
        # 归一化
        prob = values / values.sum()
        # 计算熵
        entropy = -np.sum(prob * np.log(prob + 1e-8))
        return entropy

    def analyze_gradient_evolution(self, train_loader, total_epochs, sample_epochs=5):
        """分析梯度在训练过程中的演化"""
        print("分析梯度演化过程...")

        # 采样几个关键epoch
        sample_points = [
            1,  # 开始
            max(2, total_epochs // 4),  # 25%
            max(3, total_epochs // 2),  # 50%
            max(4, total_epochs * 3 // 4),  # 75%
            total_epochs  # 结束
        ]

        epoch_gradients = {}
        for epoch in sample_points:
            gradients = self._compute_epoch_gradients(train_loader, epoch, total_epochs)
            epoch_gradients[epoch] = gradients

        # 可视化梯度演化
        self._visualize_gradient_evolution(epoch_gradients)

        return epoch_gradients

    def _visualize_gradient_evolution(self, epoch_gradients):
        """可视化梯度演化过程"""
        import matplotlib.pyplot as plt

        # 选择Top 10字段进行可视化
        all_fields = set()
        for gradients in epoch_gradients.values():
            all_fields.update(gradients.keys())

        # 计算平均重要性用于排序
        avg_importance = {}
        for field in all_fields:
            importances = [gradients.get(field, 0) for gradients in epoch_gradients.values()]
            avg_importance[field] = np.mean(importances)

        top_fields = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_field_names = [field for field, _ in top_fields]

        # 创建演化图
        plt.figure(figsize=(12, 8))
        epochs = sorted(epoch_gradients.keys())

        for field in top_field_names:
            importances = [epoch_gradients[epoch].get(field, 0) for epoch in epochs]
            plt.plot(epochs, importances, 'o-', label=field, linewidth=2, markersize=6)

        plt.xlabel('训练轮数')
        plt.ylabel('梯度重要性')
        plt.title('Top 10字段的梯度重要性演化')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 打印演化分析
        print("\n梯度演化分析:")
        print("-" * 50)
        for field in top_field_names:
            importances = [epoch_gradients[epoch].get(field, 0) for epoch in epochs]
            max_epoch = epochs[np.argmax(importances)]
            min_epoch = epochs[np.argmin(importances)]
            change = importances[-1] - importances[0]

            print(f"{field:<25}: 峰值在epoch {max_epoch}, 变化 {change:+.4f}")


class SelfAttention(nn.Module):
    """
    底层注意力机制：允许 Token (特征) 之间进行全局交互。
    修改：增加权重保存功能用于特征重要性分析。
    """

    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        # 这里的 batch_first=True 非常重要
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.attention_weights = None  # 新增：用于存储权重

    def forward(self, x):
        # x: [batch_size, num_fields, embed_dim]

        # average_attn_weights=False 让我们可以获取每个 Head 的权重
        # need_weights=True 是默认的，但显式写出来更清晰
        attn_output, attn_weights = self.mha(x, x, x, average_attn_weights=False)

        # attn_weights 形状: [batch_size, num_heads, seq_len, seq_len]
        # 我们把它存起来
        self.attention_weights = attn_weights

        # 残差连接 + 归一化
        x = self.layernorm(x + attn_output)
        return x


class AttentionAnalyzer:
    """注意力分析器，用于分析CustomTransformerEncoder模型的注意力机制"""

    def __init__(self, model, field_names, num_heads=8):
        self.model = model
        self.field_names = field_names
        self.num_heads = num_heads
        self.attention_weights = {}

    def analyze_attention(self, dataloader, num_samples=100, layer_type='transformer_last'):
        """
        分析注意力权重

        Args:
            dataloader: 数据加载器
            num_samples: 分析的样本数
            layer_type: 指定要分析的层
                - 'pre_attention': 使用 pre_attention 层
                - 'transformer_last': 使用 Transformer 最后一层
                - 'transformer_first': 使用 Transformer 第一层（备选）
        """
        # self.model.eval()
        self.attention_weights = {}  # 重置
        sample_count = 0
        target_layer_name = None

        with torch.no_grad():
            for batch_embeddings, batch_labels in dataloader:
                if sample_count >= num_samples:
                    break

                # 执行前向传播，触发权重计算
                _ = self.model(batch_embeddings)

                batch_attn_weights = None

                # 根据 layer_type 选择权重来源
                if layer_type == 'pre_attention' and hasattr(self.model, 'pre_attention'):
                    if self.model.pre_attention.attention_weights is not None:
                        batch_attn_weights = self.model.pre_attention.attention_weights.cpu()
                        target_layer_name = 'pre_attention'

                elif layer_type == 'transformer_last' and hasattr(self.model, 'transformer'):
                    if hasattr(self.model.transformer, 'attention_weights') and len(self.model.transformer.attention_weights) > 0:
                        batch_attn_weights = self.model.transformer.attention_weights[-1].cpu()
                        target_layer_name = 'last_transformer_layer'

                # 如果指定的层未找到，尝试第一层作为备选
                if batch_attn_weights is None and hasattr(self.model, 'transformer'):
                    if hasattr(self.model.transformer, 'attention_weights') and len(self.model.transformer.attention_weights) > 0:
                        batch_attn_weights = self.model.transformer.attention_weights[0].cpu()
                        target_layer_name = 'first_transformer_layer'
                        print(f"警告: 未找到指定的层 '{layer_type}'，使用 Transformer 第一层代替。")

                if batch_attn_weights is None:
                    print(f"警告: 未能获取任何注意力权重，跳过当前批次。")
                    continue

                # 处理权重并存储
                batch_size = batch_attn_weights.size(0)

                # 初始化存储列表
                if target_layer_name not in self.attention_weights:
                    self.attention_weights[target_layer_name] = {'full': [], 'mean': []}

                for i in range(min(batch_size, num_samples - sample_count)):
                    # 获取单个样本的权重 [num_heads, seq_len, seq_len]
                    layer_attn_per_sample = batch_attn_weights[i].numpy()
                    self.attention_weights[target_layer_name]['full'].append(layer_attn_per_sample)
                    self.attention_weights[target_layer_name]['mean'].append(layer_attn_per_sample.mean(axis=0))

                sample_count += batch_size

        print(f"成功收集 {sample_count} 个样本的注意力权重，来自层: {target_layer_name}")
        return self._aggregate_attention()

    def _aggregate_attention(self):
        """聚合注意力权重"""
        aggregated = {}

        for layer_name, attn_dict in self.attention_weights.items():
            if attn_dict['full']:
                # 平均所有样本的注意力权重
                mean_full_attention = np.mean(attn_dict['full'], axis=0)  # [num_heads, seq_len, seq_len]
                mean_mean_attention = np.mean(attn_dict['mean'], axis=0)  # [seq_len, seq_len]

                aggregated[layer_name] = {
                    'full': mean_full_attention,
                    'mean': mean_mean_attention
                }

        return aggregated

    def get_field_importance_variance(self, attention_maps):
        """基于注意力权重的方差计算字段重要性"""
        if not attention_maps:
            return {}

        # 获取第一个层（通常只有一个层）
        layer_name = list(attention_maps.keys())[0]
        attention_matrix_full = attention_maps[layer_name]['full']  # [num_heads, seq_len, seq_len]

        # 计算每个字段在不同注意力头中的关注度方差
        # 高方差表示该字段在某些头中很重要，在某些头中不重要
        # 低方差表示该字段在所有头中关注度一致

        field_variances = []

        for field_idx in range(attention_matrix_full.shape[2]):  # 遍历每个字段
            # 获取所有头对该字段的关注度（入度）
            field_attention = attention_matrix_full[:, :, field_idx]  # [num_heads, seq_len]
            field_in_degree = field_attention.sum(axis=1)  # [num_heads] - 每个头对该字段的总关注度

            # 计算方差 - 高方差表示该字段在某些头中特别重要
            variance = np.var(field_in_degree)
            field_variances.append(variance)

        importance_scores = np.array(field_variances)

        # 归一化
        if importance_scores.sum() > 0:
            importance_scores = importance_scores / importance_scores.sum()

        # 创建字段名称到重要性分数的映射
        field_importance = {}
        for i, field_name in enumerate(self.field_names):
            if i < len(importance_scores):
                field_importance[field_name] = importance_scores[i]

        return field_importance

    def get_field_importance(self, attention_maps, method='in_degree', head_aggregation='mean', include_cls=False):
        """
        计算字段重要性分数 - 适配新模型结构
        """
        if not attention_maps:
            return {}

        # 获取第一个层（通常只有一个层）
        layer_name = list(attention_maps.keys())[0]

        # 注意：新模型中序列长度是 num_fields + 1（包含CLS token）
        if include_cls:
            # 包含CLS token
            seq_len = len(self.field_names) + 1
        else:
            # 不包含CLS token，只分析原始字段
            seq_len = len(self.field_names)

        if head_aggregation == 'full':
            # 使用完整的注意力权重 [num_heads, seq_len, seq_len]
            attention_matrix_full = attention_maps[layer_name]['full']

            if not include_cls:
                # 去除CLS token对应的行和列（第0个位置）
                attention_matrix_full = attention_matrix_full[:, 1:, 1:]

            importance_scores_per_head = []

            for head_idx in range(attention_matrix_full.shape[0]):
                head_attention = attention_matrix_full[head_idx]  # [seq_len, seq_len]

                if method == 'in_degree':
                    # 方法1：入度重要性 - 其他字段对该字段的关注程度
                    importance_scores = head_attention.sum(axis=0)
                elif method == 'out_degree':
                    # 方法2：出度重要性 - 该字段对其他字段的关注程度
                    importance_scores = head_attention.sum(axis=1)
                elif method == 'combined':
                    # 方法3：综合重要性 - 入度和出度的加权平均
                    in_degree = head_attention.sum(axis=0)
                    out_degree = head_attention.sum(axis=1)
                    importance_scores = (in_degree + out_degree) / 2
                elif method == 'max_attention':
                    # 方法4：最大关注度 - 该字段受到的最大关注程度
                    importance_scores = head_attention.max(axis=0)
                elif method == 'centrality':
                    # 方法5：中心性 - 基于特征向量的中心性度量
                    try:
                        eigenvalues, eigenvectors = np.linalg.eig(head_attention)
                        principal_eigenvector = np.abs(eigenvectors[:, np.argmax(eigenvalues)])
                        importance_scores = principal_eigenvector
                    except:
                        # 如果特征值分解失败，回退到入度
                        importance_scores = head_attention.sum(axis=0)
                else:
                    # 默认使用入度
                    importance_scores = head_attention.sum(axis=0)

                importance_scores_per_head.append(importance_scores)

            # 平均所有注意力头的重要性分数
            importance_scores = np.mean(importance_scores_per_head, axis=0)
        else:
            # 使用平均后的注意力权重 [seq_len, seq_len]
            attention_matrix = attention_maps[layer_name]['mean']

            if not include_cls:
                # 去除CLS token对应的行和列
                attention_matrix = attention_matrix[1:, 1:]

            if method == 'in_degree':
                importance_scores = attention_matrix.sum(axis=0)
            elif method == 'out_degree':
                importance_scores = attention_matrix.sum(axis=1)
            elif method == 'combined':
                in_degree = attention_matrix.sum(axis=0)
                out_degree = attention_matrix.sum(axis=1)
                importance_scores = (in_degree + out_degree) / 2
            elif method == 'max_attention':
                importance_scores = attention_matrix.max(axis=0)
            elif method == 'centrality':
                try:
                    eigenvalues, eigenvectors = np.linalg.eig(attention_matrix)
                    principal_eigenvector = np.abs(eigenvectors[:, np.argmax(eigenvalues)])
                    importance_scores = principal_eigenvector
                except:
                    importance_scores = attention_matrix.sum(axis=0)
            else:
                importance_scores = attention_matrix.sum(axis=0)

        # 归一化到[0,1]
        if importance_scores.sum() > 0:
            importance_scores = importance_scores / importance_scores.sum()

        # 创建字段名称到重要性分数的映射
        field_importance = {}

        if include_cls:
            # 包含CLS token
            field_names_with_cls = ['CLS'] + self.field_names
            for i, field_name in enumerate(field_names_with_cls):
                if i < len(importance_scores):
                    field_importance[field_name] = importance_scores[i]
        else:
            # 只包含原始字段
            for i, field_name in enumerate(self.field_names):
                if i < len(importance_scores):
                    field_importance[field_name] = importance_scores[i]

        return field_importance

    def analyze_cls_attention(self, attention_maps):
        """
        分析CLS token的注意力模式
        """
        if not attention_maps:
            return {}

        # 获取第一个层
        layer_name = list(attention_maps.keys())[0]
        attention_matrix_full = attention_maps[layer_name]['full']  # [num_heads, seq_len, seq_len]

        # CLS token是第0个位置
        # 分析CLS token对其他字段的关注程度
        cls_attention_scores = {}

        for head_idx in range(attention_matrix_full.shape[0]):
            head_attention = attention_matrix_full[head_idx]  # [seq_len, seq_len]

            # CLS token对其他字段的注意力（行0，列1:）
            cls_to_fields = head_attention[0, 1:]  # 排除CLS token自身

            # 计算每个字段被CLS token关注的程度
            for field_idx, field_name in enumerate(self.field_names):
                if field_idx < len(cls_to_fields):
                    if field_name not in cls_attention_scores:
                        cls_attention_scores[field_name] = []
                    cls_attention_scores[field_name].append(cls_to_fields[field_idx])

        # 计算平均CLS注意力
        avg_cls_attention = {}
        for field_name, scores in cls_attention_scores.items():
            avg_cls_attention[field_name] = np.mean(scores)

        # 归一化
        total = sum(avg_cls_attention.values())
        if total > 0:
            for field_name in avg_cls_attention:
                avg_cls_attention[field_name] /= total

        return avg_cls_attention

    # def visualize_attention(self, attention_maps, field_importance, save_path=None):
    #     if not attention_maps:
    #         print("没有可用的注意力数据")
    #         return
    #
    #     layer_name = list(attention_maps.keys())[0]
    #     fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # 调整 figsize 高度为 8 以适应一行
    #     fig.suptitle(f'Transformer注意力分析 (层: {layer_name})', fontsize=16, fontweight='bold')
    #
    #     # 1. 注意力热力图（平均所有头）
    #     attention_matrix = attention_maps[layer_name]['mean']
    #     field_names_with_cls = ['CLS'] + self.field_names
    #
    #     im = axes[0].imshow(attention_matrix, cmap='viridis', aspect='auto')
    #     axes[0].set_title('注意力热力图（平均所有头，包含CLS）', fontsize=12)
    #     axes[0].set_xlabel('Key Fields')
    #     axes[0].set_ylabel('Query Fields')
    #     axes[0].set_xticks(range(len(field_names_with_cls)))
    #     axes[0].set_yticks(range(len(field_names_with_cls)))
    #     axes[0].set_xticklabels(field_names_with_cls, rotation=45, ha='right', fontsize=8)
    #     axes[0].set_yticklabels(field_names_with_cls, fontsize=8)
    #     plt.colorbar(im, ax=axes[0])
    #
    #     # 2. 字段重要性条形图
    #     if field_importance:
    #         sorted_fields = sorted(field_importance.items(), key=lambda x: x[1], reverse=True)
    #         top_fields = sorted_fields[:15]
    #         field_names = [field[0] for field in top_fields]
    #         importance_scores = [field[1] for field in top_fields]
    #
    #         bars = axes[1].barh(range(len(field_names)), importance_scores)
    #         axes[1].set_title('字段重要性排名 (前15个)', fontsize=12)
    #         axes[1].set_yticks(range(len(field_names)))
    #         axes[1].set_yticklabels(field_names, fontsize=9)
    #         axes[1].set_xlabel('重要性分数')
    #         for i, (bar, score) in enumerate(zip(bars, importance_scores)):
    #             axes[1].text(score + 0.001, bar.get_y() + bar.get_height() / 2,
    #                          f'{score:.3f}', ha='left', va='center', fontsize=8)
    #
    #     # 3. CLS token注意力分析
    #     cls_attention = self.analyze_cls_attention(attention_maps)
    #     if cls_attention:
    #         sorted_cls_attention = sorted(cls_attention.items(), key=lambda x: x[1], reverse=True)
    #         top_cls_fields = sorted_cls_attention[:15]
    #         field_names = [field[0] for field in top_cls_fields]
    #         attention_scores = [field[1] for field in top_cls_fields]
    #
    #         bars = axes[2].barh(range(len(field_names)), attention_scores, color='orange')
    #         axes[2].set_title('CLS Token关注度最高的字段 (Top 15)', fontsize=12)
    #         axes[2].set_yticks(range(len(field_names)))
    #         axes[2].set_yticklabels(field_names, fontsize=9)
    #         axes[2].set_xlabel('CLS注意力分数')
    #         for i, (bar, score) in enumerate(zip(bars, attention_scores)):
    #             axes[2].text(score + 0.001, bar.get_y() + bar.get_height() / 2,
    #                          f'{score:.3f}', ha='left', va='center', fontsize=8)
    #
    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         print(f"注意力分析图已保存至: {save_path}")
    #     plt.show()
    def visualize_attention(self, attention_maps, field_importance, save_path=None):
        if not attention_maps:
            print("没有可用的注意力数据")
            return

        layer_name = list(attention_maps.keys())[0]

        # 设置总画布尺寸：宽 32 (16+8+8)，高 16
        fig = plt.figure(figsize=(32, 16))
        fig.suptitle(f'Transformer Attention Analysis (Layer: {layer_name})', fontsize=28, fontweight='bold', y=0.98)

        # 使用 GridSpec 创建 1行3列 的网格布局，通过 width_ratios 保持宽度比例 2:1:1
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])

        # 第一张图：占据第 0 行，第 0 列 (相对宽度2，等同于 16x16 的空间)
        ax1 = fig.add_subplot(gs[0, 0])
        # 第二张图：占据第 0 行，第 1 列 (相对宽度1，等同于 8x16 的空间)
        ax2 = fig.add_subplot(gs[0, 1])
        # 第三张图：占据第 0 行，第 2 列 (相对宽度1，等同于 8x16 的空间)
        ax3 = fig.add_subplot(gs[0, 2])

        # 1. 注意力热力图（平均所有头）
        attention_matrix = attention_maps[layer_name]['mean']
        field_names_with_cls = ['CLS'] + self.field_names

        im = ax1.imshow(attention_matrix, cmap='viridis', aspect='auto')
        ax1.set_title('Attention Hot Map', fontsize=18, pad=15)
        ax1.set_xlabel('Key Fields', fontsize=16)
        ax1.set_ylabel('Query Fields', fontsize=16)
        ax1.set_xticks(range(len(field_names_with_cls)))
        ax1.set_yticks(range(len(field_names_with_cls)))
        ax1.set_xticklabels(field_names_with_cls, rotation=45, ha='right', fontsize=9)
        ax1.set_yticklabels(field_names_with_cls, fontsize=9)
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # 2. 字段重要性条形图
        if field_importance:
            sorted_fields = sorted(field_importance.items(), key=lambda x: x[1], reverse=True)
            top_fields = sorted_fields[:15]
            field_names = [field[0] for field in top_fields]
            importance_scores = [field[1] for field in top_fields]

            bars = ax2.barh(range(len(field_names)), importance_scores)
            ax2.set_title('Field Importance Ranking (Top 15)', fontsize=18, pad=15)
            ax2.set_yticks(range(len(field_names)))
            ax2.set_yticklabels(field_names, fontsize=14)
            ax2.set_xlabel('Importance Score', fontsize=16)
            ax2.invert_yaxis()  # 反转Y轴，让排名第一的显示在最上方

            for i, (bar, score) in enumerate(zip(bars, importance_scores)):
                ax2.text(score + 0.001, bar.get_y() + bar.get_height() / 2,
                         f'{score:.3f}', ha='left', va='center', fontsize=10)

        # 3. CLS token注意力分析
        cls_attention = self.analyze_cls_attention(attention_maps)
        if cls_attention:
            sorted_cls_attention = sorted(cls_attention.items(), key=lambda x: x[1], reverse=True)
            top_cls_fields = sorted_cls_attention[:15]
            field_names = [field[0] for field in top_cls_fields]
            attention_scores = [field[1] for field in top_cls_fields]

            bars = ax3.barh(range(len(field_names)), attention_scores, color='orange')
            ax3.set_title('Fields with Highest CLS Token Attention (Top 15)', fontsize=18, pad=15)
            ax3.set_yticks(range(len(field_names)))
            ax3.set_yticklabels(field_names, fontsize=14)
            ax3.set_xlabel('CLS Attention Score', fontsize=16)
            ax3.invert_yaxis()  # 反转Y轴，让排名第一的显示在最上方

            for i, (bar, score) in enumerate(zip(bars, attention_scores)):
                ax3.text(score + 0.001, bar.get_y() + bar.get_height() / 2,
                         f'{score:.3f}', ha='left', va='center', fontsize=10)

        # 调整布局以防止重叠，并为suptitle预留顶部空间
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力分析图已保存至: {save_path}")

        plt.show()  # 显示组合图

        # ========== 新增：单独绘制字段重要性条形图 ==========
        if field_importance:
            plt.figure(figsize=(8, 16))  # 单独图尺寸
            sorted_fields = sorted(field_importance.items(), key=lambda x: x[1], reverse=True)
            top_fields = sorted_fields[:15]
            field_names = [field[0] for field in top_fields]
            importance_scores = [field[1] for field in top_fields]

            bars = plt.barh(range(len(field_names)), importance_scores)
            plt.yticks(range(len(field_names)), field_names)
            plt.xlabel('Importance Score', fontsize=16)
            plt.title('Field Importance Ranking (Top 15)', fontsize=18, pad=15)
            plt.gca().invert_yaxis()  # 让排名第一的显示在最上方

            for i, (bar, score) in enumerate(zip(bars, importance_scores)):
                plt.text(score + 0.001, bar.get_y() + bar.get_height() / 2,
                         f'{score:.3f}', ha='left', va='center', fontsize=10)

            plt.tight_layout()
            if save_path:
                # 生成单独保存路径（在原文件名基础上添加 "_field_importance"）
                base, ext = os.path.splitext(save_path)
                field_save_path = base + '_field_importance' + ext
                plt.savefig(field_save_path, dpi=300, bbox_inches='tight')
                print(f"字段重要性图已单独保存至: {field_save_path}")

            plt.show()  # 显示单独图

    def print_field_importance_report(self, field_importance, attention_maps=None):
        """打印字段重要性报告"""
        if not field_importance:
            print("没有可用的字段重要性数据")
            return

        print("\n" + "=" * 80)
        print("字段重要性分析报告（适配HealthDataTransformer2）")
        print("=" * 80)

        # 按重要性排序
        sorted_fields = sorted(field_importance.items(), key=lambda x: x[1], reverse=True)

        print(f"\n总字段数: {len(sorted_fields)}")
        print("\nTop 20 重要字段:")
        print("-" * 60)
        print(f"{'排名':<4} {'字段名':<30} {'重要性分数':<12} {'百分比':<10}")
        print("-" * 60)

        for i, (field_name, importance) in enumerate(sorted_fields[:20], 1):
            percentage = importance * 100
            print(f"{i:<4} {field_name:<30} {importance:.6f}     {percentage:.2f}%")

        # 统计信息
        top_10_importance = sum(importance for _, importance in sorted_fields[:10])
        top_20_importance = sum(importance for _, importance in sorted_fields[:20])

        print(f"\n统计信息:")
        print(f"前10个字段累计重要性: {top_10_importance:.3f} ({top_10_importance * 100:.1f}%)")
        print(f"前20个字段累计重要性: {top_20_importance:.3f} ({top_20_importance * 100:.1f}%)")

        # 识别关键字段（重要性超过平均值的2倍）
        mean_importance = np.mean([imp for _, imp in sorted_fields])
        key_fields = [(field, imp) for field, imp in sorted_fields if imp > 2 * mean_importance]

        print(f"\n关键字段 (重要性 > {2 * mean_importance:.4f}):")
        for field, imp in key_fields:
            print(f"  - {field}: {imp:.4f}")

        # CLS注意力分析
        if attention_maps:
            cls_attention = self.analyze_cls_attention(attention_maps)
            if cls_attention:
                sorted_cls = sorted(cls_attention.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\nCLS Token最关注的5个字段:")
                for field, imp in sorted_cls:
                    print(f"  - {field}: {imp:.4f}")

        print("=" * 80)