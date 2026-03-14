"""
数据预处理与加载模块
用于处理卒中预测的电子健康记录数据
支持EasyEnsemble方法处理类别不平衡
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import gc
import os

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 数据集类定义
# ==========================================

class HealthDataset(Dataset):
    """基础健康数据Dataset类"""

    def __init__(self, field_embeddings, labels):
        """
        健康数据Dataset类
        Args:
            field_embeddings: 字段嵌入张量 [num_samples, num_fields, feature_dim]
            labels: 标签张量 [num_samples]
        """
        self.field_embeddings = field_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.field_embeddings[idx], self.labels[idx]


class EasyEnsembleTrainDataset(Dataset):
    """EasyEnsemble训练数据集，使用装袋方法处理类别不平衡"""

    def __init__(self, positive_embeddings, negative_embeddings, K=10, min_subset_size=None):
        """
        Args:
            positive_embeddings: 阳性样本（中风）嵌入 [num_positive, num_fields, feature_dim]
            negative_embeddings: 阴性样本（健康）嵌入 [num_negative, num_fields, feature_dim]
            K: 阴性样本子集数量
            min_subset_size: 最小子集大小（如果为None，则使用阳性样本数）
        """
        self.positive_embeddings = positive_embeddings.to(device)
        self.negative_embeddings = negative_embeddings.to(device)
        self.K = K
        self.positive_labels = torch.ones(len(positive_embeddings), dtype=torch.long, device=device)
        self.negative_labels = torch.zeros(len(negative_embeddings), dtype=torch.long, device=device)

        # 设置最小子集大小
        if min_subset_size is None:
            self.min_subset_size = len(positive_embeddings)
        else:
            self.min_subset_size = min_subset_size

        # 划分阴性样本为K个子集
        self.negative_subsets = self.split_negative_samples()

        # 初始化当前子集索引
        self.current_subset_idx = 0
        self.current_negative_indices = self.negative_subsets[0]

        # 难负样本池
        self.hard_negative_pool = []

        print(f"EasyEnsemble初始化:")
        print(f"  阳性样本数: {len(self.positive_embeddings)}")
        print(f"  阴性样本总数: {len(self.negative_embeddings)}")
        print(f"  划分成 {K} 个子集")

        # 打印每个子集的大小
        for i, subset in enumerate(self.negative_subsets):
            print(f"    子集 {i + 1}: {len(subset)} 个样本")

        self.update_dataset()

    def split_negative_samples(self):
        """
        将阴性样本划分为K个不相交的子集
        优化版本：使用所有阴性样本，动态调整子集大小
        """
        num_negative = len(self.negative_embeddings)
        num_positive = len(self.positive_embeddings)

        print(f"  阳性样本数: {num_positive}")
        print(f"  阴性样本数: {num_negative}")
        print(f"  目标子集数: {self.K}")
        print(f"  最小子集大小: {self.min_subset_size}")

        # 计算实际可用的子集数量（基于阴性样本数）
        max_possible_K = num_negative // self.min_subset_size
        if max_possible_K < self.K:
            print(f"  警告: 阴性样本不足，实际可用子集数: {max_possible_K} (目标: {self.K})")
            self.K = max(1, max_possible_K)

        # 重新计算每个子集的目标大小
        # 尽量让每个子集大小相近，使用所有阴性样本
        base_size = num_negative // self.K
        remainder = num_negative % self.K

        print(f"  每个子集基础大小: {base_size}")
        print(f"  剩余样本数: {remainder}")

        # 打乱阴性样本
        indices = torch.randperm(num_negative, device=device)

        # 划分索引
        subsets = []
        start_idx = 0

        for i in range(self.K):
            # 计算当前子集大小
            subset_size = base_size + 1 if i < remainder else base_size

            # 确保子集大小不小于最小要求
            if subset_size < self.min_subset_size and num_negative >= self.min_subset_size:
                # 如果样本足够，调整子集数量
                new_K = num_negative // self.min_subset_size
                print(f"  子集大小不足，调整子集数为: {new_K}")
                self.K = new_K
                return self.split_negative_samples()  # 重新计算

            # 获取当前子集的索引
            end_idx = start_idx + subset_size
            subset = indices[start_idx:end_idx]
            subsets.append(subset)
            start_idx = end_idx

        # 验证所有样本都被使用
        total_used = sum(len(subset) for subset in subsets)
        print(f"  使用的总阴性样本数: {total_used}/{num_negative}")

        # 验证子集之间没有重叠
        all_indices = torch.cat(subsets)
        unique_indices = torch.unique(all_indices)
        if len(unique_indices) != len(all_indices):
            print(f"  警告: 子集之间有重叠!")
        else:
            print(f"  验证通过: 所有子集不相交")

        return subsets

    def update_dataset(self):
        """更新当前数据集（使用当前阴性子集）"""
        # 获取当前子集的阴性样本
        negative_indices = self.current_negative_indices

        # 如果有难负样本，优先加入
        if self.hard_negative_pool:
            # 计算需要添加的难负样本数量（不超过阴性样本的20%）
            max_hard_samples = min(len(negative_indices) // 5, len(self.hard_negative_pool))
            if max_hard_samples > 0:
                hard_samples = torch.tensor(self.hard_negative_pool[:max_hard_samples], device=device)
                negative_indices = torch.cat([negative_indices, hard_samples])

                # 去重
                negative_indices = torch.unique(negative_indices)
                print(f"  添加了 {len(hard_samples)} 个难负样本")

        # 合并阳性样本和当前阴性子集
        self.current_embeddings = torch.cat([
            self.positive_embeddings,
            self.negative_embeddings[negative_indices]
        ])

        self.current_labels = torch.cat([
            self.positive_labels,
            self.negative_labels[negative_indices]
        ])

        # 打乱顺序
        indices = torch.randperm(len(self.current_labels), device=device)
        self.current_embeddings = self.current_embeddings[indices]
        self.current_labels = self.current_labels[indices]

        # 打印当前分布
        pos_count = (self.current_labels == 1).sum().item()
        neg_count = (self.current_labels == 0).sum().item()
        print(f"  当前批次: 阳性 {pos_count}, 阴性 {neg_count}, 总数 {len(self.current_labels)}")
        print(f"  类别比例 (阴:阳): {neg_count / pos_count:.2f}:1" if pos_count > 0 else "  类别比例: N/A")

    def next_subset(self):
        """切换到下一个阴性子集"""
        self.current_subset_idx = (self.current_subset_idx + 1) % self.K
        self.current_negative_indices = self.negative_subsets[self.current_subset_idx]
        self.update_dataset()
        print(f"切换到阴性子集 {self.current_subset_idx + 1}/{self.K}")

    def update_hard_negatives(self, model, device, num_samples=2000, top_percent=0.1, max_pool_size=5000):
        """
        优化后的滚动难负样本挖掘
        Args:
            model: 当前训练的模型
            device: 设备
            num_samples: 随机采样的评估样本数
            top_percent: 每次挖掘保留的比例
            max_pool_size: 难负样本池的最大容量
        """
        model.eval()

        # 1. 采样并预测
        if len(self.negative_embeddings) > num_samples:
            indices = torch.randperm(len(self.negative_embeddings), device=device)[:num_samples]
        else:
            indices = torch.arange(len(self.negative_embeddings), device=device)

        negative_subset = self.negative_embeddings[indices]

        with torch.no_grad():
            outputs = model(negative_subset)
            # 获取中风（类别1）的概率
            probabilities = torch.softmax(outputs, dim=1)[:, 1]

        # 2. 识别本次挖掘中最难的样本
        num_hard = max(1, int(len(probabilities) * top_percent))
        hard_probs, hard_idx_in_subset = torch.topk(probabilities, num_hard)

        # 映射回原始阴性样本池的索引
        new_hard_indices = indices[hard_idx_in_subset].cpu().numpy().tolist()
        new_hard_probs = hard_probs.cpu().numpy().tolist()

        # 3. 滚动更新策略
        # 将现有池中的样本与新样本合并 (存储格式改为: {index: probability})
        # 这样可以更新旧样本在当前模型下的最新概率
        current_pool_dict = getattr(self, 'hard_pool_with_probs', {})

        # 将新挖掘的样本加入或更新字典
        for idx, prob in zip(new_hard_indices, new_hard_probs):
            current_pool_dict[idx] = prob

        # 按概率降序排列，仅保留前 max_pool_size 个最难样本
        sorted_pool = sorted(current_pool_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_pool = sorted_pool[:max_pool_size]

        # 重新构建池和概率记录
        self.hard_pool_with_probs = dict(sorted_pool)
        self.hard_negative_pool = list(self.hard_pool_with_probs.keys())

        avg_prob = np.mean(list(self.hard_pool_with_probs.values())) if self.hard_pool_with_probs else 0
        print(
            f"滚动更新完成: 池大小 {len(self.hard_negative_pool)}/{max_pool_size}, 当前池平均中风概率: {avg_prob:.4f}")

        model.train()
        return self.hard_negative_pool

    def __len__(self):
        return len(self.current_labels)

    def __getitem__(self, idx):
        return self.current_embeddings[idx], self.current_labels[idx]


class BalancedValidationDataset(Dataset):
    """平衡验证数据集，对多数类进行欠采样"""

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings.to(device)
        self.labels = labels.to(device)
        self.balance_dataset()

    def balance_dataset(self):
        """对验证集进行平衡采样"""
        # 获取两类样本的索引
        class_0_indices = torch.where(self.labels == 0)[0]
        class_1_indices = torch.where(self.labels == 1)[0]

        print(f"平衡前验证集分布 - 类别0: {len(class_0_indices)}, 类别1: {len(class_1_indices)}")

        # 确定少数类样本数量
        min_class_count = min(len(class_0_indices), len(class_1_indices))

        # 对多数类进行随机欠采样
        if len(class_0_indices) > min_class_count:
            # 类别0是多数类，进行欠采样
            selected_0_indices = class_0_indices[torch.randperm(len(class_0_indices), device=device)[:min_class_count]]
            selected_1_indices = class_1_indices
        else:
            # 类别1是多数类，进行欠采样
            selected_0_indices = class_0_indices
            selected_1_indices = class_1_indices[torch.randperm(len(class_1_indices), device=device)[:min_class_count]]

        # 合并采样后的索引
        balanced_indices = torch.cat([selected_0_indices, selected_1_indices])

        # 打乱顺序
        balanced_indices = balanced_indices[torch.randperm(len(balanced_indices), device=device)]

        # 更新数据集
        self.embeddings = self.embeddings[balanced_indices]
        self.labels = self.labels[balanced_indices]

        print(
            f"平衡后验证集分布 - 类别0: {len(torch.where(self.labels == 0)[0])}, 类别1: {len(torch.where(self.labels == 1)[0])}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class BalancedTestDataset(Dataset):
    """平衡测试数据集，对多数类进行欠采样"""

    def __init__(self, embeddings, labels):
        self.embeddings = embeddings.to(device)
        self.labels = labels.to(device)
        self.balance_dataset()

    def balance_dataset(self):
        """对测试集进行平衡采样"""
        # 获取两类样本的索引
        class_0_indices = torch.where(self.labels == 0)[0]
        class_1_indices = torch.where(self.labels == 1)[0]

        print(f"平衡前测试集分布 - 类别0: {len(class_0_indices)}, 类别1: {len(class_1_indices)}")

        # 确定少数类样本数量
        min_class_count = min(len(class_0_indices), len(class_1_indices))

        # 对多数类进行随机欠采样
        if len(class_0_indices) > min_class_count:
            # 类别0是多数类，进行欠采样
            selected_0_indices = class_0_indices[torch.randperm(len(class_0_indices), device=device)[:min_class_count]]
            selected_1_indices = class_1_indices
        else:
            # 类别1是多数类，进行欠采样
            selected_0_indices = class_0_indices
            selected_1_indices = class_1_indices[torch.randperm(len(class_1_indices), device=device)[:min_class_count]]

        # 合并采样后的索引
        balanced_indices = torch.cat([selected_0_indices, selected_1_indices])

        # 打乱顺序
        balanced_indices = balanced_indices[torch.randperm(len(balanced_indices), device=device)]

        # 更新数据集
        self.embeddings = self.embeddings[balanced_indices]
        self.labels = self.labels[balanced_indices]

        print(
            f"平衡后测试集分布 - 类别0: {len(torch.where(self.labels == 0)[0])}, 类别1: {len(torch.where(self.labels == 1)[0])}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ==========================================
# 数据处理类
# ==========================================

class RobustHealthDataProcessor:
    """稳健的健康数据处理器"""

    def __init__(self, feature_dim=64):
        self.scalers = {}
        self.imputers = {}
        self.feature_info = {}
        self.feature_dim = feature_dim
        self.empty_fields = []
        self.field_names = []
        self.all_field_features = {}
        self.field_embeddings = nn.ModuleDict()
        self.selected_field_names = []  # 所有字段名列表（包括全空字段）
        self.field_feature_maps = {}  # 存储每个字段的特征映射

    def analyze_features(self, df):
        """分析特征类型 - 更稳健的版本"""
        categorical_features = []
        numerical_features = []
        date_features = []
        empty_fields = []

        for col in df.columns:
            print(f"分析列: {col}, 类型: {df[col].dtype}, 非空值: {df[col].notna().sum()}")

            # 跳过ID列
            if col in ['用户标识', 'PERSON_ID', 'ROWNO', 'id', 'ID', 'Id']:
                print(f"  跳过标识列: {col}")
                continue

            # 检查是否全为空
            if df[col].notna().sum() == 0:
                empty_fields.append(col)
                print(f"  列为空: {col}")
                continue

            # 检查是否为日期列（放宽条件）
            date_keywords = ['日期', '时间', 'Date', 'DATE', 'Time', 'TIME', '出生', 'Birth']
            if any(keyword in col for keyword in date_keywords):
                date_features.append(col)
                print(f"  识别为日期列: {col}")
                continue

            # 检查数值列
            if df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                numerical_features.append(col)
                print(f"  识别为数值列: {col}")
                continue

            # 尝试转换object类型为数值
            if df[col].dtype == 'object':
                try:
                    # 尝试转换为数值
                    pd.to_numeric(df[col], errors='raise')
                    numerical_features.append(col)
                    print(f"  object列转换为数值列: {col}")
                    continue
                except:
                    pass

            # 其余视为分类特征
            categorical_features.append(col)
            print(f"  识别为分类列: {col}")

        self.empty_fields = empty_fields

        print(f"\n特征分析完成:")
        print(f"  分类特征: {categorical_features}")
        print(f"  数值特征: {numerical_features}")
        print(f"  日期特征: {date_features}")
        print(f"  空字段: {empty_fields}")

        return {
            'categorical': categorical_features,
            'numerical': numerical_features,
            'date': date_features
        }

    def preprocess_numerical_column(self, df, col):
        """预处理数值列，处理混合数据类型"""
        # 复制列数据
        col_data = df[col].copy()

        # 如果列是object类型，尝试转换为数值
        if col_data.dtype == 'object':
            # 尝试转换为数值，无法转换的设为NaN
            col_data = pd.to_numeric(col_data, errors='coerce')

        return col_data

    def create_numerical_features(self, df, col):
        """为数值字段创建特征 - 处理全空字段"""
        features = {}

        # 预处理数值列
        processed_col = self.preprocess_numerical_column(df, col)

        # 检查是否有有效数据
        has_data = processed_col.notna().sum() > 0

        # 创建缺失指示器
        missing_mask = processed_col.isna()
        features[f'{col}_missing'] = missing_mask.astype(float).values

        # 用零填充缺失值
        filled = processed_col.fillna(0).values.reshape(-1, 1)

        # 如果字段全空或没有有效数据，使用零特征
        if not has_data:
            features[f'{col}_value'] = filled.ravel()
            features[f'{col}_scaled'] = np.zeros_like(filled.ravel())
            features[f'{col}_tokenized'] = np.ones_like(filled.ravel(), dtype=int)
            return features

        # 有有效数据，正常处理
        if col not in self.scalers:
            self.scalers[col] = StandardScaler()

        # 检查是否有有效数据
        if filled.shape[1] > 0 and not np.all(np.isnan(filled)):
            try:
                # 仅对非缺失值进行标准化器拟合
                non_missing_values = processed_col[~missing_mask].values.reshape(-1, 1)
                if len(non_missing_values) > 0:
                    # 使用非缺失值拟合标准化器
                    self.scalers[col].fit(non_missing_values)

                    # 对所有值进行转换
                    scaled = self.scalers[col].transform(filled)

                    # 限制数值范围，防止极端值
                    scaled = np.clip(scaled, -5, 5)

                    scaled_clean = np.nan_to_num(scaled.ravel(), nan=0.0)

                    # 改进的离散化，避免边界问题
                    bins = np.linspace(-3, 3, 10)
                    tokenized = np.digitize(scaled_clean, bins)
                    tokenized = np.clip(tokenized, 1, len(bins) - 1)  # 确保在有效范围内

                    features[f'{col}_value'] = filled.ravel()
                    features[f'{col}_scaled'] = scaled.ravel()
                    features[f'{col}_tokenized'] = tokenized
            except Exception as e:
                print(f"处理数值列 {col} 时出错: {e}")
                # 如果标准化失败，使用原始值
                features[f'{col}_value'] = filled.ravel()
                features[f'{col}_scaled'] = np.zeros_like(filled.ravel())
                features[f'{col}_tokenized'] = np.ones_like(filled.ravel(), dtype=int)

        return features

    def create_categorical_features(self, df, col):
        """为分类字段创建特征 - 处理全空字段"""
        features = {}

        # 检查是否有有效数据
        has_data = df[col].notna().sum() > 0

        # 创建缺失指示器
        missing_indicator = df[col].isna().astype(float)
        features[f'{col}_missing'] = missing_indicator.values

        # 如果字段全空或没有有效数据，使用默认特征
        if not has_data:
            # 创建简单的数值编码（全0）
            features[f'{col}_encoded'] = np.zeros(len(df))

            # 创建二元特征（全0）
            max_categories = 10
            for i in range(max_categories):
                features[f'{col}_is_{i}'] = np.zeros(len(df))

            return features

        # 有有效数据，正常处理
        # 处理分类数据 - 使用明确的编码策略
        filled_data = df[col].fillna('MISSING').astype(str)

        # 创建简单的数值编码
        unique_vals = filled_data.unique()
        encoding_dict = {val: i for i, val in enumerate(unique_vals)}

        # 应用编码
        encoded = filled_data.map(encoding_dict).values

        # 归一化到[0,1]范围
        if len(unique_vals) > 1:
            encoded_normalized = encoded / max(1, np.max(encoded))
        else:
            encoded_normalized = encoded.astype(float)

        features[f'{col}_encoded'] = encoded_normalized

        # 为所有唯一值创建二元特征（限制数量避免维度爆炸）
        max_categories = 10  # 限制最大类别数
        value_counts = filled_data.value_counts()
        top_values = value_counts.head(max_categories).index

        for i, val in enumerate(top_values):
            features[f'{col}_is_{i}'] = (filled_data == val).astype(float).values

        return features

    def preprocess_dates(self, df, date_features):
        """处理日期特征"""
        date_features_dict = {}

        for date_col in date_features:
            try:
                # 转换为datetime
                dates = pd.to_datetime(df[date_col], errors='coerce')

                # 检查是否有有效数据
                has_data = dates.notna().sum() > 0

                if not has_data:
                    # 字段全空，使用默认值
                    date_features_dict[f'{date_col}_year'] = np.zeros(len(df))
                    date_features_dict[f'{date_col}_month'] = np.zeros(len(df))
                    date_features_dict[f'{date_col}_day'] = np.zeros(len(df))
                    date_features_dict[f'{date_col}_dayofweek'] = np.zeros(len(df))
                else:
                    # 提取多个时间特征
                    date_features_dict[f'{date_col}_year'] = dates.dt.year.fillna(dates.dt.year.median()).values
                    date_features_dict[f'{date_col}_month'] = dates.dt.month.fillna(dates.dt.month.median()).values
                    date_features_dict[f'{date_col}_day'] = dates.dt.day.fillna(dates.dt.day.median()).values
                    date_features_dict[f'{date_col}_dayofweek'] = dates.dt.dayofweek.fillna(
                        dates.dt.dayofweek.median()).values

                    # 计算年龄（如果列名包含出生日期）
                    if '出生' in date_col:
                        # 使用体检时间作为参考日期
                        if '体检时间' in df.columns:
                            exam_date = pd.to_datetime(df['体检时间'], errors='coerce')
                            age_days = (exam_date - dates).dt.days
                            age_years = age_days / 365.0
                            date_features_dict['age'] = age_years.fillna(age_years.median()).values
                        else:
                            reference_date = pd.to_datetime('today')
                            age_days = (reference_date - dates).dt.days
                            age_years = age_days / 365.0
                            date_features_dict['age'] = age_years.fillna(age_years.median()).values

            except Exception as e:
                print(f"处理日期列 {date_col} 时出错: {e}")
                # 出错时使用默认值
                date_features_dict[f'{date_col}_year'] = np.zeros(len(df))
                date_features_dict[f'{date_col}_month'] = np.zeros(len(df))
                date_features_dict[f'{date_col}_day'] = np.zeros(len(df))
                date_features_dict[f'{date_col}_dayofweek'] = np.zeros(len(df))

        return date_features_dict

    def fit(self, df):
        """在训练数据上拟合处理器，保留所有字段（包括全空字段）"""
        print("分析训练数据特征类型...")
        self.feature_info = self.analyze_features(df)

        # 收集所有字段特征（包括全空字段）
        all_field_features = {}

        # 首先，处理所有在 feature_info 中的字段
        print("处理数值特征...")
        for col in self.feature_info['numerical']:
            features = self.create_numerical_features(df, col)
            all_field_features[col] = features
            if col in self.empty_fields:
                print(f"  数值字段 '{col}': 全空字段，创建 {len(features)} 个默认特征")
            else:
                print(f"  数值字段 '{col}': {len(features)} 个特征")

        print("处理分类特征...")
        for col in self.feature_info['categorical']:
            features = self.create_categorical_features(df, col)
            all_field_features[col] = features
            if col in self.empty_fields:
                print(f"  分类字段 '{col}': 全空字段，创建 {len(features)} 个默认特征")
            else:
                print(f"  分类字段 '{col}': {len(features)} 个特征")

        print("处理日期特征...")
        date_features = self.preprocess_dates(df, self.feature_info['date'])
        for col in self.feature_info['date']:
            # 提取与该日期相关的所有特征
            date_col_features = {}
            for feat_name, feat_values in date_features.items():
                if col in feat_name:
                    date_col_features[feat_name] = feat_values

            all_field_features[col] = date_col_features
            if col in self.empty_fields:
                print(f"  日期字段 '{col}': 全空字段，创建 {len(date_col_features)} 个默认特征")
            else:
                print(f"  日期字段 '{col}': {len(date_col_features)} 个特征")

        # 关键修复：确保所有字段都被处理，包括那些没有被分析出的字段
        # 获取所有列名
        all_columns = set(df.columns.tolist())

        # 移除ID列
        id_columns = ['用户标识', 'PERSON_ID', 'ROWNO', 'id', 'ID', 'Id']
        for id_col in id_columns:
            if id_col in all_columns:
                all_columns.remove(id_col)

        # 已经处理过的字段
        processed_columns = set(all_field_features.keys())

        # 需要处理但尚未处理的字段（包括全空字段）
        remaining_columns = all_columns - processed_columns

        if remaining_columns:
            print(f"\n处理剩余字段（包括全空字段）...")
            for col in remaining_columns:
                # 检查是否全空
                is_empty = df[col].notna().sum() == 0

                # 根据字段类型创建特征
                if is_empty:
                    # 全空字段，创建默认数值特征
                    features = self.create_numerical_features(df, col)
                    all_field_features[col] = features
                    print(f"  字段 '{col}': 全空字段，创建 {len(features)} 个默认特征")
                else:
                    # 有数据的字段，尝试推断类型
                    if df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                        # 数值类型
                        features = self.create_numerical_features(df, col)
                        all_field_features[col] = features
                        print(f"  数值字段 '{col}': {len(features)} 个特征")
                    else:
                        try:
                            # 尝试转换为数值
                            pd.to_numeric(df[col], errors='raise')
                            features = self.create_numerical_features(df, col)
                            all_field_features[col] = features
                            print(f"  数值字段 '{col}': {len(features)} 个特征")
                        except:
                            # 作为分类字段处理
                            features = self.create_categorical_features(df, col)
                            all_field_features[col] = features
                            print(f"  分类字段 '{col}': {len(features)} 个特征")

        # 记录选择的字段和特征映射（包括全空字段）
        self.selected_field_names = list(all_field_features.keys())
        self.field_names = list(all_field_features.keys())
        self.all_field_features = all_field_features

        # 为每个字段存储特征映射信息
        for field_name, features in all_field_features.items():
            self.field_feature_maps[field_name] = {
                'feature_names': list(features.keys()),
                'feature_count': len(features),
                'is_empty': field_name in self.empty_fields or (
                    df[field_name].notna().sum() == 0 if field_name in df.columns else True)
            }

        # 为每个字段创建特征映射器（包括全空字段）
        print("\n创建字段特征映射器...")
        self.field_embeddings = nn.ModuleDict()

        for field_name in self.selected_field_names:
            features = all_field_features[field_name]
            feature_count = len(features)
            if feature_count > 0:
                self.field_embeddings[field_name] = nn.Linear(feature_count, self.feature_dim).to(device)
                is_empty = field_name in self.empty_fields or (
                            field_name in df.columns and df[field_name].notna().sum() == 0)
                if is_empty:
                    print(f"字段 '{field_name}': 全空字段映射器: {feature_count} -> {self.feature_dim} 维")
                else:
                    print(f"字段 '{field_name}' 映射器: {feature_count} -> {self.feature_dim} 维")
            else:
                print(f"警告: 字段 '{field_name}' 没有特征，跳过映射器创建")

        print(f"\n字段统计:")
        print(f"  总字段数: {len(self.selected_field_names)}")
        empty_count = sum(1 for field_name in self.selected_field_names
                          if self.field_feature_maps[field_name]['is_empty'])
        print(f"  全空字段数: {empty_count}")
        print(f"  有效字段数: {len(self.selected_field_names) - empty_count}")

        return self

    def transform(self, df):
        """使用已拟合的处理器转换数据，使用与训练集相同的字段（包括全空字段）"""
        print(f"转换数据，使用与训练集相同的 {len(self.selected_field_names)} 个字段")

        if not self.selected_field_names:
            raise ValueError("处理器尚未拟合训练数据，请先调用fit方法")

        batch_size = len(df)
        num_fields = len(self.selected_field_names)

        # 初始化输出张量
        field_embeddings = torch.zeros(batch_size, num_fields, self.feature_dim)

        # 为每个字段计算特征并生成嵌入（包括全空字段）
        for i, field_name in enumerate(self.selected_field_names):
            is_empty_field = self.field_feature_maps[field_name]['is_empty']
            print(f"  处理字段 {i + 1}/{num_fields}: '{field_name}' {'(全空字段)' if is_empty_field else ''}", end='\r')

            # 检查数据集中是否有该字段
            if field_name not in df.columns:
                print(f"\n警告: 字段 '{field_name}' 在数据中不存在，使用零特征")
                # 创建零特征矩阵
                feature_info = self.field_feature_maps[field_name]
                feature_count = feature_info['feature_count']
                feature_matrix = np.zeros((batch_size, feature_count))
            else:
                # 根据字段类型创建特征
                if field_name in self.feature_info.get('numerical', []):
                    features = self.create_numerical_features(df, field_name)
                elif field_name in self.feature_info.get('categorical', []):
                    features = self.create_categorical_features(df, field_name)
                elif field_name in self.feature_info.get('date', []):
                    # 对于日期字段，需要重新处理
                    date_features = self.preprocess_dates(df, [field_name])
                    features = {}
                    for feat_name, feat_values in date_features.items():
                        if field_name in feat_name:
                            features[feat_name] = feat_values
                else:
                    # 未知字段类型，尝试创建数值特征
                    features = self.create_numerical_features(df, field_name)

                # 构建特征矩阵
                feature_matrix = []
                expected_features = self.field_feature_maps[field_name]['feature_names']

                for feat_name in expected_features:
                    if feat_name in features:
                        feat_values = features[feat_name]
                        if isinstance(feat_values, pd.Series):
                            feat_values = feat_values.values
                        feat_values = np.nan_to_num(feat_values, nan=0.0)
                        feature_matrix.append(feat_values)
                    else:
                        # 特征不存在，填充零
                        feature_matrix.append(np.zeros(batch_size))

                feature_matrix = np.column_stack(feature_matrix) if feature_matrix else np.zeros((batch_size, 1))

            # 分批处理特征矩阵以减少内存使用
            batch_size_processing = min(1000, batch_size)

            for start_idx in range(0, batch_size, batch_size_processing):
                end_idx = min(start_idx + batch_size_processing, batch_size)
                batch_feature_matrix = feature_matrix[start_idx:end_idx]

                # 将特征矩阵移动到GPU
                feature_tensor = torch.FloatTensor(batch_feature_matrix).to(device)

                # 通过线性层映射到特征维度
                with torch.no_grad():
                    field_embedding = self.field_embeddings[field_name](feature_tensor)
                    field_embeddings[start_idx:end_idx, i, :] = field_embedding.cpu()

                # 清理GPU内存
                del feature_tensor, field_embedding
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print()  # 换行
        # 最后将整个张量移动到GPU
        field_embeddings = field_embeddings.to(device)

        # 应用tanh激活函数并缩放
        field_embeddings = torch.tanh(field_embeddings) * 3

        return field_embeddings

    def fit_transform(self, df):
        """拟合并转换数据"""
        self.fit(df)
        return self.transform(df)


# ==========================================
# 数据加载函数
# ==========================================

def load_data_by_year(healthy_files, stroke_files, feature_dim=64):
    """
    按年份加载数据并处理

    Args:
        healthy_files: 健康数据文件路径字典，格式为 {年份: 文件路径}
        stroke_files: 中风数据文件路径字典，格式为 {年份: 文件路径}
        feature_dim: 特征维度

    Returns:
        包含各年份数据的字典
    """
    print("初始化数据处理器...")
    processor = RobustHealthDataProcessor(feature_dim=feature_dim)

    # 首先处理训练数据以拟合处理器
    print("处理训练数据以拟合处理器...")

    # 加载所有训练数据（00-16年健康数据，07-16年中风数据）
    train_healthy_dfs = []
    train_stroke_dfs = []

    # 加载健康数据（2000-2016年）
    for year in range(2000, 2017):
        if year in healthy_files:
            try:
                healthy_df = pd.read_csv(healthy_files[year], encoding='gbk')
                train_healthy_dfs.append(healthy_df)
                print(f"  加载健康数据 {year}: {healthy_df.shape}")
            except FileNotFoundError:
                print(f"  警告: 健康数据文件 {healthy_files[year]} 不存在，跳过")

    # 加载中风数据（2007-2016年）
    for year in range(2007, 2017):
        if year in stroke_files:
            try:
                stroke_df = pd.read_csv(stroke_files[year], encoding='gbk')
                train_stroke_dfs.append(stroke_df)
                print(f"  加载中风数据 {year}: {stroke_df.shape}")
            except FileNotFoundError:
                print(f"  警告: 中风数据文件 {stroke_files[year]} 不存在，跳过")

    # 合并训练数据
    if train_healthy_dfs:
        train_healthy_df = pd.concat(train_healthy_dfs, ignore_index=True)
    else:
        train_healthy_df = pd.DataFrame()

    if train_stroke_dfs:
        train_stroke_df = pd.concat(train_stroke_dfs, ignore_index=True)
    else:
        train_stroke_df = pd.DataFrame()

    print(f"训练健康数据形状: {train_healthy_df.shape}")
    print(f"训练中风数据形状: {train_stroke_df.shape}")

    # 检查是否有训练数据
    if train_healthy_df.empty and train_stroke_df.empty:
        raise ValueError("没有训练数据可用")

    # 合并所有训练数据以拟合处理器
    if not train_healthy_df.empty and not train_stroke_df.empty:
        combined_train_df = pd.concat([train_stroke_df, train_healthy_df], ignore_index=True)
    elif not train_stroke_df.empty:
        combined_train_df = train_stroke_df
    elif not train_healthy_df.empty:
        combined_train_df = train_healthy_df
    else:
        raise ValueError("没有训练数据可用")

    # 使用处理器拟合训练数据
    print("使用处理器拟合训练数据...")
    processor.fit(combined_train_df)

    # 转换训练数据
    print("转换训练数据...")
    train_embeddings = processor.transform(combined_train_df)

    # 分离训练数据的嵌入
    train_stroke_embeddings = train_embeddings[:len(train_stroke_df)]
    train_healthy_embeddings = train_embeddings[len(train_stroke_df):]

    print(f"训练中风嵌入: {train_stroke_embeddings.shape}")
    print(f"训练健康嵌入: {train_healthy_embeddings.shape}")
    print(f"训练集字段数: {train_stroke_embeddings.shape[1]} (包含全空字段)")

    # 处理验证数据（2017年）
    val_data = {}
    if 2017 in healthy_files and 2017 in stroke_files:
        print("\n处理验证数据（2017年）...")
        try:
            val_healthy_df = pd.read_csv(healthy_files[2017], encoding='gbk')
            val_stroke_df = pd.read_csv(stroke_files[2017], encoding='gbk')

            print(f"验证健康数据形状: {val_healthy_df.shape}")
            print(f"验证中风数据形状: {val_stroke_df.shape}")

            # 平衡采样：从健康数据中随机选择与中风数据相同数量的样本
            if len(val_healthy_df) > len(val_stroke_df):
                val_healthy_df = val_healthy_df.sample(n=len(val_stroke_df), random_state=42)
                print(f"平衡后验证健康数据形状: {val_healthy_df.shape}")

            # 合并验证数据
            combined_val_df = pd.concat([val_stroke_df, val_healthy_df], ignore_index=True)

            # 使用已拟合的处理器转换验证数据
            val_embeddings = processor.transform(combined_val_df)

            # 创建验证标签
            val_labels = torch.cat([
                torch.ones(len(val_stroke_df), dtype=torch.long),
                torch.zeros(len(val_healthy_df), dtype=torch.long)
            ]).to(device)

            val_data = {
                'embeddings': val_embeddings,
                'labels': val_labels
            }
            print(f"验证嵌入: {val_embeddings.shape}")
            print(f"验证字段数: {val_embeddings.shape[1]} (与训练集一致)")
        except FileNotFoundError as e:
            print(f"  警告: 验证数据文件不存在: {e}")

    # 处理测试数据（2018年）
    test_data = {}
    if 2018 in healthy_files and 2018 in stroke_files:
        print("\n处理测试数据（2018年）...")
        try:
            test_healthy_df = pd.read_csv(healthy_files[2018], encoding='gbk')
            test_stroke_df = pd.read_csv(stroke_files[2018], encoding='gbk')

            print(f"测试健康数据形状: {test_healthy_df.shape}")
            print(f"测试中风数据形状: {test_stroke_df.shape}")

            # 平衡采样：从健康数据中随机选择与中风数据相同数量的样本
            if len(test_healthy_df) > len(test_stroke_df):
                test_healthy_df = test_healthy_df.sample(n=len(test_stroke_df), random_state=42)
                print(f"平衡后测试健康数据形状: {test_healthy_df.shape}")

            # 合并测试数据
            combined_test_df = pd.concat([test_stroke_df, test_healthy_df], ignore_index=True)

            # 使用已拟合的处理器转换测试数据
            test_embeddings = processor.transform(combined_test_df)

            # 创建测试标签
            test_labels = torch.cat([
                torch.ones(len(test_stroke_df), dtype=torch.long),
                torch.zeros(len(test_healthy_df), dtype=torch.long)
            ]).to(device)

            test_data = {
                'embeddings': test_embeddings,
                'labels': test_labels
            }
            print(f"测试嵌入: {test_embeddings.shape}")
            print(f"测试字段数: {test_embeddings.shape[1]} (与训练集一致)")
        except FileNotFoundError as e:
            print(f"  警告: 测试数据文件不存在: {e}")

    # 验证字段一致性
    print("\n=== 字段一致性验证 ===")
    train_field_count = train_embeddings.shape[1] if train_embeddings is not None else 0
    val_field_count = val_data.get('embeddings', torch.Tensor([])).shape[1] if val_data else 0
    test_field_count = test_data.get('embeddings', torch.Tensor([])).shape[1] if test_data else 0

    if train_field_count > 0 and val_field_count > 0:
        print(f"训练集字段数: {train_field_count}")
        print(f"验证集字段数: {val_field_count}")
        if train_field_count == val_field_count:
            print("✓ 训练集和验证集字段数一致")
        else:
            print(f"✗ 训练集和验证集字段数不一致: {train_field_count} vs {val_field_count}")

    if train_field_count > 0 and test_field_count > 0:
        print(f"训练集字段数: {train_field_count}")
        print(f"测试集字段数: {test_field_count}")
        if train_field_count == test_field_count:
            print("✓ 训练集和测试集字段数一致")
        else:
            print(f"✗ 训练集和测试集字段数不一致: {train_field_count} vs {test_field_count}")

    return {
        'train': {
            'stroke_embeddings': train_stroke_embeddings,
            'healthy_embeddings': train_healthy_embeddings
        },
        'val': val_data,
        'test': test_data,
        'processor': processor
    }


def create_data_loaders_with_easyensemble(train_stroke_embeddings, train_healthy_embeddings,
                                          val_embeddings, val_labels, test_embeddings, test_labels,
                                          batch_size=32, K=10):
    """
    创建使用EasyEnsemble的训练数据加载器和平衡的验证/测试数据加载器

    Args:
        train_stroke_embeddings: 训练阳性样本嵌入
        train_healthy_embeddings: 训练阴性样本嵌入
        val_embeddings: 验证集嵌入
        val_labels: 验证集标签
        test_embeddings: 测试集嵌入
        test_labels: 测试集标签
        batch_size: 批次大小
        K: EasyEnsemble的阴性样本子集数量

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        train_dataset: EasyEnsemble训练数据集
    """
    print("\n" + "=" * 80)
    print("创建数据加载器")
    print("=" * 80)

    # 验证字段维度一致性
    print("验证字段维度一致性...")
    train_field_dim = train_stroke_embeddings.shape[1]
    val_field_dim = val_embeddings.shape[1]
    test_field_dim = test_embeddings.shape[1]

    if train_field_dim == val_field_dim == test_field_dim:
        print(f"✓ 所有数据集字段维度一致: {train_field_dim}")
    else:
        print(f"✗ 字段维度不一致: 训练集={train_field_dim}, 验证集={val_field_dim}, 测试集={test_field_dim}")
        raise ValueError("数据集字段维度不一致")

    # 创建EasyEnsemble训练数据集
    print("创建EasyEnsemble训练数据集...")
    train_dataset = EasyEnsembleTrainDataset(
        positive_embeddings=train_stroke_embeddings,
        negative_embeddings=train_healthy_embeddings,
        K=K
    )

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 在每个epoch内打乱
        pin_memory=False
    )

    # 创建验证数据集（平衡）
    print("\n创建验证数据集...")
    val_dataset = BalancedValidationDataset(val_embeddings, val_labels)

    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False
    )

    # 创建测试数据集（平衡）
    print("\n创建测试数据集...")
    test_dataset = BalancedTestDataset(test_embeddings, test_labels)

    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False
    )

    print(f"\n数据加载器创建完成:")
    print(f"  训练集: {len(train_dataset)} 个样本 (EasyEnsemble, {train_dataset.K}个子集)")
    print(f"  验证集: {len(val_dataset)} 个样本 (平衡)")
    print(f"  测试集: {len(test_dataset)} 个样本 (平衡)")
    print(f"  字段维度: {train_field_dim} (所有数据集一致，包含全空字段)")

    return train_loader, val_loader, test_loader, train_dataset


def load_and_process_data_by_year(healthy_files, stroke_files, feature_dim=64, batch_size=32, K=10):
    """
    按年份加载数据并创建数据加载器

    Args:
        healthy_files: 健康数据文件路径字典
        stroke_files: 中风数据文件路径字典
        feature_dim: 特征维度
        batch_size: 批次大小
        K: EasyEnsemble的阴性样本子集数量

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        train_dataset: EasyEnsemble训练数据集
        processor: 数据处理器
    """
    # 加载按年份划分的数据
    data_dict = load_data_by_year(healthy_files, stroke_files, feature_dim)

    # 检查验证和测试数据是否存在
    if not data_dict['val'] or not data_dict['test']:
        raise ValueError("验证数据或测试数据加载失败")

    # 创建数据加载器
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders_with_easyensemble(
        train_stroke_embeddings=data_dict['train']['stroke_embeddings'],
        train_healthy_embeddings=data_dict['train']['healthy_embeddings'],
        val_embeddings=data_dict['val']['embeddings'],
        val_labels=data_dict['val']['labels'],
        test_embeddings=data_dict['test']['embeddings'],
        test_labels=data_dict['test']['labels'],
        batch_size=batch_size,
        K=K
    )

    return train_loader, val_loader, test_loader, train_dataset, data_dict['processor']
