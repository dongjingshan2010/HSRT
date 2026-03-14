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
import torch.nn.functional as F
from GradientAnalyzer import SelfAttention


# ==========================================
# 模型定义部分
# ==========================================
class CustomMultiheadAttention(nn.Module):
    """自定义多头注意力层，用于捕获注意力权重"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=batch_first
        )

    def forward(self, query, key, value, attn_mask, key_padding_mask):
        output, attn_weights = self.self_attn(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        return output, attn_weights


class CustomTransformerEncoderLayer(nn.Module):
    """自定义Transformer编码层，返回注意力权重"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=True):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, src, src_mask, src_key_padding_mask=None):
        # 自注意力机制
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src2 = self.dropout1(src2)
        src = src + src2
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)
        src = src + src2
        src = self.norm2(src)

        return src, attn_weights


import copy
class CustomTransformerEncoder(nn.Module):
    """自定义Transformer编码器，用于捕获注意力权重并支持可视化"""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # ⚠️ 必须使用 copy.deepcopy 复制层，否则所有层将共享同一组物理参数
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.attention_weights = []  # 存储各层注意力权重

    def forward(self, src, mask=None, src_key_padding_mask=None, visualize=False):
        """
        Args:
            src: 输入序列张量
            mask: 注意力掩码
            src_key_padding_mask: 填充掩码 (True表示被Mask)
            visualize (bool): 是否在每次前向传播时绘制并显示注意力热图
        """
        output = src
        self.attention_weights = []  # 重置注意力权重

        for i, layer in enumerate(self.layers):
            # 依次经过每一层
            output, attn_weights = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            self.attention_weights.append(attn_weights.detach())

            # ===== 实时显示注意力热图逻辑 =====
            if visualize:
                # attn_weights 的形状为: [batch_size, num_heads, seq_len, seq_len]
                # 取 batch 中第 0 个样本，在 num_heads 维度（维度0）求平均
                # 结果形状变为: [seq_len, seq_len]
                attn_matrix = attn_weights.detach()[0].mean(dim=0).cpu().numpy()

                plt.figure(figsize=(8, 6))
                # 使用 vmin=0, vmax=1 确保颜色映射真实反映概率分布
                im = plt.imshow(attn_matrix, cmap='viridis', aspect='auto')

                # 设置标题和轴标签
                plt.title(f'Layer {i + 1} Attention Heatmap (Sample 0, Avg over Heads)', fontsize=14)
                plt.xlabel('Key Fields (被关注的特征)', fontsize=12)
                plt.ylabel('Query Fields (当前特征)', fontsize=12)

                # 添加颜色条
                plt.colorbar(im, label='Attention Probability')
                plt.tight_layout()
                plt.show()

        if self.norm is not None:
            output = self.norm(output)

        return output


class HealthDataTransformeruniLSTM(nn.Module):
    def __init__(self, num_fields, feature_dim=64, hidden_dim=128, num_layers=4, num_heads=8, num_classes=2,
                 lstm_layers=2, tau=0.01):
        super().__init__()
        self.num_fields = num_fields
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.tau = tau  # Gumbel Softmax温度参数

        # 可学习的CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # 改进的位置编码初始化 - 现在需要num_fields+1个位置（为CLS token预留位置）
        self.positional_encoding = nn.Parameter(torch.randn(1, num_fields + 1, hidden_dim) * 0.02)

        # 移除了SelfAttention层
        self.pre_attention = SelfAttention(hidden_dim, num_heads=num_heads)
        # 添加LayerNorm稳定训练
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # LSTM层
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if lstm_layers > 1 else 0
        )

        # 顺序无关编码的注意力参数
        self.W_q = nn.Linear(hidden_dim, hidden_dim)  # 查询变换
        self.W_k = nn.Linear(hidden_dim, hidden_dim)  # 键变换
        self.W_v = nn.Linear(hidden_dim, hidden_dim)  # 值变换
        self.scale_factor = np.sqrt(hidden_dim)  # 缩放因子

        # 可训练的掩膜向量 - 现在需要num_fields+1个位置（包括CLS token）
        self.mask_logits = nn.Parameter(torch.randn(num_fields + 1))

        # 初始化掩膜logits
        nn.init.constant_(self.mask_logits, 0.0)  # 初始化为0，让模型自己学习

        # 使用自定义的Transformer编码器
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)

        self.field_projection = nn.Linear(feature_dim, hidden_dim)

        # 改进分类头初始化
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # 添加LayerNorm
            nn.GELU(),  # 使用GELU
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        # 初始化CLS token
        nn.init.normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        # 初始化LSTM权重
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    # def _gumbel_sigmoid(self, logits, hard=True, threshold=0.5):
    #     """
    #     Gumbel-Sigmoid采样
    #     logits: [seq_len] - 每个位置独立决策的对数几率
    #     返回: [seq_len] 二值掩膜
    #     """
    #     if not self.training:
    #         probs = torch.sigmoid(logits / self.tau)
    #         return (probs > threshold).float()
    #     # 更简洁的实现：使用伯努利分布的Gumbel-Softmax重参数化
    #     # 采样Gumbel噪声
    #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    #
    #     # 计算选择概率：sigmoid(logits + Gumbel噪声 - Gumbel噪声')
    #     # 通过重参数化，我们可以直接计算概率
    #     # 这里我们使用等效形式：sigmoid(logits + ε)/[sigmoid(logits + ε) + sigmoid(ε')]
    #     # 其中ε和ε'是独立同分布的Gumbel噪声
    #
    #     # 方法1: 直接计算（数值稳定）
    #     eps = 1e-8
    #     logits_with_noise = logits + gumbel_noise
    #
    #     # 计算正类的概率（基于Gumbel-Max技巧）
    #     # y_soft = exp(logits_with_noise/tau) / [exp(logits_with_noise/tau) + exp(gumbel_noise2/tau)]
    #     # 为了数值稳定，我们使用log-space计算
    #
    #     # 采样第二个独立的Gumbel噪声
    #     gumbel_noise2 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    #
    #     # 在温度τ下的softmax
    #     positive = (logits + gumbel_noise) / self.tau
    #     negative = gumbel_noise2 / self.tau  # 假设负类logit为0
    #
    #     # 使用logsumexp避免数值溢出
    #     max_val = torch.max(positive, negative)
    #     positive_exp = torch.exp(positive - max_val)
    #     negative_exp = torch.exp(negative - max_val)
    #     y_soft = positive_exp / (positive_exp + negative_exp + eps)
    #
    #     if hard:
    #         # 二值化决策
    #         y_hard = (y_soft > threshold).float()
    #         # Straight-Through Estimator
    #         return y_hard - y_soft.detach() + y_soft
    #     else:
    #         return y_soft
    def _gumbel_sigmoid(self, logits, hard=True, threshold=0.5):
        """
        优化的 Gumbel-Sigmoid 采样
        """
        if not self.training:
            # 推理阶段直接根据 logits 给出确定的二值结果
            # 注意：这里 / self.tau 对 hard 阈值判断其实没有影响，但保留也无妨
            probs = torch.sigmoid(logits / self.tau)
            return (probs > threshold).float()

        # 1. 采样 Uniform(0, 1) 噪声
        u = torch.rand_like(logits)
        eps = 1e-8

        # 2. 计算 Logistic 噪声 (即 gumbel_1 - gumbel_2)
        logistic_noise = torch.log(u + eps) - torch.log(1 - u + eps)

        # 3. 计算 y_soft (利用 PyTorch 原生 sigmoid 保证数值稳定)
        y_soft = torch.sigmoid((logits + logistic_noise) / self.tau)

        if hard:
            # 4. 二值化决策与 STE (Straight-Through Estimator)
            y_hard = (y_soft > threshold).float()
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft
    def _order_agnostic_encoding(self, x):
        """
        顺序无关编码：使用注意力机制从无序特征集合中动态检索信息
        x: [batch_size, seq_len, hidden_dim]
        返回: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 使用Gumbel Sigmoid获取二值掩膜
        mask_values = self._gumbel_sigmoid(self.mask_logits)  # [seq_len]

        # 初始化LSTM隐藏状态
        h_0 = torch.zeros(self.lstm_layers, batch_size, hidden_dim, device=x.device)
        c_0 = torch.zeros(self.lstm_layers, batch_size, hidden_dim, device=x.device)

        # 准备键和值（对特征进行变换）
        K = self.W_k(x)  # [batch_size, seq_len, hidden_dim]
        V = self.W_v(x)  # [batch_size, seq_len, hidden_dim]

        # 存储LSTM的输出
        lstm_outputs = []
        current_h = h_0
        current_c = c_0

        # 固定步数处理（与序列长度相同）
        for t in range(seq_len):
            # 从LSTM隐藏状态生成查询向量 q_t = W_q * H_{t-1}
            # 取最后一层的隐藏状态作为查询
            h_last = current_h[-1]  # [batch_size, hidden_dim]
            q_t = self.W_q(h_last)  # [batch_size, hidden_dim]

            # 计算注意力分数 α_{t,i} = Softmax(q_t^T * k_i / sqrt(d))
            # 扩展维度以便广播计算
            q_t_expanded = q_t.unsqueeze(1)  # [batch_size, 1, hidden_dim]

            # 计算注意力分数
            scores = torch.bmm(q_t_expanded, K.transpose(1, 2)) / self.scale_factor  # [batch_size, 1, seq_len]
            alpha = F.softmax(scores, dim=-1)  # [batch_size, 1, seq_len]

            # 计算上下文向量 c_t = Σ α_{t,i} * v_i
            c_t = torch.bmm(alpha, V)  # [batch_size, 1, hidden_dim]

            # 将上下文向量作为LSTM的输入
            output_t, (h_t, c_t_lstm) = self.lstm(c_t, (current_h, current_c))

            # 应用二值掩膜：在mask=1的位置正常传播，在mask=0的位置重置状态
            mask_t = mask_values[t]

            # 更新隐藏状态：mask=0时保持LSTM状态，mask=1时重置为初始状态
            current_h = (1 - mask_t) * h_t + mask_t * h_0
            current_c = (1 - mask_t) * c_t_lstm + mask_t * c_0
            # current_h =  mask_t * h_t + (1 - mask_t) * h_0
            # current_c =  mask_t * c_t_lstm + (1 - mask_t) * c_0

            # 只有当mask_t为1时才将输出加入队列
            if mask_t >= 0.5:  # 使用0.5作为阈值
                lstm_outputs.append(output_t)
            else:
                # 使用零向量
                zero_output = torch.zeros_like(output_t)
                lstm_outputs.append(zero_output)

        # 拼接所有时间步的输出
        lstm_out = torch.cat(lstm_outputs, dim=1)  # [batch_size, seq_len, hidden_dim]

        sparse_mask = mask_values.unsqueeze(0)  # 先添加batch维度: [1, seq_len]
        sparse_mask = sparse_mask.expand(batch_size, -1)

        return lstm_out, sparse_mask

    def forward(self, field_embeddings):
        batch_size = field_embeddings.size(0)

        # 投影到隐藏维度
        x = self.field_projection(field_embeddings)  # [batch_size, num_fields, hidden_dim]

        # 添加LayerNorm
        # x = self.layer_norm(x)

        # 添加CLS token到序列开头
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, num_fields+1, hidden_dim]

        # 应用顺序无关编码（取代原来的pre_attention和LSTM）
        x = self.pre_attention(x)

        x, mask = self._order_agnostic_encoding(x)

        self.binary_mask_values = mask[0].detach().cpu().numpy()

        bool_padding_mask = (mask < 0.5).bool()

        # 添加位置编码
        x = x + self.positional_encoding[:, :x.size(1), :]

        # 通过Transformer
        x = self.transformer(x, mask=None, src_key_padding_mask=bool_padding_mask)  # [batch_size, num_fields+1, hidden_dim]

        # 使用CLS token进行分类
        cls_output = x[:, 0, :]  # 取第一个位置的CLS token [batch_size, hidden_dim]

        # 分类
        logits = self.classifier(cls_output)

        return logits