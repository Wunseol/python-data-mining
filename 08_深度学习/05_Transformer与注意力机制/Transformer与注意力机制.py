"""
Transformer与注意力机制模块 - Transformer & Attention Mechanism
================================================================
涵盖 Transformer 架构的核心组件：
1. 注意力机制原理（缩放点积注意力、多头注意力）
2. 缩放点积注意力手动实现
3. 多头注意力手动实现
4. 位置编码（Sinusoidal）
5. Transformer 编码器层手动实现
6. 简单序列分类实验
7. 可视化（注意力热力图、位置编码、多头注意力模式）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from utils import setup_chinese_font

setup_chinese_font()


# ============================================================
# 1. 注意力机制原理
# ============================================================
def print_attention_principles():
    """打印注意力机制核心原理"""
    print("=" * 60)
    print("注意力机制原理")
    print("=" * 60)

    print("\n--- 缩放点积注意力 (Scaled Dot-Product Attention) ---")
    print("  公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V")
    print("  - Q (Query): 查询矩阵，形状 (seq_len, d_k)")
    print("  - K (Key):   键矩阵，形状   (seq_len, d_k)")
    print("  - V (Value): 值矩阵，形状   (seq_len, d_v)")
    print("  - d_k:       键向量维度，用于缩放防止梯度消失")
    print("  - 缩放因子 sqrt(d_k) 使得点积结果不会过大")

    print("\n--- 多头注意力 (Multi-Head Attention) ---")
    print("  步骤:")
    print("  1. 将 Q, K, V 分别通过线性变换得到 h 组子空间表示")
    print("  2. 每组独立计算缩放点积注意力 (并行)")
    print("  3. 将 h 个头的输出拼接 (Concat)")
    print("  4. 通过输出线性变换 W_O 得到最终输出")
    print("  公式: MultiHead(Q,K,V) = Concat(head_1,...,head_h) W_O")
    print("        其中 head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)")
    print("  - 优势: 不同头关注不同位置的不同表示子空间信息")


# ============================================================
# 2. 缩放点积注意力手动实现
# ============================================================
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    参数:
        Q: 查询矩阵 (seq_len_q, d_k)
        K: 键矩阵   (seq_len_k, d_k)
        V: 值矩阵   (seq_len_v, d_v), seq_len_k == seq_len_v
        mask: 可选掩码 (seq_len_q, seq_len_k)

    返回:
        output: 注意力输出 (seq_len_q, d_v)
        attention_weights: 注意力权重 (seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        scores = scores + (mask * -1e9)

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = attention_weights @ V
    return output, attention_weights


# ============================================================
# 3. 多头注意力手动实现
# ============================================================
class MultiHeadAttention:
    """多头注意力机制 — 纯 NumPy 实现"""

    def __init__(self, d_model, n_heads):
        """
        d_model: 模型维度
        n_heads: 注意力头数
        """
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        np.random.seed(42)
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def forward(self, X, mask=None):
        """
        前向传播

        参数:
            X: 输入序列 (seq_len, d_model)
            mask: 可选掩码

        返回:
            output: 多头注意力输出 (seq_len, d_model)
            attn_weights: 各头注意力权重列表, 每个形状 (seq_len, seq_len)
        """
        seq_len = X.shape[0]

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        Q = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        attn_weights = []
        head_outputs = []
        for i in range(self.n_heads):
            head_out, weights = scaled_dot_product_attention(Q[i], K[i], V[i], mask)
            head_outputs.append(head_out)
            attn_weights.append(weights)

        concat = np.concatenate(head_outputs, axis=-1)
        output = concat @ self.W_O
        return output, attn_weights


# ============================================================
# 4. 位置编码
# ============================================================
def positional_encoding(seq_len, d_model):
    """
    Sinusoidal 位置编码

    参数:
        seq_len: 序列长度
        d_model: 模型维度

    返回:
        PE: 位置编码矩阵 (seq_len, d_model)
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )

    PE = np.zeros((seq_len, d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)
    return PE


# ============================================================
# 5. Transformer 编码器层手动实现
# ============================================================
class TransformerEncoderLayer:
    """Transformer 编码器层 — 纯 NumPy 实现"""

    def __init__(self, d_model, n_heads, d_ff):
        """
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff:    前馈网络隐藏层维度
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.mha = MultiHeadAttention(d_model, n_heads)

        np.random.seed(42)
        scale_ff1 = np.sqrt(2.0 / d_model)
        scale_ff2 = np.sqrt(2.0 / d_ff)
        self.W_ff1 = np.random.randn(d_model, d_ff) * scale_ff1
        self.b_ff1 = np.zeros((1, d_ff))
        self.W_ff2 = np.random.randn(d_ff, d_model) * scale_ff2
        self.b_ff2 = np.zeros((1, d_model))

        self.eps = 1e-6

    def _layer_norm(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps)

    def _feed_forward(self, x):
        hidden = np.maximum(0, x @ self.W_ff1 + self.b_ff1)
        return hidden @ self.W_ff2 + self.b_ff2

    def forward(self, X, mask=None):
        """
        前向传播

        参数:
            X: 输入 (seq_len, d_model)
            mask: 可选掩码

        返回:
            output: 编码器层输出 (seq_len, d_model)
            attn_weights: 各头注意力权重
        """
        attn_output, attn_weights = self.mha.forward(X, mask)
        X = self._layer_norm(X + attn_output)

        ff_output = self._feed_forward(X)
        X = self._layer_norm(X + ff_output)

        return X, attn_weights


# ============================================================
# 6. 简单序列分类实验
# ============================================================
class SimpleTransformerClassifier:
    """简单 Transformer 编码器分类器 — 纯 NumPy"""

    def __init__(self, seq_len, d_model, n_heads, d_ff, n_classes, n_layers=2, lr=0.001):
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.lr = lr

        np.random.seed(42)
        self.layers = []
        for _ in range(n_layers):
            layer = TransformerEncoderLayer(d_model, n_heads, d_ff)
            self.layers.append(layer)

        scale = np.sqrt(2.0 / d_model)
        self.W_cls = np.random.randn(d_model, n_classes) * scale
        self.b_cls = np.zeros((1, n_classes))

        self.losses = []

    def _softmax(self, z):
        z_max = np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def _cross_entropy_loss(self, probs, y):
        eps = 1e-15
        n = len(y)
        log_probs = -np.log(probs[np.arange(n), y] + eps)
        return np.mean(log_probs)

    def forward(self, X):
        for layer in self.layers:
            X, _ = layer.forward(X)
        cls_token = X[0:1, :]
        logits = cls_token @ self.W_cls + self.b_cls
        probs = self._softmax(logits)
        return probs, X

    def fit(self, X_data, y_data, n_iters=100):
        """
        简化训练：使用数值梯度进行参数更新

        参数:
            X_data: 输入数据列表, 每个元素形状 (seq_len, d_model)
            y_data: 标签数组
            n_iters: 训练迭代次数
        """
        n_samples = len(X_data)
        eps = 1e-5

        for iteration in range(n_iters):
            total_loss = 0
            correct = 0

            for idx in range(n_samples):
                X = X_data[idx].copy()
                y = y_data[idx]

                probs, _ = self.forward(X)
                loss = self._cross_entropy_loss(probs, np.array([y]))
                total_loss += loss
                pred = np.argmax(probs, axis=1)[0]
                if pred == y:
                    correct += 1

                params = [self.W_cls, self.b_cls]
                grads = []

                for param in params:
                    grad = np.zeros_like(param)
                    it = np.nditer(param, flags=['multi_index'])
                    while not it.finished:
                        mi = it.multi_index
                        orig = param[mi]
                        param[mi] = orig + eps
                        probs_p, _ = self.forward(X)
                        loss_p = self._cross_entropy_loss(probs_p, np.array([y]))
                        param[mi] = orig - eps
                        probs_m, _ = self.forward(X)
                        loss_m = self._cross_entropy_loss(probs_m, np.array([y]))
                        param[mi] = orig
                        grad[mi] = (loss_p - loss_m) / (2 * eps)
                        it.iternext()
                    grads.append(grad)

                for param, grad in zip(params, grads):
                    param -= self.lr * grad

            avg_loss = total_loss / n_samples
            acc = correct / n_samples
            self.losses.append(avg_loss)

            if (iteration + 1) % 10 == 0:
                print(f"  迭代 {iteration + 1:3d}/{n_iters} | "
                      f"损失: {avg_loss:.4f} | 准确率: {acc:.2%}")

        return self

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)[0]


class BaselineClassifier:
    """简单基线分类器 — 均值特征 + 线性分类"""

    def __init__(self, d_model, n_classes, lr=0.01, n_iters=200):
        self.d_model = d_model
        self.n_classes = n_classes
        self.lr = lr
        self.n_iters = n_iters

        np.random.seed(42)
        self.W = np.random.randn(d_model, n_classes) * np.sqrt(2.0 / d_model)
        self.b = np.zeros((1, n_classes))
        self.losses = []

    def _softmax(self, z):
        z_max = np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def _cross_entropy_loss(self, probs, y):
        eps = 1e-15
        return -np.log(probs[0, y] + eps)

    def fit(self, X_data, y_data):
        n_samples = len(X_data)
        eps = 1e-5

        for iteration in range(self.n_iters):
            total_loss = 0
            correct = 0

            for idx in range(n_samples):
                X = X_data[idx]
                y = y_data[idx]

                feat = np.mean(X, axis=0, keepdims=True)
                logits = feat @ self.W + self.b
                probs = self._softmax(logits)
                loss = self._cross_entropy_loss(probs, y)
                total_loss += loss
                pred = np.argmax(probs, axis=1)[0]
                if pred == y:
                    correct += 1

                params = [self.W, self.b]
                grads = []
                for param in params:
                    grad = np.zeros_like(param)
                    it = np.nditer(param, flags=['multi_index'])
                    while not it.finished:
                        mi = it.multi_index
                        orig = param[mi]
                        param[mi] = orig + eps
                        feat = np.mean(X, axis=0, keepdims=True)
                        logits = feat @ self.W + self.b
                        probs_p = self._softmax(logits)
                        loss_p = self._cross_entropy_loss(probs_p, y)
                        param[mi] = orig - eps
                        feat = np.mean(X, axis=0, keepdims=True)
                        logits = feat @ self.W + self.b
                        probs_m = self._softmax(logits)
                        loss_m = self._cross_entropy_loss(probs_m, y)
                        param[mi] = orig
                        grad[mi] = (loss_p - loss_m) / (2 * eps)
                        it.iternext()
                    grads.append(grad)

                for param, grad in zip(params, grads):
                    param -= self.lr * grad

            avg_loss = total_loss / n_samples
            acc = correct / n_samples
            self.losses.append(avg_loss)

            if (iteration + 1) % 50 == 0:
                print(f"  [基线] 迭代 {iteration + 1:3d}/{self.n_iters} | "
                      f"损失: {avg_loss:.4f} | 准确率: {acc:.2%}")

        return self

    def predict(self, X):
        feat = np.mean(X, axis=0, keepdims=True)
        logits = feat @ self.W + self.b
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)[0]


def generate_sequence_data(n_samples=80, seq_len=10, d_model=16, n_classes=3, random_state=42):
    """
    生成合成序列分类数据

    规则: 根据序列前半段与后半段的均值差异分类
    - 类别 0: 前半段均值 > 后半段均值 + margin
    - 类别 1: 后半段均值 > 前半段均值 + margin
    - 类别 2: 两者接近
    """
    np.random.seed(random_state)
    half = seq_len // 2
    margin = 1.0
    X_list = []
    y_list = []

    while len(X_list) < n_samples:
        seq = np.random.randn(seq_len, d_model) * 0.5
        diff = np.mean(seq[:half]) - np.mean(seq[half:])
        if diff > margin:
            y_list.append(0)
        elif diff < -margin:
            y_list.append(1)
        else:
            y_list.append(2)
        X_list.append(seq)

    return X_list, np.array(y_list)


# ============================================================
# 7. 可视化
# ============================================================
def plot_attention_heatmap(attn_weights, title='注意力权重热力图', tokens=None):
    """绘制单头注意力权重热力图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_weights, cmap='YlOrRd', aspect='auto')

    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

    ax.set_xlabel('Key 位置')
    ax.set_ylabel('Query 位置')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='注意力权重')
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_positional_encoding(pe, title='位置编码可视化'):
    """绘制位置编码矩阵色图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pe.T, cmap='RdBu', aspect='auto', interpolation='nearest')
    ax.set_xlabel('序列位置')
    ax.set_ylabel('维度索引')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='编码值')
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_multihead_attention(attn_weights_list, title='多头注意力模式'):
    """绘制多头注意力权重（多个热力图）"""
    n_heads = len(attn_weights_list)
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (ax, weights) in enumerate(zip(axes, attn_weights_list)):
        im = ax.imshow(weights, cmap='YlOrRd', aspect='auto')
        ax.set_title(f'Head {i + 1}')
        ax.set_xlabel('Key 位置')
        ax.set_ylabel('Query 位置')
        plt.colorbar(im, ax=ax, fraction=0.046)

    for j in range(n_heads, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_comparison(transformer_losses, baseline_losses, title='训练损失对比'):
    """绘制 Transformer 与基线模型的训练损失对比"""
    plt.figure(figsize=(10, 6))
    plt.plot(transformer_losses, label='Transformer 编码器', linewidth=1.5)
    plt.plot(baseline_losses, label='均值基线', linewidth=1.5)
    plt.xlabel('迭代次数')
    plt.ylabel('交叉熵损失')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Transformer 与注意力机制完整流程演示")
    print("=" * 60)

    # --- 1. 注意力机制原理 ---
    print_attention_principles()

    # --- 2. 缩放点积注意力演示 ---
    print("\n--- 缩放点积注意力演示 ---")
    np.random.seed(42)
    seq_len = 5
    d_k = 8
    d_v = 8
    Q_demo = np.random.randn(seq_len, d_k)
    K_demo = np.random.randn(seq_len, d_k)
    V_demo = np.random.randn(seq_len, d_v)

    attn_output, attn_weights = scaled_dot_product_attention(Q_demo, K_demo, V_demo)
    print(f"  Q 形状: {Q_demo.shape}")
    print(f"  K 形状: {K_demo.shape}")
    print(f"  V 形状: {V_demo.shape}")
    print(f"  注意力输出形状: {attn_output.shape}")
    print(f"  注意力权重形状: {attn_weights.shape}")
    print(f"  注意力权重行和: {attn_weights.sum(axis=1)}")
    print(f"  注意力权重矩阵:\n{np.round(attn_weights, 3)}")

    # --- 3. 多头注意力演示 ---
    print("\n--- 多头注意力演示 ---")
    d_model = 16
    n_heads = 4
    mha = MultiHeadAttention(d_model, n_heads)
    X_demo = np.random.randn(seq_len, d_model)
    mha_output, mha_weights = mha.forward(X_demo)
    print(f"  输入形状: {X_demo.shape}")
    print(f"  多头注意力输出形状: {mha_output.shape}")
    print(f"  头数: {n_heads}, 每头维度: {d_model // n_heads}")
    for i, w in enumerate(mha_weights):
        print(f"  Head {i + 1} 权重行和: {np.round(w.sum(axis=1), 4)}")

    # --- 4. 位置编码演示 ---
    print("\n--- 位置编码演示 ---")
    pe_seq_len = 50
    pe_d_model = 32
    PE = positional_encoding(pe_seq_len, pe_d_model)
    print(f"  位置编码形状: {PE.shape}")
    print(f"  位置 0 前 6 维: {np.round(PE[0, :6], 4)}")
    print(f"  位置 1 前 6 维: {np.round(PE[1, :6], 4)}")

    # --- 5. Transformer 编码器层演示 ---
    print("\n--- Transformer 编码器层演示 ---")
    enc_layer = TransformerEncoderLayer(d_model=16, n_heads=4, d_ff=64)
    X_enc = np.random.randn(seq_len, d_model)
    enc_output, enc_weights = enc_layer.forward(X_enc)
    print(f"  输入形状: {X_enc.shape}")
    print(f"  编码器输出形状: {enc_output.shape}")
    print(f"  编码器层注意力头数: {len(enc_weights)}")

    # --- 6. 简单序列分类实验 ---
    print("\n--- 简单序列分类实验 ---")
    print("  生成合成序列数据...")
    X_train, y_train = generate_sequence_data(
        n_samples=60, seq_len=10, d_model=16, n_classes=3, random_state=42
    )
    X_test, y_test = generate_sequence_data(
        n_samples=20, seq_len=10, d_model=16, n_classes=3, random_state=123
    )

    print(f"  训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    print(f"  类别分布 (训练): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    print("\n  [1] Transformer 编码器分类器训练:")
    transformer_clf = SimpleTransformerClassifier(
        seq_len=10, d_model=16, n_heads=4, d_ff=64, n_classes=3, n_layers=1, lr=0.005
    )
    transformer_clf.fit(X_train, y_train, n_iters=50)

    transformer_correct = sum(
        1 for i in range(len(X_test))
        if transformer_clf.predict(X_test[i]) == y_test[i]
    )
    transformer_acc = transformer_correct / len(X_test)
    print(f"  Transformer 测试准确率: {transformer_acc:.2%}")

    print("\n  [2] 均值基线分类器训练:")
    baseline_clf = BaselineClassifier(d_model=16, n_classes=3, lr=0.01, n_iters=100)
    baseline_clf.fit(X_train, y_train)

    baseline_correct = sum(
        1 for i in range(len(X_test))
        if baseline_clf.predict(X_test[i]) == y_test[i]
    )
    baseline_acc = baseline_correct / len(X_test)
    print(f"  基线模型测试准确率: {baseline_acc:.2%}")

    print(f"\n  对比结果: Transformer {transformer_acc:.2%} vs 基线 {baseline_acc:.2%}")

    # --- 7. 可视化 ---
    print("\n--- 可视化 ---")

    print("  [1] 注意力权重热力图...")
    np.random.seed(42)
    vis_seq_len = 8
    vis_d_model = 16
    X_vis = np.random.randn(vis_seq_len, vis_d_model)
    _, vis_weights = scaled_dot_product_attention(
        X_vis @ np.random.randn(vis_d_model, vis_d_model),
        X_vis @ np.random.randn(vis_d_model, vis_d_model),
        X_vis @ np.random.randn(vis_d_model, vis_d_model),
    )
    tokens = [f'Token_{i}' for i in range(vis_seq_len)]
    plot_attention_heatmap(vis_weights, title='缩放点积注意力权重', tokens=tokens)

    print("  [2] 位置编码可视化...")
    PE_vis = positional_encoding(50, 32)
    plot_positional_encoding(PE_vis, title='Sinusoidal 位置编码')

    print("  [3] 多头注意力模式可视化...")
    mha_vis = MultiHeadAttention(vis_d_model, n_heads=4)
    _, mha_vis_weights = mha_vis.forward(X_vis)
    plot_multihead_attention(mha_vis_weights, title='多头注意力模式 (4头)')

    print("  [4] 训练损失对比...")
    plot_training_comparison(
        transformer_clf.losses, baseline_clf.losses,
        title='Transformer vs 基线 训练损失对比'
    )

    print("\n" + "=" * 60)
    print("Transformer 与注意力机制演示完成")
    print("=" * 60)
