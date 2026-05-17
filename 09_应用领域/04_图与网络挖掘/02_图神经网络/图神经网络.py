"""
图神经网络模块 - Graph Neural Networks (GNN)
============================================
涵盖图神经网络的核心概念与实现：
1. 图神经网络基础 (消息传递框架、邻接矩阵归一化、GNN通用公式)
2. GCN (图卷积网络) 手动实现
3. GAT (图注意力网络) 核心思想与简化实现
4. GraphSAGE 核心思想
5. 节点分类实验
6. 可视化分析
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import setup_chinese_font

setup_chinese_font()


# ============================================================
# 1. 图神经网络基础
# ============================================================
def demo_gnn_basics():
    """图神经网络基础概念演示"""
    print("=" * 60)
    print("1. 图神经网络基础")
    print("=" * 60)

    print("\n--- 消息传递框架 (Message Passing Framework) ---")
    print("  核心思想: 每个节点通过聚合邻居信息来更新自身表示")
    print("  三个步骤:")
    print("    1) 消息生成 (Message): m_ij = MSG(h_i, h_j, e_ij)")
    print("    2) 消息聚合 (Aggregate): M_i = AGG({m_ij : j ∈ N(i)})")
    print("    3) 节点更新 (Update): h_i' = UPDATE(h_i, M_i)")

    print("\n--- 邻接矩阵归一化 ---")
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    print(f"  原始邻接矩阵 A:\n{A}")
    A_hat = A + np.eye(A.shape[0])
    D = np.diag(A_hat.sum(axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    print(f"\n  加入自环后 A_hat:\n{A_hat}")
    print(f"\n  度矩阵 D:\n{D}")
    print(f"\n  对称归一化 D^(-1/2) A_hat D^(-1/2):\n{np.round(A_norm, 4)}")

    print("\n--- GNN 通用公式 ---")
    print("  H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))")
    print("  其中:")
    print("    H^(l): 第l层节点特征矩阵 (n × d_l)")
    print("    W^(l): 第l层可学习权重矩阵 (d_l × d_{l+1})")
    print("    σ: 非线性激活函数 (如 ReLU)")
    print("    D^(-1/2) A D^(-1/2): 对称归一化邻接矩阵")


# ============================================================
# 2. GCN (图卷积网络) 手动实现
# ============================================================
class GCNManual:
    """图卷积网络 (GCN) 手动实现"""

    def __init__(self, n_features, n_hidden, n_classes, lr=0.01):
        np.random.seed(42)
        self.W1 = np.random.randn(n_features, n_hidden) * np.sqrt(2.0 / n_features)
        self.W2 = np.random.randn(n_hidden, n_classes) * np.sqrt(2.0 / n_hidden)
        self.lr = lr

    def _normalize_adj(self, A):
        A_hat = A + np.eye(A.shape[0])
        D = np.diag(A_hat.sum(axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X, A):
        self.A_norm = self._normalize_adj(A)
        self.Z1 = self.A_norm @ X @ self.W1
        self.H1 = self._relu(self.Z1)
        self.Z2 = self.A_norm @ self.H1 @ self.W2
        self.output = self._softmax(self.Z2)
        return self.output

    def fit(self, X, A, y, epochs=200):
        n = X.shape[0]
        n_classes = self.W2.shape[1]
        y_onehot = np.zeros((n, n_classes))
        y_onehot[np.arange(n), y] = 1.0

        train_mask = np.zeros(n, dtype=bool)
        train_indices = np.random.choice(n, max(1, int(n * 0.2)), replace=False)
        train_mask[train_indices] = True

        losses = []
        for epoch in range(epochs):
            output = self.forward(X, A)

            loss = -np.mean(y_onehot[train_mask] * np.log(output[train_mask] + 1e-8))
            losses.append(loss)

            dZ2 = (output - y_onehot)
            dZ2[~train_mask] = 0
            dW2 = self.H1.T @ dZ2 / train_mask.sum()

            dH1 = dZ2 @ self.W2.T
            dZ1 = dH1 * self._relu_grad(self.Z1)
            dZ1[~train_mask] = 0
            dW1 = X.T @ self.A_norm.T @ dZ1 / train_mask.sum()

            self.W2 -= self.lr * dW2
            self.W1 -= self.lr * dW1

            if (epoch + 1) % 50 == 0:
                pred = self.predict(X, A)
                acc = (pred[train_mask] == y[train_mask]).mean()
                print(f"  Epoch {epoch+1:4d} | Loss: {loss:.4f} | Train Acc: {acc:.4f}")

        return losses

    def predict(self, X, A):
        output = self.forward(X, A)
        return np.argmax(output, axis=1)


# ============================================================
# 3. GAT (图注意力网络) 核心思想与简化实现
# ============================================================
def demo_gat_concepts():
    """GAT 核心思想演示"""
    print("\n" + "=" * 60)
    print("3. GAT (图注意力网络) 核心思想")
    print("=" * 60)

    print("\n--- 注意力机制 ---")
    print("  核心公式: α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))")
    print("  步骤:")
    print("    1) 线性变换: h_i' = W · h_i")
    print("    2) 计算注意力系数: e_ij = LeakyReLU(a^T [h_i' || h_j'])")
    print("    3) 归一化: α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k exp(e_ik)")
    print("    4) 加权聚合: h_i'' = σ(Σ_j α_ij · h_j')")

    print("\n--- 多头注意力 ---")
    print("  K个注意力头并行计算，然后拼接或平均:")
    print("    拼接: h_i = ||_{k=1}^K σ(Σ_j α_ij^k W^k h_j)")
    print("    平均: h_i = σ(1/K Σ_{k=1}^K Σ_j α_ij^k W^k h_j)")


class GATManual:
    """图注意力网络 (GAT) 简化单头实现"""

    def __init__(self, n_features, n_hidden, n_classes, lr=0.01):
        np.random.seed(42)
        self.W = np.random.randn(n_features, n_hidden) * np.sqrt(2.0 / n_features)
        self.a = np.random.randn(2 * n_hidden) * np.sqrt(2.0 / (2 * n_hidden))
        self.W_out = np.random.randn(n_hidden, n_classes) * np.sqrt(2.0 / n_hidden)
        self.lr = lr
        self.n_hidden = n_hidden

    def _leaky_relu(self, x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)

    def _leaky_relu_grad(self, x, alpha=0.2):
        return np.where(x > 0, 1.0, alpha)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def _compute_attention(self, H, A):
        n = H.shape[0]
        e = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if A[i, j] > 0 or i == j:
                    concat = np.concatenate([H[i], H[j]])
                    e[i, j] = self._leaky_relu(self.a @ concat)
                else:
                    e[i, j] = -1e9
        alpha = self._softmax(e)
        return alpha

    def forward(self, X, A):
        self.X = X
        self.A = A
        self.H = X @ self.W
        self.alpha = self._compute_attention(self.H, A)
        self.H_agg = self.alpha @ self.H
        self.H_act = np.maximum(0, self.H_agg)
        self.Z = self.H_act @ self.W_out
        self.output = self._softmax(self.Z)
        return self.output

    def fit(self, X, A, y, epochs=100):
        n = X.shape[0]
        n_classes = self.W_out.shape[1]
        y_onehot = np.zeros((n, n_classes))
        y_onehot[np.arange(n), y] = 1.0

        train_mask = np.zeros(n, dtype=bool)
        train_indices = np.random.choice(n, max(1, int(n * 0.2)), replace=False)
        train_mask[train_indices] = True

        losses = []
        for epoch in range(epochs):
            output = self.forward(X, A)

            loss = -np.mean(y_onehot[train_mask] * np.log(output[train_mask] + 1e-8))
            losses.append(loss)

            dZ = (output - y_onehot)
            dZ[~train_mask] = 0
            dW_out = self.H_act.T @ dZ / train_mask.sum()

            dH_act = dZ @ self.W_out.T
            dH_agg = dH_act * (self.H_agg > 0).astype(float)
            dW = X.T @ (self.alpha.T @ dH_agg) / train_mask.sum()

            self.W_out -= self.lr * dW_out
            self.W -= self.lr * dW

            if (epoch + 1) % 25 == 0:
                pred = self.predict(X, A)
                acc = (pred[train_mask] == y[train_mask]).mean()
                print(f"  Epoch {epoch+1:4d} | Loss: {loss:.4f} | Train Acc: {acc:.4f}")

        return losses

    def predict(self, X, A):
        output = self.forward(X, A)
        return np.argmax(output, axis=1)


# ============================================================
# 4. GraphSAGE 核心思想
# ============================================================
def demo_graphsage_concepts():
    """GraphSAGE 核心思想演示"""
    print("\n" + "=" * 60)
    print("4. GraphSAGE 核心思想")
    print("=" * 60)

    print("\n--- 采样-聚合策略 ---")
    print("  GraphSAGE = SAmple + aggreGatE")
    print("  步骤:")
    print("    1) 采样 (Sample): 对每个节点随机采样固定数量的邻居")
    print("       h_N(i) = {h_j : j ∈ Sample(N(i), K)}")
    print("    2) 聚合 (Aggregate): 将采样邻居的特征聚合")
    print("    3) 拼接/更新 (Concat/Update): h_i' = σ(W · [h_i || AGG(h_N(i))])")

    print("\n--- 聚合器类型 ---")
    print("  Mean Aggregator:")
    print("    h_i' = σ(W · MEAN({h_i} ∪ {h_j, j ∈ N(i)}))")
    print("    等价于 GCN 的归纳版本")
    print()
    print("  MaxPool Aggregator:")
    print("    h_N(i) = max({σ(W_pool · h_j + b), j ∈ N(i)})")
    print("    h_i' = σ(W · [h_i || h_N(i)])")
    print()
    print("  LSTM Aggregator:")
    print("    h_N(i) = LSTM({h_j, j ∈ N(i)})")
    print("    对邻居顺序敏感，需要随机排列")

    print("\n--- GraphSAGE vs GCN ---")
    print("  GCN: 使用全部邻居 (转导式，需要全图)")
    print("  GraphSAGE: 采样固定数量邻居 (归纳式，可泛化到新节点)")


# ============================================================
# 5. 节点分类实验
# ============================================================
def create_synthetic_graph(n_nodes_per_community=30, n_communities=4, n_features=8,
                           intra_prob=0.3, inter_prob=0.02, random_state=42):
    """使用随机块模型 (SBM) 生成合成图数据"""
    np.random.seed(random_state)

    sizes = [n_nodes_per_community] * n_communities
    n_total = sum(sizes)

    p = np.full((n_communities, n_communities), inter_prob)
    np.fill_diagonal(p, intra_prob)

    G = nx.stochastic_block_model(sizes, p, seed=random_state)

    A = nx.to_numpy_array(G)

    y = np.zeros(n_total, dtype=int)
    for i in range(n_communities):
        start = i * n_nodes_per_community
        y[start:start + n_nodes_per_community] = i

    X = np.random.randn(n_total, n_features) * 0.5
    for i in range(n_communities):
        start = i * n_nodes_per_community
        end = start + n_nodes_per_community
        center = np.random.randn(n_features) * 2.0
        X[start:end] += center

    return A, X, y, G


def run_node_classification():
    """节点分类实验"""
    print("\n" + "=" * 60)
    print("5. 节点分类实验")
    print("=" * 60)

    A, X, y, G = create_synthetic_graph(
        n_nodes_per_community=30, n_communities=4, n_features=8,
        intra_prob=0.3, inter_prob=0.02, random_state=42
    )
    n = X.shape[0]
    n_classes = len(np.unique(y))

    print(f"\n  图统计信息:")
    print(f"    节点数: {n}")
    print(f"    边数: {int(A.sum() / 2)}")
    print(f"    社区数: {n_classes}")
    print(f"    特征维度: {X.shape[1]}")
    print(f"    平均度: {A.sum(axis=1).mean():.2f}")

    np.random.seed(42)
    test_mask = np.zeros(n, dtype=bool)
    test_indices = np.random.choice(n, int(n * 0.2), replace=False)
    test_mask[test_indices] = True
    train_val_mask = ~test_mask

    print(f"\n  训练+验证集: {train_val_mask.sum()} 节点, 测试集: {test_mask.sum()} 节点")

    print("\n  --- GCN 训练 ---")
    gcn = GCNManual(n_features=X.shape[1], n_hidden=16, n_classes=n_classes, lr=0.05)
    gcn_losses = gcn.fit(X, A, y, epochs=300)

    gcn_pred = gcn.predict(X, A)
    gcn_test_acc = (gcn_pred[test_mask] == y[test_mask]).mean()
    print(f"\n  GCN 测试准确率: {gcn_test_acc:.4f}")

    print("\n  --- GAT 训练 ---")
    gat = GATManual(n_features=X.shape[1], n_hidden=16, n_classes=n_classes, lr=0.01)
    gat_losses = gat.fit(X, A, y, epochs=100)

    gat_pred = gat.predict(X, A)
    gat_test_acc = (gat_pred[test_mask] == y[test_mask]).mean()
    print(f"\n  GAT 测试准确率: {gat_test_acc:.4f}")

    np.random.seed(42)
    random_pred = np.random.randint(0, n_classes, size=n)
    random_acc = (random_pred[test_mask] == y[test_mask]).mean()
    print(f"  随机基线准确率: {random_acc:.4f}")

    return A, X, y, G, gcn, gat, gcn_pred, gat_pred, gcn_losses, gat_losses, test_mask, random_acc, gcn_test_acc, gat_test_acc


# ============================================================
# 6. 可视化
# ============================================================
def visualize_results(A, X, y, G, gcn, gat, gcn_pred, gat_pred,
                      gcn_losses, gat_losses, test_mask, random_acc,
                      gcn_test_acc, gat_test_acc):
    """可视化分析"""
    print("\n" + "=" * 60)
    print("6. 可视化分析")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    pos = nx.spring_layout(G, seed=42)
    node_colors_true = [y[i] for i in range(len(y))]
    nx.draw(G, pos, ax=ax1, node_color=node_colors_true,
            cmap=plt.cm.Set2, node_size=80, width=0.5, alpha=0.8,
            with_labels=False)
    ax1.set_title("真实社区标签", fontsize=14)

    ax2 = axes[0, 1]
    node_colors_pred = [gcn_pred[i] for i in range(len(gcn_pred))]
    nx.draw(G, pos, ax=ax2, node_color=node_colors_pred,
            cmap=plt.cm.Set2, node_size=80, width=0.5, alpha=0.8,
            with_labels=False)
    ax2.set_title("GCN 预测标签", fontsize=14)

    ax3 = axes[1, 0]
    gcn_forward = gcn.forward(X, A)
    hidden = gcn.A_norm @ X @ gcn.W1
    hidden_act = np.maximum(0, hidden)
    ax3.scatter(hidden_act[:, 0], hidden_act[:, 1], c=y, cmap=plt.cm.Set2,
                s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
    ax3.set_xlabel("隐藏层维度 1")
    ax3.set_ylabel("隐藏层维度 2")
    ax3.set_title("GCN 节点嵌入 (前2维)", fontsize=14)

    ax4 = axes[1, 1]
    ax4.plot(gcn_losses, label='GCN Loss', color='#2196F3', linewidth=1.5)
    ax4.plot(gat_losses, label='GAT Loss', color='#FF5722', linewidth=1.5)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Cross-Entropy Loss")
    ax4.set_title("训练损失曲线", fontsize=14)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '图神经网络_可视化.png'), dpi=150, bbox_inches='tight')
    plt.show()

    fig2, ax = plt.subplots(figsize=(8, 5))
    methods = ['Random', 'GCN', 'GAT']
    accuracies = [random_acc, gcn_test_acc, gat_test_acc]
    colors = ['#9E9E9E', '#2196F3', '#FF5722']
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel("准确率")
    ax.set_title("GCN vs GAT vs 随机基线 — 测试集准确率", fontsize=14)
    ax.set_ylim(0, max(accuracies) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '图神经网络_准确率对比.png'), dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    demo_gnn_basics()
    demo_gat_concepts()
    demo_graphsage_concepts()

    results = run_node_classification()
    visualize_results(*results)
