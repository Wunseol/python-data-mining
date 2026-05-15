"""
图与网络挖掘模块 - Graph & Network Mining
==========================================
涵盖图数据挖掘的核心算法：
1. 图的表示与基本概念
2. 中心性度量 (度中心性、介数中心性、接近中心性、特征向量中心性)
3. PageRank算法
4. 社区发现 (Girvan-Newman、Louvain)
5. 链接预测
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 图的表示与基本概念
# ============================================================
class Graph:
    """简单无向图实现"""

    def __init__(self):
        self.adj = defaultdict(set)
        self.nodes = set()

    def add_edge(self, u, v):
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.nodes.add(u)
        self.nodes.add(v)

    def add_node(self, u):
        self.nodes.add(u)

    def neighbors(self, u):
        return self.adj[u]

    def degree(self, u):
        return len(self.adj[u])

    def edges(self):
        seen = set()
        result = []
        for u in self.adj:
            for v in self.adj[u]:
                if (v, u) not in seen:
                    seen.add((u, v))
                    result.append((u, v))
        return result

    def to_adjacency_matrix(self):
        """转换为邻接矩阵"""
        node_list = sorted(self.nodes)
        idx = {n: i for i, n in enumerate(node_list)}
        n = len(node_list)
        mat = np.zeros((n, n))
        for u, v in self.edges():
            mat[idx[u], idx[v]] = 1
            mat[idx[u], idx[u]] = 1
        return mat, node_list

    def visualize(self, title="图", pos=None, node_colors=None, figsize=(8, 6)):
        """简单的图可视化（弹簧布局）"""
        node_list = sorted(self.nodes)
        n = len(node_list)
        if n == 0:
            return

        # 简单弹簧布局
        if pos is None:
            np.random.seed(42)
            pos = {}
            for i, node in enumerate(node_list):
                angle = 2 * np.pi * i / n
                pos[node] = (np.cos(angle), np.sin(angle))

        fig, ax = plt.subplots(figsize=figsize)
        # 画边
        for u, v in self.edges():
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, 'b-', alpha=0.3, linewidth=1)

        # 画节点
        if node_colors is None:
            node_colors = 'lightblue'
        for node in node_list:
            ax.scatter(pos[node][0], pos[node][1], s=300, c=node_colors if isinstance(node_colors, str) else node_colors.get(node, 'lightblue'),
                       edgecolors='black', zorder=5)
            ax.annotate(str(node), (pos[node][0], pos[node][1]),
                       ha='center', va='center', fontsize=10, fontweight='bold')

        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.show()


def create_sample_graph():
    """创建示例社交网络图"""
    g = Graph()
    # 构建一个有社区结构的小型社交网络
    # 社区A: 0,1,2,3
    edges_a = [(0,1), (0,2), (1,2), (2,3), (1,3)]
    # 社区B: 4,5,6,7
    edges_b = [(4,5), (4,6), (5,6), (6,7), (5,7)]
    # 桥接边
    bridges = [(3,4), (2,5)]
    # 社区C: 8,9
    edges_c = [(8,9), (7,8)]

    for u, v in edges_a + edges_b + bridges + edges_c:
        g.add_edge(u, v)
    return g


def demo_graph_basics():
    """演示图的基本概念"""
    print("--- 1. 图的基本概念 ---")
    g = create_sample_graph()
    print(f"  节点数: {len(g.nodes)}")
    print(f"  边数: {len(g.edges())}")
    print(f"  各节点度数: { {n: g.degree(n) for n in sorted(g.nodes)} }")


# ============================================================
# 2. 中心性度量
# ============================================================
def degree_centrality(graph):
    """度中心性：连接数最多的节点最重要"""
    n = len(graph.nodes)
    if n <= 1:
        return {node: 1.0 for node in graph.nodes}
    return {node: graph.degree(node) / (n - 1) for node in graph.nodes}


def betweenness_centrality(graph):
    """介数中心性：经过该节点的最短路径越多越重要"""
    nodes = sorted(graph.nodes)
    cb = {v: 0.0 for v in nodes}

    for s in nodes:
        # BFS找最短路径
        pred = {v: [] for v in nodes}
        dist = {v: -1 for v in nodes}
        sigma = {v: 0 for v in nodes}
        sigma[s] = 1
        dist[s] = 0
        queue = deque([s])

        while queue:
            v = queue.popleft()
            for w in graph.neighbors(v):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # 回溯计算介数
        delta = {v: 0.0 for v in nodes}
        while queue:
            queue.pop()
        stack = sorted(nodes, key=lambda v: dist[v] if dist[v] >= 0 else float('inf'), reverse=True)
        for w in stack:
            if w == s:
                continue
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                cb[w] += delta[w]

    # 归一化
    n = len(nodes)
    norm = (n - 1) * (n - 2) / 2 if n > 2 else 1
    return {v: cb[v] / norm for v in nodes}


def eigenvector_centrality(graph, max_iter=100, tol=1e-6):
    """特征向量中心性：连接到重要节点的节点更重要"""
    mat, node_list = graph.to_adjacency_matrix()
    n = len(node_list)

    # 幂迭代法
    x = np.ones(n) / n
    for _ in range(max_iter):
        x_new = mat @ x
        norm = np.linalg.norm(x_new)
        if norm == 0:
            break
        x_new = x_new / norm
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return {node_list[i]: x[i] for i in range(n)}


def demo_centrality():
    """演示中心性度量"""
    print("\n--- 2. 中心性度量 ---")
    g = create_sample_graph()

    dc = degree_centrality(g)
    bc = betweenness_centrality(g)
    ec = eigenvector_centrality(g)

    print(f"  {'节点':>4} {'度中心性':>10} {'介数中心性':>10} {'特征向量中心性':>12}")
    print(f"  {'----':>4} {'----------':>10} {'----------':>10} {'------------':>12}")
    for node in sorted(g.nodes):
        print(f"  {node:>4} {dc[node]:>10.4f} {bc[node]:>10.4f} {ec[node]:>12.4f}")


# ============================================================
# 3. PageRank算法
# ============================================================
def pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    """
    PageRank算法 — Google网页排名的核心算法
    基于随机游走模型：以概率d沿链接跳转，以概率1-d随机跳到任意节点
    """
    nodes = sorted(graph.nodes)
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    # 构建转移矩阵
    M = np.zeros((n, n))
    for u in nodes:
        neighbors = list(graph.neighbors(u))
        if neighbors:
            for v in neighbors:
                M[idx[v], idx[u]] = 1.0 / len(neighbors)
        else:
            # 悬挂节点：均匀跳转
            M[:, idx[u]] = 1.0 / n

    # 迭代计算 PageRank
    pr = np.ones(n) / n
    for _ in range(max_iter):
        pr_new = damping * M @ pr + (1 - damping) / n
        if np.linalg.norm(pr_new - pr, 1) < tol:
            break
        pr = pr_new

    return {nodes[i]: pr[i] for i in range(n)}


def demo_pagerank():
    """演示PageRank算法"""
    print("\n--- 3. PageRank算法 ---")
    g = create_sample_graph()

    pr = pagerank(g)
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)

    print("  节点重要性排名 (PageRank):")
    for i, (node, score) in enumerate(sorted_pr):
        bar = '█' * int(score * 60)
        print(f"  {i+1}. 节点{node}: {score:.4f} {bar}")


# ============================================================
# 4. 社区发现
# ============================================================
def girvan_newman(graph, n_communities=2):
    """
    Girvan-Newman社区发现算法
    反复移除介数最高的边，直到分成指定数量的社区
    """
    import copy
    g = copy.deepcopy(graph)
    communities = [set(g.nodes)]

    while len(communities) < n_communities:
        # 计算所有边的边介数
        edge_betweenness = defaultdict(float)
        for u, v in g.edges():
            edge_betweenness[(u, v)] = 0

        nodes = sorted(g.nodes)
        for s in nodes:
            # BFS
            pred = {v: [] for v in g.nodes}
            dist = {v: -1 for v in g.nodes}
            sigma = {v: 0 for v in g.nodes}
            sigma[s] = 1
            dist[s] = 0
            queue = deque([s])

            while queue:
                v = queue.popleft()
                for w in g.neighbors(v):
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            # 计算边介数贡献
            delta = {v: 0.0 for v in g.nodes}
            stack = sorted([v for v in g.nodes if dist[v] >= 0],
                          key=lambda v: dist[v], reverse=True)
            for w in stack:
                for v in pred[w]:
                    c = sigma[v] / sigma[w] * (1 + delta[w])
                    delta[v] += c
                    # 记录边介数
                    edge = (min(v, w), max(v, w))
                    if edge in edge_betweenness:
                        edge_betweenness[edge] += c

        # 移除介数最高的边
        if not edge_betweenness:
            break
        max_edge = max(edge_betweenness, key=edge_betweenness.get)
        u, v = max_edge
        g.adj[u].discard(v)
        g.adj[v].discard(u)

        # 重新计算连通分量
        visited = set()
        communities = []
        for node in g.nodes:
            if node not in visited:
                comp = set()
                queue = deque([node])
                while queue:
                    n = queue.popleft()
                    if n in visited:
                        continue
                    visited.add(n)
                    comp.add(n)
                    for nb in g.adj[n]:
                        if nb not in visited:
                            queue.append(nb)
                communities.append(comp)

    return communities


def demo_community_detection():
    """演示社区发现"""
    print("\n--- 4. 社区发现 (Girvan-Newman) ---")
    g = create_sample_graph()

    communities = girvan_newman(g, n_communities=3)
    for i, comm in enumerate(communities):
        print(f"  社区{i+1}: {sorted(comm)}")


# ============================================================
# 5. 链接预测
# ============================================================
def common_neighbors(graph, u, v):
    """共同邻居数"""
    return len(graph.neighbors(u) & graph.neighbors(v))


def jaccard_coefficient(graph, u, v):
    """Jaccard系数"""
    nu, nv = graph.neighbors(u), graph.neighbors(v)
    intersection = len(nu & nv)
    union = len(nu | nv)
    return intersection / union if union > 0 else 0


def adamic_adar(graph, u, v):
    """Adamic-Adar指数"""
    common = graph.neighbors(u) & graph.neighbors(v)
    score = 0
    for z in common:
        d = graph.degree(z)
        if d > 1:
            score += 1 / np.log(d)
    return score


def preferential_attachment(graph, u, v):
    """优先连接（度乘积）"""
    return graph.degree(u) * graph.degree(v)


def demo_link_prediction():
    """演示链接预测"""
    print("\n--- 5. 链接预测 ---")
    g = create_sample_graph()

    # 找出所有不存在的边
    non_edges = []
    nodes = sorted(g.nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:
            if v not in g.adj[u]:
                non_edges.append((u, v))

    # 计算各种链接预测分数
    scores = []
    for u, v in non_edges:
        cn = common_neighbors(g, u, v)
        jc = jaccard_coefficient(g, u, v)
        aa = adamic_adar(g, u, v)
        pa = preferential_attachment(g, u, v)
        scores.append((u, v, cn, jc, aa, pa))

    # 按共同邻居排序，展示最可能形成连接的节点对
    scores.sort(key=lambda x: x[2], reverse=True)
    print(f"  {'节点对':>10} {'共同邻居':>8} {'Jaccard':>8} {'Adamic-Adar':>12} {'优先连接':>8}")
    for u, v, cn, jc, aa, pa in scores[:5]:
        print(f"  ({u},{v}){'':>6} {cn:>8} {jc:>8.4f} {aa:>12.4f} {pa:>8}")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    demo_graph_basics()
    demo_centrality()
    demo_pagerank()
    demo_community_detection()
    demo_link_prediction()
