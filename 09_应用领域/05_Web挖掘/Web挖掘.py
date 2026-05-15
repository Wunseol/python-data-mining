"""
Web挖掘模块 - Web Mining
========================
涵盖Web挖掘的三个主要方向：
1. Web内容挖掘（网页信息提取、文本分类）
2. Web结构挖掘（PageRank、HITS算法）
3. Web日志挖掘（用户行为模式分析）

参考：Han & Kamber 第13.1.2节、DaKM 2025专题
"""

import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. Web结构挖掘 - PageRank算法
# ============================================================
def pagerank(graph, damping=0.85, max_iter=100, tol=1e-6):
    """
    PageRank算法实现

    核心思想：一个网页的重要性 = 所有指向它的网页重要性之和 / 指出链接数
    PR(A) = (1-d)/N + d * Σ[PR(Ti)/C(Ti)]
    其中 Ti 指向A，C(Ti)是Ti的出链数，d是阻尼因子

    Parameters:
        graph: dict, {节点: [指向的节点列表]}
        damping: float, 阻尼因子(通常0.85)
        max_iter: int, 最大迭代次数
        tol: float, 收敛阈值
    """
    nodes = set()
    for src, targets in graph.items():
        nodes.add(src)
        nodes.update(targets)
    nodes = sorted(nodes)
    N = len(nodes)

    # 初始化：每个节点等权
    pr = {node: 1.0 / N for node in nodes}

    # 计算入链和出链
    out_links = {node: len(graph.get(node, [])) for node in nodes}
    in_links = defaultdict(list)
    for src, targets in graph.items():
        for tgt in targets:
            in_links[tgt].append(src)

    print("=" * 60)
    print("PageRank算法 - Web结构挖掘")
    print("=" * 60)
    print(f"网页数: {N}, 阻尼因子: {damping}")
    print(f"图结构: {dict(graph)}")

    for iteration in range(max_iter):
        new_pr = {}
        for node in nodes:
            rank_sum = sum(pr[src] / out_links[src] for src in in_links[node]
                          if out_links[src] > 0)
            new_pr[node] = (1 - damping) / N + damping * rank_sum

        # 检查收敛
        diff = sum(abs(new_pr[node] - pr[node]) for node in nodes)
        pr = new_pr

        if diff < tol:
            print(f"\n收敛于第 {iteration + 1} 次迭代 (变化量={diff:.2e})")
            break

    # 排序输出
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    print("\nPageRank值排名:")
    for node, score in sorted_pr:
        bar = '█' * int(score * 200)
        print(f"  {node}: {score:.4f} {bar}")

    return pr


# ============================================================
# 2. Web结构挖掘 - HITS算法
# ============================================================
def hits_algorithm(graph, max_iter=100, tol=1e-6):
    """
    HITS算法 (Hyperlink-Induced Topic Search)

    核心思想：
    - Authority页面：被很多Hub页面指向的页面（内容权威）
    - Hub页面：指向很多Authority页面的页面（导航权威）
    - 互相增强：好的Hub指向好的Authority，好的Authority被好的Hub指向

    Parameters:
        graph: dict, {节点: [指向的节点列表]}
    """
    nodes = set()
    for src, targets in graph.items():
        nodes.add(src)
        nodes.update(targets)
    nodes = sorted(nodes)

    # 初始化
    auth = {node: 1.0 for node in nodes}  # Authority值
    hub = {node: 1.0 for node in nodes}   # Hub值

    in_links = defaultdict(list)
    out_links_map = defaultdict(list)
    for src, targets in graph.items():
        for tgt in targets:
            in_links[tgt].append(src)
            out_links_map[src].append(tgt)

    print("\n" + "=" * 60)
    print("HITS算法 - Hub与Authority分析")
    print("=" * 60)

    for iteration in range(max_iter):
        # 更新Authority: auth(p) = Σ hub(q), q→p
        new_auth = {}
        for node in nodes:
            new_auth[node] = sum(hub[src] for src in in_links[node])

        # 更新Hub: hub(p) = Σ auth(q), p→q
        new_hub = {}
        for node in nodes:
            new_hub[node] = sum(new_auth[tgt] for tgt in out_links_map[node])

        # 归一化
        auth_norm = max(new_auth.values()) if new_auth.values() else 1
        hub_norm = max(new_hub.values()) if new_hub.values() else 1
        new_auth = {k: v / auth_norm for k, v in new_auth.items()}
        new_hub = {k: v / hub_norm for k, v in new_hub.items()}

        # 检查收敛
        auth_diff = sum(abs(new_auth[n] - auth[n]) for n in nodes)
        hub_diff = sum(abs(new_hub[n] - hub[n]) for n in nodes)
        auth, hub = new_auth, new_hub

        if auth_diff + hub_diff < tol:
            print(f"收敛于第 {iteration + 1} 次迭代")
            break

    print("\nAuthority排名 (内容权威):")
    sorted_auth = sorted(auth.items(), key=lambda x: x[1], reverse=True)
    for node, score in sorted_auth:
        print(f"  {node}: {score:.4f}")

    print("\nHub排名 (导航权威):")
    sorted_hub = sorted(hub.items(), key=lambda x: x[1], reverse=True)
    for node, score in sorted_hub:
        print(f"  {node}: {score:.4f}")

    return auth, hub


# ============================================================
# 3. Web内容挖掘 - TF-IDF文本特征提取
# ============================================================
def web_content_mining():
    """Web内容挖掘：TF-IDF特征提取与网页相似度计算"""

    print("\n" + "=" * 60)
    print("Web内容挖掘 - TF-IDF特征提取")
    print("=" * 60)

    # 模拟网页文本
    web_pages = {
        "页A": "数据挖掘 机器学习 分类算法",
        "页B": "数据挖掘 聚类分析 无监督学习",
        "页C": "深度学习 神经网络 图像识别",
        "页D": "数据挖掘 关联规则 购物篮分析",
        "页E": "自然语言处理 深度学习 文本分类",
    }

    # 构建词表
    all_terms = set()
    for text in web_pages.values():
        all_terms.update(text.split())
    all_terms = sorted(all_terms)

    # 计算TF
    N = len(web_pages)
    tf = {}
    for page, text in web_pages.items():
        words = text.split()
        for term in all_terms:
            tf[(page, term)] = words.count(term) / len(words) if words else 0

    # 计算IDF
    df = defaultdict(int)
    for text in web_pages.values():
        for term in set(text.split()):
            df[term] += 1

    idf = {term: np.log(N / df[term]) for term in all_terms}

    # 计算TF-IDF
    tfidf = {}
    for page in web_pages:
        for term in all_terms:
            tfidf[(page, term)] = tf[(page, term)] * idf[term]

    # 输出TF-IDF矩阵
    print("\nTF-IDF矩阵:")
    header = f"{'网页':<6}" + "".join(f"{t:<8}" for t in all_terms)
    print(header)
    for page in web_pages:
        row = f"{page:<6}" + "".join(f"{tfidf[(page, t)]:.3f}   " for t in all_terms)
        print(row)

    # 计算网页相似度（余弦）
    print("\n网页相似度(余弦相似度) - 发现内容相关网页:")
    page_names = list(web_pages.keys())
    for i in range(len(page_names)):
        for j in range(i + 1, len(page_names)):
            p1, p2 = page_names[i], page_names[j]
            v1 = np.array([tfidf[(p1, t)] for t in all_terms])
            v2 = np.array([tfidf[(p2, t)] for t in all_terms])
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            sim = dot / norm if norm > 0 else 0
            if sim > 0.1:
                print(f"  {p1} ↔ {p2}: {sim:.3f}")

    return tfidf


# ============================================================
# 4. Web日志挖掘 - 用户行为模式分析
# ============================================================
def web_log_mining():
    """Web日志挖掘：发现用户访问模式"""

    print("\n" + "=" * 60)
    print("Web日志挖掘 - 用户访问模式分析")
    print("=" * 60)

    # 模拟Web访问日志
    web_logs = [
        ("用户1", "首页", "10:00"),
        ("用户1", "产品列表", "10:01"),
        ("用户1", "产品详情", "10:03"),
        ("用户1", "购物车", "10:05"),
        ("用户2", "首页", "10:02"),
        ("用户2", "产品列表", "10:03"),
        ("用户2", "搜索页", "10:05"),
        ("用户2", "产品详情", "10:06"),
        ("用户3", "首页", "10:10"),
        ("用户3", "产品列表", "10:11"),
        ("用户3", "产品详情", "10:13"),
        ("用户3", "购物车", "10:15"),
        ("用户4", "首页", "10:20"),
        ("用户4", "搜索页", "10:21"),
        ("用户4", "产品详情", "10:23"),
    ]

    # 提取访问路径
    user_paths = defaultdict(list)
    for user, page, time in web_logs:
        user_paths[user].append(page)

    print("\n用户访问路径:")
    for user, path in user_paths.items():
        print(f"  {user}: {' → '.join(path)}")

    # 频繁页面序列挖掘（简化版）
    print("\n频繁2-项访问序列:")
    pair_count = defaultdict(int)
    for user, path in user_paths.items():
        for i in range(len(path) - 1):
            pair = (path[i], path[i + 1])
            pair_count[pair] += 1

    min_support = 2
    for pair, count in sorted(pair_count.items(), key=lambda x: x[1], reverse=True):
        if count >= min_support:
            print(f"  {pair[0]} → {pair[1]}: 支持度={count}")

    # 页面访问频率统计
    print("\n页面热度排名:")
    page_count = defaultdict(int)
    for _, page, _ in web_logs:
        page_count[page] += 1
    for page, count in sorted(page_count.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * count
        print(f"  {page}: {count}次 {bar}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║          Web挖掘 - Web Mining                ║")
    print("╚══════════════════════════════════════════════╝")

    # Web结构挖掘
    web_graph = {
        'A': ['B', 'C'],
        'B': ['C'],
        'C': ['A'],
        'D': ['C'],
        'E': ['A', 'C'],
    }
    pagerank(web_graph)
    hits_algorithm(web_graph)

    # Web内容挖掘
    web_content_mining()

    # Web日志挖掘
    web_log_mining()
