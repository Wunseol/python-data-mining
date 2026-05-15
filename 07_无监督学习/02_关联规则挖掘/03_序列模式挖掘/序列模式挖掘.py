"""
序列模式挖掘模块 - Sequential Pattern Mining
=============================================
涵盖从序列数据中发现频繁模式的方法：
1. 序列数据概念与表示
2. AprioriAll算法 (基于Apriori的序列挖掘)
3. PrefixSpan算法 (基于前缀投影的序列挖掘)
4. 序列规则与应用场景
"""

from collections import defaultdict
from itertools import combinations


# ============================================================
# 1. 序列数据概念与表示
# ============================================================
def print_sequence_concepts():
    """展示序列模式挖掘的基本概念"""
    concepts = {
        "序列 (Sequence)": "有序的事件/项集列表，如 <(a)(b,c)(d)>",
        "项集 (Itemset)": "序列中同一时间出现的项的集合，如 (b,c)",
        "序列长度": "序列中项集的个数",
        "子序列": "序列α是β的子序列，若α中每个项集都是β中对应项集的子集且保持顺序",
        "支持度": "包含该子序列的序列数占总序列数的比例",
        "频繁序列": "支持度不小于最小支持度阈值的子序列",
    }

    print("=" * 60)
    print("序列模式挖掘基本概念")
    print("=" * 60)
    for term, definition in concepts.items():
        print(f"  • {term}：{definition}")


# ============================================================
# 2. AprioriAll算法
# ============================================================
def is_subsequence(sub, seq):
    """
    检查sub是否为seq的子序列
    sub和seq都是项集列表，如 sub=[['a'], ['b','c']], seq=[['a'], ['d'], ['b','c'], ['e']]
    """
    len_sub = len(sub)
    len_seq = len(seq)
    if len_sub == 0:
        return True
    if len_sub > len_seq:
        return False

    # 使用动态规划/贪心匹配
    sub_idx = 0
    for itemset in seq:
        if sub_idx < len_sub:
            # 检查sub[sub_idx]是否是itemset的子集
            if set(sub[sub_idx]).issubset(set(itemset)):
                sub_idx += 1
    return sub_idx == len_sub


def count_support(sequences, pattern):
    """计算模式在序列数据库中的支持度"""
    count = 0
    for seq in sequences:
        if is_subsequence(pattern, seq):
            count += 1
    return count / len(sequences) if sequences else 0


def apriori_all(sequences, min_support=0.5):
    """
    AprioriAll算法 — 基于Apriori思想的序列模式挖掘
    逐层产生频繁序列：1-序列 → 2-序列 → ... → k-序列
    """
    n_seq = len(sequences)

    # 生成候选1-序列
    items = set()
    for seq in sequences:
        for itemset in seq:
            for item in itemset:
                items.add(item)

    # 频繁1-序列
    freq_1 = {}
    for item in sorted(items):
        pattern = [[item]]
        sup = count_support(sequences, pattern)
        if sup >= min_support:
            freq_1[tuple(tuple(p) for p in pattern)] = sup

    if not freq_1:
        return {}

    all_frequent = dict(freq_1)
    prev_frequent = dict(freq_1)
    k = 2

    while prev_frequent:
        # 候选生成：连接步
        candidates = set()
        prev_patterns = [list(list(p) for p in pat) for pat in prev_frequent.keys()]

        for i, p1 in enumerate(prev_patterns):
            for j, p2 in enumerate(prev_patterns):
                # 连接：p1去掉第一个元素 ≈ p2去掉最后一个元素
                # 简化实现：尝试两种扩展方式
                # 方式1：将p2最后一个项添加为新项集
                new_seq = [list(x) for x in p1] + [list(p2[-1])]
                candidates.add(tuple(tuple(x) for x in new_seq))

                # 方式2：将p2最后一个项添加到p1最后一个项集中
                new_seq2 = [list(x) for x in p1]
                new_seq2[-1] = sorted(set(new_seq2[-1]) | set(p2[-1]))
                candidates.add(tuple(tuple(x) for x in new_seq2))

        # 剪枝与支持度计算
        new_frequent = {}
        for cand in candidates:
            pattern = [list(p) for p in cand]
            sup = count_support(sequences, pattern)
            if sup >= min_support:
                new_frequent[cand] = sup

        if not new_frequent:
            break

        all_frequent.update(new_frequent)
        prev_frequent = new_frequent
        k += 1

    return all_frequent


def demo_apriori_all():
    """演示AprioriAll算法"""
    print("--- 2. AprioriAll 序列模式挖掘 ---")

    # 示例：客户购买序列数据库
    sequences = [
        [['a'], ['b', 'c'], ['d']],        # 客户1
        [['a'], ['c'], ['b', 'd']],        # 客户2
        [['a', 'b'], ['c'], ['d']],        # 客户3
        [['b'], ['c'], ['d']],             # 客户4
    ]

    print("  序列数据库:")
    for i, seq in enumerate(sequences):
        print(f"    客户{i+1}: {' → '.join(str(set(s)) for s in seq)}")

    min_sup = 0.5
    freq_seqs = apriori_all(sequences, min_support=min_sup)

    print(f"\n  频繁序列 (min_support={min_sup}):")
    for seq, sup in sorted(freq_seqs.items(), key=lambda x: (-x[1], str(x[0]))):
        seq_str = ' → '.join(str(set(s)) for s in seq)
        print(f"    {seq_str}  支持度={sup:.2f}")


# ============================================================
# 3. PrefixSpan算法
# ============================================================
def prefix_span(sequences, min_support=0.5):
    """
    PrefixSpan算法 — 基于前缀投影的序列模式挖掘
    不需要候选生成，直接通过投影数据库递归挖掘
    """
    n = len(sequences)
    min_count = min_support * n
    all_patterns = {}

    def get_projected_db(sequences, prefix_item):
        """构建前缀投影数据库"""
        projected = []
        for seq in sequences:
            # 找到prefix_item在序列中的位置
            for i, itemset in enumerate(seq):
                if prefix_item in itemset:
                    # 投影：去掉匹配项集之前的部分和匹配项本身
                    if len(itemset) == 1:
                        # 整个项集匹配，取之后的部分
                        suffix = seq[i+1:]
                    else:
                        # 部分匹配，当前项集去掉匹配项
                        remaining = [x for x in itemset if x != prefix_item]
                        suffix = [remaining] + seq[i+1:] if remaining else seq[i+1:]
                    if suffix:
                        projected.append(suffix)
                    break
        return projected

    def mine_recursive(prefix, projected_db):
        """递归挖掘"""
        # 统计投影数据库中各项的出现次数
        item_counts = defaultdict(int)
        for seq in projected_db:
            seen_in_seq = set()
            for itemset in seq:
                for item in itemset:
                    if item not in seen_in_seq:
                        item_counts[item] += 1
                        seen_in_seq.add(item)

        # 对满足最小支持度的项递归
        for item, count in sorted(item_counts.items()):
            if count >= min_count:
                new_prefix = prefix + [[item]]
                sup = count / n
                all_patterns[tuple(tuple(p) for p in new_prefix)] = sup

                # 构建新的投影数据库
                new_projected = get_projected_db(projected_db, item)
                if new_projected:
                    mine_recursive(new_prefix, new_projected)

    # 统计所有项的支持度
    item_counts = defaultdict(int)
    for seq in sequences:
        seen = set()
        for itemset in seq:
            for item in itemset:
                if item not in seen:
                    item_counts[item] += 1
                    seen.add(item)

    # 从频繁1-序列开始递归
    for item, count in sorted(item_counts.items()):
        if count >= min_count:
            prefix = [[item]]
            all_patterns[tuple(tuple(p) for p in prefix)] = count / n
            projected = get_projected_db(sequences, item)
            if projected:
                mine_recursive(prefix, projected)

    return all_patterns


def demo_prefix_span():
    """演示PrefixSpan算法"""
    print("\n--- 3. PrefixSpan 序列模式挖掘 ---")

    sequences = [
        [['a'], ['b', 'c'], ['d']],
        [['a'], ['c'], ['b', 'd']],
        [['a', 'b'], ['c'], ['d']],
        [['b'], ['c'], ['d']],
    ]

    min_sup = 0.5
    freq_seqs = prefix_span(sequences, min_support=min_sup)

    # 按序列长度和支持度排序
    sorted_patterns = sorted(freq_seqs.items(), key=lambda x: (len(x[0]), -x[1]))

    print(f"  频繁序列 (min_support={min_sup}):")
    for seq, sup in sorted_patterns:
        seq_str = ' → '.join(str(set(s)) for s in seq)
        print(f"    {seq_str}  支持度={sup:.2f}")


# ============================================================
# 4. 序列规则与应用场景
# ============================================================
def print_applications():
    """展示序列模式挖掘的应用场景"""
    applications = {
        "电商购买行为": "发现用户购买序列中的规律，如'买手机→买手机壳→买贴膜'",
        "Web浏览路径": "分析用户页面访问顺序，优化网站导航设计",
        "金融交易监控": "检测异常交易序列模式，用于反欺诈",
        "医疗病程分析": "挖掘疾病发展过程中的用药序列模式",
        "网络入侵检测": "识别系统日志中的异常操作序列",
    }

    print("\n--- 4. 序列模式挖掘应用场景 ---")
    for domain, desc in applications.items():
        print(f"  • {domain}：{desc}")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print_sequence_concepts()
    print()
    demo_apriori_all()
    demo_prefix_span()
    print_applications()
