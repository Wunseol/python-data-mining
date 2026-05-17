"""
公共工具模块 - Common Utilities
================================
提供项目各模块共用的工具函数：
1. Matplotlib 中文字体配置
2. 非交互后端支持
3. 通用数据生成函数
4. 通用打印辅助函数
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def setup_chinese_font():
    """配置 Matplotlib 中文字体，支持 Windows/macOS/Linux"""
    font_candidates = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    available = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    for font in font_candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            break
    else:
        plt.rcParams['font.sans-serif'] = font_candidates
    plt.rcParams['axes.unicode_minus'] = False


def setup_non_interactive_backend():
    """设置非交互式后端，适用于无 GUI 环境（CI/CD、服务器）"""
    if os.environ.get('MPLBACKEND') is None:
        matplotlib.use('Agg')
    setup_chinese_font()


def generate_classification_data(n_samples=200, n_features=2, n_classes=2, random_state=42):
    """生成分类示例数据"""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=min(n_features, 2), n_redundant=0,
        n_classes=n_classes, random_state=random_state
    )
    return X, y


def generate_regression_data(n_samples=100, n_features=1, noise=10, random_state=42):
    """生成回归示例数据"""
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        noise=noise, random_state=random_state
    )
    return X, y


def generate_cluster_data(n_samples=300, n_centers=3, random_state=42):
    """生成聚类示例数据"""
    from sklearn.datasets import make_blobs
    X, y = make_blobs(
        n_samples=n_samples, centers=n_centers,
        random_state=random_state
    )
    return X, y


def print_section(title, width=60):
    """打印分节标题"""
    print("=" * width)
    print(title)
    print("=" * width)


def print_subsection(title):
    """打印子节标题"""
    print(f"\n--- {title} ---")


setup_chinese_font()
