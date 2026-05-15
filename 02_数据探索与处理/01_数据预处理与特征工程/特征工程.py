"""
特征工程模块 - Feature Engineering
===================================
涵盖数据挖掘中常见的特征工程技术：
1. 特征选择 (方差阈值、相关系数、互信息、模型选择)
2. 特征构造 (多项式特征、交叉特征)
3. 特征变换 (对数变换、Box-Cox变换)
4. 降维 (PCA、LDA)
5. 文本特征提取 (TF-IDF、词袋模型)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif,
    mutual_info_classif, RFE
)
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# ============================================================
# 1. 特征选择
# ============================================================
def feature_selection_variance(X, threshold=0.0):
    """基于方差阈值去除低方差特征"""
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    kept = selector.get_support(indices=True)
    print(f"方差阈值选择: {X.shape[1]} -> {X_selected.shape[1]} 特征 (阈值={threshold})")
    return X_selected, kept


def feature_selection_correlation(df, target_col, threshold=0.05):
    """基于与目标变量的相关系数选择特征"""
    corr = df.corr()[target_col].abs().sort_values(ascending=False)
    selected = corr[corr > threshold].index.tolist()
    selected.remove(target_col)
    print(f"相关系数选择: 保留 {len(selected)} 个特征 (阈值={threshold})")
    return selected, corr


def feature_selection_mutual_info(X, y, k=10):
    """基于互信息选择Top-K特征"""
    mi = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=range(X.shape[1])).sort_values(ascending=False)
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    print(f"互信息选择: Top-{k} 特征")
    print(mi_series.head(k))
    return X_selected, mi_series


def feature_selection_rfe(X, y, n_features=10):
    """递归特征消除 (RFE)"""
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    rfe = RFE(estimator, n_features_to_select=n_features)
    X_selected = rfe.fit_transform(X, y)
    print(f"RFE选择: {n_features} 个特征, 排名:")
    print(pd.Series(rfe.ranking_, index=range(X.shape[1])).sort_values().head(n_features))
    return X_selected, rfe


def feature_selection_model(X, y, top_k=10):
    """基于模型(随机森林)的特征重要性选择"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=range(X.shape[1]))
    importance = importance.sort_values(ascending=False)
    top_features = importance.head(top_k).index.tolist()
    print(f"随机森林特征重要性 Top-{top_k}:")
    print(importance.head(top_k))
    return X[:, top_features], importance


# ============================================================
# 2. 特征构造
# ============================================================
def create_polynomial_features(X, degree=2):
    """多项式特征构造"""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    print(f"多项式特征构造: {X.shape[1]} -> {X_poly.shape[1]} 特征 (degree={degree})")
    return X_poly, poly


def create_interaction_features(df, columns):
    """交叉特征构造 — 两两特征相乘"""
    df_result = df.copy()
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a, col_b = columns[i], columns[j]
            df_result[f"{col_a}_x_{col_b}"] = df_result[col_a] * df_result[col_b]
    print(f"交叉特征构造: 新增 {len(columns) * (len(columns) - 1) // 2} 个特征")
    return df_result


# ============================================================
# 3. 特征变换
# ============================================================
def transform_log(df, columns):
    """对数变换 — 处理右偏分布"""
    df_result = df.copy()
    for col in columns:
        df_result[f"{col}_log"] = np.log1p(df_result[col].clip(lower=0))
    print(f"对数变换: 处理 {len(columns)} 列")
    return df_result


def transform_boxcox(X):
    """Box-Cox变换 — 自动寻找最优变换使数据接近正态分布"""
    pt = PowerTransformer(method='box-cox', standardize=True)
    X_transformed = pt.fit_transform(X)
    print("Box-Cox变换完成")
    return X_transformed, pt


def transform_yeojohnson(X):
    """Yeo-Johnson变换 — 支持负值的Box-Cox推广"""
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_transformed = pt.fit_transform(X)
    print("Yeo-Johnson变换完成")
    return X_transformed, pt


# ============================================================
# 4. 降维
# ============================================================
def reduce_pca(X, n_components=0.95):
    """PCA主成分分析降维"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    if isinstance(n_components, float):
        print(f"PCA降维: {X.shape[1]} -> {X_pca.shape[1]} 维 (保留{n_components*100:.0f}%方差)")
    else:
        print(f"PCA降维: {X.shape[1]} -> {X_pca.shape[1]} 维")
    print(f"各主成分方差解释比: {pca.explained_variance_ratio_}")
    return X_pca, pca


def reduce_lda(X, y, n_components=None):
    """LDA线性判别分析降维（有监督）"""
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X, y)
    print(f"LDA降维: {X.shape[1]} -> {X_lda.shape[1]} 维")
    return X_lda, lda


# ============================================================
# 5. 文本特征提取
# ============================================================
def extract_tfidf(texts, max_features=1000):
    """TF-IDF文本特征提取"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    print(f"TF-IDF特征: {X_tfidf.shape[1]} 个词项")
    return X_tfidf, vectorizer, feature_names


def extract_bow(texts, max_features=1000):
    """词袋模型特征提取"""
    vectorizer = CountVectorizer(max_features=max_features)
    X_bow = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    print(f"词袋模型特征: {X_bow.shape[1]} 个词项")
    return X_bow, vectorizer, feature_names


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("特征工程完整流程演示")
    print("=" * 60)

    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    print(f"\n数据集: {X.shape[0]} 样本, {X.shape[1]} 特征")

    # 1. 特征选择
    print("\n--- 1. 方差阈值特征选择 ---")
    X_var, kept_idx = feature_selection_variance(X, threshold=0.5)

    print("\n--- 2. 互信息特征选择 ---")
    X_mi, mi_scores = feature_selection_mutual_info(X, y, k=10)

    print("\n--- 3. 随机森林特征重要性 ---")
    X_rf, importance = feature_selection_model(X, y, top_k=10)

    # 2. 特征构造
    print("\n--- 4. 多项式特征构造 ---")
    X_small = X[:, :5]
    X_poly, poly = create_polynomial_features(X_small, degree=2)

    # 3. 特征变换
    print("\n--- 5. Yeo-Johnson变换 ---")
    X_yj, pt = transform_yeojohnson(X)

    # 4. 降维
    print("\n--- 6. PCA降维 ---")
    X_pca, pca = reduce_pca(X, n_components=0.95)

    print("\n--- 7. LDA降维 ---")
    X_lda, lda = reduce_lda(X, y)

    # 5. 文本特征
    print("\n--- 8. TF-IDF文本特征 ---")
    texts = [
        "数据挖掘是从大量数据中发现有用模式的过程",
        "机器学习是人工智能的一个重要分支",
        "深度学习使用多层神经网络进行特征学习",
        "自然语言处理研究计算机理解人类语言",
        "数据预处理是数据挖掘的重要步骤",
    ]
    X_tfidf, vec, names = extract_tfidf(texts, max_features=20)
