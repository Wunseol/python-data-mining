"""
数据预处理模块 - Data Preprocessing
===================================
涵盖数据挖掘中常见的数据预处理技术：
1. 缺失值处理
2. 异常值检测与处理
3. 数据标准化与归一化
4. 类别型变量编码
5. 数据分割
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split


def create_sample_data():
    """创建含缺失值、异常值、类别变量的示例数据"""
    np.random.seed(42)
    n = 200
    data = {
        'age': np.random.randint(18, 70, n),
        'income': np.random.normal(50000, 15000, n),
        'score': np.random.uniform(0, 100, n),
        'city': np.random.choice(['北京', '上海', '广州', '深圳'], n),
        'education': np.random.choice(['高中', '本科', '硕士', '博士'], n),
        'gender': np.random.choice(['男', '女'], n),
    }
    df = pd.DataFrame(data)

    # 人为添加缺失值
    for col in ['age', 'income', 'score']:
        mask = np.random.random(n) < 0.05
        df.loc[mask, col] = np.nan

    # 人为添加异常值
    outlier_idx = np.random.choice(n, 5, replace=False)
    df.loc[outlier_idx, 'income'] = df.loc[outlier_idx, 'income'] * 5

    return df


# ============================================================
# 1. 缺失值处理
# ============================================================
def handle_missing_values(df, strategy='mean', columns=None):
    """
    缺失值处理
    strategy: 'mean'(均值), 'median'(中位数), 'most_frequent'(众数), 'constant'(常量), 'knn'(KNN填充)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    df_result = df.copy()

    if strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        df_result[columns] = imputer.fit_transform(df_result[columns])
    else:
        imputer = SimpleImputer(strategy=strategy)
        df_result[columns] = imputer.fit_transform(df_result[columns])

    return df_result


# ============================================================
# 2. 异常值检测与处理
# ============================================================
def detect_outliers_iqr(df, column, factor=1.5):
    """基于IQR(四分位距)方法检测异常值"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(df, column, threshold=3):
    """基于Z-Score方法检测异常值"""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = df[z_scores > threshold]
    return outliers


def handle_outliers(df, column, method='clip', factor=1.5):
    """
    异常值处理
    method: 'clip'(截断), 'remove'(删除), 'replace'(用中位数替换)
    """
    df_result = df.copy()
    Q1 = df_result[column].quantile(0.25)
    Q3 = df_result[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    if method == 'clip':
        df_result[column] = df_result[column].clip(lower, upper)
    elif method == 'remove':
        df_result = df_result[(df_result[column] >= lower) & (df_result[column] <= upper)]
    elif method == 'replace':
        median_val = df_result[column].median()
        df_result.loc[(df_result[column] < lower) | (df_result[column] > upper), column] = median_val

    return df_result


# ============================================================
# 3. 数据标准化与归一化
# ============================================================
def scale_features(df, columns=None, method='standard'):
    """
    特征缩放
    method: 'standard'(标准化), 'minmax'(归一化), 'robust'(鲁棒缩放)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    df_result = df.copy()

    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
    }

    scaler = scalers[method]
    df_result[columns] = scaler.fit_transform(df_result[columns])

    print(f"[{method}] 缩放后统计:")
    print(df_result[columns].describe().loc[['mean', 'std', 'min', 'max']])

    return df_result, scaler


# ============================================================
# 4. 类别型变量编码
# ============================================================
def encode_categorical(df, columns=None, method='onehot'):
    """
    类别变量编码
    method: 'onehot'(独热编码), 'label'(标签编码), 'ordinal'(有序编码)
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    df_result = df.copy()

    if method == 'onehot':
        df_result = pd.get_dummies(df_result, columns=columns, drop_first=False)
    elif method == 'label':
        le = LabelEncoder()
        for col in columns:
            df_result[col] = le.fit_transform(df_result[col].astype(str))
    elif method == 'ordinal':
        oe = OrdinalEncoder()
        df_result[columns] = oe.fit_transform(df_result[columns].astype(str))

    return df_result


# ============================================================
# 5. 数据分割
# ============================================================
def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    数据集分割：训练集 / 验证集 / 测试集
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y
    )
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - relative_val_size, random_state=random_state, stratify=y_temp
    )
    print(f"训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("数据预处理完整流程演示")
    print("=" * 60)

    # 创建示例数据
    df = create_sample_data()
    print("\n--- 原始数据概览 ---")
    print(df.info())
    print(f"\n缺失值统计:\n{df.isnull().sum()}")

    # 1. 缺失值处理
    print("\n--- 1. 缺失值处理 (KNN填充) ---")
    df_filled = handle_missing_values(df, strategy='knn')
    print(f"处理后缺失值: {df_filled.isnull().sum().sum()}")

    # 2. 异常值检测与处理
    print("\n--- 2. 异常值检测 (IQR法) ---")
    outliers, low, high = detect_outliers_iqr(df_filled, 'income')
    print(f"检测到异常值: {len(outliers)} 条, 范围: [{low:.0f}, {high:.0f}]")

    df_clean = handle_outliers(df_filled, 'income', method='clip')
    outliers_after, _, _ = detect_outliers_iqr(df_clean, 'income')
    print(f"处理后异常值: {len(outliers_after)} 条")

    # 3. 特征缩放
    print("\n--- 3. 特征标准化 ---")
    df_scaled, scaler = scale_features(df_clean, method='standard')

    # 4. 类别编码
    print("\n--- 4. 类别变量独热编码 ---")
    df_encoded = encode_categorical(df_scaled, method='onehot')
    print(f"编码后特征数: {df_encoded.shape[1]}")

    # 5. 数据分割
    print("\n--- 5. 数据集分割 ---")
    # 构造一个简单的二分类标签
    y = (df_encoded['age'] > df_encoded['age'].median()).astype(int)
    X = df_encoded.drop(columns=['age'])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
