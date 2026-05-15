"""
自然语言处理模块 - NLP Basics
=================================
涵盖NLP的基础技术：
1. 文本预处理 (分词、去停用词、词干化)
2. 词袋模型与TF-IDF
3. 文本分类
4. 情感分析
5. 主题模型 (LDA)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 文本预处理
# ============================================================
class TextPreprocessor:
    """文本预处理器"""

    def __init__(self):
        # 简单英文停用词表
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        ])

    def clean_text(self, text):
        """清洗文本：去HTML标签、特殊字符、数字"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text

    def tokenize(self, text):
        """简单分词（空格分割）"""
        return text.split()

    def remove_stopwords(self, tokens):
        """去除停用词"""
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def preprocess(self, text):
        """完整预处理流程"""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return tokens


# ============================================================
# 2. 词袋模型 (手动实现)
# ============================================================
class BoWVectorizer:
    """词袋模型向量化器"""

    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocabulary = {}

    def fit(self, documents):
        word_counts = Counter()
        for doc in documents:
            word_counts.update(doc)

        if self.max_features:
            vocab = word_counts.most_common(self.max_features)
        else:
            vocab = word_counts.most_common()

        self.vocabulary = {word: idx for idx, (word, _) in enumerate(vocab)}
        return self

    def transform(self, documents):
        n_docs = len(documents)
        n_features = len(self.vocabulary)
        X = np.zeros((n_docs, n_features))

        for i, doc in enumerate(documents):
            for word in doc:
                if word in self.vocabulary:
                    X[i, self.vocabulary[word]] += 1
        return X

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


# ============================================================
# 3. TF-IDF (手动实现)
# ============================================================
class TfidfVectorizerManual:
    """TF-IDF向量化器"""

    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = None

    def fit(self, documents):
        n_docs = len(documents)
        word_doc_freq = Counter()

        for doc in documents:
            unique_words = set(doc)
            for word in unique_words:
                word_doc_freq[word] += 1

        if self.max_features:
            vocab = word_doc_freq.most_common(self.max_features)
        else:
            vocab = word_doc_freq.most_common()

        self.vocabulary = {word: idx for idx, (word, _) in enumerate(vocab)}
        self.idf = np.zeros(len(self.vocabulary))

        for word, idx in self.vocabulary.items():
            self.idf[idx] = np.log((n_docs + 1) / (word_doc_freq[word] + 1)) + 1

        return self

    def transform(self, documents):
        n_docs = len(documents)
        n_features = len(self.vocabulary)
        X = np.zeros((n_docs, n_features))

        for i, doc in enumerate(documents):
            word_counts = Counter(doc)
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    X[i, idx] = count

            # TF: 词频 / 文档总词数
            if len(doc) > 0:
                X[i] = X[i] / len(doc)

            # TF * IDF
            X[i] = X[i] * self.idf

        return X

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


# ============================================================
# 4. 文本分类 (朴素贝叶斯)
# ============================================================
class NaiveBayesText:
    """朴素贝叶斯文本分类器"""

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        n_docs, n_features = X.shape

        self.class_prior = np.zeros(self.n_classes)
        self.likelihood = np.zeros((self.n_classes, n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_prior[i] = X_c.shape[0] / n_docs
            # 拉普拉斯平滑
            self.likelihood[i] = (X_c.sum(axis=0) + 1) / (X_c.sum() + n_features)

        return self

    def predict(self, X):
        log_prior = np.log(self.class_prior)
        log_likelihood = np.log(self.likelihood)
        scores = X @ log_likelihood.T + log_prior
        return self.classes[np.argmax(scores, axis=1)]


# ============================================================
# 5. 情感分析
# ============================================================
def demo_sentiment_analysis():
    """简单情感分析示例"""
    print("--- 情感分析 ---")

    # 模拟影评数据
    reviews = [
        ("This movie is absolutely wonderful and amazing", "positive"),
        ("Great acting and beautiful cinematography", "positive"),
        ("I loved every moment of this fantastic film", "positive"),
        ("Excellent performance by the entire cast", "positive"),
        ("A masterpiece of modern cinema", "positive"),
        ("Terrible movie, waste of time and money", "negative"),
        ("Awful plot and horrible acting", "negative"),
        ("The worst film I have ever seen", "negative"),
        ("Boring and disappointing experience", "negative"),
        ("Do not watch this terrible movie", "negative"),
    ]

    preprocessor = TextPreprocessor()

    # 预处理
    documents = [preprocessor.preprocess(text) for text, _ in reviews]
    labels = np.array([1 if label == "positive" else 0 for _, label in reviews])

    # TF-IDF向量化
    tfidf = TfidfVectorizerManual(max_features=100)
    X = tfidf.fit_transform(documents)

    # 训练朴素贝叶斯
    clf = NaiveBayesText()
    clf.fit(X, labels)

    # 测试
    test_reviews = [
        "This is a wonderful and amazing movie",
        "Terrible and boring waste of time",
        "Great performance and beautiful story",
        "Awful and horrible experience",
    ]
    for review in test_reviews:
        tokens = preprocessor.preprocess(review)
        X_test = tfidf.transform([tokens])
        pred = clf.predict(X_test)[0]
        sentiment = "正面" if pred == 1 else "负面"
        print(f"  '{review}' -> {sentiment}")


# ============================================================
# 6. 主题模型 (LDA)
# ============================================================
def demo_topic_modeling():
    """LDA主题模型"""
    print("\n--- LDA主题模型 ---")
    try:
        from sklearn.feature_extraction.text import CountVectorizer as SkCountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        documents = [
            "Machine learning algorithms process data to find patterns",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing analyzes text data",
            "Computer vision processes images and video data",
            "Data mining discovers patterns in large datasets",
            "Neural networks learn representations from data",
            "Text classification categorizes documents automatically",
            "Image recognition identifies objects in pictures",
            "Statistical learning theory provides mathematical foundations",
            "Recurrent neural networks handle sequential data",
        ]

        vectorizer = SkCountVectorizer(max_features=50, stop_words='english')
        X = vectorizer.fit_transform(documents)

        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-5:]]
            print(f"  主题{topic_idx}: {', '.join(top_words)}")

    except ImportError:
        print("  [提示] 需安装sklearn以使用LDA")


# ============================================================
# 7. 词频可视化
# ============================================================
def plot_word_frequency(documents, top_n=20):
    """词频统计与可视化"""
    all_words = [word for doc in documents for word in doc]
    word_counts = Counter(all_words).most_common(top_n)

    words, counts = zip(*word_counts)
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(words)), counts, color='steelblue')
    plt.yticks(range(len(words)), words)
    plt.xlabel('频次')
    plt.title(f'词频Top-{top_n}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('词频统计.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序演示
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("自然语言处理(NLP)完整流程演示")
    print("=" * 60)

    # 示例文本数据
    texts = [
        "Data mining is the process of discovering patterns in large datasets",
        "Machine learning algorithms learn from training data",
        "Natural language processing helps computers understand human language",
        "Deep learning uses multi-layer neural networks for feature learning",
        "Text classification automatically categorizes documents",
        "Sentiment analysis determines the emotional tone of text",
        "Topic modeling discovers abstract topics in document collections",
        "Word embeddings represent words as dense numerical vectors",
        "Named entity recognition identifies proper nouns in text",
        "Language models predict the probability of word sequences",
    ]

    preprocessor = TextPreprocessor()

    # 1. 预处理
    print("\n--- 1. 文本预处理 ---")
    documents = [preprocessor.preprocess(text) for text in texts]
    for i, (orig, tokens) in enumerate(zip(texts[:3], documents[:3])):
        print(f"  原文: {orig}")
        print(f"  分词: {tokens}\n")

    # 2. 词袋模型
    print("--- 2. 词袋模型 ---")
    bow = BoWVectorizer(max_features=30)
    X_bow = bow.fit_transform(documents)
    print(f"  词袋矩阵形状: {X_bow.shape}")
    print(f"  词汇表大小: {len(bow.vocabulary)}")

    # 3. TF-IDF
    print("\n--- 3. TF-IDF ---")
    tfidf = TfidfVectorizerManual(max_features=30)
    X_tfidf = tfidf.fit_transform(documents)
    print(f"  TF-IDF矩阵形状: {X_tfidf.shape}")

    # 4. 词频可视化
    print("\n--- 4. 词频可视化 ---")
    plot_word_frequency(documents, top_n=15)

    # 5. 情感分析
    demo_sentiment_analysis()

    # 6. 主题模型
    demo_topic_modeling()
