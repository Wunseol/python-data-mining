import importlib
import unittest
import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg')

_apriori_dir = os.path.join(os.path.dirname(__file__), '..', '07_无监督学习', '02_关联规则挖掘', '01_Apriori算法')
sys.path.insert(0, os.path.abspath(_apriori_dir))
_apriori = importlib.import_module('Apriori')
loadDataSet = _apriori.loadDataSet
createC1 = _apriori.createC1
scanD = _apriori.scanD
apriori = _apriori.apriori

_anomaly_dir = os.path.join(os.path.dirname(__file__), '..', '07_无监督学习', '04_异常检测')
sys.path.insert(0, os.path.abspath(_anomaly_dir))
_anomaly = importlib.import_module('异常检测')
detect_zscore = _anomaly.detect_zscore
detect_iqr = _anomaly.detect_iqr


class TestApriori(unittest.TestCase):

    def test_loadDataSet(self):
        data = loadDataSet()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)
        for row in data:
            self.assertIsInstance(row, list)

    def test_createC1(self):
        data = loadDataSet()
        C1 = createC1(data)
        self.assertIsInstance(C1, list)
        for item in C1:
            self.assertIsInstance(item, frozenset)
            self.assertEqual(len(item), 1)

    def test_createC1_unique(self):
        data = [[1, 2, 3], [2, 3, 4]]
        C1 = createC1(data)
        items = set(C1)
        self.assertEqual(len(items), len(C1))

    def test_scanD(self):
        data = loadDataSet()
        C1 = createC1(data)
        D = list(map(set, data))
        L1, supportData = scanD(D, C1, 0.5)
        self.assertIsInstance(L1, list)
        self.assertIsInstance(supportData, dict)
        for key in supportData:
            self.assertGreaterEqual(supportData[key], 0)
            self.assertLessEqual(supportData[key], 1.0)

    def test_apriori_returns_frequent_itemsets(self):
        data = loadDataSet()
        L, supportData = apriori(data, minSupport=0.5)
        self.assertIsInstance(L, list)
        self.assertGreater(len(L), 0)
        self.assertIsInstance(supportData, dict)

    def test_apriori_first_level(self):
        data = loadDataSet()
        L, _ = apriori(data, minSupport=0.5)
        self.assertGreater(len(L[0]), 0)


class TestAnomalyDetection(unittest.TestCase):

    def test_detect_zscore_returns_correct_length(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        labels = detect_zscore(X, threshold=3)
        self.assertEqual(labels.shape, (50,))

    def test_detect_zscore_labels_binary(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        labels = detect_zscore(X, threshold=3)
        self.assertTrue(set(labels).issubset({0, 1}))

    def test_detect_iqr_returns_correct_length(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        labels = detect_iqr(X, factor=1.5)
        self.assertEqual(labels.shape, (50,))

    def test_detect_iqr_labels_binary(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        labels = detect_iqr(X, factor=1.5)
        self.assertTrue(set(labels).issubset({0, 1}))

    def test_detect_zscore_with_outliers(self):
        np.random.seed(42)
        X_normal = np.random.randn(40, 2) * 0.5
        X_outliers = np.array([[10.0, 10.0], [20.0, -20.0]])
        X = np.vstack([X_normal, X_outliers])
        labels = detect_zscore(X, threshold=3)
        self.assertGreater(labels.sum(), 0)
