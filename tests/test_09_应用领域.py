import importlib
import unittest
import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg')

_gnn_dir = os.path.join(os.path.dirname(__file__), '..', '09_应用领域', '04_图与网络挖掘', '02_图神经网络')
sys.path.insert(0, os.path.abspath(_gnn_dir))
_gnn = importlib.import_module('图神经网络')
GCNManual = _gnn.GCNManual

_fl_dir = os.path.join(os.path.dirname(__file__), '..', '09_应用领域', '07_联邦学习与隐私保护')
sys.path.insert(0, os.path.abspath(_fl_dir))
_fl = importlib.import_module('联邦学习与隐私保护')
laplace_mechanism = _fl.laplace_mechanism
gaussian_mechanism = _fl.gaussian_mechanism
split_iid = _fl.split_iid


class TestGCNManual(unittest.TestCase):

    def test_init(self):
        gcn = GCNManual(n_features=8, n_hidden=4, n_classes=3, lr=0.01)
        self.assertEqual(gcn.W1.shape, (8, 4))
        self.assertEqual(gcn.W2.shape, (4, 3))

    def test_normalize_adj_shape(self):
        gcn = GCNManual(n_features=8, n_hidden=4, n_classes=3, lr=0.01)
        A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        A_norm = gcn._normalize_adj(A)
        self.assertEqual(A_norm.shape, (3, 3))

    def test_normalize_adj_symmetric(self):
        gcn = GCNManual(n_features=8, n_hidden=4, n_classes=3, lr=0.01)
        A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        A_norm = gcn._normalize_adj(A)
        np.testing.assert_array_almost_equal(A_norm, A_norm.T)

    def test_forward_shape(self):
        gcn = GCNManual(n_features=4, n_hidden=8, n_classes=2, lr=0.01)
        np.random.seed(42)
        X = np.random.randn(10, 4)
        A = np.random.randint(0, 2, (10, 10))
        A = ((A + A.T) > 0).astype(float)
        output = gcn.forward(X, A)
        self.assertEqual(output.shape, (10, 2))

    def test_forward_output_probabilities(self):
        gcn = GCNManual(n_features=4, n_hidden=8, n_classes=2, lr=0.01)
        np.random.seed(42)
        X = np.random.randn(10, 4)
        A = np.random.randint(0, 2, (10, 10))
        A = ((A + A.T) > 0).astype(float)
        output = gcn.forward(X, A)
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1.0 + 1e-6))
        np.testing.assert_array_almost_equal(output.sum(axis=1), np.ones(10))


class TestLaplaceMechanism(unittest.TestCase):

    def test_adds_noise(self):
        np.random.seed(42)
        value = np.array([1.0, 2.0, 3.0])
        result = laplace_mechanism(value, sensitivity=1.0, epsilon=1.0)
        self.assertFalse(np.allclose(result, value))

    def test_preserves_shape(self):
        np.random.seed(42)
        value = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = laplace_mechanism(value, sensitivity=1.0, epsilon=1.0)
        self.assertEqual(result.shape, value.shape)

    def test_high_epsilon_less_noise(self):
        np.random.seed(42)
        value = np.array([5.0])
        results_low_eps = []
        results_high_eps = []
        for _ in range(1000):
            r_low = laplace_mechanism(value, sensitivity=1.0, epsilon=0.1)
            results_low_eps.append(np.abs(r_low[0] - value[0]))
            r_high = laplace_mechanism(value, sensitivity=1.0, epsilon=10.0)
            results_high_eps.append(np.abs(r_high[0] - value[0]))
        self.assertGreater(np.mean(results_low_eps), np.mean(results_high_eps))


class TestGaussianMechanism(unittest.TestCase):

    def test_adds_noise(self):
        np.random.seed(42)
        value = np.array([1.0, 2.0, 3.0])
        result = gaussian_mechanism(value, sensitivity=1.0, epsilon=1.0, delta=1e-5)
        self.assertFalse(np.allclose(result, value))

    def test_preserves_shape(self):
        np.random.seed(42)
        value = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = gaussian_mechanism(value, sensitivity=1.0, epsilon=1.0, delta=1e-5)
        self.assertEqual(result.shape, value.shape)

    def test_high_epsilon_less_noise(self):
        np.random.seed(42)
        value = np.array([5.0])
        results_low_eps = []
        results_high_eps = []
        for _ in range(1000):
            r_low = gaussian_mechanism(value, sensitivity=1.0, epsilon=0.1, delta=1e-5)
            results_low_eps.append(np.abs(r_low[0] - value[0]))
            r_high = gaussian_mechanism(value, sensitivity=1.0, epsilon=10.0, delta=1e-5)
            results_high_eps.append(np.abs(r_high[0] - value[0]))
        self.assertGreater(np.mean(results_low_eps), np.mean(results_high_eps))


class TestSplitIID(unittest.TestCase):

    def test_returns_correct_number_of_clients(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_list, y_list = split_iid(X, y, n_clients=5)
        self.assertEqual(len(X_list), 5)
        self.assertEqual(len(y_list), 5)

    def test_total_samples_preserved(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_list, y_list = split_iid(X, y, n_clients=5)
        total = sum(len(x) for x in X_list)
        self.assertEqual(total, 100)

    def test_feature_dim_preserved(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_list, y_list = split_iid(X, y, n_clients=5)
        for x in X_list:
            self.assertEqual(x.shape[1], 5)
