import importlib
import numpy as np
import pytest
import sys
import os

_module_dir = os.path.join(os.path.dirname(__file__), '..', '03_回归分析')
sys.path.insert(0, os.path.abspath(_module_dir))
_lr = importlib.import_module('01_线性回归')
SimpleLinearRegression = _lr.SimpleLinearRegression
LinearRegressionGD = _lr.LinearRegressionGD


class TestSimpleLinearRegression:

    def test_perfect_line(self):
        X = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 * X + 1
        model = SimpleLinearRegression()
        model.fit(X, y)
        assert abs(model.w - 2.0) < 1e-6
        assert abs(model.b - 1.0) < 1e-6

    def test_r2_perfect(self):
        X = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 3 * X + 2
        model = SimpleLinearRegression()
        model.fit(X, y)
        assert abs(model.r2(X, y) - 1.0) < 1e-6

    def test_predict(self):
        X = np.array([1, 2, 3, 4, 5], dtype=float)
        y = 2 * X + 1
        model = SimpleLinearRegression()
        model.fit(X, y)
        y_pred = model.predict(np.array([6.0]))
        assert abs(y_pred[0] - 13.0) < 1e-6


class TestLinearRegressionGD:

    def test_convergence(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        true_W = np.array([3.0, -1.0])
        y = X @ true_W + 2.0
        model = LinearRegressionGD(lr=0.01, n_iters=5000)
        model.fit(X, y)
        assert abs(model.W[0] - true_W[0]) < 0.5
        assert abs(model.W[1] - true_W[1]) < 0.5
        assert abs(model.b - 2.0) < 0.5

    def test_loss_decreases(self):
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2 * X[:, 0] + 1
        model = LinearRegressionGD(lr=0.01, n_iters=500)
        model.fit(X, y)
        assert model.losses[-1] < model.losses[0]

    def test_predict_shape(self):
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model = LinearRegressionGD(lr=0.01, n_iters=100)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (20,)
