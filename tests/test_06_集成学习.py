import importlib
import unittest
import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg')

_ens_dir = os.path.join(os.path.dirname(__file__), '..', '06_集成学习')
sys.path.insert(0, os.path.abspath(_ens_dir))
_ens = importlib.import_module('集成学习')
BaggingManual = _ens.BaggingManual
AdaBoostManual = _ens.AdaBoostManual

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


class TestBaggingManual(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.X, cls.y = make_classification(
            n_samples=100, n_features=10, random_state=42
        )

    def test_fit_returns_self(self):
        model = BaggingManual(n_estimators=5, random_state=42)
        result = model.fit(self.X, self.y)
        self.assertIs(result, model)

    def test_predict_shape(self):
        model = BaggingManual(n_estimators=5, random_state=42)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (100,))

    def test_accuracy_above_random(self):
        model = BaggingManual(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        acc = accuracy_score(self.y, preds)
        self.assertGreater(acc, 0.5)

    def test_estimators_created(self):
        model = BaggingManual(n_estimators=7, random_state=42)
        model.fit(self.X, self.y)
        self.assertEqual(len(model.estimators_), 7)

    def test_predict_values_binary(self):
        model = BaggingManual(n_estimators=5, random_state=42)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertTrue(set(preds).issubset({0, 1}))


class TestAdaBoostManual(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.X, cls.y = make_classification(
            n_samples=100, n_features=10, random_state=42
        )

    def test_fit_returns_self(self):
        model = AdaBoostManual(n_estimators=10, random_state=42)
        result = model.fit(self.X, self.y)
        self.assertIs(result, model)

    def test_predict_shape(self):
        model = AdaBoostManual(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (100,))

    def test_predict_values_binary(self):
        model = AdaBoostManual(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_estimator_weights_shape(self):
        model = AdaBoostManual(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        self.assertEqual(len(model.estimator_weights_), len(model.estimators_))

    def test_estimator_weights_positive(self):
        model = AdaBoostManual(n_estimators=10, random_state=42)
        model.fit(self.X, self.y)
        self.assertTrue(np.all(model.estimator_weights_ > 0))

    def test_accuracy_above_random(self):
        model = AdaBoostManual(n_estimators=20, random_state=42)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        acc = accuracy_score(self.y, preds)
        self.assertGreater(acc, 0.5)
