import importlib
import unittest
import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg')

_xai_dir = os.path.join(os.path.dirname(__file__), '..', '05_模型评估与调优', '03_可解释AI')
sys.path.insert(0, os.path.abspath(_xai_dir))
_xai = importlib.import_module('可解释AI')
LIMEManual = _xai.LIMEManual
SHAPManual = _xai.SHAPManual
compute_pdp = _xai.compute_pdp

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TestLIMEManual(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(X_train)
        cls.X_test = scaler.transform(X_test)
        cls.model = RandomForestClassifier(n_estimators=50, random_state=42)
        cls.model.fit(cls.X_train, y_train)

    def test_init(self):
        lime = LIMEManual(self.model, self.X_train)
        self.assertEqual(lime.feature_means.shape, (self.X_train.shape[1],))
        self.assertEqual(lime.feature_stds.shape, (self.X_train.shape[1],))

    def test_init_stds_no_zero(self):
        lime = LIMEManual(self.model, self.X_train)
        self.assertTrue(np.all(lime.feature_stds > 0))

    def test_explain_returns_dict(self):
        lime = LIMEManual(self.model, self.X_train)
        result = lime.explain(self.X_test[0], n_samples=100, kernel_width=1.0)
        self.assertIn('feature_contributions', result)
        self.assertIn('intercept', result)
        self.assertIn('weights', result)

    def test_explain_contributions_shape(self):
        lime = LIMEManual(self.model, self.X_train)
        result = lime.explain(self.X_test[0], n_samples=100, kernel_width=1.0)
        self.assertEqual(result['feature_contributions'].shape, (self.X_train.shape[1],))

    def test_explain_intercept_is_scalar(self):
        lime = LIMEManual(self.model, self.X_train)
        result = lime.explain(self.X_test[0], n_samples=100, kernel_width=1.0)
        self.assertIsInstance(result['intercept'], (float, np.floating))


class TestSHAPManual(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(X_train)
        cls.X_test = scaler.transform(X_test)
        cls.model = RandomForestClassifier(n_estimators=50, random_state=42)
        cls.model.fit(cls.X_train, y_train)

    def test_init(self):
        shap = SHAPManual(self.model, self.X_train, n_samples=10)
        self.assertEqual(shap.background.shape[1], self.X_train.shape[1])
        self.assertLessEqual(shap.background.shape[0], 50)

    def test_init_n_samples(self):
        shap = SHAPManual(self.model, self.X_train, n_samples=10)
        self.assertEqual(shap.n_samples, 10)

    def test_explain_returns_dict(self):
        shap = SHAPManual(self.model, self.X_train, n_samples=5)
        result = shap.explain(self.X_test[0])
        self.assertIn('shap_values', result)
        self.assertIn('base_value', result)

    def test_explain_shap_values_shape(self):
        shap = SHAPManual(self.model, self.X_train, n_samples=5)
        result = shap.explain(self.X_test[0])
        self.assertEqual(result['shap_values'].shape, (self.X_train.shape[1],))

    def test_explain_base_value_is_scalar(self):
        shap = SHAPManual(self.model, self.X_train, n_samples=5)
        result = shap.explain(self.X_test[0])
        self.assertIsInstance(result['base_value'], (float, np.floating))


class TestComputePDP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = load_breast_cancer()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        cls.X_train = scaler.fit_transform(X_train)
        cls.X_test = scaler.transform(X_test)
        cls.model = RandomForestClassifier(n_estimators=50, random_state=42)
        cls.model.fit(cls.X_train, y_train)

    def test_pdp_returns_grid_and_values(self):
        grid, pdp_vals = compute_pdp(self.model, self.X_test, feature_idx=0, grid_points=20)
        self.assertEqual(grid.shape, (20,))
        self.assertEqual(pdp_vals.shape, (20,))

    def test_pdp_values_in_range(self):
        grid, pdp_vals = compute_pdp(self.model, self.X_test, feature_idx=0, grid_points=20)
        self.assertTrue(np.all(pdp_vals >= 0))
        self.assertTrue(np.all(pdp_vals <= 1))

    def test_pdp_grid_monotonic(self):
        grid, _ = compute_pdp(self.model, self.X_test, feature_idx=0, grid_points=20)
        self.assertTrue(np.all(np.diff(grid) > 0))

    def test_pdp_custom_grid_points(self):
        grid, pdp_vals = compute_pdp(self.model, self.X_test, feature_idx=1, grid_points=10)
        self.assertEqual(len(grid), 10)
        self.assertEqual(len(pdp_vals), 10)

    def test_pdp_different_features(self):
        grid0, pdp0 = compute_pdp(self.model, self.X_test, feature_idx=0, grid_points=10)
        grid1, pdp1 = compute_pdp(self.model, self.X_test, feature_idx=1, grid_points=10)
        self.assertFalse(np.allclose(pdp0, pdp1))
