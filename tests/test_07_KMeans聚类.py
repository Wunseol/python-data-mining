import importlib
import numpy as np
import pytest
import sys
import os

_module_dir = os.path.join(os.path.dirname(__file__), '..', '07_无监督学习', '01_聚类分析')
sys.path.insert(0, os.path.abspath(_module_dir))
_mod = importlib.import_module('KMeans聚类')
KMeansManual = _mod.KMeansManual


class TestKMeansManual:

    def test_fit_returns_self(self):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        model = KMeansManual(n_clusters=3, random_state=42)
        result = model.fit(X)
        assert result is model

    def test_labels_shape(self):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        model = KMeansManual(n_clusters=3, random_state=42)
        model.fit(X)
        assert model.labels.shape == (30,)

    def test_labels_range(self):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        model = KMeansManual(n_clusters=3, random_state=42)
        model.fit(X)
        assert set(model.labels).issubset({0, 1, 2})

    def test_inertia_positive(self):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        model = KMeansManual(n_clusters=3, random_state=42)
        model.fit(X)
        assert model.inertia_ > 0

    def test_predict_shape(self):
        np.random.seed(42)
        X = np.random.randn(30, 2)
        model = KMeansManual(n_clusters=3, random_state=42)
        model.fit(X)
        predictions = model.predict(X)
        assert predictions.shape == (30,)

    def test_well_separated_clusters(self):
        np.random.seed(42)
        c1 = np.random.randn(20, 2) + [5, 5]
        c2 = np.random.randn(20, 2) + [-5, -5]
        c3 = np.random.randn(20, 2) + [5, -5]
        X = np.vstack([c1, c2, c3])
        model = KMeansManual(n_clusters=3, random_state=42)
        model.fit(X)
        unique_labels = set(model.labels)
        assert len(unique_labels) == 3
