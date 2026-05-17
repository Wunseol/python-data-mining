import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    setup_chinese_font,
    setup_non_interactive_backend,
    generate_classification_data,
    generate_regression_data,
    generate_cluster_data,
    print_section,
    print_subsection,
)


class TestSetupChineseFont:

    def test_rcparams_set(self):
        setup_chinese_font()
        import matplotlib.pyplot as plt
        assert len(plt.rcParams['font.sans-serif']) > 0
        assert plt.rcParams['axes.unicode_minus'] is False


class TestGenerateClassificationData:

    def test_shape(self):
        X, y = generate_classification_data(n_samples=100, n_features=4)
        assert X.shape == (100, 4)
        assert y.shape == (100,)

    def test_reproducible(self):
        X1, y1 = generate_classification_data(random_state=42)
        X2, y2 = generate_classification_data(random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestGenerateRegressionData:

    def test_shape(self):
        X, y = generate_regression_data(n_samples=50, n_features=3)
        assert X.shape == (50, 3)
        assert y.shape == (50,)

    def test_reproducible(self):
        X1, y1 = generate_regression_data(random_state=42)
        X2, y2 = generate_regression_data(random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestGenerateClusterData:

    def test_shape(self):
        X, y = generate_cluster_data(n_samples=150, n_centers=3)
        assert X.shape[0] == 150
        assert y.shape[0] == 150

    def test_centers(self):
        X, y = generate_cluster_data(n_centers=4)
        assert len(set(y)) == 4


class TestPrintHelpers:

    def test_print_section(self, capsys):
        print_section("Test Title")
        captured = capsys.readouterr()
        assert "Test Title" in captured.out

    def test_print_subsection(self, capsys):
        print_subsection("Sub Title")
        captured = capsys.readouterr()
        assert "Sub Title" in captured.out
