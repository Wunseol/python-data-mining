import importlib
import numpy as np
import pytest
import sys
import os

_module_dir = os.path.join(os.path.dirname(__file__), '..', '00_数据挖掘导论')
sys.path.insert(0, os.path.abspath(_module_dir))
_mod = importlib.import_module('数据挖掘导论')
euclidean_distance = _mod.euclidean_distance
manhattan_distance = _mod.manhattan_distance
minkowski_distance = _mod.minkowski_distance
cosine_similarity = _mod.cosine_similarity
jaccard_similarity = _mod.jaccard_similarity
pearson_correlation = _mod.pearson_correlation


class TestDistanceMetrics:

    def test_euclidean_distance_same_vector(self):
        x = np.array([1, 2, 3])
        assert euclidean_distance(x, x) == 0.0

    def test_euclidean_distance_known(self):
        x = np.array([0, 0])
        y = np.array([3, 4])
        assert euclidean_distance(x, y) == 5.0

    def test_manhattan_distance_same_vector(self):
        x = np.array([1, 2, 3])
        assert manhattan_distance(x, x) == 0.0

    def test_manhattan_distance_known(self):
        x = np.array([0, 0])
        y = np.array([3, 4])
        assert manhattan_distance(x, y) == 7.0

    def test_minkowski_p2_equals_euclidean(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        assert abs(minkowski_distance(x, y, p=2) - euclidean_distance(x, y)) < 1e-10

    def test_minkowski_p1_equals_manhattan(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        assert abs(minkowski_distance(x, y, p=1) - manhattan_distance(x, y)) < 1e-10

    def test_cosine_similarity_same_direction(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        assert abs(cosine_similarity(x, y) - 1.0) < 1e-10

    def test_cosine_similarity_opposite(self):
        x = np.array([1, 0])
        y = np.array([-1, 0])
        assert abs(cosine_similarity(x, y) - (-1.0)) < 1e-10

    def test_cosine_similarity_zero_vector(self):
        x = np.array([0, 0])
        y = np.array([1, 2])
        assert cosine_similarity(x, y) == 0.0

    def test_jaccard_similarity_identical(self):
        assert jaccard_similarity([1, 2, 3], [1, 2, 3]) == 1.0

    def test_jaccard_similarity_disjoint(self):
        assert jaccard_similarity([1, 2], [3, 4]) == 0.0

    def test_jaccard_similarity_known(self):
        result = jaccard_similarity([1, 2, 3, 4], [3, 4, 5, 6])
        assert abs(result - 2 / 6) < 1e-10

    def test_jaccard_similarity_empty(self):
        assert jaccard_similarity([], []) == 0.0

    def test_pearson_correlation_perfect_positive(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        assert abs(pearson_correlation(x, y) - 1.0) < 1e-10

    def test_pearson_correlation_perfect_negative(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])
        assert abs(pearson_correlation(x, y) - (-1.0)) < 1e-10

    def test_pearson_correlation_zero_std(self):
        x = np.array([1, 1, 1])
        y = np.array([2, 3, 4])
        assert pearson_correlation(x, y) == 0.0
