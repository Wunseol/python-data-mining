import importlib
import unittest
import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg')

_knn_dir = os.path.join(os.path.dirname(__file__), '..', '04_分类算法', '01_K近邻算法')
sys.path.insert(0, os.path.abspath(_knn_dir))
_knn = importlib.import_module('K近邻算法')
classify0 = _knn.classify0
autoNorm = _knn.autoNorm
createDataSet = _knn.createDataSet

_nb_dir = os.path.join(os.path.dirname(__file__), '..', '04_分类算法', '02_朴素贝叶斯')
sys.path.insert(0, os.path.abspath(_nb_dir))
_nb = importlib.import_module('朴素贝叶斯算法')
createVocabList = _nb.createVocabList
setOfWords2Vec = _nb.setOfWords2Vec
bagOfWords2VecMN = _nb.bagOfWords2VecMN
loadDataSet_nb = _nb.loadDataSet

_tree_dir = os.path.join(os.path.dirname(__file__), '..', '04_分类算法', '03_决策树', '01_ID3决策树')
sys.path.insert(0, os.path.abspath(_tree_dir))
_trees = importlib.import_module('trees')
calcShannonEnt = _trees.calcShannonEnt
splitDataSet = _trees.splitDataSet
chooseBestFeatureToSplit = _trees.chooseBestFeatureToSplit
createDataSet_tree = _trees.createDataSet


class TestKNN(unittest.TestCase):

    def test_classify0_simple(self):
        group, labels = createDataSet()
        result = classify0([0.2, 0.2], group, labels, 3)
        self.assertIn(result, ['A', 'B'])

    def test_classify0_near_A(self):
        group, labels = createDataSet()
        result = classify0([1.0, 1.0], group, labels, 3)
        self.assertEqual(result, 'A')

    def test_classify0_near_B(self):
        group, labels = createDataSet()
        result = classify0([0.0, 0.0], group, labels, 3)
        self.assertEqual(result, 'B')

    def test_autoNorm_ranges(self):
        np.random.seed(42)
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        norm_data, ranges, min_vals = autoNorm(data)
        self.assertTrue(np.all(norm_data >= 0))
        self.assertTrue(np.all(norm_data <= 1.0 + 1e-10))
        np.testing.assert_array_almost_equal(ranges, [2.0, 20.0])

    def test_autoNorm_min_vals(self):
        data = np.array([[5.0, 50.0], [10.0, 100.0], [15.0, 150.0]])
        _, ranges, min_vals = autoNorm(data)
        np.testing.assert_array_almost_equal(min_vals, [5.0, 50.0])


class TestNaiveBayes(unittest.TestCase):

    def test_createVocabList(self):
        docs = [['hello', 'world'], ['hello', 'python']]
        vocab = createVocabList(docs)
        self.assertIsInstance(vocab, list)
        self.assertEqual(len(vocab), 3)
        self.assertEqual(set(vocab), {'hello', 'world', 'python'})

    def test_setOfWords2Vec(self):
        vocab = ['a', 'b', 'c', 'd']
        result = setOfWords2Vec(vocab, ['a', 'c'])
        self.assertEqual(result, [1, 0, 1, 0])

    def test_setOfWords2Vec_all_present(self):
        vocab = ['x', 'y', 'z']
        result = setOfWords2Vec(vocab, ['x', 'y', 'z'])
        self.assertEqual(result, [1, 1, 1])

    def test_bagOfWords2VecMN(self):
        vocab = ['a', 'b', 'c']
        result = bagOfWords2VecMN(vocab, ['a', 'a', 'b'])
        self.assertEqual(result, [2, 1, 0])

    def test_bagOfWords2VecMN_no_duplicates(self):
        vocab = ['a', 'b', 'c']
        result = bagOfWords2VecMN(vocab, ['a', 'b'])
        self.assertEqual(result, [1, 1, 0])

    def test_loadDataSet(self):
        posts, classes = loadDataSet_nb()
        self.assertEqual(len(posts), 6)
        self.assertEqual(len(classes), 6)
        self.assertEqual(len(posts[0]), 7)


class TestDecisionTree(unittest.TestCase):

    def test_calcShannonEnt_uniform(self):
        data = [[1, 1, 'yes'], [1, 1, 'no']]
        ent = calcShannonEnt(data)
        self.assertAlmostEqual(ent, 1.0, places=5)

    def test_calcShannonEnt_pure(self):
        data = [[1, 1, 'yes'], [1, 1, 'yes']]
        ent = calcShannonEnt(data)
        self.assertAlmostEqual(ent, 0.0, places=5)

    def test_splitDataSet(self):
        data = [[1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no']]
        result = splitDataSet(data, 0, 1)
        self.assertEqual(len(result), 2)
        for row in result:
            self.assertEqual(len(row), 2)

    def test_splitDataSet_empty(self):
        data = [[1, 1, 'yes'], [1, 0, 'no']]
        result = splitDataSet(data, 0, 0)
        self.assertEqual(len(result), 0)

    def test_chooseBestFeatureToSplit(self):
        data, _ = createDataSet_tree()
        best = chooseBestFeatureToSplit(data)
        self.assertIsInstance(best, int)
        self.assertGreaterEqual(best, 0)

    def test_chooseBestFeatureToSplit_known(self):
        data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
        best = chooseBestFeatureToSplit(data)
        self.assertEqual(best, 0)
