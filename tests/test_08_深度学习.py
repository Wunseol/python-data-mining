import importlib
import unittest
import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg')

_ae_dir = os.path.join(os.path.dirname(__file__), '..', '08_深度学习', '03_自编码器与生成模型')
sys.path.insert(0, os.path.abspath(_ae_dir))
_ae = importlib.import_module('自编码器与VAE')
AutoencoderManual = _ae.AutoencoderManual
min_max_scale = _ae.min_max_scale

_attn_dir = os.path.join(os.path.dirname(__file__), '..', '08_深度学习', '05_Transformer与注意力机制')
sys.path.insert(0, os.path.abspath(_attn_dir))
_attn = importlib.import_module('Transformer与注意力机制')
scaled_dot_product_attention = _attn.scaled_dot_product_attention
positional_encoding = _attn.positional_encoding


class TestAutoencoderManual(unittest.TestCase):

    def test_init(self):
        ae = AutoencoderManual(input_dim=10, hidden_dim=8, latent_dim=2, random_state=42)
        self.assertEqual(ae.input_dim, 10)
        self.assertEqual(ae.hidden_dim, 8)
        self.assertEqual(ae.latent_dim, 2)

    def test_encode_shape(self):
        ae = AutoencoderManual(input_dim=10, hidden_dim=8, latent_dim=2, random_state=42)
        np.random.seed(42)
        X = np.random.rand(20, 10)
        X_scaled, _, _ = min_max_scale(X)
        Z = ae.encode(X_scaled)
        self.assertEqual(Z.shape, (20, 2))

    def test_decode_shape(self):
        ae = AutoencoderManual(input_dim=10, hidden_dim=8, latent_dim=2, random_state=42)
        Z = np.random.randn(20, 2)
        X_hat = ae.decode(Z)
        self.assertEqual(X_hat.shape, (20, 10))

    def test_reconstruct_shape(self):
        ae = AutoencoderManual(input_dim=10, hidden_dim=8, latent_dim=2, random_state=42)
        np.random.seed(42)
        X = np.random.rand(20, 10)
        X_scaled, _, _ = min_max_scale(X)
        X_hat = ae.reconstruct(X_scaled)
        self.assertEqual(X_hat.shape, (20, 10))

    def test_fit_reduces_loss(self):
        ae = AutoencoderManual(input_dim=10, hidden_dim=8, latent_dim=2, random_state=42)
        np.random.seed(42)
        X = np.random.rand(20, 10)
        X_scaled, _, _ = min_max_scale(X)
        ae.fit(X_scaled, epochs=10, lr=0.01)
        self.assertLess(ae.loss_history[-1], ae.loss_history[0])

    def test_output_range(self):
        ae = AutoencoderManual(input_dim=10, hidden_dim=8, latent_dim=2, random_state=42)
        np.random.seed(42)
        X = np.random.rand(20, 10)
        X_scaled, _, _ = min_max_scale(X)
        X_hat = ae.reconstruct(X_scaled)
        self.assertTrue(np.all(X_hat >= 0))
        self.assertTrue(np.all(X_hat <= 1.0 + 1e-6))


class TestScaledDotProductAttention(unittest.TestCase):

    def test_output_shape(self):
        np.random.seed(42)
        Q = np.random.randn(5, 8)
        K = np.random.randn(5, 8)
        V = np.random.randn(5, 8)
        output, weights = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(output.shape, (5, 8))
        self.assertEqual(weights.shape, (5, 5))

    def test_weights_sum_to_one(self):
        np.random.seed(42)
        Q = np.random.randn(5, 8)
        K = np.random.randn(5, 8)
        V = np.random.randn(5, 8)
        _, weights = scaled_dot_product_attention(Q, K, V)
        row_sums = weights.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(5))

    def test_weights_non_negative(self):
        np.random.seed(42)
        Q = np.random.randn(5, 8)
        K = np.random.randn(5, 8)
        V = np.random.randn(5, 8)
        _, weights = scaled_dot_product_attention(Q, K, V)
        self.assertTrue(np.all(weights >= 0))

    def test_different_dk_dv(self):
        np.random.seed(42)
        Q = np.random.randn(4, 6)
        K = np.random.randn(4, 6)
        V = np.random.randn(4, 10)
        output, weights = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(output.shape, (4, 10))
        self.assertEqual(weights.shape, (4, 4))


class TestPositionalEncoding(unittest.TestCase):

    def test_output_shape(self):
        PE = positional_encoding(seq_len=20, d_model=32)
        self.assertEqual(PE.shape, (20, 32))

    def test_different_params(self):
        PE = positional_encoding(seq_len=10, d_model=16)
        self.assertEqual(PE.shape, (10, 16))

    def test_values_bounded(self):
        PE = positional_encoding(seq_len=50, d_model=64)
        self.assertTrue(np.all(PE >= -1.0 - 1e-10))
        self.assertTrue(np.all(PE <= 1.0 + 1e-10))

    def test_deterministic(self):
        PE1 = positional_encoding(seq_len=20, d_model=32)
        PE2 = positional_encoding(seq_len=20, d_model=32)
        np.testing.assert_array_equal(PE1, PE2)
