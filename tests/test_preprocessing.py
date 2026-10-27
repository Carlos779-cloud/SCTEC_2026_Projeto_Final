"""Testes unitários de normalização e pré-processamento."""

import unittest
import numpy as np
from src.preprocessing import normalizar_pixels


class TestPreprocessing(unittest.TestCase):
    def test_normalizar_pixels_faixa(self):
        arr = np.array([[0, 127.5, 255]], dtype=np.float32)
        norm = normalizar_pixels(arr)
        self.assertAlmostEqual(float(norm.min()), 0.0, places=4)
        self.assertAlmostEqual(float(norm.max()), 1.0, places=4)
        self.assertEqual(norm.dtype, np.float32)

    def test_dimensoes_imagem(self):
        arr = np.zeros((28, 28), dtype=np.uint8)
        norm = normalizar_pixels(arr)
        self.assertEqual(norm.shape, (28, 28))


if __name__ == "__main__":
    unittest.main()
