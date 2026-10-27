"""Testes unitários de banco de dados SQLite e codificação/decodificação de BLOB."""

import unittest
import numpy as np
from src.database import (
    verificar_integridade_banco,
    obter_exemplo_por_digito,
    codificar_imagem_para_blob,
    decodificar_blob_para_imagem,
    obter_amostras_paginadas,
)


class TestDatabase(unittest.TestCase):
    def test_integridade_sqlite(self):
        info = verificar_integridade_banco()
        self.assertEqual(info["integridade"], "ok")
        self.assertGreater(info["total_mnist_images"], 0)
        self.assertGreater(info["total_registros_demo"], 0)

    def test_blob_codificacao_e_decodificacao(self):
        img_original = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        blob = codificar_imagem_para_blob(img_original)
        self.assertEqual(len(blob), 784)

        img_recuperada = decodificar_blob_para_imagem(blob, normalizado=False)
        self.assertEqual(img_recuperada.shape, (28, 28))
        np.testing.assert_array_equal(img_original, img_recuperada)

        img_norm = decodificar_blob_para_imagem(blob, normalizado=True)
        self.assertEqual(img_norm.shape, (28, 28))
        self.assertEqual(img_norm.dtype, np.float32)
        self.assertTrue(0.0 <= img_norm.min() <= img_norm.max() <= 1.0)

    def test_busca_exemplo_por_digito(self):
        for digito in range(10):
            exemplo = obter_exemplo_por_digito(digit=digito, split="test")
            self.assertIsNotNone(exemplo)
            self.assertEqual(exemplo["digit"], digito)
            self.assertEqual(exemplo["width"], 28)
            self.assertEqual(exemplo["height"], 28)
            self.assertEqual(len(exemplo["vector_784"]), 784)

    def test_amostras_paginadas(self):
        df_mnist = obter_amostras_paginadas("mnist_images", limite=10, offset=0)
        self.assertEqual(len(df_mnist), 10)
        self.assertIn("blob_bytes", df_mnist.columns)

        df_demo = obter_amostras_paginadas("registros_demo", limite=5, offset=0)
        self.assertEqual(len(df_demo), 5)
        self.assertIn("rua", df_demo.columns)
        self.assertIn("setor", df_demo.columns)


if __name__ == "__main__":
    unittest.main()
