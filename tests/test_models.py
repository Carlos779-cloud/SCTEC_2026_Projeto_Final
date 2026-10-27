"""Testes unitários de saída dos modelos, formato e soma de probabilidades."""

import unittest
import numpy as np
from src.inference import MotorInferencia
from src.database import obter_exemplo_por_digito


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.motor = MotorInferencia.get_instancia()

    def test_todos_modelos_carregados(self):
        esperados = ["Random Forest", "KNN", "SVM", "Rede Neural MLP", "Rede Neural Keras"]
        for nome in esperados:
            self.assertIn(nome, self.motor.modelos)

    def test_inferencia_probabilidades_soma_100(self):
        amostra = obter_exemplo_por_digito(digit=0, split="test")
        self.assertIsNotNone(amostra)

        resultado = self.motor.prever(amostra["vector_784"])
        predicoes = resultado["predicoes"]

        self.assertEqual(len(predicoes), 5)

        for nome, d in predicoes.items():
            self.assertIn("classe_prevista", d)
            self.assertIn(d["classe_prevista"], list(range(10)))
            self.assertTrue(0.0 <= d["confianca_max_pct"] <= 100.0)

            probs = d["probabilidades_pct"]
            self.assertEqual(len(probs), 10)

            # Probabilidades não negativas
            for c, p in probs.items():
                self.assertGreaterEqual(p, 0.0)

            # Soma próxima a 100%
            soma_total = sum(probs.values())
            self.assertAlmostEqual(soma_total, 100.0, delta=1.0)

    def test_consenso_estrutura(self):
        amostra = obter_exemplo_por_digito(digit=1, split="test")
        res = self.motor.prever(amostra["vector_784"])
        consenso = res["consenso"]

        self.assertIn("digito_previsto", consenso)
        self.assertIn("votos", consenso)
        self.assertIn("concordancia_pct", consenso)
        self.assertGreaterEqual(consenso["concordancia_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()
