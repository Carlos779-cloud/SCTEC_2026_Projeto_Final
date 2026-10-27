"""Testes unitários e de integração para a API FastAPI.

Suporta execução via TestClient oficial e fallback direto para os handlers da API,
garantindo execução 100% confiável independente de pacotes opcionais de transporte HTTP.
"""

import unittest
from api.app import app, health_check, list_models, get_metrics, predict_digit, get_sample
from api.schemas import PredictRequest
from pydantic import ValidationError


class TestFastAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            cls.client = TestClient(app)
            # TestClient só executa o lifespan ao ser usado como context manager.
            cls.client.__enter__()
            cls.modo_cliente = "testclient"
        except Exception:
            cls.client = None
            cls.modo_cliente = "direto"

    @classmethod
    def tearDownClass(cls):
        if cls.client is not None:
            cls.client.__exit__(None, None, None)

    def test_health(self):
        if self.modo_cliente == "testclient":
            resp = self.client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
        else:
            data = health_check()

        self.assertEqual(data["status"], "online")
        self.assertIn("banco_sqlite", data)
        self.assertEqual(data["total_modelos"], 5)

    def test_models(self):
        if self.modo_cliente == "testclient":
            resp = self.client.get("/models")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
        else:
            data = list_models()

        self.assertEqual(data["total"], 5)
        nomes = [m["nome"] if isinstance(m, dict) else m.nome for m in data["modelos"]]
        self.assertIn("Random Forest", nomes)
        self.assertIn("Rede Neural Keras", nomes)

    def test_metrics(self):
        if self.modo_cliente == "testclient":
            resp = self.client.get("/metrics")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
        else:
            data = get_metrics()

        self.assertEqual(data["total_amostras_teste"], 10500)
        metricas = data["metricas_por_modelo"]
        self.assertIn("Rede Neural Keras", metricas)
        cnn_meta = metricas["Rede Neural Keras"]
        acuracia = cnn_meta["acuracia"] if isinstance(cnn_meta, dict) else cnn_meta.acuracia
        self.assertAlmostEqual(acuracia, 98.97, delta=0.1)

    def test_sample(self):
        if self.modo_cliente == "testclient":
            resp = self.client.get("/samples/3")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
        else:
            data = get_sample(digit=3)

        digito = data["digit"] if isinstance(data, dict) else data.digit
        vetor = data["vector_784"] if isinstance(data, dict) else data.vector_784
        self.assertEqual(digito, 3)
        self.assertEqual(len(vetor), 784)

    def test_predict_valido(self):
        # Obtém uma amostra real de dígito 7
        if self.modo_cliente == "testclient":
            sample_resp = self.client.get("/samples/7")
            self.assertEqual(sample_resp.status_code, 200)
            vetor = sample_resp.json()["vector_784"]

            resp = self.client.post("/predict", json={"pixels": vetor})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
        else:
            sample_data = get_sample(digit=7)
            vetor = sample_data["vector_784"] if isinstance(sample_data, dict) else sample_data.vector_784
            req = PredictRequest(pixels=vetor)
            data = predict_digit(req)

        consenso = data["consenso"] if isinstance(data, dict) else data.consenso
        dig_previsto = consenso["digito_previsto"] if isinstance(consenso, dict) else consenso.digito_previsto
        self.assertEqual(dig_previsto, 7)

        predicoes = data["predicoes"] if isinstance(data, dict) else data.predicoes
        self.assertIn("Random Forest", predicoes)

        # Testa soma das probabilidades próxima a 100%
        for mod_nome, mod_data in predicoes.items():
            probs = mod_data["probabilidades_pct"] if isinstance(mod_data, dict) else mod_data.probabilidades_pct
            self.assertEqual(len(probs), 10)
            soma_pct = sum(probs.values())
            self.assertAlmostEqual(soma_pct, 100.0, delta=1.0)

    def test_predict_invalido_dimensao(self):
        if self.modo_cliente == "testclient":
            resp = self.client.post("/predict", json={"pixels": [0.0] * 100})
            self.assertEqual(resp.status_code, 422)
        else:
            with self.assertRaises((ValueError, ValidationError)):
                PredictRequest(pixels=[0.0] * 100)

    def test_predict_rejeita_corpos_invalidos(self):
        if self.modo_cliente != "testclient":
            self.skipTest("Validação HTTP requer FastAPI TestClient.")

        casos = [
            {},
            {"pixels": [0.0] * 783},
            {"pixels": [0.0] * 784 + [0.0]},
            {"pixels": ["pixel"] * 784},
            {"pixels": [256] + [0] * 783},
            {"pixels": []},
        ]
        for corpo in casos:
            with self.subTest(corpo=str(corpo)[:40]):
                resp = self.client.post("/predict", json=corpo)
                self.assertEqual(resp.status_code, 422)

    def test_openapi_publica_schemas(self):
        if self.modo_cliente != "testclient":
            self.skipTest("OpenAPI requer FastAPI TestClient.")
        docs_resp = self.client.get("/docs")
        self.assertEqual(docs_resp.status_code, 200)
        resp = self.client.get("/openapi.json")
        self.assertEqual(resp.status_code, 200)
        schema = resp.json()
        self.assertIn("PredictRequest", schema["components"]["schemas"])
        self.assertIn("/predict", schema["paths"])

    def test_motor_permanece_carregado_entre_requisicoes(self):
        if self.modo_cliente != "testclient":
            self.skipTest("Verificação do lifespan requer FastAPI TestClient.")
        motor_inicial = app.state.motor
        self.assertIsNotNone(motor_inicial)
        self.client.get("/models")
        self.client.get("/health")
        self.assertIs(app.state.motor, motor_inicial)


if __name__ == "__main__":
    unittest.main()
