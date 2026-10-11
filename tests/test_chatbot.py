"""Testes unitários para o Chatbot local e respostas verificáveis."""

import unittest
from chatbot.bot import ChatbotLocal


class TestChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = ChatbotLocal()

    def test_pergunta_objetivo(self):
        resp = self.bot.responder("Qual é o objetivo deste projeto?")
        self.assertIn("SCTEC 2026", resp)
        self.assertIn("MNIST", resp)

    def test_pergunta_dataset(self):
        resp = self.bot.responder("Fale sobre o dataset MNIST e seu tamanho.")
        self.assertIn("28 × 28", resp)
        self.assertIn("70.000", resp)
        self.assertIn("CC BY-SA 3.0", resp)

    def test_pergunta_modelos(self):
        resp = self.bot.responder("Quais modelos foram implementados no projeto?")
        self.assertIn("Random Forest", resp)
        self.assertIn("KNN", resp)
        self.assertIn("SVM", resp)
        self.assertIn("Rede Neural MLP", resp)
        self.assertIn("Keras CNN", resp)

    def test_pergunta_melhor_modelo(self):
        resp = self.bot.responder("Qual foi o melhor modelo e qual sua acuracia?")
        self.assertIn("98.97%", resp)
        self.assertIn("Keras (CNN)", resp)

    def test_pergunta_api_fastapi(self):
        resp = self.bot.responder("Quais os endpoints da API FastAPI?")
        self.assertIn("/health", resp)
        self.assertIn("/predict", resp)
        self.assertIn("8000", resp)

    def test_pergunta_sqlite_dbeaver(self):
        resp = self.bot.responder("Como funciona o SQLite e o DBeaver com registros_demo?")
        self.assertIn("mnist_images", resp)
        self.assertIn("registros_demo", resp)
        self.assertIn("queries_dbeaver.sql", resp)

    def test_fallback_honesto(self):
        resp = self.bot.responder("Qual é a capital da Austrália e a previsão do tempo para amanhã?")
        self.assertIn("não localizei uma resposta", resp.lower())
        self.assertIn("não invento dados", resp.lower())
        self.assertIn("docs/etapas/", resp)


if __name__ == "__main__":
    unittest.main()
