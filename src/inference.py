
"""
Módulo Central de Inferência Unificada — Projeto SCTEC MNIST.

Modelos:
- Random Forest
- KNN
- SVM
- Rede Neural MLP (scikit-learn)
- Rede Neural Keras
"""

import os
import time
import joblib
import numpy as np
import keras

from scipy.special import softmax


class MotorInferencia:
    """Gerencia o carregamento e a inferência dos cinco modelos."""

    _instancia = None

    def __init__(self, diretorio_modelos="models"):
        self.diretorio_modelos = diretorio_modelos
        self.modelos = {}
        self.metricas = self._carregar_metricas_estaticas()
        self._carregar_modelos()

    @classmethod
    def get_instancia(cls, diretorio_modelos="models"):
        """Retorna uma única instância do motor."""
        if cls._instancia is None:
            cls._instancia = cls(diretorio_modelos)
        return cls._instancia

    def _carregar_modelos(self):
        """Carrega os cinco modelos disponíveis."""

        caminhos = {
            "Random Forest": ("random_forest.pkl", "sklearn"),
            "KNN": ("knn.pkl", "sklearn"),
            "SVM": ("svm.pkl", "sklearn_svm"),
            "Rede Neural MLP": ("mlp.pkl", "sklearn"),
            "Rede Neural Keras": ("keras_cnn.keras", "keras"),
        }

        print("\n[MotorInferencia] Carregando modelos...")

        for nome, (arquivo, tipo) in caminhos.items():
            caminho = os.path.join(
                self.diretorio_modelos, arquivo
            )

            if not os.path.exists(caminho):
                print(f"[!] Modelo não encontrado: {caminho}")
                continue

            try:
                if tipo == "keras":
                    modelo = keras.models.load_model(caminho)
                else:
                    modelo = joblib.load(caminho)

                self.modelos[nome] = {
                    "modelo": modelo,
                    "tipo": tipo,
                }

                print(f"[OK] Modelo carregado: {nome}")

            except Exception as erro:
                print(f"[ERRO] Falha ao carregar {nome}: {erro}")

        if not self.modelos:
            raise RuntimeError(
                "Nenhum modelo foi carregado. "
                "Verifique os arquivos da pasta models."
            )

        print(
            f"\nModelos carregados: "
            f"{len(self.modelos)}/5"
        )

    @staticmethod
    def _carregar_metricas_estaticas():
        """Métricas de avaliação previamente registradas."""

        return {
            "Rede Neural Keras": {
                "acuracia": 98.97,
                "precisao_ponderada": 98.98,
                "recall_ponderado": 98.97,
                "f1_ponderado": 98.97,
                "tempo_treino_s": 94.25,
                "latencia_ms": 0.124,
                "amostras_teste": 10500,
                "descricao": (
                    "CNN com Conv2D, MaxPool, Dropout "
                    "e Dense Softmax."
                ),
            },
            "Rede Neural MLP": {
                "acuracia": 97.47,
                "precisao_ponderada": 97.47,
                "recall_ponderado": 97.47,
                "f1_ponderado": 97.47,
                "tempo_treino_s": 24.64,
                "latencia_ms": 0.004,
                "amostras_teste": 10500,
                "descricao": (
                    "Perceptron multicamadas com camadas "
                    "128 e 64, ativação ReLU."
                ),
            },
            "SVM": {
                "acuracia": 97.26,
                "precisao_ponderada": 97.26,
                "recall_ponderado": 97.26,
                "f1_ponderado": 97.25,
                "tempo_treino_s": 43.39,
                "latencia_ms": 4.255,
                "amostras_teste": 10500,
                "descricao": "SVM com kernel RBF.",
            },
            "KNN": {
                "acuracia": 97.12,
                "precisao_ponderada": 97.15,
                "recall_ponderado": 97.12,
                "f1_ponderado": 97.12,
                "tempo_treino_s": 0.03,
                "latencia_ms": 1.155,
                "amostras_teste": 10500,
                "descricao": "K-Nearest Neighbors.",
            },
            "Random Forest": {
                "acuracia": 96.63,
                "precisao_ponderada": 96.63,
                "recall_ponderado": 96.63,
                "f1_ponderado": 96.63,
                "tempo_treino_s": 13.68,
                "latencia_ms": 0.031,
                "amostras_teste": 10500,
                "descricao": "Conjunto de árvores de decisão.",
            },
        }

    def prever(self, imagem_ou_vetor):
        """
        Classifica uma imagem MNIST com os modelos disponíveis.

        Entrada:
            Matriz 28x28 ou vetor com 784 pixels.
            Aceita valores entre 0 e 255 ou entre 0 e 1.

        Retorno:
            Predições, probabilidades, latências e consenso.
        """

        arr = np.asarray(
            imagem_ou_vetor,
            dtype=np.float32,
        )

        if arr.size != 784:
            raise ValueError(
                f"Esperados 784 pixels; recebidos {arr.size}."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(
                "A imagem contém valores NaN ou infinitos."
            )

        if arr.min() < 0 or arr.max() > 255:
            raise ValueError(
                "Os pixels devem estar entre 0 e 255."
            )

        if arr.max() > 1.0:
            arr = arr / 255.0

        vetor_784 = arr.reshape(1, 784)
        vetor_cnn = arr.reshape(1, 28, 28, 1)

        predicoes = {}
        votos_classe = []

        ordem_modelos = [
            "Random Forest",
            "KNN",
            "SVM",
            "Rede Neural MLP",
            "Rede Neural Keras",
        ]

        for nome in ordem_modelos:
            if nome not in self.modelos:
                continue

            info = self.modelos[nome]
            modelo = info["modelo"]
            tipo = info["tipo"]

            inicio = time.perf_counter()

            try:
                if tipo == "sklearn":
                    probs_raw = modelo.predict_proba(
                        vetor_784
                    )[0]
                    classes = modelo.classes_

                elif tipo == "sklearn_svm":
                    decisao = modelo.decision_function(
                        vetor_784
                    )

                    probs_raw = softmax(
                        decisao,
                        axis=1,
                    )[0]

                    classes = modelo.classes_

                elif tipo == "keras":
                    saida = modelo.predict(
                        vetor_cnn,
                        verbose=0,
                    )

                    probs_raw = np.asarray(
                        saida
                    ).reshape(-1)

                    classes = np.arange(
                        len(probs_raw)
                    )

                else:
                    continue

                latencia_ms = (
                    time.perf_counter() - inicio
                ) * 1000.0

                probs_10 = np.zeros(
                    10,
                    dtype=np.float64,
                )

                for classe, prob in zip(
                    classes, probs_raw
                ):
                    classe = int(classe)

                    if 0 <= classe <= 9:
                        probs_10[classe] = float(prob)

                soma = probs_10.sum()

                if soma > 0:
                    probs_10 /= soma
                else:
                    raise ValueError(
                        "O modelo retornou probabilidades inválidas."
                    )

                classe_prevista = int(
                    np.argmax(probs_10)
                )

                confianca = (
                    float(probs_10[classe_prevista])
                    * 100
                )

                probabilidades_pct = {
                    str(d): round(
                        float(probs_10[d] * 100), 2
                    )
                    for d in range(10)
                }

                votos_classe.append(classe_prevista)

                predicoes[nome] = {
                    "classe_prevista": classe_prevista,
                    "confianca_max_pct": round(
                        confianca, 2
                    ),
                    "probabilidades_pct": probabilidades_pct,
                    "latencia_ms": round(
                        latencia_ms, 3
                    ),
                    "tipo_saida": (
                        "Softmax dos escores de decisão; "
                        "não calibrada"
                        if tipo == "sklearn_svm"
                        else "Saída probabilística do modelo"
                    ),
                }

            except Exception as erro:
                print(
                    f"[ERRO] Falha na previsão de "
                    f"{nome}: {erro}"
                )

        if not votos_classe:
            raise RuntimeError(
                "Nenhum modelo conseguiu realizar a previsão."
            )

        valores, contagens = np.unique(
            votos_classe,
            return_counts=True,
        )

        indice = int(np.argmax(contagens))
        digito_consenso = int(valores[indice])
        votos_consenso = int(contagens[indice])

        concordancia = (
            votos_consenso / len(votos_classe)
        ) * 100.0

        return {
            "predicoes": predicoes,
            "consenso": {
                "digito_previsto": digito_consenso,
                "votos": (
                    f"{votos_consenso}/{len(votos_classe)}"
                ),
                "concordancia_pct": round(
                    concordancia, 1
                ),
            },
            "aviso_metodologico": (
                "A confiança não garante acerto. No SVM, "
                "a softmax dos escores de decisão não é "
                "uma calibração probabilística formal."
            ),
        }
