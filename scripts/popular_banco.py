"""Script utilitário para popular o banco de dados SQLite com amostras reais do MNIST e dados sintéticos.

Pode ser executado diretamente via terminal:
    python -m scripts.popular_banco
ou
    python scripts/popular_banco.py
"""

import os
import sys
import numpy as np

# Garante acesso aos módulos de src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import (
    inicializar_banco,
    popular_registros_demo,
    salvar_amostras_mnist_no_banco,
    verificar_integridade_banco,
    DEFAULT_DB_PATH,
)
from src.eda import carregar_dados
from src.preprocessing import normalizar_pixels, dividir_dados_estratificado


def executar_populacao_banco(limite_por_classe_teste=200, limite_por_classe_treino=300, db_path=None):
    caminho = db_path or DEFAULT_DB_PATH
    print(f"\n=======================================================")
    print(f" POPULANDO BANCO DE DADOS SQLITE ({caminho})")
    print(f"=======================================================")

    inicializar_banco(caminho)
    popular_registros_demo(n_registros=120, seed=42, db_path=caminho)

    print("\nCarregando dataset MNIST do OpenML/Cache para extração de amostras...")
    X, y = carregar_dados()

    # Divisão idêntica à do projeto
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados_estratificado(
        X, y, proporcao_teste=0.15, proporcao_val=0.15, random_state=42
    )

    print("\nInserindo amostras de TESTE na tabela 'mnist_images'...")
    salvar_amostras_mnist_no_banco(
        X_test, y_test, split="test", limite_por_classe=limite_por_classe_teste, db_path=caminho
    )

    print("\nInserindo amostras de TREINO na tabela 'mnist_images'...")
    salvar_amostras_mnist_no_banco(
        X_train, y_train, split="train", limite_por_classe=limite_por_classe_treino, db_path=caminho
    )

    status = verificar_integridade_banco(caminho)
    print("\n--- Auditoria do Banco de Dados ---")
    print(f"Status da Integridade : {status['integridade']}")
    print(f"Total Imagens MNIST   : {status['total_mnist_images']:,}")
    print(f"Total Registros Demo  : {status['total_registros_demo']:,}")
    print(f"Arquivo               : {status['db_path']}")
    print("=======================================================\n")
    return status


if __name__ == "__main__":
    executar_populacao_banco()
