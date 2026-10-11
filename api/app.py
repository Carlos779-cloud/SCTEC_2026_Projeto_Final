"""API REST FastAPI para o Projeto SCTEC 2026: Classificação de Dígitos MNIST.

Fornece:
- GET  /health          : Estado do serviço, modelos e integridade do banco SQLite.
- GET  /models          : Lista dos 5 modelos e seus metadados.
- GET  /metrics         : Métricas preditivas detalhadas calculadas no teste oficial.
- POST /predict         : Inferência em vetor de 784 pixels com probabilidades percentuais (0-9).
- GET  /samples/{digit} : Recupera exemplo real do conjunto MNIST persistido no SQLite.
"""

import os
import sys
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Garante importações a partir da raiz do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    HealthResponse,
    ModelsResponse,
    ModelInfo,
    MetricsResponse,
    MetricDetail,
    PredictRequest,
    PredictResponse,
    SampleResponse,
)
from src.inference import MotorInferencia
from src.database import verificar_integridade_banco, obter_exemplo_por_digito

app = FastAPI(
    title="API SCTEC MNIST — Classificação de Dígitos Manuscritos",
    description="API REST de alta performance para inferência em modelos de Machine Learning e Deep Learning sobre o dataset MNIST.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Habilita CORS para consumo via frontend web ou Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa singleton do motor de inferência
motor = MotorInferencia.get_instancia()


@app.get("/health", response_model=HealthResponse, summary="Verificação de Saúde e Conectividade")
def health_check():
    """Retorna o estado do serviço, integridade da base SQLite e modelos carregados em memória."""
    try:
        status_db = verificar_integridade_banco()
    except Exception as e:
        status_db = {"status": "erro", "detalhe": str(e)}

    modelos_ativos = list(motor.modelos.keys())

    return {
        "status": "online",
        "versao": "1.0.0",
        "banco_sqlite": status_db,
        "modelos_carregados": modelos_ativos,
        "total_modelos": len(modelos_ativos),
    }


@app.get("/models", response_model=ModelsResponse, summary="Lista de Modelos Disponíveis")
def list_models():
    """Retorna a relação dos 5 modelos de classificação suportados com seus tipos e descrições."""
    lista = []
    for nome, meta in motor.metricas.items():
        tipo_map = {
            "Random Forest": "Ensemble (Árvores)",
            "KNN": "Baseado em Instâncias",
            "SVM": "Kernel RBF",
            "Rede Neural MLP": "Rede Neural Densa",
            "Rede Neural Keras": "Rede Convolucional (CNN)",
        }
        lista.append(
            ModelInfo(
                nome=nome,
                tipo=tipo_map.get(nome, "Classificador"),
                descricao=meta["descricao"],
                acuracia_pct=meta["acuracia"],
                latencia_ms=meta["latencia_ms"],
            )
        )

    return {"modelos": lista, "total": len(lista)}


@app.get("/metrics", response_model=MetricsResponse, summary="Métricas de Avaliação Preditiva")
def get_metrics():
    """Retorna métricas consolidadas (acurácia, precisão, recall, F1-macro e latência) calculadas no conjunto de teste."""
    metricas_formatadas = {}
    for nome, d in motor.metricas.items():
        metricas_formatadas[nome] = MetricDetail(**d)

    return {
        "total_amostras_teste": 10500,
        "metricas_por_modelo": metricas_formatadas,
    }


@app.post("/predict", response_model=PredictResponse, summary="Classificar Imagem de Dígito")
def predict_digit(request: PredictRequest):
    """Recebe um vetor com exatamente 784 pixels e retorna a predição de todos os 5 modelos.
    
    Cada modelo fornece sua distribuição de probabilidade percentual para as 10 classes (0 a 9)
    com soma interna garantida de 100%.
    """
    try:
        resultado = motor.prever(request.pixels)
        return resultado
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de inferência: {str(e)}")


@app.get("/samples/{digit}", response_model=SampleResponse, summary="Obter Amostra Real do SQLite por Dígito")
def get_sample(
    digit: int = Path(..., ge=0, le=9, description="Dígito desejado entre 0 e 9")
):
    """Busca uma imagem real de teste MNIST para o dígito fornecido diretamente do SQLite."""
    amostra = obter_exemplo_por_digito(digit=digit, split="test")
    if not amostra:
        raise HTTPException(status_code=404, detail=f"Nenhuma amostra encontrada para o dígito {digit}.")

    return {
        "id": amostra["id"],
        "digit": amostra["digit"],
        "split": amostra["split"],
        "width": amostra["width"],
        "height": amostra["height"],
        "source": amostra["source"],
        "created_at": str(amostra["created_at"]),
        "vector_784": amostra["vector_784"].tolist(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)
