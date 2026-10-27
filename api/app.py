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
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware

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

logger = logging.getLogger(__name__)


def _motor_da_requisicao(request: Request = None):
    """Obtém o motor do lifespan; mantém compatibilidade com chamadas diretas locais."""
    if request is None:
        return MotorInferencia.get_instancia()
    return getattr(request.app.state, "motor", None)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Carrega os modelos uma vez e mantém o serviço inicializável em caso de falha."""
    application.state.motor = None
    try:
        application.state.motor = MotorInferencia.get_instancia()
    except Exception:
        logger.exception("Falha ao carregar os modelos durante a inicialização")
    yield

app = FastAPI(
    title="API SCTEC MNIST — Classificação de Dígitos Manuscritos",
    description="API REST de alta performance para inferência em modelos de Machine Learning e Deep Learning sobre o dataset MNIST.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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


@app.get("/health", response_model=HealthResponse, summary="Verificação de Saúde e Conectividade")
def health_check(request: Request = None):
    """Retorna o estado do serviço, integridade da base SQLite e modelos carregados em memória."""
    try:
        status_db = verificar_integridade_banco()
    except Exception:
        logger.exception("Falha ao verificar a integridade do SQLite")
        status_db = {"status": "erro"}

    motor = _motor_da_requisicao(request)
    modelos_ativos = list(motor.modelos.keys()) if motor else []
    modelo_disponivel = bool(modelos_ativos)
    banco_disponivel = status_db.get("integridade") == "ok"
    status_db_publico = {chave: valor for chave, valor in status_db.items() if chave != "db_path"}

    return {
        "status": "online" if modelo_disponivel and banco_disponivel else "degraded",
        "versao": "1.0.0",
        "banco_sqlite": status_db_publico,
        "modelos_carregados": modelos_ativos,
        "total_modelos": len(modelos_ativos),
        "modelo_disponivel": modelo_disponivel,
    }


@app.get("/models", response_model=ModelsResponse, summary="Lista de Modelos Disponíveis")
def list_models(request: Request = None):
    """Retorna a relação dos 5 modelos de classificação suportados com seus tipos e descrições."""
    lista = []
    motor = _motor_da_requisicao(request)
    metricas = motor.metricas if motor else MotorInferencia._carregar_metricas_estaticas()
    for nome, meta in metricas.items():
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
def get_metrics(request: Request = None):
    """Retorna métricas consolidadas (acurácia, precisão, recall, F1 ponderado e latência) calculadas no conjunto de teste."""
    metricas_formatadas = {}
    motor = _motor_da_requisicao(request)
    metricas = motor.metricas if motor else MotorInferencia._carregar_metricas_estaticas()
    for nome, d in metricas.items():
        metricas_formatadas[nome] = MetricDetail(**d)

    return {
        "total_amostras_teste": 10500,
        "metricas_por_modelo": metricas_formatadas,
    }


@app.post("/predict", response_model=PredictResponse, summary="Classificar Imagem de Dígito")
def predict_digit(payload: PredictRequest, request: Request = None):
    """Recebe 784 pixels achatados e retorna previsões dos modelos carregados.
    
    Cada modelo fornece sua distribuição de probabilidade percentual para as 10 classes (0 a 9)
    normalizadas para somar aproximadamente 100% após o arredondamento.
    """
    motor = _motor_da_requisicao(request)
    if motor is None:
        raise HTTPException(status_code=503, detail="Modelos de inferência indisponíveis.")
    try:
        resultado = motor.prever(payload.pixels)
        return resultado
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception:
        logger.exception("Falha inesperada durante a inferência")
        raise HTTPException(status_code=500, detail="Falha interna durante a inferência.")


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
