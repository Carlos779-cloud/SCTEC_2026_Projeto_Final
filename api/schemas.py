"""Esquemas Pydantic para validação e documentação dos endpoints da API FastAPI."""

import math
from pydantic import BaseModel, Field, StrictFloat, StrictInt, field_validator
from typing import Dict, List, Any


class HealthResponse(BaseModel):
    status: str = Field(..., example="online")
    versao: str = Field(..., example="1.0.0")
    banco_sqlite: Dict[str, Any]
    modelos_carregados: List[str]
    total_modelos: int
    modelo_disponivel: bool


class ModelInfo(BaseModel):
    nome: str
    tipo: str
    descricao: str
    acuracia_pct: float
    latencia_ms: float


class ModelsResponse(BaseModel):
    modelos: List[ModelInfo]
    total: int


class MetricDetail(BaseModel):
    acuracia: float
    precisao_ponderada: float
    recall_ponderado: float
    f1_ponderado: float
    tempo_treino_s: float
    latencia_ms: float
    amostras_teste: int
    descricao: str


class MetricsResponse(BaseModel):
    total_amostras_teste: int
    metricas_por_modelo: Dict[str, MetricDetail]


class PredictRequest(BaseModel):
    pixels: List[StrictFloat | StrictInt] = Field(
        ...,
        min_length=784,
        max_length=784,
        examples=[[0] * 784],
        description="Lista contendo exatamente 784 valores numéricos (pixels) no intervalo [0.0, 1.0] ou [0, 255].",
    )

    @field_validator("pixels")
    @classmethod
    def validar_pixels(cls, v):
        if any(not math.isfinite(pixel) for pixel in v):
            raise ValueError("Os pixels devem ser valores numéricos finitos.")
        if any(pixel < 0 or pixel > 255 for pixel in v):
            raise ValueError("Os pixels devem estar entre 0 e 255.")
        return v


class ModelPrediction(BaseModel):
    classe_prevista: int
    confianca_max_pct: float
    probabilidades_pct: Dict[str, float]
    latencia_ms: float
    tipo_saida: str


class ConsensusInfo(BaseModel):
    digito_previsto: int
    votos: str
    concordancia_pct: float


class PredictResponse(BaseModel):
    predicoes: Dict[str, ModelPrediction]
    consenso: ConsensusInfo
    aviso_metodologico: str


class SampleResponse(BaseModel):
    id: int
    digit: int
    split: str
    width: int
    height: int
    source: str
    created_at: str
    vector_784: List[float]
