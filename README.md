# Classificação de Dígitos Manuscritos com Machine Learning

**Projeto Final — SCTEC 2026**

**Entrega Prevista: 03/11/2026**

**Autor:** Carlos João Reinert

**Orientação e Engenharia de Software:** SCTEC

Este projeto desenvolve, avalia e compara múltiplos modelos de Machine Learning e Deep Learning para o reconhecimento de dígitos manuscritos utilizando o consagrado conjunto de dados **MNIST**. A solução reúne um pipeline completo de Visão Computacional (OpenCV), persistência relacional em **SQLite** consultável via **DBeaver**, uma **API REST** de alto desempenho com **FastAPI**, uma aplicação web interativa em **Streamlit** e um **Chatbot local** especializado no próprio projeto.

---

## 🎯 Objetivo e Destaques do Projeto

- **Modelos de Ponta a Ponta:** Comparação rigorosa de 5 algoritmos supervisionados:
  1. **Random Forest** (Scikit-Learn — 100 árvores)
  2. **K-Nearest Neighbors (KNN)** (Scikit-Learn — $k=3$, ponderação por distância)
  3. **Support Vector Machine (SVM RBF)** (Scikit-Learn — $C=5.0$, softmax heurística sobre escores; saída não calibrada)
  4. **Rede Neural MLP** (Scikit-Learn — camadas $128, 64$, ativação ReLU, Adam)
  5. **Rede Neural Convolucional (Keras CNN)** (TensorFlow/Keras — 2 blocos Conv2D + MaxPool + Dropout)
- **Correção Conceitual Rigorosa:** A seleção numérica de um dígito ($0$ a $9$) não realiza inferência sobre o texto digitado; ela busca uma **imagem real de teste MNIST dessa classe no SQLite** e calcula as probabilidades reais sobre a matriz $28 \times 28$.
- **Saídas dos classificadores:** Os modelos retornam valores normalizados para as classes $0$ a $9$. No SVM, a softmax dos escores não é uma probabilidade calibrada; confiança não equivale a acurácia.
- **Visão Computacional OpenCV:** Pré-processamento adaptativo de imagens desenhadas no Paint (detecção de fundo, binarização Otsu, enquadramento $20 \times 20$ e alinhamento por Centro de Massa na tela $28 \times 28$).
- **Experimento de rejeição (Open-Set):** na avaliação de robustez, um Random Forest treinado sem algumas classes rejeita previsões com confiança abaixo de 50%. Esse mecanismo experimental não está integrado às predições normais da API ou do Streamlit.
- **Banco de Dados SQLite & DBeaver:** Persistência compacta de imagens em campo binário `image_blob` de 784 bytes uint8 e tabela de demonstração administrativa `registros_demo` (dados sintéticos determinísticos com os campos solicitados *"rua, A, B, setor"*).
- **Analise por setor:** a pagina Dados e banco do Streamlit mostra as medias de vendas (Valor A) e frete (Valor B), calculadas a partir de dados sinteticos de `registros_demo`.
- **Chatbot local:** responde duvidas tecnicas com base na documentacao do projeto.

---

## Clonar o repositório

Copie e execute este comando no terminal:

```bash
git clone https://github.com/Carlos779-cloud/SCTEC_2026_Projeto_Final.git
cd SCTEC_2026_Projeto_Final
```

## Link do vídeo

[Assistir Vídeo Demo Coloquei um código no link para ele tentar abrir diretamente em alta definição (720p). Se por acaso a imagem ainda parecer um pouco borrada ou desfocada no seu início, basta clicar no ícone da Engrenagem (Configurações) no próprio player do vídeo, ir em Qualidade e selecionar 720p ou 1080p. Muito obrigado!]

(https://youtu.be/r5gGV-pp_1U?vq=hd720)

---

## Estrutura do Repositório

```text
SCTEC_2026_Projeto_final/
|-- api/
|   |-- app.py
|   `-- schemas.py
|-- chatbot/
|   `-- bot.py
|-- data/
|   |-- mnist.db
|   `-- queries_dbeaver.sql
|-- docs/
|   |-- etapas/              # Documentação das fases 0 a 6
|   `-- graficos/            # Gráficos documentados
|-- imagem/                  # Imagens de entrada para análise de robustez
|-- models/
|   `-- .gitkeep             # Modelos treinados ficam fora do Git
|-- outputs/                 # Gráficos e relatórios gerados pelo pipeline
|-- scripts/
|   `-- popular_banco.py
|-- src/                     # EDA, preparação, modelos, inferência e SQLite
|-- tests/                   # Testes unittest
|-- app_streamlit.py
|-- main.py                  # Pipeline de treinamento e avaliação
|-- requirements.txt
`-- README.md
```

Os arquivos dos modelos (`*.pkl` e `*.keras`) são ignorados pelo Git por serem artefatos gerados. Para usar a API ou o Streamlit em um clone novo, primeiro treine os modelos com `python main.py`.

## 🚀 Como Instalar e Executar (Passo a Passo)

### 1. Pré-requisitos
- **Python 3.10 a 3.12** instalado.
- Gerenciador de pacotes `pip`.
- Ambiente virtual recomendado (`venv`).
- As dependências e suas faixas/versões estão declaradas em `requirements.txt`.

### 2. Instalação das Dependências
No terminal do projeto:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Inicializar e Popular o Banco de Dados SQLite
Em um banco vazio, a configuracao padrao insere ate 5.000 imagens MNIST (2.000 de teste e 3.000 de treino), alem dos registros de demonstracao. A rotina acrescenta imagens e nao remove registros existentes; evite executa-la repetidamente para nao duplicar as imagens. O banco incluido contem 15.000 imagens (9.000 treino e 6.000 teste) e 120 registros sinteticos de demonstracao.
```powershell
python scripts/popular_banco.py
```

### 4. Preparar os modelos
Os arquivos serializados ficam em `models/` e são ignorados pelo Git por causa do tamanho. Execute o pipeline antes dos testes e dos serviços. Se os modelos não estiverem disponíveis, o pipeline os treina e salva; os dados são obtidos via OpenML ou do cache local:
```powershell
python main.py
```


### 5. Executar os Testes Automatizados
Executa a suíte automatizada de pré-processamento, SQLite, modelos, API e chatbot:
```powershell
python -m unittest discover -s tests -v
```
> A documentacao da Fase 6 registra 25 testes aprovados em uma execucao anterior. O resultado pode variar apos alteracoes no projeto.

### 6. Iniciar os Serviços (Terminais Separados)

#### Terminal 1 — API REST FastAPI:
```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```
- **Documentação Interativa Swagger:** [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **Documentação ReDoc:** [`http://localhost:8000/redoc`](http://localhost:8000/redoc)
- **Health Check:** [`http://localhost:8000/health`](http://localhost:8000/health)

Exemplo de predição: a entrada é um objeto JSON com `pixels`, uma lista achatada de 784 números em ordem por linha. A API aceita a escala `[0, 1]` ou `[0, 255]`; imagens em arquivo, Base64 ou matriz 28×28 não são aceitas nesse endpoint.

```powershell
$body = @{ pixels = @(0) * 784 } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```

A resposta contém `predicoes` (classe, confiança, probabilidades e latência por modelo), `consenso` e `aviso_metodologico`. Os números dependem da imagem enviada. A saída do SVM é uma softmax heurística dos escores e não deve ser interpretada como probabilidade calibrada.

Rotas disponíveis: `GET /health`, `GET /models`, `GET /metrics`, `GET /samples/{digit}` e `POST /predict`. A entrada inválida retorna HTTP 422; se os modelos não forem carregados, `/predict` retorna HTTP 503. A documentação interativa também está em [`/docs`](http://localhost:8000/docs), e o esquema OpenAPI em [`/openapi.json`](http://localhost:8000/openapi.json).

Trecho de uma resposta real obtida ao enviar a amostra de teste do dígito 7 do SQLite (a latência varia entre execuções):
```json
{
  "predicoes": {
    "Random Forest": {
      "classe_prevista": 7,
      "confianca_max_pct": 99.0,
      "probabilidades_pct": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0, "5": 0.0, "6": 0.0, "7": 99.0, "8": 0.0, "9": 0.0},
      "latencia_ms": 48.34,
      "tipo_saida": "Saída probabilística do modelo"
    }
  },
  "consenso": {"digito_previsto": 7, "votos": "5/5", "concordancia_pct": 100.0},
  "aviso_metodologico": "A confiança não garante acerto. No SVM, a softmax dos escores de decisão não é uma calibração probabilística formal."
}
```

#### Terminal 2 — Interface Web Streamlit:
```powershell
python -m streamlit run app_streamlit.py
```
- **URL Local:** [`http://localhost:8501`](http://localhost:8501). Abra **Dados e banco** para consultar o grafico de medias por setor, feito com dados sinteticos.

---

## 🗄️ Integração com SQLite e DBeaver

O banco relacional está em **`data/mnist.db`**.

### Como Conectar no DBeaver:
1. Abra o **DBeaver** e clique em **Nova Conexão** (`Ctrl + N` ou ícone de tomada).
2. Selecione **SQLite** e avance.
3. No campo **Path**, aponte para:
   - Caminho relativo: `data/mnist.db`
   - Caminho absoluto: `C:\SCTEC_2026_Projeto_final\data\mnist.db`
4. Clique em **Test Connection** e depois em **Concluir**.
5. Abra e execute o script [`data/queries_dbeaver.sql`](data/queries_dbeaver.sql).

### Tabelas Criadas:
- **`mnist_images`**: Contém `id`, `digit` (0-9), `split` (train/test), `width` (28), `height` (28), `image_blob` (784 bytes uint8), `source`, `created_at`.
  - *Decodificação:* No Python, `np.frombuffer(row['image_blob'], dtype=np.uint8).reshape((28, 28))`.
- **`registros_demo`**: Contém `id`, `rua`, `valor_a`, `valor_b`, `setor`, `observacao`, `data_criacao`.
  - *Aviso:* Tabela administrativa sintética gerada deterministicamente com semente fixa (`seed=42`) para atender aos campos solicitados *"rua, A, B, setor"* sem poluir o conjunto MNIST.

---

## 🏆 Tabela Comparativa Oficial de Desempenho (10.500 imagens de teste)

| Posição | Modelo | Acurácia (%) | Precisão ponderada (%) | Recall ponderado (%) | F1-Score ponderado (%) | Tempo Treino (s) | Latência Média (ms/img) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Rede Neural Keras (CNN)** | **98.97%** | **98.98%** | **98.97%** | **98.97%** | 94.25 s | 0.124 ms |
| 🥈 | **Rede Neural MLP** | **97.47%** | **97.47%** | **97.47%** | **97.47%** | 24.64 s | **0.004 ms** ⚡ |
| 🥉 | **SVM (Kernel RBF)** | **97.26%** | **97.26%** | **97.26%** | **97.25%** | 43.39 s | 4.255 ms |
| 4 | **KNN ($k=3$)** | **97.12%** | **97.15%** | **97.12%** | **97.12%** | **0.03 s** | 1.155 ms |
| 5 | **Random Forest** | **96.63%** | **96.63%** | **96.63%** | **96.63%** | 13.68 s | 0.031 ms |

---

## Limitações conhecidas

- A API recebe somente 784 pixels em JSON; não recebe diretamente arquivos ou imagens coloridas.
- A softmax aplicada aos escores do SVM não é calibração probabilística formal.
- O banco pode acumular imagens duplicadas se `scripts/popular_banco.py` for executado mais de uma vez sobre o mesmo arquivo.
- O pipeline depende de obter o conjunto `mnist_784` via OpenML ou do cache local.

---

## 📜 Rastreabilidade, Fonte e Licença do MNIST

- **Fonte Oficial:** Repositório OpenML (`mnist_784`, ID: 554), derivado do trabalho de Yann LeCun, Corinna Cortes e Christopher J.C. Burges.
- **Licença:** **Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0)**.
- **Diferenciação:** Não confundir com o dataset de brinquedo `digits` ($8 \times 8$) do scikit-learn.
---

## 🎓 Conclusão e Apresentação Acadêmica

As métricas de precisão, recall e F1 deste README usam média ponderada (`average="weighted"`), conforme `src/evaluation.py`. Os valores são resultados registrados de uma execução anterior; tempos e latência variam entre computadores.

O projeto reúne os seguintes componentes técnicos:
1. Pipeline reproduzível com sementes fixas e possibilidade de recriar os modelos.
2. Camada corporativa com FastAPI documentada via Swagger e Pydantic.
3. Persistência relacional limpa com SQLite e DBeaver.
4. Interface Streamlit didática e acessível com Chatbot local.
5. Documentação completa por etapas em `docs/etapas/00_auditoria.md` a `06_testes_entrega.md`.
