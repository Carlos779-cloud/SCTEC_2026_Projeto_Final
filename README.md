# Classificação de Dígitos Manuscritos com Machine Learning

**Projeto Final — SCTEC 2026**  
**Autor:** Carlos João Reinert  
**Orientação e Engenharia de Software:** SCTEC  

Este projeto desenvolve, avalia e compara múltiplos modelos de Machine Learning e Deep Learning para o reconhecimento de dígitos manuscritos utilizando o consagrado conjunto de dados **MNIST**. A solução reúne um pipeline completo de Visão Computacional (OpenCV), persistência relacional em **SQLite** consultável via **DBeaver**, uma **API REST** de alto desempenho com **FastAPI**, uma aplicação web interativa em **Streamlit** e um **Chatbot local** especializado no próprio projeto.

---

## 🎯 Objetivo e Destaques do Projeto

- **Modelos de Ponta a Ponta:** Comparação rigorosa de 5 algoritmos supervisionados:
  1. **Random Forest** (Scikit-Learn — 100 árvores)
  2. **K-Nearest Neighbors (KNN)** (Scikit-Learn — $k=3$, ponderação por distância)
  3. **Support Vector Machine (SVM RBF)** (Scikit-Learn — $C=5.0$, calibração probabilística via Softmax)
  4. **Rede Neural MLP** (Scikit-Learn — camadas $128, 64$, ativação ReLU, Adam)
  5. **Rede Neural Convolucional (Keras CNN)** (TensorFlow/Keras — 2 blocos Conv2D + MaxPool + Dropout)
- **Correção Conceitual Rigorosa:** A seleção numérica de um dígito ($0$ a $9$) não realiza inferência sobre o texto digitado; ela busca uma **imagem real de teste MNIST dessa classe no SQLite** e calcula as probabilidades reais sobre a matriz $28 \times 28$.
- **Probabilidades Calibradas:** Todos os modelos retornam probabilidades percentuais para as classes $0$ a $9$ que somam estritamente $100\%$, diferenciando claramente confiança matemática de acurácia.
- **Visão Computacional OpenCV:** Pré-processamento adaptativo de imagens desenhadas no Paint (detecção de fundo, binarização Otsu, enquadramento $20 \times 20$ e alinhamento por Centro de Massa na tela $28 \times 28$).
- **Open-Set Recognition:** Rejeição sob incerteza com limiar de confiança $\ge 50\%$ para atribuir *"Não encontrado"* quando exposto a classes desconhecidas.
- **Banco de Dados SQLite & DBeaver:** Persistência compacta de imagens em campo binário `image_blob` de 784 bytes uint8 e tabela de demonstração administrativa `registros_demo` (dados sintéticos determinísticos com os campos solicitados *"rua, A, B, setor"*).
- **Chatbot Local Privado:** Assistente em linguagem natural que responde dúvidas técnicas sem envio de dados ou dependência de chaves de API externas.

---

## 📁 Estrutura do Repositório

```text
SCTEC_2026_Projeto_final/
│
├── api/                                # Camada de Serviços REST FastAPI
│   ├── __init__.py
│   ├── app.py                         # Endpoints REST (/health, /models, /metrics, /predict, /samples)
│   └── schemas.py                     # Contratos de validação de dados Pydantic
│
├── chatbot/                            # Motor do Assistente Inteligente Local
│   ├── __init__.py
│   └── bot.py                         # Base de conhecimento e correspondência semântica
│
├── data/                               # Persistência Relacional e Scripts SQL
│   ├── mnist.db                       # Banco de dados SQLite estruturado
│   └── queries_dbeaver.sql            # Consultas de exemplo e guia para o DBeaver
│
├── docs/                               # Documentação Oficial Organizada por Fases
│   ├── etapas/
│   │   ├── 00_auditoria.md            # Fase 0: Inventário e diagnóstico
│   │   ├── 01_escopo_arquitetura.md   # Fase 1: Arquitetura e fluxos Mermaid
│   │   ├── 02_dados_banco.md          # Fase 2: Rastreabilidade MNIST e SQLite
│   │   ├── 03_modelos_avaliacao.md    # Fase 3: Modelagem e probabilidades
│   │   ├── 04_api_sqlite.md           # Fase 4: Endpoints FastAPI e persistência
│   │   ├── 05_streamlit_chatbot.md    # Fase 5: Interface Streamlit e Chatbot
│   │   └── 06_testes_entrega.md       # Fase 6: Suíte de testes e validações
│   └── graficos/                      # Figuras e gráficos reais gerados programaticamente
│
├── imagem/                             # Imagens desenhadas no Paint para testes
│   ├── Imagem 1.png a Imagem 12.png
│
├── models/                             # Modelos treinados e persistidos em disco
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── mlp.pkl
│   └── keras_cnn.keras
│
├── outputs/                            # Gráficos consolidados do pipeline
├── scripts/                            # Scripts utilitários de suporte
│   └── popular_banco.py               # População do banco SQLite com imagens reais e demo
│
├── src/                                # Módulos centrais de lógica e ML
│   ├── __init__.py
│   ├── database.py                    # Gerenciador de conexão e queries SQLite
│   ├── inference.py                   # Motor unificado de inferência e calibração
│   ├── eda.py                         # Carregamento e análise exploratória
│   ├── preprocessing.py               # Normalização e divisão estratificada
│   ├── models.py                      # Definição e treinamento dos modelos
│   ├── evaluation.py                  # Métricas e matrizes de confusão
│   └── robustness.py                  # Robustez OpenCV e Open-Set Recognition
│
├── tests/                              # Suíte completa de testes automatizados
│   ├── test_preprocessing.py
│   ├── test_database.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_chatbot.py
│
├── app_streamlit.py                    # Aplicação Web Interativa Streamlit
├── main.py                            # Pipeline completo (Fases 1 a 5)
├── requirements.txt                   # Dependências do projeto
└── README.md                          # Este documento consolidado
```

---

## 🚀 Como Instalar e Executar (Passo a Passo)

### 1. Pré-requisitos
- **Python 3.10 a 3.12** instalado.
- Gerenciador de pacotes `pip`.

### 2. Instalação das Dependências
No terminal do projeto:
```powershell
pip install -r requirements.txt
```

### 3. Inicializar e Popular o Banco de Dados SQLite
Gera o arquivo `data/mnist.db` com 5.000 amostras MNIST reais (BLOBs de 784 bytes) e a tabela demonstrativa `registros_demo`:
```powershell
python scripts/popular_banco.py
```

### 4. Executar os Testes Automatizados
Valida 100% dos componentes (pré-processamento, SQLite, modelos, API e chatbot):
```powershell
python -m unittest discover tests
```
> *Resultado esperado:* 22 testes executados e aprovados com sucesso (`OK`).

### 5. Iniciar os Serviços (Terminais Separados)

#### Terminal 1 — API REST FastAPI:
```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```
- **Documentação Interativa Swagger:** [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **Documentação ReDoc:** [`http://localhost:8000/redoc`](http://localhost:8000/redoc)
- **Health Check:** [`http://localhost:8000/health`](http://localhost:8000/health)

#### Terminal 2 — Interface Web Streamlit:
```powershell
streamlit run app_streamlit.py
```
- **URL Local:** [`http://localhost:8501`](http://localhost:8501)

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

| Posição | Modelo | Acurácia (%) | Precisão Macro (%) | Recall Macro (%) | F1-Score Macro (%) | Tempo Treino (s) | Latência Média (ms/img) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Rede Neural Keras (CNN)** | **98.97%** | **98.98%** | **98.97%** | **98.97%** | 94.25 s | 0.124 ms |
| 🥈 | **Rede Neural MLP** | **97.47%** | **97.47%** | **97.47%** | **97.47%** | 24.64 s | **0.004 ms** ⚡ |
| 🥉 | **SVM (Kernel RBF)** | **97.26%** | **97.26%** | **97.26%** | **97.25%** | 43.39 s | 4.255 ms |
| 4 | **KNN ($k=3$)** | **97.12%** | **97.15%** | **97.12%** | **97.12%** | **0.03 s** | 1.155 ms |
| 5 | **Random Forest** | **96.63%** | **96.63%** | **96.63%** | **96.63%** | 13.68 s | 0.031 ms |

---

## 📜 Rastreabilidade, Fonte e Licença do MNIST

- **Fonte Oficial:** Repositório OpenML (`mnist_784`, ID: 554), derivado do trabalho de Yann LeCun, Corinna Cortes e Christopher J.C. Burges.
- **Licença:** **Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0)**.
- **Diferenciação:** Não confundir com o dataset de brinquedo `digits` ($8 \times 8$) do scikit-learn.

---

## 🎓 Conclusão e Apresentação Acadêmica

O projeto atende integralmente a todas as exigências do **MEGA PROMPT SCTEC**, oferecendo:
1. Reprodutibilidade total com sementes fixas e modelos auditados.
2. Camada corporativa com FastAPI documentada via Swagger e Pydantic.
3. Persistência relacional limpa com SQLite e DBeaver.
4. Interface Streamlit didática e acessível com Chatbot local.
5. Documentação completa por etapas em `docs/etapas/00_auditoria.md` a `06_testes_entrega.md`.