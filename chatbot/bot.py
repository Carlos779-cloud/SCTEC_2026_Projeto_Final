"""Motor de Chatbot Local para o Projeto SCTEC 2026 MNIST.

Características:
- 100% local, independente de chaves de API externas e sem envio de dados para fora.
- Baseado em base de conhecimento estruturada sobre o projeto, dados MNIST, modelos, métricas, API e SQLite/DBeaver.
- Mecanismo de busca semântica por correspondência de intenções, palavras-chave e similaridade textual.
- Respostas verificáveis e fallback honesto apontando a documentação quando a pergunta fugir do escopo.
"""

import re
import unicodedata


def normalizar_texto(texto):
    """Remove acentos, pontuações e converte para minúsculas para correspondência robusta."""
    if not texto:
        return ""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^\w\s]", " ", texto)
    return " ".join(texto.split())


class ChatbotLocal:
    def __init__(self):
        self.base_conhecimento = self._construir_base()

    def _construir_base(self):
        """Constrói a base de conhecimento estruturada do projeto."""
        return [
            {
                "intencao": "objetivo_projeto",
                "gatilhos": ["objetivo", "sobre o que e", "proposito", "do que se trata", "projeto", "sctec", "escopo"],
                "resposta": (
                    "**Objetivo do Projeto (SCTEC 2026):**\n"
                    "O projeto consiste no desenvolvimento, avaliação e comparação de múltiplos modelos de Machine Learning "
                    "e Deep Learning para classificação de dígitos manuscritos (0 a 9) com o dataset MNIST.\n"
                    "Ele integra um pipeline de Visão Computacional (OpenCV), persistência relacional em SQLite consultável via DBeaver, "
                    "uma API REST construída com FastAPI e esta interface interativa em Streamlit com assistente inteligente local."
                ),
            },
            {
                "intencao": "dataset_mnist",
                "gatilhos": ["mnist", "dataset mnist", "dados", "tamanho do dataset", "dimensao", "pixels", "amostras", "origem", "licenca"],
                "resposta": (
                    "**Sobre o Dataset MNIST:**\n"
                    "- **Dimensões:** Imagens em escala de cinza de 28 × 28 pixels (vetor de 784 características).\n"
                    "- **Amostras:** 70.000 imagens no total (padrão OpenML `mnist_784`), divididas em Treino (70%), Validação (15%) e Teste (15% = 10.500 imagens).\n"
                    "- **Classes:** 10 dígitos (0 a 9), perfeitamente balanceadas (~10% por classe).\n"
                    "- **Licença:** Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0) (Yann LeCun et al.).\n"
                    "> *Atenção:* O MNIST não deve ser confundido com o conjunto 'digits' 8×8 do scikit-learn."
                ),
            },
            {
                "intencao": "modelos_implementados",
                "gatilhos": ["modelos", "quais modelos", "algoritmos", "classificadores", "arquiteturas", "familias", "pytorch", "torch"],
                "resposta": (
                    "**Modelos Implementados no Projeto (5 no total):**\n"
                    "1. **Rede Neural Convolucional (Keras CNN):** 2 blocos Conv2D + MaxPool + Dropout + Dense Softmax — modelo campeão.\n"
                    "2. **Rede Neural MLP:** Perceptron Multicamadas (128, 64 neurônios, ativação ReLU, otimizador Adam) — mais rápido.\n"
                    "3. **SVM (Kernel RBF):** Support Vector Machine com C=5.0 e calibração probabilística via softmax.\n"
                    "4. **KNN (K-Nearest Neighbors):** Classificador baseado em 3 vizinhos com ponderação por distância.\n"
                    "5. **Random Forest:** Ensemble de 100 árvores de decisão balanceadas (Scikit-Learn).\n"
                    "PyTorch não faz parte da implementação atual."
                ),
            },
            {
                "intencao": "resultados_metricas",
                "gatilhos": ["melhor modelo", "acuracia", "campeao", "metricas", "f1 score", "resultado", "desempenho", "latencia", "tempo de treino"],
                "resposta": (
                    "**Resultados de Desempenho no Conjunto de Teste:**\n"
                    "1. 🥇 **Rede Neural Keras (CNN):** **98.97% de acurácia** (F1: 98.97%, latência: 0.124 ms) — *Modelo Campeão*.\n"
                    "2. 🥈 **Rede Neural MLP:** **97.47% de acurácia** — latência ultrarrápida de **0.004 ms** (melhor custo-benefício).\n"
                    "3. 🥉 **SVM (Kernel RBF):** **97.26% de acurácia** (latência: 4.255 ms).\n"
                    "4. **KNN (k=3):** **97.12% de acurácia** (latência: 1.155 ms).\n"
                    "5. **Random Forest:** **96.63% de acurácia** (latência: 0.031 ms).\n"
                ),
            },
            {
                "intencao": "api_fastapi",
                "gatilhos": ["api", "fastapi", "endpoints", "rotas", "docs", "swagger", "post", "get", "predict"],
                "resposta": (
                    "**API REST FastAPI:**\n"
                    "A API roda na porta `8000` e possui documentação interativa em `http://localhost:8000/docs`.\n"
                    "Principais Endpoints:\n"
                    "- `GET /health`: Estado do serviço, modelos ativos e conexão SQLite.\n"
                    "- `GET /models`: Lista dos 5 modelos e descrições técnicas.\n"
                    "- `GET /metrics`: Métricas de acurácia, F1 ponderado e latência de cada modelo.\n"
                    "- `POST /predict`: Recebe vetor de 784 pixels e retorna predições dos 5 modelos com probabilidades percentuais (0-9).\n"
                    "- `GET /samples/{digit}`: Retorna uma imagem real do conjunto MNIST persistida no SQLite."
                ),
            },
            {
                "intencao": "sqlite_dbeaver",
                "gatilhos": ["sqlite", "dbeaver", "banco", "tabela", "blob", "registros_demo", "rua", "setor"],
                "resposta": (
                    "**Banco SQLite e DBeaver:**\n"
                    "- **Arquivo:** `data/mnist.db`.\n"
                    "- **Tabela `mnist_images`:** Armazena imagens em BLOB binário de 784 bytes uint8 (28×28 pixels), dígito, partição (train/test) e índices.\n"
                    "- **Tabela `registros_demo`:** Contém os campos demonstrativos 'rua, A, B, setor' gerados de forma sintética determinística (seed=42) para testes no DBeaver sem misturar com o MNIST.\n"
                    "- **Script SQL de Consultas:** Disponível em `data/queries_dbeaver.sql` com instruções completas de conexão."
                ),
            },
            {
                "intencao": "como_executar",
                "gatilhos": ["como rodar", "executar", "comando", "iniciar", "instalacao", "passo a passo", "start"],
                "resposta": (
                    "**Como Executar o Projeto:**\n"
                    "1. **Instalar Dependências:**\n"
                    "   `pip install -r requirements.txt`\n"
                    "2. **Popular o Banco SQLite:**\n"
                    "   `python scripts/popular_banco.py`\n"
                    "3. **Iniciar a API FastAPI (Terminal 1):**\n"
                    "   `uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload`\n"
                    "4. **Iniciar o Streamlit (Terminal 2):**\n"
                    "   `streamlit run app_streamlit.py`"
                ),
            },
            {
                "intencao": "correcao_conceitual",
                "gatilhos": ["digitar numero", "digitar", "rotulo", "inferencia de numero", "conceitual", "distincao"],
                "resposta": (
                    "**Correção Conceitual Obrigatória:**\n"
                    "Digitar apenas o número '7' (um caractere) não é uma imagem e não possui pixels. "
                    "Um classificador de visão computacional requer uma matriz 28×28 para inferência.\n"
                    "Por isso, no modo de teste por dígito, o número selecionado é usado para buscar uma **imagem real de teste MNIST** "
                    "daquela classe no SQLite/dataset. A predição e as probabilidades são então calculadas legitimamente sobre os pixels dessa imagem real!"
                ),
            },
            {
                "intencao": "robustez_paint",
                "gatilhos": ["paint", "opencv", "classes ausentes", "rejeicao", "nao encontrado", "open set", "externa"],
                "resposta": (
                    "**Robustez e Imagens do Paint:**\n"
                    "- **Experimento Open-Set:** a etapa de robustez testa rejeição abaixo de 50% com um Random Forest treinado sem algumas classes. Esse mecanismo não faz parte das predições normais da API ou do Streamlit.\n"
                    "- **Pipeline OpenCV:** As imagens feitas no Paint são normalizadas por: detecção de fundo, binarização Otsu, enquadramento proporcional 20×20 e centralização por Centro de Massa na tela 28×28."
                ),
            },
            {
                "intencao": "canvas_interativo",
                "gatilhos": ["canvas", "desenhar", "desenho", "pincel", "paint interativo", "tela", "digitar numero", "escrever numero"],
                "resposta": (
                    "**Canvas Interativo de Desenho (Seção 🎨):**\n"
                    "Na seção **'Teste por Desenho / Imagem'**, há uma tela estilo Paint onde você pode desenhar um dígito diretamente com o mouse.\n"
                    "- O componente usado é o `streamlit-drawable-canvas` (fundo preto, traço branco).\n"
                    "- Após desenhar, o traço é capturado como imagem RGBA, processado pelo pipeline OpenCV (binarização Otsu + Centro de Massa) e transformado em um vetor 28×28.\n"
                    "- Os 5 modelos implementados (Keras CNN, MLP, SVM, KNN e Random Forest) fazem a inferência simultaneamente.\n"
                    "- O resultado exibe a imagem original, a matriz 28×28, as previsões e confianças dos modelos, o gráfico de probabilidades e a tabela comparativa."
                ),
            },
            {
                "intencao": "relatorio_pdf",
                "gatilhos": ["pdf", "relatorio", "exportar", "baixar", "documento", "gerar pdf", "download"],
                "resposta": (
                    "**Relatórios PDF:**\n"
                    "O projeto contém o módulo `src/pdf_generator.py`, que monta um relatório técnico com a imagem, o consenso, as previsões dos modelos e as probabilidades. "
                    "No momento, esse gerador ainda não está conectado a um botão de download na interface nem a uma rota da API."
                ),
            },
        ]

    def responder(self, pergunta_usuario):
        """Processa a mensagem do usuário e formula a resposta."""
        texto_norm = normalizar_texto(pergunta_usuario)
        if not texto_norm:
            return "Por favor, digite uma pergunta sobre o projeto MNIST SCTEC."

        stopwords = {
            "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
            "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas",
            "para", "pra", "com", "sem", "sob", "sobre", "e", "ou", "mas", "que", "se",
            "qual", "quais", "como", "onde", "quando", "quem", "por que", "porque",
            "este", "esta", "esse", "essa", "aquele", "aquela", "isto", "isso", "aquilo",
            "me", "te", "se", "lhe", "nos", "vos", "lhes", "meu", "sua", "seu", "suas",
            "amanha", "hoje", "ontem", "tempo"
        }
        melhor_match = None
        maior_pontuacao = 0

        for item in self.base_conhecimento:
            pontos = 0
            for gatilho in item["gatilhos"]:
                gatilho_norm = normalizar_texto(gatilho)
                # Verifica termo exato com word boundaries
                padrao = r"\b" + re.escape(gatilho_norm) + r"\b"
                if re.search(padrao, texto_norm):
                    # Frases completas valem muito mais que termos de uma letra
                    if len(gatilho_norm) >= 3 and gatilho_norm not in stopwords:
                        pontos += len(gatilho_norm.split()) * 5 + 5
                else:
                    palavras_gatilho = set(gatilho_norm.split()) - stopwords
                    palavras_user = set(texto_norm.split()) - stopwords
                    intersecao = palavras_gatilho.intersection(palavras_user)
                    if intersecao:
                        pontos += len(intersecao) * 3

            if pontos > maior_pontuacao:
                maior_pontuacao = pontos
                melhor_match = item

        if melhor_match and maior_pontuacao >= 5:
            return melhor_match["resposta"]

        # Fallback honesto e verificável
        return (
            "Desculpe, não localizei uma resposta exata para sua dúvida na base de conhecimento local do projeto.\n\n"
            "Como assistente técnico verificável, **não invento dados**. Sugiro consultar a documentação oficial organizada por fases:\n"
            "- `docs/etapas/00_auditoria.md` (Auditoria e inventário)\n"
            "- `docs/etapas/01_escopo_arquitetura.md` (Arquitetura e fluxos)\n"
            "- `docs/etapas/02_dados_banco.md` (Dataset MNIST e SQLite)\n"
            "- `docs/etapas/03_modelos_avaliacao.md` (Métricas e hiperparâmetros)\n"
            "- `docs/etapas/04_api_sqlite.md` (FastAPI e DBeaver)\n"
            "- `README.md` (Guia geral do projeto)"
        )
