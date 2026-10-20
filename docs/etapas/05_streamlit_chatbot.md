# Fase 5 — Interface Streamlit e Chatbot Local

**Projeto:** Classificação de Dígitos Manuscritos com Machine Learning  
**Evento:** SCTEC 2026 — Projeto Final  
**Documento:** `docs/etapas/05_streamlit_chatbot.md`  

---

## 1. Visão Geral da Interface Visual

A aplicação desenvolvida em **Streamlit** (`app_streamlit.py`) oferece uma experiência interativa, didática e em português do Brasil, estruturada em cinco páginas de projeto e sete páginas de documentação de fases:
1. **Visão Geral:** resumo do projeto, métricas registradas e estado dos serviços.
2. **Comparação dos Modelos:** compara acurácia, métricas ponderadas e latência registrada.
3. **Teste por Dígito:** busca uma imagem MNIST real no SQLite e executa a inferência.
4. **Desenhar ou enviar imagem:** processa a imagem com OpenCV e consulta os modelos.
5. **Dados e banco:** consulta as tabelas `mnist_images` e `registros_demo`; inclui o gráfico de médias por setor.

As sete fases documentadas estão disponíveis em páginas próprias. O chatbot local fica no painel lateral.
---

## 2. Diagrama de Fluxo da Interface Streamlit (Mermaid)

```mermaid
flowchart TD
    subgraph UI ["Navegação Streamlit"]
        N1["Visão geral"]
        N2["Comparação dos modelos"]
        N3["Teste por dígito"]
        N4["Desenhar ou enviar imagem"]
        N5["Dados e banco"]
        DOCS["Sete fases documentadas"]
        CHAT["Chatbot no painel lateral"]
    end

    subgraph Processamento ["Serviços e motores"]
        DB["SQLite (data/mnist.db)"]
        INF["MotorInferencia (src/inference.py)"]
        CV["Pré-processamento OpenCV"]
        CB["ChatbotLocal (chatbot/bot.py)"]
    end

    N3 -->|Busca imagem real| DB
    DB -->|Vetor 784| INF
    N4 -->|Imagem Paint ou upload| CV
    CV -->|Imagem 28x28| INF
    N5 -->|Consultas e gráfico por setor| DB
    CHAT -->|Pergunta| CB
    INF -->|Predições e consenso| N3
```

---

## 3. Especificação do Chatbot Local e Fallback

### Principais Temas Suportados pelo Chatbot:
- **Objetivo do Projeto:** Propósito acadêmico do SCTEC 2026, escopo e tecnologias.
- **Dataset MNIST:** 70.000 imagens, resolução $28 \times 28$, balanceamento, licença CC BY-SA 3.0 e diferença para o dataset `digits` 8x8.
- **Modelos e Hiperparâmetros:** Random Forest, KNN, SVM, MLP e Keras CNN.
- **Resultados e Métricas:** Modelo campeão (Keras CNN com 98.97%), modelo mais rápido (MLP com 0.004 ms/img), latências e acurácias.
- **API FastAPI:** Endpoints `/health`, `/models`, `/metrics`, `/predict` e `/samples/{digit}`.
- **SQLite e DBeaver:** Arquivo `data/mnist.db`, tabelas `mnist_images` e `registros_demo`, consultas SQL.
- **Como Executar:** Comandos exatos para instalação e inicialização da API e Streamlit.
- **Correção Conceitual:** Explicação de que digitar um número busca imagem real no teste e não faz predição sobre caractere texto.

### Política de Fallback Honesto:
Se o usuário fizer uma pergunta que fuja do escopo ou que não possa ser verificada com rigor:
> O chatbot informa explicitamente que não possui a informação na base local e **não inventa dados**, direcionando o usuário para os documentos correspondentes em `docs/etapas/` e no `README.md`.

---

## 4. Como Iniciar a Interface Streamlit

Em um terminal dedicado:
```powershell
streamlit run app_streamlit.py
```
A aplicação abrirá automaticamente no navegador em: **`http://localhost:8501`**.
