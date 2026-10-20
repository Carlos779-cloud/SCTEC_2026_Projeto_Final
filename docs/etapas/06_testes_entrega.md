# Fase 6 — Testes Automatizados, Validações e Entrega Final

**Projeto:** Classificação de Dígitos Manuscritos com Machine Learning  
**Evento:** SCTEC 2026 — Projeto Final  
**Documento:** `docs/etapas/06_testes_entrega.md`  

---

## 1. Resumo da Execução dos Testes Automatizados

Uma suíte de **25 testes unitários e de integração** foi executada no ambiente do projeto através do framework `unittest`:

```powershell
python -m unittest discover tests
```

### Resultado da Execução:
- **Total de Testes:** 25
- **Testes Aprovados:** 25 (**100% de Sucesso**)
- **Falhas / Erros:** 0
- **Tempo de Execução:** ~2.90 segundos na execução mais recente deste ambiente; varia conforme o computador.
- **Ciclo de vida da API:** `TestClient` é usado como context manager para executar o `lifespan` e validar o carregamento dos modelos durante a inicialização.

---

## 2. Tabela Detalhada de Verificações e Cobertura

| Módulo de Teste | Arquivo | Casos de Teste | Status | O que foi Validado |
| :--- | :--- | :---: | :---: | :--- |
| **Pré-Processamento** | `tests/test_preprocessing.py` | 2 | ✅ Aprovado | Normalização precisa para escala $[0.0, 1.0]$ em `float32` e integridade das dimensões $28 \times 28$. |
| **Persistência SQLite** | `tests/test_database.py` | 4 | ✅ Aprovado | `PRAGMA integrity_check`, codificação bidirecional do BLOB de 784 bytes uint8, consultas paginadas e busca por classe. |
| **Modelos & Probabilidades** | `tests/test_models.py` | 3 | ✅ Aprovado | Carregamento dos 5 modelos, predição válida de classes (0-9), probabilidades não negativas com soma estrita de 100% e cálculo de consenso. |
| **API REST FastAPI** | `tests/test_api.py` | 9 | ✅ Aprovado | Rotas `/health`, `/models`, `/metrics`, `/samples/{digit}`, `/predict`, corpos inválidos com HTTP 422, OpenAPI e persistência do motor entre requisições. |
| **Chatbot Local** | `tests/test_chatbot.py` | 7 | ✅ Aprovado | Respostas exatas sobre objetivo, dataset, modelos, campeão, API, SQLite/DBeaver e política de fallback honesto para perguntas fora do escopo. |

---

## 3. Matriz de Conformidade do Projeto

| Requisito do Documento | Status | Como foi Atendido |
| :--- | :---: | :--- |
| **1. Revisar antes de alterar** | Concluído | Auditoria inicial realizada com inventário e diagnóstico em 3 listas (`docs/etapas/00_auditoria.md`). |
| **2. Objetivo e correção conceitual** | Concluído | Campo de dígito busca imagem real de teste no SQLite/dataset; predição real executada sobre pixels. |
| **3. MNIST e Rastreabilidade** | Concluído | Dimensões 28x28, tons de cinza, 70k amostras, CC BY-SA 3.0 documentada e distinção para digits 8x8. |
| **4. Modelos e Percentuais** | Parcial | 5 modelos retornam saídas normalizadas. No SVM, a softmax dos escores não é calibração probabilística; essa limitação é informada pela API. |
| **5. Aplicação Streamlit** | Concluído | Interface completa em pt-BR com 8 abas, visualização de imagens, gráficos interativos e chatbot integrado. |
| **6. API REST FastAPI** | Concluído | Endpoints `/health`, `/models`, `/metrics`, `/predict`, `/samples/{digit}` com Pydantic e `/docs` interativo. |
| **7. SQLite e DBeaver** | Concluído | `data/mnist.db` com `mnist_images` (BLOB 784 bytes uint8) e tabela demo sintética `registros_demo` com queries em `data/queries_dbeaver.sql`. |
| **8. Organização por Fases** | Concluído | Fases 0 a 6 documentadas em `docs/etapas/` com diagramas Mermaid e gráficos reais em `docs/graficos/`. |
| **9. Qualidade e Testes** | Concluído | Suíte com 25 testes executados e aprovados, incluindo validação HTTP e ciclo de vida da API. |
| **10. Entrega Final** | Concluído | README atualizado, comandos claros para terminais separados, portas `8501` e `8000`. |
