from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import unicodedata

import cv2
import keras
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

from chatbot.bot import ChatbotLocal
from src.database import obter_contagem_por_classe, obter_exemplo_por_digito, verificar_integridade_banco
from src.inference import MotorInferencia


ROOT = Path(__file__).resolve().parent
PHASE_DOCS = [
    ("Fase 0 — Auditoria", "docs/etapas/00_auditoria.md", ":material/fact_check:"),
    ("Fase 1 — Escopo e arquitetura", "docs/etapas/01_escopo_arquitetura.md", ":material/account_tree:"),
    ("Fase 2 — Dados e banco", "docs/etapas/02_dados_banco.md", ":material/database:"),
    ("Fase 3 — Modelos e avaliação", "docs/etapas/03_modelos_avaliacao.md", ":material/analytics:"),
    ("Fase 4 — API e SQLite", "docs/etapas/04_api_sqlite.md", ":material/api:"),
    ("Fase 5 — Streamlit e chatbot", "docs/etapas/05_streamlit_chatbot.md", ":material/dashboard:"),
    ("Fase 6 — Testes e entrega", "docs/etapas/06_testes_entrega.md", ":material/checklist:"),
]

st.set_page_config(
    page_title="SCTEC | Classificação MNIST",
    page_icon=":material/numbers:",
    layout="wide",
)


@st.cache_resource
def carregar_motor() -> MotorInferencia:
    return MotorInferencia.get_instancia("models")


@st.cache_resource
def carregar_chatbot() -> ChatbotLocal:
    return ChatbotLocal()


def inicializar_estado() -> None:
    st.session_state.setdefault("resultado_analise", None)
    st.session_state.setdefault("resultado_digito", None)
    st.session_state.setdefault("mensagens_chat", [])
    st.session_state.setdefault("upload_version", 0)
    st.session_state.setdefault("canvas_version", 0)
    st.session_state.setdefault("modo_version", 0)
    st.session_state.setdefault("digit_number", 0)


def limpar_analise() -> None:
    st.session_state["resultado_analise"] = None
    st.session_state["resultado_digito"] = None
    st.session_state["upload_version"] += 1
    st.session_state["canvas_version"] += 1
    st.session_state["modo_version"] += 1
    st.session_state["digit_number"] = 0


def limpar_chat() -> None:
    st.session_state["mensagens_chat"] = []


def preparar_imagem_mnist(imagem: Image.Image | np.ndarray) -> np.ndarray:
    """Converte uma imagem para um dígito branco, centralizado, em 28 × 28 pixels."""
    if isinstance(imagem, Image.Image):
        imagem = np.asarray(imagem.convert("RGB"))

    imagem = np.asarray(imagem)
    if imagem.ndim == 3:
        if imagem.shape[2] == 4:
            imagem = cv2.cvtColor(imagem, cv2.COLOR_RGBA2GRAY)
        else:
            imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    if imagem.ndim != 2:
        raise ValueError("A imagem precisa ser colorida ou em escala de cinza.")

    imagem = np.clip(imagem, 0, 255).astype(np.uint8)
    if float(np.mean(imagem)) > 127:
        imagem = 255 - imagem

    _, binaria = cv2.threshold(
        imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    coordenadas = cv2.findNonZero(binaria)
    if coordenadas is None:
        raise ValueError("Nenhum traço foi encontrado. Escolha outra imagem ou desenhe um dígito.")

    x, y, largura, altura = cv2.boundingRect(coordenadas)
    recorte = binaria[y:y + altura, x:x + largura]
    escala = 20.0 / max(largura, altura)
    nova_largura = max(1, int(round(largura * escala)))
    nova_altura = max(1, int(round(altura * escala)))
    digito = cv2.resize(recorte, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)

    resultado = np.zeros((28, 28), dtype=np.uint8)
    inicio_x = (28 - nova_largura) // 2
    inicio_y = (28 - nova_altura) // 2
    resultado[inicio_y:inicio_y + nova_altura, inicio_x:inicio_x + nova_largura] = digito
    return resultado


def criar_resultado(imagem: np.ndarray, origem: str) -> dict:
    imagem_28 = preparar_imagem_mnist(imagem)
    previsao = carregar_motor().prever(imagem_28)
    return {"imagem": imagem_28, "origem": origem, "previsao": previsao}


def dataframe_predicoes(previsao: dict, digito_consultado: int | None = None) -> pd.DataFrame:
    linhas = []
    for nome, dados in previsao.get("predicoes", {}).items():
        probabilidades = dados.get("probabilidades_pct", {})
        linha = {
            "Modelo": nome,
            "Previsão": int(dados["classe_prevista"]),
            "Confiança máxima (%)": float(dados["confianca_max_pct"]),
            "Latência (ms)": float(dados["latencia_ms"]),
        }
        if digito_consultado is not None:
            linha[f"P(dígito {digito_consultado}) (%)"] = float(
                probabilidades.get(str(digito_consultado), 0)
            )
        linhas.append(linha)
    return pd.DataFrame(linhas)


def renderizar_progresso_keras(imagem: np.ndarray) -> None:
    """Mostra ativações reais das camadas principais da CNN Keras."""
    info = carregar_motor().modelos.get("Rede Neural Keras")
    if not info or info.get("tipo") != "keras":
        return

    modelo = info["modelo"]
    camadas = [
        camada for camada in modelo.layers
        if camada.__class__.__name__ in {"Conv2D", "Dense"}
        and camada is not modelo.layers[-1]
    ]
    if len(camadas) < 3:
        return

    st.subheader("Caminho da imagem pela rede Keras")
    st.caption(
        "A visualização acompanha as ativações calculadas pela CNN: entrada 28 × 28, "
        "filtros das duas convoluções e probabilidades dos 10 dígitos na saída."
    )
    entrada = (imagem.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
    barra = st.progress(0, text="Entrada: imagem preparada em 28 × 28 pixels")
    colunas = st.columns(len(camadas) + 2)
    with colunas[0]:
        st.image(imagem, caption="Entrada · 28 × 28", width=110, clamp=True)

    for indice, camada in enumerate(camadas):
        extrator = keras.Model(inputs=modelo.inputs, outputs=camada.output)
        ativacao = np.asarray(extrator(entrada, training=False))[0]
        percentual = int((indice + 1) / (len(camadas) + 1) * 100)
        barra.progress(percentual, text=f"Processando {camada.name} · {camada.__class__.__name__}")
        with colunas[indice + 1]:
            st.write(f"**{camada.name}**")
            if camada.__class__.__name__ == "Conv2D" and ativacao.ndim == 3:
                # Quatro mapas são exibidos como amostra dos filtros reais da camada.
                canais = min(4, ativacao.shape[-1])
                mosaico = np.concatenate(
                    [ativacao[:, :, canal] for canal in range(canais)], axis=1
                )
                # st.image espera floats entre 0 e 1; ativações ReLU podem exceder esse intervalo.
                mosaico = np.nan_to_num(mosaico, nan=0.0, posinf=0.0, neginf=0.0)
                minimo, maximo = float(mosaico.min()), float(mosaico.max())
                if maximo > minimo:
                    mosaico = (mosaico - minimo) / (maximo - minimo)
                mosaico = (mosaico * 255).astype(np.uint8)
                st.image(
                    mosaico,
                    caption=f"Conv2D · {ativacao.shape[-1]} filtros (4 exibidos)",
                    width="stretch",
                )
            elif ativacao.ndim == 1:
                amostra = ativacao[: min(8, ativacao.shape[0])]
                st.caption(f"Dense · {ativacao.shape[0]} neurônios")
                st.bar_chart(
                    pd.DataFrame({"Neurônio": list(range(len(amostra))), "Ativação": amostra}),
                    x="Neurônio", y="Ativação", sort=False,
                    alt=f"Ativações de neurônios da camada {camada.name}",
                )

    probabilidades = modelo(entrada, training=False).numpy()[0]
    barra.progress(100, text="Saída: probabilidades calculadas para os dígitos 0 a 9")
    barra.empty()
    with colunas[-1]:
        st.write("**Saída**")
        st.caption("10 neurônios · dígitos 0–9")
        st.bar_chart(
            pd.DataFrame({"Dígito": list(range(10)), "Probabilidade": probabilidades}),
            x="Dígito", y="Probabilidade", sort=False,
            alt="Probabilidades de saída da rede Keras para os dígitos de zero a nove",
        )


def renderizar_detalhe_resultado(resultado: dict, digito_consultado: int | None = None) -> None:
    previsao = resultado["previsao"]
    consenso = previsao.get("consenso", {})
    tabela = dataframe_predicoes(previsao, digito_consultado)

    with st.container(border=True):
        if digito_consultado is None:
            st.subheader("Previsões dos cinco modelos")
            st.caption(f"Entrada: {resultado['origem']}")
        else:
            st.subheader(f"Comportamento dos modelos para o dígito {digito_consultado}")
            st.caption(
                "O dígito selecionado serve para buscar uma imagem real dessa classe no MNIST. "
                "Os modelos recebem a imagem, não o número digitado como texto."
            )

        col_imagem, col_consenso = st.columns([1, 2], gap="medium")
        with col_imagem:
            st.image(
                resultado["imagem"],
                caption="Imagem processada para o MNIST",
                width=150,
                clamp=True,
                alt="Dígito processado em escala de cinza, tamanho 28 por 28 pixels",
            )
        with col_consenso:
            st.metric("Consenso dos modelos", consenso.get("digito_previsto", "—"))
            st.caption(
                f"Concordância: {consenso.get('concordancia_pct', 0)}% "
                f"({consenso.get('votos', '0/0')} votos)"
            )

        st.dataframe(tabela, hide_index=True, width="stretch", alt="Previsões e confiança de cada modelo")
        if digito_consultado is not None and not tabela.empty:
            coluna_prob = f"P(dígito {digito_consultado}) (%)"
            st.bar_chart(
                tabela,
                x="Modelo",
                y=coluna_prob,
                sort=False,
                alt=f"Probabilidade atribuída por cada modelo ao dígito {digito_consultado}",
            )

        modelos = list(previsao.get("predicoes", {}))
        if modelos:
            modelo_selecionado = st.selectbox(
                "Ver probabilidades por classe",
                modelos,
                key=f"modelo_probabilidades_{digito_consultado}_{resultado['origem']}",
            )
            probabilidades = previsao["predicoes"][modelo_selecionado]["probabilidades_pct"]
            distribuicao = pd.DataFrame({
                "Dígito": list(range(10)),
                "Probabilidade (%)": [float(probabilidades.get(str(i), 0)) for i in range(10)],
            })
            st.bar_chart(
                distribuicao,
                x="Dígito",
                y="Probabilidade (%)",
                sort=False,
                alt=f"Probabilidades de 0 a 9 para {modelo_selecionado}",
            )


def renderizar_chatbot() -> None:
    with st.container(border=True):
        st.subheader("Pergunte sobre o projeto", help="Assistente local baseado na documentação SCTEC.")
        st.caption("Chatbot local · sem envio de perguntas para serviços externos")
        if st.button("Limpar conversa", icon=":material/delete_sweep:", key="limpar_chat"):
            limpar_chat()
            st.rerun()

        if not st.session_state["mensagens_chat"]:
            st.caption("Exemplos: quais modelos são usados? Como funciona o banco? O que foi feito na fase 3?")

        for mensagem in st.session_state["mensagens_chat"][-8:]:
            with st.chat_message(mensagem["role"]):
                st.markdown(mensagem["content"])

        pergunta = st.chat_input("Escreva sua pergunta", key="pergunta_chat")
        if pergunta:
            resposta = carregar_chatbot().responder(pergunta)
            st.session_state["mensagens_chat"].extend([
                {"role": "user", "content": pergunta},
                {"role": "assistant", "content": resposta},
            ])
            st.rerun()


def pagina_com_chat(titulo: str, icone: str, conteudo) -> None:
    principal, painel_chat = st.columns([0.69, 0.31], gap="large")
    with principal:
        st.title(titulo, icon=icone)
        conteudo()
    with painel_chat:
        renderizar_chatbot()


def pagina_visao_geral() -> None:
    metricas = carregar_motor().metricas
    ranking = sorted(metricas.items(), key=lambda item: item[1]["acuracia"], reverse=True)
    total_modelos = len(carregar_motor().modelos)
    total_registros = verificar_integridade_banco()["total_mnist_images"]

    st.write("Projeto SCTEC 2026 · classificação de dígitos manuscritos do MNIST")
    destaques = st.columns(3)
    destaques[0].metric("Modelos carregados", total_modelos)
    destaques[1].metric("Imagens no banco", f"{total_registros:,}".replace(",", "."))
    destaques[2].metric("Modelo com maior acurácia registrada", f"{ranking[0][0]} · {ranking[0][1]['acuracia']:.2f}%")

    st.subheader("Explore o projeto")
    st.markdown(
        "Use a navegação à esquerda para abrir a visão geral, comparar os modelos, "
        "testar um dígito, desenhar ou enviar uma imagem e ler cada fase do projeto. "
        "O painel de perguntas fica à direita."
    )
    st.subheader("Modelos disponíveis")
    st.dataframe(
        pd.DataFrame([
            {"Modelo": nome, "Acurácia (%)": dados["acuracia"], "F1 macro (%)": dados["f1_macro"], "Latência (ms)": dados["latencia_ms"]}
            for nome, dados in ranking
        ]),
        hide_index=True,
        width="stretch",
        alt="Resumo de desempenho dos cinco modelos MNIST",
    )


def pagina_comparacao() -> None:
    metricas = carregar_motor().metricas
    df = pd.DataFrame([
        {
            "Modelo": nome,
            "Acurácia (%)": dados["acuracia"],
            "Precisão macro (%)": dados["precisao_macro"],
            "Recall macro (%)": dados["recall_macro"],
            "F1 macro (%)": dados["f1_macro"],
            "Latência (ms)": dados["latencia_ms"],
            "Treino (s)": dados["tempo_treino_s"],
        }
        for nome, dados in metricas.items()
    ])
    st.write("Compare as métricas registradas para os cinco modelos.")
    metrica = st.selectbox(
        "Métrica do gráfico",
        ["Acurácia (%)", "Precisão macro (%)", "Recall macro (%)", "F1 macro (%)", "Latência (ms)"],
        key="metrica_comparacao",
    )
    st.bar_chart(df, x="Modelo", y=metrica, sort=False, alt=f"Comparação dos modelos por {metrica}")
    st.dataframe(df, hide_index=True, width="stretch", alt="Tabela completa de métricas dos modelos")
    st.caption("As métricas são valores registrados na avaliação do projeto; a latência varia com o computador.")


def pagina_por_digito() -> None:
    st.write("Informe um dígito. O sistema procura uma imagem real de teste dessa classe no banco e consulta os cinco modelos.")
    with st.form("form_digito"):
        digito = st.number_input("Dígito de 0 a 9", min_value=0, max_value=9, step=1, key="digit_number")
        buscar = st.form_submit_button("Buscar imagem e comparar", type="primary", icon=":material/search:")

    if buscar:
        try:
            amostra = obter_exemplo_por_digito(int(digito), split="test")
            if amostra is None:
                st.warning(f"Não há imagem de teste para o dígito {digito} no banco.")
                st.session_state["resultado_digito"] = None
            else:
                resultado = carregar_motor().prever(amostra["image_uint8"])
                st.session_state["resultado_digito"] = {
                    "imagem": amostra["image_uint8"],
                    "previsao": resultado,
                    "id": amostra["id"],
                    "rotulo": amostra["digit"],
                    "digito_consultado": int(digito),
                }
        except Exception as erro:
            st.error(f"Não foi possível consultar os modelos: {erro}")

    resultado = st.session_state.get("resultado_digito")
    if resultado:
        st.caption(f"Amostra MNIST #{resultado['id']} · rótulo real: {resultado['rotulo']}")
        renderizar_detalhe_resultado(
            {"imagem": resultado["imagem"], "origem": "Amostra real MNIST", "previsao": resultado["previsao"]},
            digito_consultado=resultado["digito_consultado"],
        )


def imagem_upload_rgb(conteudo: bytes) -> Image.Image:
    with Image.open(BytesIO(conteudo)) as imagem:
        if imagem.width * imagem.height > 25_000_000:
            raise ValueError("A imagem excede o limite de 25 megapixels.")
        if imagem.width < 2 or imagem.height < 2:
            raise ValueError("A imagem é pequena demais para ser processada.")
        imagem = ImageOps.exif_transpose(imagem)
        imagem.load()
        if "A" in imagem.getbands() or "transparency" in imagem.info:
            imagem_rgba = imagem.convert("RGBA")
            fundo = Image.new("RGBA", imagem_rgba.size, "white")
            fundo.alpha_composite(imagem_rgba)
            return fundo.convert("RGB")
        return imagem.convert("RGB")


def pagina_imagem() -> None:
    st.write("Desenhe um dígito ou escolha um arquivo para comparar as previsões de todos os modelos.")
    modo = st.segmented_control(
        "Tipo de entrada",
        ["Enviar arquivo", "Desenhar"],
        default="Enviar arquivo",
        key=f"modo_entrada_{st.session_state['modo_version']}",
        width="stretch",
    )
    imagem_original = None
    origem = None

    if modo == "Enviar arquivo":
        arquivo = st.file_uploader(
            "Escolha uma imagem PNG ou outro formato compatível",
            type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
            max_upload_size=10,
            key=f"arquivo_imagem_{st.session_state['upload_version']}",
            help="Formatos aceitos: PNG, JPG, JPEG, BMP, WebP e TIFF. Limite: 10 MB.",
        )
        if arquivo is not None:
            try:
                imagem_original = imagem_upload_rgb(arquivo.getvalue())
                st.image(imagem_original, caption="Imagem escolhida", width=260, alt="Imagem enviada para classificação")
                origem = f"Arquivo: {arquivo.name}"
            except Exception as erro:
                st.error(f"Não foi possível abrir essa imagem: {erro}")
    elif modo == "Desenhar":
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            return_image_data=True,
            key=f"canvas_mnist_{st.session_state['canvas_version']}",
        )
        try:
            imagem_canvas = canvas.image_data
        except RuntimeError:
            imagem_canvas = None
        if imagem_canvas is not None and np.any(imagem_canvas[:, :, 3] > 0):
            imagem_original = imagem_canvas
            origem = "Desenho no canvas"

    if imagem_original is not None and st.button(
        "Classificar com os cinco modelos",
        type="primary",
        icon=":material/psychology:",
        key=f"classificar_{st.session_state['modo_version']}",
    ):
        try:
            with st.spinner("Comparando os modelos…"):
                st.session_state["resultado_analise"] = criar_resultado(imagem_original, origem or "Imagem")
        except Exception as erro:
            st.error(f"Não foi possível classificar essa entrada: {erro}")

    resultado = st.session_state.get("resultado_analise")
    if resultado:
        renderizar_detalhe_resultado(resultado)
        renderizar_progresso_keras(resultado["imagem"])


def pagina_dados() -> None:
    integridade = verificar_integridade_banco()
    st.write("Resumo do banco SQLite local usado pelo projeto.")
    resumo = st.columns(3)
    resumo[0].metric("Integridade SQLite", integridade["integridade"])
    resumo[1].metric("Imagens MNIST", f"{integridade['total_mnist_images']:,}".replace(",", "."))
    resumo[2].metric("Registros de demonstração", f"{integridade['total_registros_demo']:,}".replace(",", "."))
    st.caption(f"Arquivo do banco: `{integridade['db_path']}`")
    st.subheader("Amostras por dígito e divisão")
    st.dataframe(obter_contagem_por_classe(), hide_index=True, width="stretch", alt="Contagem de imagens por dígito e divisão MNIST")
    st.info("Para inspecionar tabelas e executar consultas, abra `data/mnist.db` no DBeaver. Exemplos estão em `data/queries_dbeaver.sql`.")


def pagina_fase(titulo: str, caminho: str) -> None:
    arquivo = ROOT / caminho
    if not arquivo.exists():
        st.warning(f"O documento desta fase não foi encontrado: {caminho}")
        return
    st.markdown(arquivo.read_text(encoding="utf-8"))


def criar_pagina(titulo: str, icone: str, renderizar) -> st.Page:
    def executar() -> None:
        pagina_com_chat(titulo, icone, renderizar)

    caminho_url = unicodedata.normalize("NFKD", titulo)
    caminho_url = caminho_url.encode("ascii", "ignore").decode("ascii").lower()
    caminho_url = re.sub(r"[^a-z0-9]+", "-", caminho_url).strip("-")
    return st.Page(executar, title=titulo, icon=icone, url_path=caminho_url)


inicializar_estado()

try:
    carregar_motor()
except Exception as erro:
    st.error(f"Não foi possível carregar os modelos: {erro}")
    st.info("Confira se os arquivos dos modelos estão disponíveis na pasta `models/`.")
    st.stop()

paginas_projeto = [
    criar_pagina("Visão geral", ":material/home:", pagina_visao_geral),
    criar_pagina("Comparação dos modelos", ":material/bar_chart:", pagina_comparacao),
    criar_pagina("Testar por dígito", ":material/dialpad:", pagina_por_digito),
    criar_pagina("Desenhar ou enviar imagem", ":material/upload_file:", pagina_imagem),
    criar_pagina("Dados e banco", ":material/database:", pagina_dados),
]
paginas_fases = [
    criar_pagina(titulo, icone, lambda caminho=caminho, titulo=titulo: pagina_fase(titulo, caminho))
    for titulo, caminho, icone in PHASE_DOCS
]

navegacao = st.navigation(
    {"Projeto": paginas_projeto, "Fases documentadas": paginas_fases},
    position="sidebar",
    expanded=True,
)
with st.sidebar:
    st.divider()
    st.caption("Projeto Final SCTEC 2026 · MNIST")
    st.button(
        "Limpar imagem e resultados",
        icon=":material/delete_sweep:",
        on_click=limpar_analise,
        width="stretch",
    )
navegacao.run()
