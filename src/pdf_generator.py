"""Módulo Utilitário de Geração de Relatório PDF do Reconhecimento de Dígitos.

Gera um PDF estruturado contendo:
1. Cabeçalho Oficial do Projeto SCTEC 2026.
2. Visualização do Dígito (Desenho no Canvas original + Tratamento 28x28).
3. Explicação Didática da Arquitetura Neural:
   - Entrada: 784 pixels normalizados
   - Processamento Oculto / Intermediário: Camadas de convolução / densas com ReLU
   - Saída: Softmax gerando probabilidades de 0 a 9
4. Tabela de Predições e Consenso dos cinco modelos implementados.
5. Gráfico de Barras com as Porcentagens de Confiança.
6. Conclusão Técnica e Diagnóstico.
"""

import io
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)


def gerar_relatorio_pdf(
    imagem_canvas_bytes,
    imagem_28x28_array,
    resultados_predicao,
    modelo_destaque="Rede Neural Keras",
    caminho_saida=None
):
    """Gera um PDF completo com a explicação das etapas da rede neural e as métricas obtidas.
    
    Args:
        imagem_canvas_bytes: Bytes da imagem original desenhada ou array uint8.
        imagem_28x28_array: Array numpy (28, 28) da imagem pré-processada.
        resultados_predicao: Dicionário retornado por motor.prever(imagem_28x28).
        modelo_destaque: Nome do modelo para focar na distribuição percentual.
        caminho_saida: Caminho para salvar em disco (opcional). Se None, retorna bytes.
        
    Returns:
        bytes: Buffer de bytes do documento PDF.
    """
    buffer = io.BytesIO() if caminho_saida is None else open(caminho_saida, "wb")
    doc = SimpleDocTemplate(
        buffer if caminho_saida is None else caminho_saida,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    
    # Estilos customizados
    titulo_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        alignment=1, # Centralizado
    )
    
    subtitulo_style = ParagraphStyle(
        "CustomSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
    )
    
    secao_style = ParagraphStyle(
        "CustomSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6,
    )
    
    texto_style = ParagraphStyle(
        "CustomText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
    )

    destaque_box_style = ParagraphStyle(
        "CustomBoxText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2C5282"),
        alignment=1,
    )

    story = []

    # 1. Cabeçalho
    story.append(Paragraph("SCTEC 2026 — Inteligência Artificial & Deep Learning", subtitulo_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Relatório Técnico de Reconhecimento de Dígitos Manuscritos", titulo_style))
    story.append(Paragraph("Diagnóstico dos Modelos de Classificação", subtitulo_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3182CE"), spaceBefore=1, spaceAfter=12))

    # 2. Imagens: Desenho Original e Tratamento 28x28
    # Prepara buffers de imagens para o ReportLab
    img_orig_buf = io.BytesIO()
    if isinstance(imagem_canvas_bytes, bytes):
        img_orig_buf.write(imagem_canvas_bytes)
        img_orig_buf.seek(0)
    elif isinstance(imagem_canvas_bytes, np.ndarray):
        pil_orig = Image.fromarray(imagem_canvas_bytes.astype(np.uint8))
        pil_orig.save(img_orig_buf, format="PNG")
        img_orig_buf.seek(0)
    else:
        # Fallback para array em branco caso nulo
        pil_orig = Image.new("L", (100, 100), color=255)
        pil_orig.save(img_orig_buf, format="PNG")
        img_orig_buf.seek(0)

    # Imagem 28x28 ampliada para visualização
    img_28_buf = io.BytesIO()
    fig_img, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(imagem_28x28_array, cmap="gray")
    ax.axis("off")
    fig_img.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig_img.savefig(img_28_buf, format="PNG", dpi=120)
    plt.close(fig_img)
    img_28_buf.seek(0)

    rl_img_orig = RLImage(img_orig_buf, width=100, height=100)
    rl_img_28 = RLImage(img_28_buf, width=100, height=100)

    consenso = resultados_predicao.get("consenso", {})
    dig_previsto = consenso.get("digito_previsto", "?")
    votos = consenso.get("votos", "0/0")
    concordancia = consenso.get("concordancia_pct", 0)

    tabela_visual = [
        [
            rl_img_orig,
            rl_img_28,
            [
                Paragraph("<b>Veredito do Consenso Neural:</b>", texto_style),
                Spacer(1, 4),
                Paragraph(f"<font size=24 color='#2B6CB0'><b>Dígito {dig_previsto}</b></font>", destaque_box_style),
                Spacer(1, 4),
                Paragraph(f"• Concordância: <b>{concordancia}%</b> ({votos} votos)", texto_style),
                Paragraph(f"• Modelo Foco: <b>{modelo_destaque}</b>", texto_style),
            ]
        ],
        [
            Paragraph("<b>Traço do Usuário (Canvas)</b>", subtitulo_style),
            Paragraph("<b>Pré-processado (28 × 28 px)</b>", subtitulo_style),
            Paragraph("<b>Status da Classificação</b>", subtitulo_style),
        ]
    ]
    t_visual = Table(tabela_visual, colWidths=[130, 130, 260])
    t_visual.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#EBF8FF")),
        ("BOX", (2, 0), (2, 0), 1, colors.HexColor("#BEE3F8")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_visual)
    story.append(Spacer(1, 10))

    # 3. Explicação Didática das Etapas da Rede Neural
    story.append(Paragraph("1. Arquitetura e Etapas da Rede Neural", secao_style))
    etapas_texto = """
    <b>Etapa 1 — Entrada (Input Layer):</b> O traço desenhado no Paint é redimensionado para uma grade de 
    <b>28 × 28 pixels</b> (784 entradas no total). Os pixels são normalizados no intervalo [0.0, 1.0] e centralizados 
    via Centro de Massa para garantir invariância posicional.<br/><br/>
    <b>Etapa 2 — Processamento Intermediário (Hidden Layers):</b> Os 784 sinais passam por transformações 
    lineares seguidas de funções de ativação não-lineares (<b>ReLU</b>), Batch Normalization e Dropout (20%). 
    Aqui a rede extrai traços primitivos (bordas, curvas e laços característicos do dígito).<br/><br/>
    <b>Etapa 3 — Saída e Probabilidades (Output Layer & Softmax):</b> A última camada conecta as representações a 
    <b>10 neurônios</b> (um para cada dígito de 0 a 9). A função <b>Softmax</b> converte os logits em uma distribuição 
    percentual cuja soma é exatamente 100%.
    """
    story.append(Paragraph(etapas_texto, texto_style))
    story.append(Spacer(1, 10))

    # 4. Tabela com as Predições de Todos os Modelos
    story.append(Paragraph("2. Comparativo de Inferência dos Modelos", secao_style))
    
    dados_tabela = [["Modelo", "Dígito Previsto", "Confiança (%)", "Latência (ms)", "Método Probabilístico"]]
    predicoes = resultados_predicao.get("predicoes", {})
    for mod_nome, d in predicoes.items():
        conf = f"{d.get('confianca_max_pct', 0):.2f}%"
        lat = f"{d.get('latencia_ms', 0):.2f} ms"
        metodo = d.get("tipo_saida", "Softmax")
        dados_tabela.append([mod_nome, str(d.get("classe_prevista", "?")), conf, lat, metodo])

    t_mod = Table(dados_tabela, colWidths=[140, 90, 85, 80, 125])
    t_mod.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F7FAFC"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_mod)
    story.append(Spacer(1, 10))

    # 5. Gráfico de Barras com as Porcentagens de 0 a 9
    story.append(Paragraph(f"3. Distribuição Percentual de Confiança — {modelo_destaque}", secao_style))
    
    probs_dict = predicoes.get(modelo_destaque, {}).get("probabilidades_pct", {})
    if not probs_dict and predicoes:
        primeiro_mod = list(predicoes.keys())[0]
        probs_dict = predicoes[primeiro_mod].get("probabilidades_pct", {})

    valores_pct = [probs_dict.get(str(i), 0.0) for i in range(10)]
    digitos_label = [str(i) for i in range(10)]

    fig_bar, ax_bar = plt.subplots(figsize=(6.5, 2.2))
    cores = ["#3182CE" if v < max(valores_pct) else "#E53E3E" for v in valores_pct]
    barras = ax_bar.bar(digitos_label, valores_pct, color=cores, width=0.6, edgecolor="#2B6CB0")
    ax_bar.set_ylabel("Certeza (%)", fontsize=8.5, fontweight="bold")
    ax_bar.set_xlabel("Dígito Avaliado (0 a 9)", fontsize=8.5, fontweight="bold")
    ax_bar.set_ylim(0, max(100.0, max(valores_pct) * 1.15))
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, pct in zip(barras, valores_pct):
        if pct > 0.5:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold"
            )

    fig_bar.tight_layout()
    chart_buf = io.BytesIO()
    fig_bar.savefig(chart_buf, format="PNG", dpi=130)
    plt.close(fig_bar)
    chart_buf.seek(0)

    rl_chart = RLImage(chart_buf, width=520, height=176)
    story.append(rl_chart)
    story.append(Spacer(1, 8))

    # 6. Conclusão Técnica
    story.append(Paragraph("4. Conclusão e Veredito Final", secao_style))
    conclusao_txt = f"""
    Os modelos integrados processaram o sinal bidimensional com sucesso. 
    O dígito classificado com maior probabilidade foi o <b>{dig_previsto}</b> com <b>{concordancia}% de consenso</b> entre 
    as arquiteturas. A distribuição de saída do Softmax comprovou que as camadas internas conseguiram isolar 
    as características estruturais do traço desenhado, separando com nitidez as classes concorrentes.
    """
    story.append(Paragraph(conclusao_txt, texto_style))

    doc.build(story)

    if caminho_saida is None:
        buffer.seek(0)
        return buffer.getvalue()
    else:
        buffer.close()
        return None
