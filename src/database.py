"""Módulo de Gerenciamento e Persistência SQLite para o MNIST e Demonstração.

Suporta:
- Criação e inicialização do esquema relacional com índices e constraints.
- Armazenamento compacto de imagens MNIST em BLOB binário de exatamente 784 bytes (uint8).
- Decodificação rápida de BLOB para matriz NumPy (28x28) e vetor normalizado [0.0, 1.0].
- Tabela sintética 'registros_demo' ('rua, A, B, setor') com semente determinística.
- Métodos de consulta paginada, busca de exemplos por classe e estatísticas.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

# Caminho padrão configurável por variável de ambiente
DEFAULT_DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mnist.db"),
)


def get_connection(db_path=None):
    """Retorna uma conexão aberta com o banco SQLite."""
    caminho = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    conn = sqlite3.connect(caminho)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def codificar_imagem_para_blob(imagem_28x28_ou_vetor784):
    """Converte matriz 28x28 ou vetor 784 (uint8 ou float 0-1) para bytes de 784 posições uint8."""
    arr = np.asarray(imagem_28x28_ou_vetor784)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    arr_flat = arr.flatten()
    if len(arr_flat) != 784:
        raise ValueError(f"Dimensão inválida: esperado 784 pixels, recebido {len(arr_flat)}")
    return arr_flat.tobytes()


def decodificar_blob_para_imagem(blob, normalizado=False):
    """Decodifica BLOB binário de 784 bytes para matriz 28x28.
    
    Args:
        blob (bytes): Bytes brutos de 784 pixels uint8.
        normalizado (bool): Se True, retorna float32 no intervalo [0.0, 1.0]. Se False, uint8 [0, 255].
    """
    if len(blob) != 784:
        raise ValueError(f"Tamanho de BLOB inválido: esperado 784 bytes, recebido {len(blob)} bytes.")
    matriz = np.frombuffer(blob, dtype=np.uint8).reshape((28, 28))
    if normalizado:
        return (matriz / 255.0).astype(np.float32)
    return matriz


def inicializar_banco(db_path=None):
    """Cria as tabelas e índices se ainda não existirem."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Tabela principal: mnist_images
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mnist_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        digit INTEGER NOT NULL CHECK(digit >= 0 AND digit <= 9),
        split TEXT NOT NULL CHECK(split IN ('train', 'val', 'test')),
        width INTEGER NOT NULL DEFAULT 28,
        height INTEGER NOT NULL DEFAULT 28,
        image_blob BLOB NOT NULL,
        source TEXT NOT NULL DEFAULT 'OpenML mnist_784',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Índices para consultas eficientes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mnist_digit ON mnist_images(digit);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mnist_split ON mnist_images(split);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mnist_digit_split ON mnist_images(digit, split);")

    # Tabela de demonstração administrativa solicitada: registros_demo (rua, A, B, setor)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS registros_demo (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rua TEXT NOT NULL,
        valor_a REAL NOT NULL,
        valor_b REAL NOT NULL,
        setor TEXT NOT NULL,
        observacao TEXT DEFAULT 'Dado sintético gerado deterministicamente',
        data_criacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_demo_setor ON registros_demo(setor);")

    conn.commit()
    conn.close()
    print(f"-> Banco de dados SQLite inicializado em: {db_path or DEFAULT_DB_PATH}")


def popular_registros_demo(n_registros=100, seed=42, db_path=None):
    """Gera registros sintéticos determinísticos na tabela 'registros_demo'."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Verifica se já há dados
    cursor.execute("SELECT COUNT(*) FROM registros_demo;")
    if cursor.fetchone()[0] > 0:
        conn.close()
        return

    np.random.seed(seed)
    nomes_ruas = [
        "Rua das Acácias", "Avenida Brasil", "Rua XV de Novembro", "Rua Santa Catarina",
        "Avenida Central", "Rua dos Pioneiros", "Rua Bela Vista", "Avenida Blumenau",
        "Rua Rio do Sul", "Avenida Tecnológica", "Rua do Saber", "Avenida Universitária"
    ]
    setores = ["Setor Norte", "Setor Sul", "Setor Leste", "Setor Oeste", "Setor Central"]

    registros = []
    for _ in range(n_registros):
        rua = str(np.random.choice(nomes_ruas))
        val_a = round(float(np.random.uniform(10.0, 500.0)), 2)
        val_b = round(float(np.random.uniform(1.0, 100.0)), 2)
        setor = str(np.random.choice(setores))
        registros.append((rua, val_a, val_b, setor))

    cursor.executemany("""
    INSERT INTO registros_demo (rua, valor_a, valor_b, setor)
    VALUES (?, ?, ?, ?);
    """, registros)

    conn.commit()
    conn.close()
    print(f"-> Inseridos {n_registros} registros sintéticos na tabela 'registros_demo'.")


def salvar_amostras_mnist_no_banco(X, y, split="test", limite_por_classe=None, db_path=None):
    """Armazena amostras do MNIST no SQLite no formato BLOB binário."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Agrupa por classe para amostragem estratificada
    classes = np.unique(y)
    total_inseridos = 0

    for digito in classes:
        indices = np.where(y == digito)[0]
        if limite_por_classe is not None and len(indices) > limite_por_classe:
            indices = indices[:limite_por_classe]

        registros = []
        for idx in indices:
            blob = codificar_imagem_para_blob(X[idx])
            registros.append((int(digito), split, 28, 28, blob, "OpenML mnist_784"))

        cursor.executemany("""
        INSERT INTO mnist_images (digit, split, width, height, image_blob, source)
        VALUES (?, ?, ?, ?, ?, ?);
        """, registros)
        total_inseridos += len(registros)

    conn.commit()
    conn.close()
    print(f"-> Inseridas {total_inseridos} imagens MNIST (split='{split}') na tabela 'mnist_images'.")


def obter_exemplo_por_digito(digit, split="test", indice_offset=0, db_path=None):
    """Retorna uma imagem real e metadados para um dígito específico."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, digit, split, width, height, image_blob, source, created_at
    FROM mnist_images
    WHERE digit = ? AND split = ?
    ORDER BY id ASC
    LIMIT 1 OFFSET ?;
    """, (int(digit), split, int(indice_offset)))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    img_uint8 = decodificar_blob_para_imagem(row[5], normalizado=False)
    img_norm = (img_uint8 / 255.0).astype(np.float32)

    return {
        "id": row[0],
        "digit": row[1],
        "split": row[2],
        "width": row[3],
        "height": row[4],
        "image_uint8": img_uint8,
        "image_norm": img_norm,
        "vector_784": img_norm.flatten(),
        "source": row[6],
        "created_at": row[7],
    }


def obter_contagem_por_classe(db_path=None):
    """Retorna a contagem de amostras por dígito e split em formato de DataFrame."""
    conn = get_connection(db_path)
    df = pd.read_sql_query("""
    SELECT digit, split, COUNT(*) as total
    FROM mnist_images
    GROUP BY digit, split
    ORDER BY digit, split;
    """, conn)
    conn.close()
    return df


def obter_amostras_paginadas(tabela="mnist_images", limite=20, offset=0, db_path=None):
    """Retorna linhas de uma tabela para exibição segura sem carregar a base inteira."""
    conn = get_connection(db_path)
    if tabela == "mnist_images":
        query = f"""
        SELECT id, digit, split, width, height, LENGTH(image_blob) as blob_bytes, source, created_at
        FROM mnist_images
        ORDER BY id ASC
        LIMIT {int(limite)} OFFSET {int(offset)};
        """
    elif tabela == "registros_demo":
        query = f"""
        SELECT id, rua, valor_a, valor_b, setor, observacao, data_criacao
        FROM registros_demo
        ORDER BY id ASC
        LIMIT {int(limite)} OFFSET {int(offset)};
        """
    else:
        conn.close()
        raise ValueError(f"Tabela desconhecida: {tabela}")

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def verificar_integridade_banco(db_path=None):
    """Testa a integridade do arquivo SQLite e contagens."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA integrity_check;")
    status = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM mnist_images;")
    total_mnist = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM registros_demo;")
    total_demo = cursor.fetchone()[0]

    conn.close()
    return {
        "integridade": status,
        "total_mnist_images": total_mnist,
        "total_registros_demo": total_demo,
        "db_path": db_path or DEFAULT_DB_PATH,
    }
