-- ==============================================================================
-- PROJETO SCTEC 2026: RECONHECIMENTO DE DÍGITOS MNIST
-- GUIA DE CONSULTAS SQL PARA EXECUÇÃO NO DBEAVER
-- ==============================================================================
--
-- COMO CONECTAR ESTE BANCO NO DBEAVER:
-- 1. Abra o DBeaver Community ou PRO.
-- 2. Clique em "Nova Conexão" (ícone de tomada com '+') ou no menu: Database -> New Database Connection.
-- 3. Selecione "SQLite" e clique em "Avançar".
-- 4. No campo "Path" (Caminho do Banco), selecione o arquivo gerado:
--    Caminho relativo: data/mnist.db
--    Caminho absoluto: C:\SCTEC_2026_Projeto_final\data\mnist.db
-- 5. Clique em "Test Connection" (o DBeaver baixará o driver JDBC SQLite se for a primeira vez).
-- 6. Clique em "Concluir".
--
-- ==============================================================================

-- 1. VERIFICAÇÃO DE INTEGRIDADE DA BASE DE DADOS
PRAGMA integrity_check;
PRAGMA foreign_keys;

-- 2. ESTATÍSTICAS GERAIS DA TABELA DE IMAGENS MNIST
-- Total de imagens armazenadas
SELECT COUNT(*) AS total_imagens FROM mnist_images;

-- Distribuição de imagens por dígito e conjunto (train, val, test)
SELECT 
    digit AS digito,
    split AS particao,
    COUNT(*) AS quantidade_imagens,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM mnist_images), 2) AS percentual_total
FROM mnist_images
GROUP BY digit, split
ORDER BY digit ASC, split ASC;

-- Resumo consolidado por classe (0 a 9)
SELECT 
    digit AS digito,
    COUNT(*) AS total_amostras,
    MIN(LENGTH(image_blob)) AS bytes_minimo,
    MAX(LENGTH(image_blob)) AS bytes_maximo,
    MAX(width) || 'x' || MAX(height) AS resolucao
FROM mnist_images
GROUP BY digit
ORDER BY digit ASC;

-- 3. INSPEÇÃO DE AMOSTRAS DE IMAGENS E METADADOS
-- Visualizar as primeiras 15 imagens do conjunto de teste com tamanho do BLOB
SELECT 
    id,
    digit,
    split,
    width,
    height,
    LENGTH(image_blob) AS tamanho_blob_bytes,
    source,
    created_at
FROM mnist_images
WHERE split = 'test'
ORDER BY id ASC
LIMIT 15;

-- Como decodificar o campo image_blob:
-- O campo image_blob contém exatamente 784 bytes brutos (28 * 28 pixels) em escala de cinza uint8 [0, 255].
-- No Python, a leitura é direta:
--   pixels = np.frombuffer(row['image_blob'], dtype=np.uint8).reshape((28, 28))

-- 4. CONSULTAS NA TABELA DE DEMONSTRAÇÃO SINTÉTICA (registros_demo)
-- Nota: Esta tabela contém dados sintéticos administrativos gerados deterministicamente (rua, A, B, setor)
-- e não se mistura ao conjunto MNIST.

-- Total de registros demo
SELECT COUNT(*) AS total_registros_demo FROM registros_demo;

-- Estatísticas por setor (agregação)
SELECT 
    setor,
    COUNT(*) AS qtd_registros,
    ROUND(AVG(valor_a), 2) AS media_valor_a,
    ROUND(AVG(valor_b), 2) AS media_valor_b,
    ROUND(MIN(valor_a), 2) AS min_valor_a,
    ROUND(MAX(valor_a), 2) AS max_valor_a
FROM registros_demo
GROUP BY setor
ORDER BY media_valor_a DESC;

-- Amostra de registros demo ordenados por data
SELECT 
    id,
    rua,
    valor_a,
    valor_b,
    setor,
    observacao,
    data_criacao
FROM registros_demo
ORDER BY id ASC
LIMIT 20;

-- 5. VALIDAÇÃO DE RESTRIÇÕES (CONSTRAINTS)
-- Confere se existem valores fora dos limites especificados
SELECT COUNT(*) AS erros_digito_invalido 
FROM mnist_images 
WHERE digit < 0 OR digit > 9;

SELECT COUNT(*) AS erros_tamanho_blob_invalido 
FROM mnist_images 
WHERE LENGTH(image_blob) != 784;
