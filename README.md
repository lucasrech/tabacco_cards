# Extracao de Tabelas para CSV

Este projeto le imagens da pasta `img/`, extrai texto de tabelas antigas e salva os dados em CSV.

## Como executar

```bash
uv run main.py --input-dir img --output-dir output
```

## Saida

- Um CSV por imagem em `output/` (ex: `output/IMG_5545.csv`)
- Um CSV consolidado com todas as imagens: `output/tabelas_extraidas.csv`

## Regras aplicadas

- Detecta automaticamente a estrutura da tabela com IA (Table Transformer): tabela, linhas e colunas.
- Preenche o texto das celulas com OCR (RapidOCR).
- Colapsa subcolunas detectadas pela IA para o layout semantico final (ex: `Sheet`, `Series Title`, `World Index Number`, `Card Nos.`, `No of Cards`).
- Se uma celula vier com simbolo de repeticao (`"`, `'`, `“`, `”`, `〃`), o valor e substituido pelo valor da celula acima na mesma coluna.

## Opcoes uteis

```bash
# Forcar quantidade de colunas
uv run main.py --input-dir img --output-dir output --columns 4

# Ajustar confianca minima do OCR
uv run main.py --input-dir img --output-dir output --min-confidence 0.40
```
