# ANN-DL 2026.2 — Entregas

Repositório de entregas da disciplina de Redes Neurais Artificiais & Deep Learning
([enunciados](https://insper.github.io/ann-dl/)), gerado como site MkDocs Material e
publicado no GitHub Pages.

**Site publicado:** <https://lucacm.github.io/ann-dl-2026-2/>

## Estrutura

```
docs/
  index.md                              # capa: identificação e status das entregas
  exercises/
    data/{index.md,code/,figures/}      # entregue
    perceptron/{index.md,code/,figures/}  # entregue
    mlp/{index.md,code/,figures/}
    vae/{index.md,code/,figures/}
  projects/                             # projeto em equipe, três entregas
mkdocs.yml
requirements.txt
```

Cada entrega tem a própria pasta, sempre no mesmo formato: `index.md` para o relatório,
`code/` para os scripts (incluídos no relatório via `--8<--`, nunca copiados e colados) e
`figures/` para as imagens commitadas.

MLP, VAE e o projeto ainda não foram entregues nesta edição; as pastas já existem
no repositório, prontas para quando cada um for feito.

## Setup

```shell
python3 -m venv env
source ./env/bin/activate          # Windows: .\env\Scripts\activate
python3 -m pip install -r requirements.txt --upgrade
```

## Rodando localmente

```shell
mkdocs serve -o
```

Antes de qualquer commit, valide com o modo estrito (o CI publica mesmo com avisos, este
comando não):

```shell
mkdocs build --strict
```

## Publicação

O workflow em [.github/workflows/main.yaml](.github/workflows/main.yaml) roda
`mkdocs gh-deploy --force` a cada push na `main`: ele constrói o HTML e o publica na branch
`gh-pages`, que o GitHub Pages serve.
