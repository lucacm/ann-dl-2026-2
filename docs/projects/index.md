# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

!!! danger "Preencha antes de qualquer entrega"

    Toda entrega do projeto é avaliada em equipe. Se os nomes não estiverem aqui, não há
    como atribuir a nota — e o mesmo vale para o `mkdocs.yml`, cujo `site_author` deve
    listar o grupo.

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Luca Cazzolato Machado | lucacm@al.insper.edu.br | [@lucacm](https://github.com/lucacm) |
| Gabriel Cavarsan | gabrielcc2@al.insper.edu.br | [@Gabriel-Cavarsan](https://github.com/Gabriel-Cavarsan) |
| Gabriel Gerner Maggion | gabrielgm1@al.insper.edu.br | [@gabrielgmaggion-dot](https://github.com/gabrielgmaggion-dot) |

Times de 2 a 3 pessoas. Repita esses nomes no cabeçalho de cada entrega — quem corrige pode
abrir uma página sozinha, sem passar por aqui.

## As três entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | Classificação **ou** Regressão | *a entregar* |
| 3 | Generativo | *a entregar* |

Datas e pesos são da sua edição — veja o
[overview](https://insper.github.io/ann-dl/){:target='_blank'}.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Deliverables bem escritos não sustentam uma equipe que não consegue explicar o que
    entregou. Escreva os relatórios de modo que você consiga defendê-los meses depois, e
    confira no overview da sua edição como a prova entra na nota.

!!! warning "Escolha uma: classificação ou regressão"

    A segunda entrega é **uma das duas**, não as duas. Este template traz as duas pastas
    para você escolher; depois de decidir, apague a que não vai usar — da pasta `docs/projects/`
    **e** da `nav` no `mkdocs.yml`.

## Dataset

O mesmo dataset atravessa as três entregas — escolhê-lo bem no EDA é o que torna as outras
duas viáveis.

| | |
|---|---|
| **Nome** | Vehicle dataset (CarDekho), arquivos `Car details v3.csv` + `car details v4.csv` |
| **Fonte (URL)** | <https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho> |
| **Licença / termos de uso** | conforme a página do dataset no Kaggle |
| **Amostras** | 8754 (após deduplicação; 6699 do v3, 2055 do v4) |
| **Features** | 11 (8 numéricas, 3 categóricas) |
| **Variável alvo** | `price`: preço pedido no anúncio, em rúpias (modelado em log) |
| **Tarefa escolhida** | Regressão |

Preço de carro usado é uma regressão com estrutura real: depende de idade, uso, motor, marca e
câmbio, com interações (o efeito da quilometragem inverte de sinal quando se controla pelo
ano). O alvo é muito assimétrico, de 30 mil a 35 milhões de rúpias, e os dois arquivos
combinados cobrem tanto o mercado popular quanto o de luxo. Isso torna a tarefa interessante
em vez de trivial: um modelo precisa acertar a ordem de grandeza nas duas pontas. Detalhes e
evidências em [1. EDA](eda/index.md).

## Status

- [x] **1. EDA**
- [ ] **2. Classificação ou Regressão**
- [ ] **3. Generativo**

## Registro de decisões

Anote aqui as decisões que atravessam as entregas — troca de dataset, mudança de alvo,
recorte de features — com a data. É o que permite reconstruir o raciocínio na prova de
projeto.

| Data | Decisão | Motivo |
|------|---------|--------|
| 2026-10-07 | Dataset CarDekho, tarefa de regressão do preço | dataset tabular público com mais de 1000 linhas, features numéricas e categóricas e alvo no arquivo |
| 2026-10-07 | Usar os arquivos v3 + v4, descartar `car data` e `CAR DETAILS FROM CAR DEKHO` | `car data` tem 301 linhas; FROM DEKHO não acrescenta colunas e repete 321 carros do v3 com preço diferente; o v4 cobre a faixa acima de 10 lakh que falta no v3 ([EDA, 1A](eda/index.md#1a-dicionario-de-dados)) |
| 2026-10-07 | Manter só as 12 colunas comuns a v3 e v4 | colunas de um arquivo só teriam 20% (`mileage`) a 80% (exclusivas do v4) de ausência estrutural |
| 2026-10-07 | Deduplicar no arquivo bruto e colapsar reanúncios no preço mediano | 1202 duplicatas exatas e 213 grupos do mesmo carro com preço diferente no v3 ([EDA, 1B](eda/index.md#1b-qualidade)) |
| 2026-10-07 | Alvo em log; `source` fora do modelo | assimetria 8,39 → 0,43; a diferença v3/v4 cai de 2,06× para 1,11× quando se comparam carros parecidos ([EDA, 3B](eda/index.md#3b-categorica-alvo)) |
