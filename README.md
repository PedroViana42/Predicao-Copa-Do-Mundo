# Projeto de IA: Predicao de Resultados da Copa do Mundo

Este projeto usa dados historicos de partidas da Copa do Mundo para treinar um modelo de classificacao capaz de prever o resultado de confrontos entre duas selecoes.

O objetivo inicial e prever partidas individuais. Como continuidade, o mesmo modelo pode ser usado para simular uma edicao inteira da Copa, incluindo fase de grupos, mata-mata e probabilidades de titulo.

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier

## Arquivos do projeto

- `matches.csv`: base historica de partidas da Copa do Mundo.
- `predicao.py`: script principal com carregamento dos dados, criacao das estatisticas, treinamento do modelo e teste de previsao.
- `copa 2026WCup_2026_4.2.5_en.xlsx`: planilha da Copa de 2026 usada para receber os placares previstos.
- `copa_2026_preenchida.xlsx`: planilha gerada automaticamente com os jogos da fase de grupos preenchidos.
- `Relatorio_Passo_a_Passo_Predicao_Copa.docx`: documento explicando o processo do projeto.

## Como executar

No terminal, dentro da pasta do projeto, execute:

```bash
python predicao.py
```

O script ira:

1. Carregar a base `matches.csv`.
2. Criar a coluna alvo `resultado`.
3. Calcular estatisticas historicas por selecao.
4. Montar a base de treinamento.
5. Treinar um modelo `RandomForestClassifier`.
6. Avaliar o modelo com `classification_report`.
7. Fazer uma previsao de exemplo para `Brazil x France`.
8. Mostrar um ranking historico das selecoes por taxa de vitoria.
9. Criar a planilha `copa_2026_preenchida.xlsx` com 72 jogos da fase de grupos preenchidos.

## Base de dados

A base contem partidas historicas da Copa do Mundo, com colunas como:

- `year`: ano da Copa.
- `stage_name`: fase da competicao.
- `team_a_name` e `team_b_name`: selecoes da partida.
- `team_a_score` e `team_b_score`: placar.
- `team_a_win`, `team_b_win` e `draw`: resultado da partida.
- `extra_time` e `penalty_shootout`: informacoes sobre prorrogacao e penaltis.

O arquivo `matches.csv` esta separado por virgulas. Por isso, a leitura correta no Pandas e:

```python
df = pd.read_csv("matches.csv")
```

## Como o modelo funciona

Primeiro, o script transforma o resultado da partida em uma unica coluna chamada `resultado`:

- `1`: vitoria do Time A.
- `0`: empate.
- `-1`: vitoria do Time B.

Depois, o codigo cria estatisticas historicas para cada selecao, considerando jogos em que ela aparece como Time A e como Time B. Entre as estatisticas usadas estao:

- quantidade de jogos;
- media de gols marcados;
- media de gols sofridos;
- taxa de vitoria;
- taxa de empate;
- taxa de derrota.

Essas estatisticas sao usadas como atributos numericos para treinar o modelo.

As estatisticas usam ponderacao por recencia: jogos mais novos recebem peso maior que jogos antigos. Assim, Copas recentes influenciam mais a previsao do que partidas muito antigas.

## Exemplo de previsao

No final do script existe um teste com:

```python
time_a_teste = "Brazil"
time_b_teste = "France"
```

O programa exibe o resultado previsto e as probabilidades de:

- vitoria do Time A;
- empate;
- vitoria do Time B.

## Preenchimento da planilha

O script tambem preenche automaticamente a planilha da Copa de 2026. Ele le os confrontos da fase de grupos no arquivo `copa 2026WCup_2026_4.2.5_en.xlsx`, usa o modelo treinado para prever os placares e salva uma nova versao chamada:

```text
copa_2026_preenchida.xlsx
```

A planilha original nao e sobrescrita.

## Limitacoes

O modelo usa apenas dados historicos da Copa do Mundo. Ele nao considera fatores atuais, como:

- elenco convocado;
- lesoes;
- momento dos jogadores;
- tecnico;
- ranking FIFA atual;
- desempenho recente fora da Copa;
- mando de campo;
- chaveamento real do torneio.

Por isso, as previsoes devem ser interpretadas como uma estimativa baseada em historico, nao como uma garantia de resultado.

## Proximos passos

Para evoluir o projeto e prever um campeao da Copa de 2026, os proximos passos seriam:

1. Adicionar a tabela de grupos e jogos da Copa de 2026.
2. Prever os jogos da fase de grupos.
3. Atribuir pontos para vitoria, empate e derrota.
4. Classificar as selecoes dentro de cada grupo.
5. Montar o mata-mata.
6. Prever cada confronto eliminatorio.
7. Repetir a simulacao varias vezes para gerar probabilidades de titulo.

## Observacao sobre erro corrigido

Durante o desenvolvimento apareceu o erro:

```text
KeyError: 'team_a_win'
```

Esse erro aconteceu porque o CSV estava sendo lido com separador de tabulacao:

```python
pd.read_csv("matches.csv", sep="\t")
```

Como o arquivo estava separado por virgulas, o Pandas nao criou a coluna `team_a_win` corretamente. A correcao foi remover o parametro `sep="\t"`:

```python
pd.read_csv("matches.csv")
```
