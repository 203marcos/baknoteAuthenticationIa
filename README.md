# AV2 - Classificadores (em Português)

Este repositório contém uma implementação didática de classificadores e validação cruzada para o dataset "banknote authentication".

## Arquivos principais

- `av2_classificador.py`: código contendo implementações de KNN, Naive Gaussian Bayes (univariado) e Gaussian Bayes multivariado, além de validação cruzada e métricas.
- `data_banknote_authentication.txt`: dataset CSV (última coluna = rótulo 0/1).

## Usando um ambiente virtual (recomendado)

Recomenda-se criar e ativar um ambiente virtual para instalar as dependências (evita misturar com o Python do sistema). Exemplo:

```bash
# criar o ambiente virtual
python3 -m venv venv
# ativar o ambiente
source venv/bin/activate
# instalar dependências necessárias (p.ex. numpy)
pip install numpy
# rodar o script
python3 av2_classificador.py
```

## Funções e classes (explicação em português)

- `carregar_dados_notas(caminho_local=None, url=None)`
  - Carrega o dataset. Se `caminho_local` for fornecido, lê o arquivo CSV local.
  - Retorna `X` (atributos) e `y` (rótulos inteiros).

- `KNN(k=3, distance='euclidean')`
  - Classificador K-Nearest Neighbors implementado do zero.
  - `fit(X, y)` armazena as amostras de treino.
  - `predict(X_test)` calcula distâncias e vota pelos k vizinhos.

- `GaussianNaiveBayes`
  - Implementação do Naive Bayes Gaussiano (assume independência entre atributos).
  - Calcula médias e variâncias por classe e usa densidade normal por dimensão.

- `MultivariateGaussianBayes`
  - Modela cada classe como uma Gaussiana multivariada (leva em conta covariância entre atributos).
  - Calcula média vetorial e matriz de covariância ML regularizada.

- `validacao_cruzada(X, y, classe_modelo, parametros_modelo, n_folds=10)`
  - Realiza validação cruzada estratificada (mantém proporção de classes nas folds).
  - Sempre instancia um novo objeto de `classe_modelo` por fold para evitar vazamento.
  - Retorna lista de resultados (por fold) com métricas e tempos.

## Métricas impressas

- **Acurácia**: proporção de previsões corretas.
- **Precisão**: TP / (TP + FP) para a classe positiva (1).
- **F1-score**: média harmônica entre precisão e recall.
- **Tempo de treino / teste**: tempo gasto em `fit` e `predict` por fold (segundos).

## Interpretação dos resultados obtidos

- **KNN (k=3)** apresentou acurácia 1.0 nas 10 folds — indica que o dataset é altamente separável pelas features, ou existem amostras muito semelhantes/duplicadas entre treino e teste nas folds.
- **Bayes Multivariado** teve acurácia ≈ 0.9847 — bom desempenho que indica que a modelagem multivariada captura bem as distribuições das classes.
- **Bayes Univariado (Naive)** obteve acurácia ≈ 0.8404 — assume independência entre atributos; se os atributos forem correlacionados isso prejudica o desempenho.

## Melhorias sugeridas

- Usar StratifiedKFold com várias seeds e média das runs para testar estabilidade.
- Checar duplicatas (o script já imprime um resumo antes de rodar) e remover/avaliar se necessário.
- Otimizar KNN (KDTree) para datasets maiores ou usar scikit-learn para produção.

## Contato

Se quiser, posso aplicar mais melhorias: cache de inversos para o Bayes multivariado, saída de matriz de confusão, ou salvar resultados em CSV.
