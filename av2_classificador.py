# av2_classifiers.py
import numpy as np
import time
from collections import Counter

def load_banknote_data(url=None, local_path=None):
  """Retorna X (n_amostras, n_atributos) e y (n_amostras,)"""
  if local_path:
    data = np.loadtxt(local_path, delimiter=',')
  X = data[:, :-1]
  y = data[:, -1].astype(int)
  return X, y

def accuracy_score(y_true, y_pred):
  return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred, positive=1):
  tp = np.sum((y_pred == positive) & (y_true == positive))
  fp = np.sum((y_pred == positive) & (y_true != positive))
  if tp + fp == 0:
    return 0.0
  return tp / (tp + fp)

def f1_score(y_true, y_pred, positive=1):
  p = precision_score(y_true, y_pred, positive)
  tp = np.sum((y_pred == positive) & (y_true == positive))
  fn = np.sum((y_pred != positive) & (y_true == positive))
  if tp + fn == 0:
    return 0.0
  recall = tp / (tp + fn)
  if p + recall == 0:
    return 0.0
  return 2 * (p * recall) / (p + recall)

class KNN:
  def __init__(self, k=3, distance='euclidean'):
    self.k = k
    if distance not in ('euclidean', 'manhattan'):
      raise ValueError("distance must be 'euclidean' or 'manhattan'")
    self.distance = distance

  def fit(self, X, y):
    self.X_train = X.copy()
    self.y_train = y.copy()

  def _distances(self, x):
    if self.distance == 'euclidean':
      dif = self.X_train - x
      dists = np.sqrt(np.sum(dif**2, axis=1))
    else:
      dists = np.sum(np.abs(self.X_train - x), axis=1)
    return dists

  def predict(self, X):
    y_pred = []
    for x in X:
      dists = self._distances(x)
      idx = np.argsort(dists)[:self.k]
      votes = self.y_train[idx]
      most_common = Counter(votes).most_common()
      top_count = most_common[0][1]
      top_labels = [lab for lab, cnt in most_common if cnt == top_count]
      pred = min(top_labels)
      y_pred.append(pred)
    return np.array(y_pred)

class GaussianNaiveBayes:
  def fit(self, X, y):
    self.classes = np.unique(y)
    self.means = {}
    self.vars = {}
    self.priors = {}
    for c in self.classes:
      Xc = X[y == c]
      self.means[c] = np.mean(Xc, axis=0)
      self.vars[c] = np.var(Xc, axis=0)
      self.priors[c] = Xc.shape[0] / X.shape[0]

  def _gaussian_prob(self, x, mean, var):
    eps = 1e-9
    coef = 1.0 / np.sqrt(2.0 * np.pi * (var + eps))
    exponent = - ((x - mean) ** 2) / (2.0 * (var + eps))
    return coef * np.exp(exponent)

  def predict(self, X):
    y_pred = []
    for x in X:
      posteriors = {}
      for c in self.classes:
        prior = np.log(self.priors[c])
        probs = self._gaussian_prob(x, self.means[c], self.vars[c])
        log_likelihood = np.sum(np.log(probs + 1e-12))
        posteriors[c] = prior + log_likelihood
      pred = max(posteriors.items(), key=lambda t: t[1])[0]
      y_pred.append(pred)
    return np.array(y_pred)

class MultivariateGaussianBayes:
  def fit(self, X, y):
    self.classes = np.unique(y)
    self.means = {}
    self.covs = {}
    self.priors = {}
    for c in self.classes:
      Xc = X[y == c]
      self.means[c] = np.mean(Xc, axis=0)
      Xc_centered = Xc - self.means[c]
      cov_ml = (Xc_centered.T @ Xc_centered) / Xc_centered.shape[0]
      cov_ml += np.eye(cov_ml.shape[0]) * 1e-6
      self.covs[c] = cov_ml
      self.priors[c] = Xc.shape[0] / X.shape[0]

  def _log_multivariate_gaussian(self, x, mean, cov):
    d = len(x)
    xc = x - mean
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    term = -0.5 * (xc.T @ inv_cov @ xc)
    log_norm = -0.5 * (d * np.log(2 * np.pi) + np.log(det_cov + 1e-12))
    return term + log_norm

  def predict(self, X):
    y_pred = []
    for x in X:
      posteriors = {}
      for c in self.classes:
        log_prior = np.log(self.priors[c])
        log_likelihood = self._log_multivariate_gaussian(x, self.means[c], self.covs[c])
        posteriors[c] = log_prior + log_likelihood
      pred = max(posteriors.items(), key=lambda t: t[1])[0]
      y_pred.append(pred)
    return np.array(y_pred)

def cross_validate(X, y, model_class, model_params=None, n_folds=10, random_state=42):
  if model_params is None:
    model_params = {}
  n_samples = X.shape[0]
  rng = np.random.RandomState(random_state)

  classes, _ = np.unique(y, return_counts=True)
  indices_per_class = {c: np.where(y == c)[0].tolist() for c in classes}
  for c in indices_per_class:
    rng.shuffle(indices_per_class[c])

  folds_idx = [[] for _ in range(n_folds)]
  for c in classes:
    idxs = indices_per_class[c]
    for i, idx in enumerate(idxs):
      folds_idx[i % n_folds].append(idx)

  results = []
  for fold_idx in range(n_folds):
    test_idx = np.array(sorted(folds_idx[fold_idx]))
    train_idx = np.array(sorted([i for i in range(n_samples) if i not in test_idx]))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = model_class(**(model_params or {}))
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    t_test = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, positive=1)
    f1 = f1_score(y_test, y_pred, positive=1)

    results.append({
      'fold': fold_idx,
      'accuracy': acc,
      'precision': prec,
      'f1': f1,
      'train_time': t_train,
      'test_time': t_test
    })

  return results

def summarize_results(results):
  metrics = ['accuracy', 'precision', 'f1', 'train_time', 'test_time']
  summary = {}
  for m in metrics:
    vals = np.array([r[m] for r in results])
    summary[m + '_mean'] = np.mean(vals)
    summary[m + '_std'] = np.std(vals, ddof=0)
  return summary

def carregar_dados_notas(url=None, caminho_local=None):
  return load_banknote_data(url=url, local_path=caminho_local)

def acuracia(y_true, y_pred):
  return accuracy_score(y_true, y_pred)

def precisao(y_true, y_pred, positivo=1):
  return precision_score(y_true, y_pred, positive=positivo)

def f1(y_true, y_pred, positivo=1):
  return f1_score(y_true, y_pred, positive=positivo)

BayesNaiveGaussiano = GaussianNaiveBayes
BayesGaussianoMultivariado = MultivariateGaussianBayes

def validacao_cruzada(X, y, classe_modelo, parametros_modelo=None, n_folds=10, random_state=42):
  return cross_validate(X, y, classe_modelo, model_params=parametros_modelo, n_folds=n_folds, random_state=random_state)

def resumir_resultados(results):
  return summarize_results(results)

if __name__ == '__main__':
  caminho = '/home/av2Ia/data_banknote_authentication.txt'
  X, y = carregar_dados_notas(caminho_local=caminho)

  data = np.loadtxt(caminho, delimiter=',')
  uniq_rows, counts = np.unique(data, axis=0, return_counts=True)
  n_total = data.shape[0]
  n_unique = uniq_rows.shape[0]
  n_dup = np.sum(counts > 1)
  from collections import Counter as _Counter
  class_counts = _Counter(y)
  print(f'Dados: amostras={n_total}, únicas={n_unique}, duplicatas={n_dup}')
  print(f'Contagem por classe: {dict(class_counts)}')

  classifiers = [
    ('KNN (Euclidiana)', KNN, {'k': 3, 'distance': 'euclidean'}),
    ('KNN (Manhattan)', KNN, {'k': 3, 'distance': 'manhattan'}),
    ('Bayes (Multivariado)', MultivariateGaussianBayes, {}),
    ('Bayes (Univariado)', GaussianNaiveBayes, {}),
  ]

  all_summaries = {}
  for name, cls, params in classifiers:
    print("Rodando:", name)
    results = validacao_cruzada(X, y, cls, parametros_modelo=params, n_folds=10, random_state=42)
    summary = resumir_resultados(results)
    all_summaries[name] = (summary, results)
    print("  accuracy: {:.4f} ± {:.4f}".format(summary['accuracy_mean'], summary['accuracy_std']))
    print("  precision: {:.4f} ± {:.4f}".format(summary['precision_mean'], summary['precision_std']))
    print("  f1: {:.4f} ± {:.4f}".format(summary['f1_mean'], summary['f1_std']))
    print("  train_time: {:.4f} ± {:.4f} s".format(summary['train_time_mean'], summary['train_time_std']))
    print("  test_time: {:.4f} ± {:.4f} s".format(summary['test_time_mean'], summary['test_time_std']))
    print()

  header = ("Classificador", "Acurácia", "Precisão", "F1-Score", "Tempo Treino (s)", "Tempo Teste (s)")
  print("{:40} {:20} {:20} {:20} {:20} {:20}".format(*header))
  for name in all_summaries:
    s = all_summaries[name][0]
    acc = f"{s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f}"
    prec = f"{s['precision_mean']:.4f} ± {s['precision_std']:.4f}"
    f1s = f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}"
    tr = f"{s['train_time_mean']:.4f} ± {s['train_time_std']:.4f}"
    te = f"{s['test_time_mean']:.4f} ± {s['test_time_std']:.4f}"
    print("{:40} {:20} {:20} {:20} {:20} {:20}".format(name, acc, prec, f1s, tr, te))

  print('\nResumo: confira os valores acima para comparar os classificadores.')
