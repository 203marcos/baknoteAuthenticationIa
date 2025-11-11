import numpy as np
from av2_classificador import load_banknote_data, cross_validate, KNN

path = '/home/av2Ia/data_banknote_authentication.txt'
X, y = load_banknote_data(local_path=path)

def summarize(results):
    import numpy as np
    accs = np.array([r['accuracy'] for r in results])
    precs = np.array([r['precision'] for r in results])
    f1s = np.array([r['f1'] for r in results])
    return accs.mean(), accs.std(), precs.mean(), f1s.mean()

print('Samples, features:', X.shape)

for k in [1,3,5]:
    print('\nK=', k, '(unscaled)')
    results = cross_validate(X, y, KNN, model_params={'k': k, 'distance': 'euclidean'}, n_folds=10, random_state=42)
    acc_mean, acc_std, prec_mean, f1_mean = summarize(results)
    print(' acc mean/std =', acc_mean, acc_std)
    print(' prec mean =', prec_mean)
    print(' f1 mean =', f1_mean)

# standardize features (zero mean, unit var)
Xs = (X - X.mean(axis=0)) / X.std(axis=0)
for k in [1,3,5]:
    print('\nK=', k, '(standardized)')
    results = cross_validate(Xs, y, KNN, model_params={'k': k, 'distance': 'euclidean'}, n_folds=10, random_state=42)
    acc_mean, acc_std, prec_mean, f1_mean = summarize(results)
    print(' acc mean/std =', acc_mean, acc_std)
    print(' prec mean =', prec_mean)
    print(' f1 mean =', f1_mean)

# try manhattan metric as well
print('\nK=3 (manhattan)')
results = cross_validate(X, y, KNN, model_params={'k': 3, 'distance': 'manhattan'}, n_folds=10, random_state=42)
acc_mean, acc_std, prec_mean, f1_mean = summarize(results)
print(' acc mean/std =', acc_mean, acc_std)
print(' prec mean =', prec_mean)
print(' f1 mean =', f1_mean)
