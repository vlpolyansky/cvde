from estimators import *
from distributions import *
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('estimator', choices=['KDE', 'CVDE'])
parser.add_argument('dataset', choices=['gaussians', 'mnist', 'frogs'])
parser.add_argument('--adaptive', dest='adaptive', action='store_true')
parser.add_argument('--no-adaptive', dest='adaptive', action='store_false')
parser.set_defaults(adaptive=False)
args = parser.parse_args()

estimator = args.estimator
adaptive = args.adaptive
dataset = args.dataset

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

num_bands = 20
runs = 5

if dataset == 'gaussians':
    n_train = 1000
    n_test = 1000
    isomap = False
    d_iso = None
    band_list = np.linspace(start=.1, stop=50, num=num_bands)
    distr = TwoGaussians(10, 100, .1, 1, alpha=0.5)
elif dataset == 'mnist':
    n_train = 30000
    n_test = 10000
    isomap = True
    d_iso = 10
    band_list = np.linspace(start=.1, stop=1., num=num_bands)
    distr = mnist()
elif dataset == 'frogs':
    n_train = 3238
    n_test = 719
    isomap = True
    d_iso = 10
    band_list = np.linspace(start=.01, stop=.2, num=num_bands)
    distr = frogs()
else:
    raise Exception(f'Unknown dataset: {dataset}')


embedding = PCA(n_components=d_iso)

scores = []

for run in range(runs):
    scores.append([])

    train_data = distr.sample(n_train, train=True)
    test_data = distr.sample(n_test, train=False)
    test_data_ood = distr.sample(n_test, train=False)
    d = train_data.shape[-1]

    if isomap:
        d = d_iso
        l = len(train_data)
        reduced = embedding.fit_transform(np.concatenate((train_data, test_data), axis=0))
        train_data = reduced[:l]
        test_data = reduced[l:]

    for band in tqdm(band_list):
        if estimator == 'CVDE':
            est = CVDE(train_data, d, band, adaptive=adaptive)
        elif estimator == 'KDE':
            est = KDE(train_data, band, adaptive=adaptive)
        else:
            raise Exception(f'Unknown estimator: {estimator}')

        scores_cur = est.logdensity(test_data).mean()
        scores[run].append(scores_cur)


scores = np.array(scores)
if adaptive:
    adpt = '_ada'
else:
    adpt = ''

np.save(f'{output_dir}/{estimator}_{dataset}{adpt}.npy', scores)
np.save(f'{output_dir}/bandwidths_{dataset}.npy', band_list)
