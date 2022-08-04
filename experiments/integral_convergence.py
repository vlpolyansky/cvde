import random

import matplotlib.cm
import vde
import numpy as np
import matplotlib.pyplot as plt
from distributions import Gaussian, scott_bw, mnist
from tqdm import tqdm, trange
import os
import distributions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['mnist', 'gaussian'])
args = parser.parse_args()

dataset = args.dataset

os.makedirs('images', exist_ok=True)

njobs = 8
seed = 239
random.seed(seed + 1)
np.random.seed(seed + 2)

n = 1000

if dataset == 'mnist':
    distr = mnist()
    data = distr.sample(n)
    bw = 1.
elif dataset == 'gaussian':
    distr = Gaussian(10, 1.)
    data = distr.sample(n)
    bw = scott_bw(data)
else:
    raise Exception(f'Unknown dataset: {dataset}')

dim = data.shape[1]

ck = vde.GaussianCellKernel(dim, bw)

si_rays_step = 500
si_rays_list = np.arange(si_rays_step, si_rays_step * 10, si_rays_step, dtype=np.int32)
n_eval = 10
eval_data = data[np.random.choice(data.shape[0], n_eval, replace=False), :]

values = np.zeros((n_eval, si_rays_list.shape[0]), dtype=np.float128)

vde = vde.VoronoiDensityEstimator(data, ck, seed, njobs, si_rays_step, 100,
                                  vde.RayStrategyType.BRUTE_FORCE_GPU, vde.Unbounded())

for i, si_rays in enumerate(tqdm(si_rays_list)):
    vde.initialize_weights()
    estimate = vde.estimate(eval_data)
    values[:, i] = np.log(estimate)

plt.figure()
for i in range(n_eval):
    plt.plot(si_rays_list, values[i], c=matplotlib.cm.get_cmap('Blues')(np.random.random()))

plt.savefig(f'images/si_{dataset}.png')
plt.show()
