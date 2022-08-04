import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['gaussians', 'mnist', 'frogs'])
args = parser.parse_args()

dataset = args.dataset

dir = 'results'

band_list = np.load(f'{dir}/bandwidths_{dataset}.npy')
if dataset == 'gaussians':
    range = [-100, -30]
elif dataset == 'mnist':
    range = [-30, 12]
elif dataset == 'frogs':
    range = [-5, 20]

estimators = ['CVDE', 'KDE', 'KDE']
adaptive = [False, False, True]
labels = ['CVDE', 'KDE', 'AdaKDE']
colors = ['tab:orange', 'tab:green', 'tab:blue']

alpha = 0.2

for est, ada, label, col in zip(estimators, adaptive, labels, colors):
    data = np.load(f'{dir}/{est}_{dataset}{"_ada" if ada else ""}.npy')
    data[np.isinf(data)] = -10000000000
    print(data)
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    plt.plot(band_list, mean, label=label, c=col, linewidth=2, zorder=2)
    plt.fill_between(band_list, mean - std, mean + std, alpha=alpha, color=col, zorder=2)


ax = plt.gca()
ax.set_ylabel('avg. log-likelihood')
ax.set_xlabel('bandwidth')
plt.ylim(range[0], range[1])
lines, names = ax.get_legend_handles_labels()
lines, names = reversed(lines), reversed(names)
plt.savefig(f'images/vskde_{dataset}.png')

fig = plt.gcf()
size = fig.get_size_inches()
size[0] = 1

plt.clf()

plt.figure(figsize=size)
plt.axis(False)
plt.legend(lines, names, loc="center", frameon=False)
plt.tight_layout()
plt.savefig("images/vskde_labels.png")
