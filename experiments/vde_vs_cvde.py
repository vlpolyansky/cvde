import random
import vde
import numpy as np
import matplotlib.pyplot as plt
from distributions import Gaussian, scott_bw
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import os

os.makedirs('images', exist_ok=True)

njobs = 8
seed = 239
si_rays = 10000
hr_rays = 1000
random.seed(seed + 1)
np.random.seed(seed + 2)

n = 1000
dim = 10

bbox = 3.5

distr = Gaussian(dim, 1)
data = distr.sample(n)
assert(np.max(np.abs(data[:, :2])) < bbox)

plt.figure(figsize=(8, 8))
plt.axes().set_aspect('equal')
plt.plot(*data[:, :2].T, '.')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("original")
plt.savefig(f'images/bbox_original_{dim}d.png')
# plt.show(block=False)

bw = scott_bw(data)
print(f'{bw=}')

vde_orig = vde.VoronoiDensityEstimator(data, vde.UniformCellKernel(dim), seed, njobs, si_rays, hr_rays,
                                  vde.RayStrategyType.BRUTE_FORCE_GPU, vde.BoundingBox(dim, bbox))
vde_orig.initialize_weights()
data_vde = vde_orig.sample(n)
plt.figure(figsize=(8, 8))
plt.axes().set_aspect('equal')
plt.plot(*data_vde[:, :2].T, '.')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("VDE w/ bounding box")
plt.plot([bbox, -bbox, -bbox, bbox, bbox], [bbox, bbox, -bbox, -bbox, bbox], '--', c='gray')
plt.savefig(f'images/bbox_vde_{dim}d.png')
# plt.show(block=False)


cvde = vde.VoronoiDensityEstimator(data, vde.GaussianCellKernel(dim, bw), seed, njobs, si_rays, hr_rays,
                                  vde.RayStrategyType.BRUTE_FORCE_GPU, vde.Unbounded())
cvde.initialize_weights()
data_vde = cvde.sample(n)
plt.figure(figsize=(8, 8))
plt.axes().set_aspect('equal')
plt.plot(*data_vde[:, :2].T, '.')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title(f"CVDE($\\sigma={bw:0.3f})$")
plt.savefig(f'images/bbox_cvde_{dim}d.png')
# plt.show()
