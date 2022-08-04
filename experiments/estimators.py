import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import vde
from numpy import random
import awkde
from scipy.special import gamma

class estimator():
    def __init__(self):
        pass
    def logdensity(self):
        pass
    def sample(self):
        pass

class CVDE(estimator):
    def __init__(self, train_data, dim, band, seed=238, num_threads=8, num_rays=5000, num_steps=10, adaptive=False):
        if adaptive:
            ck = vde.AdaptiveGaussianCellKernel(dim, band)
        else:
            ck = vde.GaussianCellKernel(dim, band)

        self.vde = vde.VoronoiDensityEstimator(train_data, ck, seed, num_threads, num_rays, num_steps, vde.RayStrategyType.BRUTE_FORCE_GPU, vde.Unbounded())
        self.vde.initialize_weights()
        self.adaptive = adaptive

    def logdensity(self, data):
        return np.log(self.vde.estimate(data))

    def sample(self, n):
        return self.vde.sample(n)


class KDE(estimator):
    def __init__(self, train_data, band, adaptive=False):
        if adaptive:
            self.kde = awkde.GaussianKDE(glob_bw=band)
            self.kde.fit(train_data)
            # print(f'{self.kde._glob_bw=}')
        else:
            self.kde = KernelDensity(kernel='gaussian', bandwidth=band).fit(train_data)
        self.adaptive = adaptive

    def logdensity(self, samples):
        if self.adaptive:
            return np.log(self.kde.predict(samples))
        else:
            return self.kde.score_samples(samples)

    def sample(self, n):
        return self.kde.sample(n)
