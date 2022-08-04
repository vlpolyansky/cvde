import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import eig
import math
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN
from torchvision import transforms
import torch
from numpy import random


def bw(data, factor):
    cov = np.cov(data.T)
    eigval, _ = eig(cov)
    tmp = np.real(np.sqrt(np.mean(eigval)))
    return factor * tmp


def scott_bw(data):
    n = data.shape[0]
    d = data.shape[1]
    factor = n ** (-1. / (d + 4))
    """Computes the covariance matrix for each Gaussian kernel using
    covariance_factor().
    """
    return bw(data, factor)


class Gaussian:
    def __init__(self, dim, sigma):
        self.name = 'single'
        self.dim = dim
        self.sigma = sigma

        self.distr = multivariate_normal(np.zeros(dim), np.eye(dim) * self.sigma)

    def sample(self, n, train=False):
        return np.random.normal(0, self.sigma, size=(n, self.dim))

    def logpdf(self, points):
        return np.log(self.distr.pdf(points))


class TwoGaussians:
    def __init__(self, dim, s1, s2, dst, alpha=0.5):
        self.name = 'double'
        self.dim = dim
        self.s1 = s1
        self.s2 = s2
        self.dst = dst
        self.alpha = alpha

        mean1 = np.zeros(dim)
        mean1[0] -= dst * 0.5
        mean2 = np.zeros(dim)
        mean2[0] += dst * 0.5
        self.distr1 = multivariate_normal(mean1, np.eye(dim) * self.s1)
        self.distr2 = multivariate_normal(mean2, np.eye(dim) * self.s2)

    def sample(self, n, train=False):
        which = np.random.random(n)
        result = np.zeros((n, self.dim))
        g1 = np.random.normal(0, self.s1, size=result.shape) + self.distr1.mean[None, :]
        g2 = np.random.normal(0, self.s2, size=result.shape) + self.distr2.mean[None, :]
        result[which < self.alpha] = g1[which < self.alpha]
        result[which >= self.alpha] = g2[which >= self.alpha]
        return result

    def logpdf(self, points):
        pdf1 = self.distr1.pdf(points)
        pdf2 = self.distr1.pdf(points)
        return np.log(self.alpha * pdf1 + (1 - self.alpha) * pdf2)


class mnist:
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_data = MNIST('./datasets/MNIST/', download=True,
            transform=transform, train=True)
        self.test_data = MNIST('./datasets/MNIST/', download=True,
            transform=transform, train=False)
        self.test_data_ood = FashionMNIST('./datasets/FashionMNIST/', download=True,
            transform=transform, train=False)
        print(self.train_data.__len__())

    def sample(self, n, train=False):
        res = []
        if train:
            dset = self.train_data
        else:
            dset = self.test_data

        idxs = random.permutation(len(dset))[:n]
        for count, i in enumerate(idxs):
            img, lbl = dset.__getitem__(i)
            res.append(img.flatten().numpy())
            if count % 1000 == 0:
                print(count)
        return np.array(res)


class frogs:
    def __init__(self):
        lbls = np.load('./datasets/frog_calls/frogs_lbls.npy')
        data = np.load('./datasets/frog_calls/frogs_data.npy')
        indices = np.random.permutation(data.shape[0])
        data = data[indices]
        lbls = lbls[indices]
        l1 = data.shape[0]
        self.train_data = data[: -int(l1/10)]
        self.test_data = data[-int(l1/10):]

    def sample(self, n, train=False):
        if train:
            dset_full = self.train_data
            l = len(dset_full)
            dset = dset_full[random.permutation(l)][:n]
        else:
            dset = self.test_data[:n]

        return dset
