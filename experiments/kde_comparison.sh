#!/bin/bash

set -e

for dataset in "gaussians" "frogs" "mnist"; do
    python train_estimator.py KDE ${dataset}
    python train_estimator.py KDE ${dataset} --adaptive
    python train_estimator.py CVDE ${dataset}
    python plot_kde_comparison.py ${dataset}
done