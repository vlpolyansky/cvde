#include "AbstractDensityEstimator.h"

AbstractDensityEstimator::AbstractDensityEstimator(const dmatrix &points, int seed, int njobs)
        : re(seed), njobs(njobs),
        weights(points.cols(), 0), is_initialized(points.cols(), false),
        data_n(points.cols()), dim(points.rows()) {
    this->points = points;
}

vec<ftype> AbstractDensityEstimator::estimate(const dmatrix &points) const {
    int n = points.cols();
    vec<ftype> results(n, 0);
    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);
    my_tqdm bar(n);
    #pragma omp parallel
    {
        re.fix_random_engines();
        #pragma omp for
        for (int i = 0; i < n; i++) {
            bar.atomic_iteration();
            results[i] = estimate_single(static_cast<dvector>(points.col(i)));
        }
    }
    bar.bar().finish();
    omp_set_num_threads(old_threads);
    return results;
}

dmatrix AbstractDensityEstimator::sample(int size) const {
    dmatrix results(dim, size);
    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);
    my_tqdm bar(size);
    #pragma omp parallel
    {
        re.fix_random_engines();
        #pragma omp for
        for (int i = 0; i < size; i++) {
            bar.atomic_iteration();
            results.col(i) = sample_single();
        }
    }
    bar.bar().finish();
    omp_set_num_threads(old_threads);
    return results;
}

const vec<ftype> &AbstractDensityEstimator::get_weights() const {
    return weights;
}

void AbstractDensityEstimator::reset_points(const dmatrix &points) {
    ensure(points.cols() == data_n && points.rows() == dim,
           "Number of points or dimensionality changed!");
    this->points = points;
    weights = vec<ftype>(points.cols(), 0);
    is_initialized = vec<bool>(points.size(), false);
}

const dmatrix &AbstractDensityEstimator::get_points() const {
    return points;
}
