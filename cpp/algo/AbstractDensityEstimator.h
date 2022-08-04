#pragma once

#include "../utils.h"
#include "../RandomEngine.h"

class AbstractDensityEstimator {
public:
    AbstractDensityEstimator(const dmatrix &points, int seed = 239, int njobs = 1);

    virtual void initialize_weights() = 0;

    virtual ftype estimate_single(const dvector &point) const = 0;

    virtual dvector sample_single() const = 0;

    virtual vec<ftype> estimate(const dmatrix &points) const;

    virtual dmatrix sample(int size) const;

    const vec<ftype> &get_weights() const;

    virtual void reset_points(const dmatrix &points);

    const dmatrix &get_points() const;

protected:
    dmatrix points;
    const int data_n;
    const int dim;

    mutable RandomEngineMultithread re;

    vec<ftype> weights;
    vec<bool> is_initialized;

    int njobs;
};
