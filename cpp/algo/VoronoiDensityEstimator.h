#pragma once

#include "AbstractDensityEstimator.h"
#include "cell_kernels.h"
#include "kernels_gpu.h"

class VoronoiDensityEstimator : public AbstractDensityEstimator {
public:
    VoronoiDensityEstimator(const dmatrix &points, const ptr<CellKernel> &cell_kernel, int seed,
                            int njobs, int nrays_weights, int nrays_sampling=5,
                            RayStrategyType strategy = BRUTE_FORCE,
                            const ptr<Bounds> &bounds = std::make_shared<Unbounded>());

    dmatrix sample(int sample_size) const override;

    void initialize_weights() override;

    void initialize_weights_uncentered(const dmatrix &ref_mat);

    void update_weights(int nrays);

    ftype estimate_single(const dvector &point) const override;

    dvector sample_single() const override;

    void centroid_smoothing(int smoothing_steps, int points_per_centroid);

    void reset_points(const dmatrix &points) override;

//    void set_max_block_size(int _max_block_size);

    dmatrix sample_masked(int sample_size, const dvector &mask) const;  // mask has NaN in free directions

private:
    dvector sample_within_cell(int cell_index, RandomEngine &engine) const;

    vec<ftype> weight_sums;
    vec<int> weight_counts;

    RayStrategyType strategy;
    ptr<EuclideanKernel> geometry_kernel;

    ptr<CellKernel> cell_kernel;

    int nrays_si;
    int nrays_hr;

    ftype norm_constant;
};
