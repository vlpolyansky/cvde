#include "../algo/VoronoiDensityEstimator.h"

int main(int argc, const char** argv) {
    using Estimator = VoronoiDensityEstimator;

    int njobs = 1;
    int number_of_rays = 100;
    int nrays_sampling = 1;

    std::string size_str = argv[1];
    std::string dim_str = argv[2];
    std::string base_fname = "data/other/gaussian_1000_2d";
//    std::string test_fname = "data/other/sampled_gaussian_1000_2d";

//    std::string base_fname = "data/hypercube_data/train_" + size_str + "_" + dim_str;
//    std::string test_fname = "data/hypercube_data/test_" + size_str + "_" + dim_str;
//    std::string base_fname = "data/laplace_data/train_" + size_str + "_" + dim_str;
//    std::string test_fname = "data/laplace_data/test_" + size_str + "_" + dim_str;

//    std::string base_fname = "data/other/gaussian_1000_8d";
//    std::string test_fname = "data/other/sampled_gaussian_1000_8d";

//    std::string base_fname = "data/mnist/train_data_8";
//    std::string test_fname = "data/mnist/test_data_89";

    int nsamples = 1000;

    dmatrix points = npy2matrix(cnpy::npy_load(base_fname + ".npy"));
    int dim = points.rows();

//    ptr<Bounds> bounds = std::make_shared<BoundingBox>(dim, 25);
    ptr<Bounds> bounds = std::make_shared<Unbounded>();

//    RayStrategyType strategy = BRUTE_FORCE;
    RayStrategyType strategy = BRUTE_FORCE_GPU;

    ftype sigma = 0.3;

    // scott's rule
//    sigma = pow(ftype(points.cols()), -1 / ftype(dim + 4));

//    ptr<CellKernel> cell_kernel = std::make_shared<UniformCellKernel>(dim);
    ptr<CellKernel> cell_kernel = std::make_shared<GaussianCellKernel>(dim, sigma);

    Estimator estimator(points, cell_kernel, 239, njobs, number_of_rays, nrays_sampling,
                        strategy, bounds);
//    estimator.centroid_smoothing(1, 1);

    std::cout << "estimating weights" << std::endl;
    estimator.initialize_weights();
//    vec<ftype> weights = estimator.get_weights();

//    std::cout << "saving weights" << std::endl;
//    cnpy::npy_save(base_fname + "_weights_vor.npy", weights.data(), {weights.size()});


//    dmatrix test_points = npy2matrix(cnpy::npy_load(test_fname + ".npy"));
//
//    std::cout << "estimating test densities" << std::endl;
//    vec<ftype> estimates = estimator.estimate(test_points);
//    cnpy::npy_save(test_fname + "_estimates.npy", estimates.data(), {estimates.size()});

//    std::cout << "sampling" << std::endl;
//    auto sample = estimator.sample(nsamples);
//
//    std::cout << "saving samples" << std::endl;
//    cnpy::npy_save(base_fname + "_sampled.npy", sample.data(), {static_cast<size_t>(nsamples),
//                                                                          static_cast<size_t>(dim)});

    return 0;
}