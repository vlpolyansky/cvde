#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "algo/cell_kernels.h"
#include "algo/kernels.h"
#include "algo/VoronoiDensityEstimator.h"

namespace py = boost::python;
namespace numpy = boost::python::numpy;

void copy_numpy_data_float(const std::string &dtype, const numpy::ndarray &array, size_t size, ftype *out) {
    if (dtype == "float32") {
        assert(sizeof(float) == 4);
        float *array_data = reinterpret_cast<float *>(array.get_data());
        std::transform(array_data, array_data + size, out, [](float a) -> ftype {return static_cast<ftype>(a);});
    } else if (dtype == "float64") {
        assert(sizeof(double) == 8);
        double *array_data = reinterpret_cast<double *>(array.get_data());
        std::transform(array_data, array_data + size, out, [](double a) -> ftype {return static_cast<ftype>(a);});
    } else if (dtype == "float128") {
        assert(sizeof(long double) == 16);
        long double *array_data = reinterpret_cast<long double *>(array.get_data());
        std::transform(array_data, array_data + size, out, [](long double a) -> ftype {return static_cast<ftype>(a);});
    } else {
        throw std::runtime_error("Unknown dtype: " + dtype);
    }
}

dmatrix numpy_to_dmatrix(const numpy::ndarray &array) {
    assert(array.get_nd() == 2);
    auto shape = array.get_shape();
    dmatrix res(shape[1], shape[0]);
    std::string dtype(py::extract<char const *>(py::str(array.get_dtype())));
    copy_numpy_data_float(dtype, array, shape[0] * shape[1], res.data());
    return res;
}

dvector numpy_to_dvector(const numpy::ndarray &_array) {
    const numpy::ndarray &array = _array.squeeze();
    assert(array.get_nd() == 1);
    auto shape = array.get_shape();
    dvector res(shape[0]);
    std::string dtype(py::extract<char const *>(py::str(array.get_dtype())));
    copy_numpy_data_float(dtype, array, shape[0], res.data());
    return res;
}

vec<ftype> numpy_to_vector(const numpy::ndarray &_array) {
    const numpy::ndarray &array = _array.squeeze();
    assert(array.get_nd() == 1);
    auto shape = array.get_shape();
    vec<ftype> res(shape[0]);
    std::string dtype(py::extract<char const *>(py::str(array.get_dtype())));
    copy_numpy_data_float(dtype, array, shape[0], res.data());
    return res;
}

numpy::ndarray vector_to_numpy(const vec<ftype> &data) {
    // At the moment, always saves to np.float128
    Py_intptr_t shape[1] = { static_cast<Py_intptr_t>(data.size()) };
    numpy::ndarray result = numpy::zeros(1, shape, numpy::dtype(py::object("float128")));
    std::transform(data.begin(), data.end(), reinterpret_cast<long double*>(result.get_data()),
                   [](ftype a) {return static_cast<long double>(a);});
    return result;
}

numpy::ndarray dmatrix_to_numpy(const dmatrix &data) {
    // At the moment, always saves to np.float128
    Py_intptr_t shape[2] = { static_cast<Py_intptr_t>(data.cols()), static_cast<Py_intptr_t>(data.rows()) };
    numpy::ndarray result = numpy::zeros(2, shape, numpy::dtype(py::object("float128")));
    std::transform(data.data(), data.data() + data.size(), reinterpret_cast<long double*>(result.get_data()),
                   [](ftype a) {return static_cast<long double>(a);});
    return result;
}

class WrapperFuncs {
public:
    static ptr<VoronoiDensityEstimator> VoronoiDensityEstimator_constructor(
            const numpy::ndarray &points, const ptr<CellKernel> &cell_kernel, int seed,
            int njobs, int nrays_weights, int nrays_sampling, RayStrategyType strategy, const ptr<Bounds> &bounds) {
        return std::make_shared<VoronoiDensityEstimator>(numpy_to_dmatrix(points), cell_kernel, seed, njobs,
                                                         nrays_weights, nrays_sampling, strategy, bounds);
    }

    static void VoronoiDensityEstimator_constructor_wrapper(py::object &self,
            const numpy::ndarray &points, const ptr<CellKernel> &cell_kernel, int seed,
            int njobs, int nrays_weights, int nrays_sampling, RayStrategyType strategy, const ptr<Bounds> &bounds) {

        auto constructor = py::make_constructor(&VoronoiDensityEstimator_constructor);
        constructor(self, points, cell_kernel, seed, njobs,
                    nrays_weights, nrays_sampling, strategy, bounds);
    }

    static numpy::ndarray VoronoiDensityEstimator_estimate(VoronoiDensityEstimator *self, const numpy::ndarray &points) {
        return vector_to_numpy(self->estimate(numpy_to_dmatrix(points)));
    }

    static numpy::ndarray VoronoiDensityEstimator_get_points(VoronoiDensityEstimator *self) {
        return dmatrix_to_numpy(self->get_points());
    }

    static numpy::ndarray VoronoiDensityEstimator_sample(VoronoiDensityEstimator *self, int size) {
        return dmatrix_to_numpy(self->sample(size));
    }

    static numpy::ndarray VoronoiDensityEstimator_sample_masked(VoronoiDensityEstimator *self, int size, const numpy::ndarray &mask) {
        return dmatrix_to_numpy(self->sample_masked(size, numpy_to_dvector(mask)));
    }

    static numpy::ndarray VoronoiDensityEstimator_get_weights(VoronoiDensityEstimator *self) {
        return vector_to_numpy(self->get_weights());
    }

    static void VoronoiDensityEstimator_initialize_weights_uncentered(VoronoiDensityEstimator *self,
                                                                                const numpy::ndarray &centroids) {
        self->initialize_weights_uncentered(numpy_to_dmatrix(centroids));
    }

    static ptr<AdaptiveGaussianCellKernel> AdaptiveGaussianCellKernel_constructor(
            int dim, ftype global_sigma) {
        return std::make_shared<AdaptiveGaussianCellKernel>(dim, global_sigma);
    }

    static void AdaptiveGaussianCellKernel_constructor_wrapper(py::object &self,
                                                               int dim, ftype global_sigma) {

        auto constructor = py::make_constructor(&AdaptiveGaussianCellKernel_constructor);
        constructor(self, dim, global_sigma);
    }

    static ptr<AdaptiveGaussianCellKernel> AdaptiveGaussianCellKernel_constructor2(
            int dim, ftype global_sigma, const numpy::ndarray &local_sigma) {
        return std::make_shared<AdaptiveGaussianCellKernel>(dim, global_sigma, numpy_to_vector(local_sigma));
    }

    static void AdaptiveGaussianCellKernel_constructor2_wrapper(py::object &self,
                                                               int dim, ftype global_sigma, const numpy::ndarray &local_sigma) {

        auto constructor = py::make_constructor(&AdaptiveGaussianCellKernel_constructor2);
        constructor(self, dim, global_sigma, local_sigma);
    }

    static void AdaptiveGaussianCellKernel_update_local_bandwidths(AdaptiveGaussianCellKernel *self,
                                                                   const numpy::ndarray &bandwidths) {
        self->update_local_bandwidths(numpy_to_vector(bandwidths));
    }


};


BOOST_PYTHON_MODULE(vde) {
    numpy::initialize();

    py::class_<CellKernel, boost::noncopyable>("CellKernel", py::no_init);
    py::class_<UniformCellKernel, py::bases<CellKernel>>("UniformCellKernel", py::init<int>());
    py::class_<GaussianCellKernel, py::bases<CellKernel>>("GaussianCellKernel", py::init<int, ftype>());
    py::class_<AdaptiveGaussianCellKernel, py::bases<CellKernel>>("AdaptiveGaussianCellKernel", py::no_init)
            .def("__init__", &WrapperFuncs::AdaptiveGaussianCellKernel_constructor_wrapper)
            .def("__init__", &WrapperFuncs::AdaptiveGaussianCellKernel_constructor2_wrapper)
            .def("update_local_bandwidths", &WrapperFuncs::AdaptiveGaussianCellKernel_update_local_bandwidths);

    py::enum_<RayStrategyType>("RayStrategyType")
            .value("BRUTE_FORCE", BRUTE_FORCE)
            .value("BIN_SEARCH", BIN_SEARCH)
            .value("BRUTE_FORCE_GPU", BRUTE_FORCE_GPU)
            ;

    py::class_<Bounds, boost::noncopyable>("Bounds", py::no_init);
    py::class_<Unbounded, py::bases<Bounds>>("Unbounded", py::init<>());
    py::class_<BoundingBox, py::bases<Bounds>>("BoundingBox", py::init<int, ftype>());
    py::class_<BoundingSphere, py::bases<Bounds>>("BoundingSphere", py::init<int, ftype>());

    py::class_<VoronoiDensityEstimator/*, py::bases<AbstractDensityEstimator>*/>("VoronoiDensityEstimator", py::no_init)
            .def("__init__", &WrapperFuncs::VoronoiDensityEstimator_constructor_wrapper,
                 py::return_internal_reference<2, py::return_internal_reference<3, py::return_internal_reference<9>>>())
            .def("initialize_weights", &VoronoiDensityEstimator::initialize_weights)
            .def("initialize_weights_uncentered", &WrapperFuncs::VoronoiDensityEstimator_initialize_weights_uncentered)
            .def("estimate", &WrapperFuncs::VoronoiDensityEstimator_estimate)
            .def("centroid_smoothing", &VoronoiDensityEstimator::centroid_smoothing)
            .def("get_points", &WrapperFuncs::VoronoiDensityEstimator_get_points)
            .def("sample", &WrapperFuncs::VoronoiDensityEstimator_sample)
//            .def("set_max_block_size", &VoronoiDensityEstimator::set_max_block_size)
            .def("sample_masked", &WrapperFuncs::VoronoiDensityEstimator_sample_masked)
            .def("get_weights", &WrapperFuncs::VoronoiDensityEstimator_get_weights)
            ;
}
