#include "cell_kernels.h"

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/distributions/normal.hpp>

CellKernel::CellKernel(int dim) : dim(dim) {}

GaussianCellKernel::GaussianCellKernel(int dim, ftype sigma) : CellKernel(dim), sigma(sigma) {}

AdaptiveGaussianCellKernel::AdaptiveGaussianCellKernel(int dim, ftype sigma) : CellKernel(dim), global_sigma(sigma),
        local_sigma(0), initialized(false) {}

AdaptiveGaussianCellKernel::AdaptiveGaussianCellKernel(int dim, ftype sigma, const vec<ftype> &local_sigma) :
        CellKernel(dim), global_sigma(sigma), local_sigma(local_sigma), initialized(true) {

}

bool AdaptiveGaussianCellKernel::is_initialized() const {
    return initialized;
}

ftype UniformCellKernel::cone_integral(int index, ftype length) const {
    if (math::isinf(length)) {
        return std::numeric_limits<ftype>::infinity();
    }
    return qpow(length, dim);
}

ftype UniformCellKernel::sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u,
                                        ftype t0, ftype t1, RandomEngine &re) const {
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    if (math::isinf(t0) || math::isinf(t1)) {
        return std::numeric_limits<ftype>::quiet_NaN();
    }
    return t0 + re.rand_float() * (t1 - t0);
}

ftype UniformCellKernel::unnormalized_pdf(int index, ftype squared_dist) const {
    return 1;
}

ftype UniformCellKernel::normalization_constant(int index) const {
    return nball_volume(dim) * dim;
}

ftype UniformCellKernel::sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                               ftype t1) const {
    if (math::isinf(t0) || math::isinf(t1)) {
        return std::numeric_limits<ftype>::quiet_NaN();
    }
    return static_cast<ftype>(0.5) * (t0 + t1);
}

ftype UniformCellKernel::cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const {
    return cone_integral(index, length);
}

ftype GaussianCellKernel::cone_integral(int index, ftype length) const {
    return boost::math::gamma_p(dim * static_cast<ftype>(0.5), length * length / (2 * sigma * sigma));
}

ftype
GaussianCellKernel::sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                                   RandomEngine &re) const {
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    ftype mean = u.dot(center - ref);
    boost::math::normal_distribution distr(mean, sigma);
    ftype cdf0 = boost::math::cdf(distr, t0);
    ftype cdf1 = boost::math::cdf(distr, t1);
    ftype cdf_ret = cdf0 + re.rand_float() * (cdf1 - cdf0);
    ftype t_ret;
    try {
        t_ret = quantile(distr, cdf_ret);
    } catch (const std::overflow_error& e) {
        std::cout << "Warning: overflow error in sampling on a segment [" << e.what() << "]" << std::endl;
        t_ret = t0 + re.rand_float() * (t1 - t0);
    }
    t_ret = std::max(t0, std::min(t1, t_ret));
    return t_ret;
}

ftype GaussianCellKernel::unnormalized_pdf(int index, ftype squared_dist) const {
    return math::exp(-static_cast<ftype>(0.5) * squared_dist / (sigma * sigma));
}

ftype GaussianCellKernel::normalization_constant(int index) const {
    return math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5));
}

ftype GaussianCellKernel::sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                                ftype t1) const {
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    ftype mean = u.dot(center - ref);
    boost::math::normal_distribution distr(mean, sigma);
    ftype cdf0 = boost::math::cdf(distr, t0);
    ftype cdf1 = boost::math::cdf(distr, t1);
    ftype cdf_ret = static_cast<ftype>(0.5) * (cdf0 + cdf1);
    ftype t_ret;
    try {
        t_ret = quantile(distr, cdf_ret);
    } catch (const std::overflow_error& e) {
        std::cout << "Warning: overflow error in sampling on a segment [" << e.what() << "]" << std::endl;
//        t_ret = static_cast<ftype>(0.5) * (t0 + t1);
        return NAN_ftype;
    }
    t_ret = std::max(t0, std::min(t1, t_ret));
    return t_ret;
}

ftype GaussianCellKernel::cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const {
    ftype sq_2 = math::sqrt(2);
    ftype sq_2pi = math::sqrt(2 * PI_ftype);
    ftype o5 = static_cast<ftype>(0.5);

    ftype vol_sphere = 2 * math::pow(PI_ftype, o5 * subspace_dim) / boost::math::tgamma(o5 * subspace_dim);

    ftype upper = (length + a) / (sq_2 * sigma);
    ftype lower = a / (sq_2 * sigma);

    ftype erf_upper = boost::math::erf(upper);
    ftype erf_lower = boost::math::erf(lower);

    ftype exp_upper2 = math::exp(-upper * upper);
    ftype exp_lower2 = math::exp(-lower * lower);

    // setting for n=1
    ftype phi_n0 = o5 * sq_2pi * sigma * (erf_upper - erf_lower);
    ftype phi_n1 = sigma * sigma * (-exp_upper2 + exp_lower2) -
                   o5 * sq_2pi * sigma * a * (erf_upper - erf_lower);

    if (subspace_dim == 1) {
        phi_n1 = phi_n0;
    }

    ftype beta_n = math::isinf(length) ? 0 : sigma * sigma * exp_upper2 * length;

    for (int n = 1; n + 2 <= subspace_dim; n++) {
        ftype phi_n2 = -a * phi_n1 + sigma * sigma * n * phi_n0 - beta_n;

        phi_n0 = phi_n1;
        phi_n1 = phi_n2;
        if (!math::isinf(length)) {
            beta_n *= length;
        }
    }


    return vol_sphere * math::exp(-(b + a * a) / (2 * sigma * sigma)) * phi_n1;
}

void AdaptiveGaussianCellKernel::update_local_bandwidths(const vec<ftype> &new_local_sigma) {
    initialized = true;
    this->local_sigma = new_local_sigma;
}

ftype AdaptiveGaussianCellKernel::cone_integral(int index, ftype length) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = global_sigma * local_sigma[index];
    return boost::math::gamma_p(dim * static_cast<ftype>(0.5), length * length / (2 * sigma * sigma));
}

ftype
AdaptiveGaussianCellKernel::sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u,
                                           ftype t0, ftype t1, RandomEngine &re) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = global_sigma * local_sigma[index];
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    ftype mean = u.dot(center - ref);
    boost::math::normal_distribution distr(mean, sigma);
    ftype cdf0 = boost::math::cdf(distr, t0);
    ftype cdf1 = boost::math::cdf(distr, t1);
    ftype cdf_ret = cdf0 + re.rand_float() * (cdf1 - cdf0);
    ftype t_ret;
    try {
        t_ret = quantile(distr, cdf_ret);
    } catch (const std::overflow_error& e) {
        std::cout << "Warning: overflow error in sampling on a segment [" << e.what() << "]" << std::endl;
        t_ret = t0 + re.rand_float() * (t1 - t0);
    }
    t_ret = std::max(t0, std::min(t1, t_ret));
    return t_ret;
}

ftype AdaptiveGaussianCellKernel::unnormalized_pdf(int index, ftype squared_dist) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = global_sigma * local_sigma[index];
    return math::exp(-static_cast<ftype>(0.5) * squared_dist / (sigma * sigma));
}

ftype AdaptiveGaussianCellKernel::normalization_constant(int index) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = global_sigma * local_sigma[index];
    return math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5));
}

ftype AdaptiveGaussianCellKernel::sample_on_line_median(int index, const dvector &ref, const dvector &center,
                                                        const dvector &u, ftype t0, ftype t1) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = global_sigma * local_sigma[index];
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    ftype mean = u.dot(center - ref);
    boost::math::normal_distribution distr(mean, sigma);
    ftype cdf0 = boost::math::cdf(distr, t0);
    ftype cdf1 = boost::math::cdf(distr, t1);
    ftype cdf_ret = static_cast<ftype>(0.5) * (cdf0 + cdf1);
    ftype t_ret;
    try {
        t_ret = quantile(distr, cdf_ret);
    } catch (const std::overflow_error& e) {
        std::cout << "Warning: overflow error in sampling on a segment [" << e.what() << "]" << std::endl;
//        t_ret = static_cast<ftype>(0.5) * (t0 + t1);
        return NAN_ftype;
    }
    t_ret = std::max(t0, std::min(t1, t_ret));
    return t_ret;
}

ftype AdaptiveGaussianCellKernel::cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = global_sigma * local_sigma[index];
    ftype sq_2 = math::sqrt(2);
    ftype sq_2pi = math::sqrt(2 * PI_ftype);
    ftype o5 = static_cast<ftype>(0.5);

    ftype vol_sphere = 2 * math::pow(PI_ftype, o5 * subspace_dim) / boost::math::tgamma(o5 * subspace_dim);

    ftype upper = (length + a) / (sq_2 * sigma);
    ftype lower = a / (sq_2 * sigma);

    ftype erf_upper = boost::math::erf(upper);
    ftype erf_lower = boost::math::erf(lower);

    ftype exp_upper2 = math::exp(-upper * upper);
    ftype exp_lower2 = math::exp(-lower * lower);

    // setting for n=1
    ftype phi_n0 = o5 * sq_2pi * sigma * (erf_upper - erf_lower);
    ftype phi_n1 = sigma * sigma * (-exp_upper2 + exp_lower2) -
                   o5 * sq_2pi * sigma * a * (erf_upper - erf_lower);

    if (subspace_dim == 1) {
        phi_n1 = phi_n0;
    }

    ftype beta_n = math::isinf(length) ? 0 : sigma * sigma * exp_upper2 * length;

    for (int n = 1; n + 2 <= subspace_dim; n++) {
        ftype phi_n2 = -a * phi_n1 + sigma * sigma * n * phi_n0 - beta_n;

        phi_n0 = phi_n1;
        phi_n1 = phi_n2;
        if (!math::isinf(length)) {
            beta_n *= length;
        }
    }


    return vol_sphere * math::exp(-(b + a * a) / (2 * sigma * sigma)) * phi_n1;
}
