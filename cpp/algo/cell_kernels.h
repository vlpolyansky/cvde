#pragma once

#include "../utils.h"
#include "../RandomEngine.h"

class CellKernel {
public:
    explicit CellKernel(int dim);

    virtual ftype unnormalized_pdf(int index, ftype squared_dist) const = 0;

    virtual ftype cone_integral(int index, ftype length) const = 0;

    virtual ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const = 0;  // (x + ut - p0)^2 = t^2 + 2at + b

    virtual ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u,
                                 ftype t0, ftype t1, RandomEngine &re) const = 0;

    virtual ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u,
                                        ftype t0, ftype t1) const = 0;

    /**
     * Returns a constant beta, such that:
     * beta * E_{x \in S^{n-1}} [cone_integral(l(x)] = \int_{Cell(x0} unnormalized_pdf(x) dx,
     * where P'(Cell(x0)) is the integral of the unnormalized pdf.
     */
    virtual ftype normalization_constant(int index) const = 0;

protected:
    int dim;
};

class UniformCellKernel : public CellKernel {
public:
    using CellKernel::CellKernel;

    /**
     * Returns 1 at 0.
     */
    ftype unnormalized_pdf(int index, ftype squared_dist) const override;

    ftype cone_integral(int index, ftype length) const override;

    ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const override;

    ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                         RandomEngine &re) const override;

    ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                ftype t1) const override;

    ftype normalization_constant(int index) const override;
};

class GaussianCellKernel : public CellKernel {
public:
    GaussianCellKernel(int dim, ftype sigma);

    ftype unnormalized_pdf(int index, ftype squared_dist) const override;

    ftype cone_integral(int index, ftype length) const override;

    ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const override;

    ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                         RandomEngine &re) const override;

    ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                ftype t1) const override;

    ftype normalization_constant(int index) const override;

private:
    ftype sigma;
};


class AdaptiveGaussianCellKernel : public CellKernel {
public:
    AdaptiveGaussianCellKernel(int dim, ftype sigma);

    AdaptiveGaussianCellKernel(int dim, ftype sigma, const vec<ftype> &local_sigma);

    bool is_initialized() const;

    void update_local_bandwidths(const vec<ftype> &new_local_sigma);

    ftype unnormalized_pdf(int index, ftype squared_dist) const override;

    ftype cone_integral(int index, ftype length) const override;

    ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const override;

    ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                         RandomEngine &re) const override;

    ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                ftype t1) const override;

    ftype normalization_constant(int index) const override;

private:
    bool initialized;
    ftype global_sigma;
    vec<ftype> local_sigma;
};
