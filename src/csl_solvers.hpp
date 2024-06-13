#ifndef CSL_SOLVERS_HPP
#define CSL_SOLVERS_HPP

#include "liblinear/linear.hpp"
#include <armadillo>
#include <stdio.h>
#include <lbfgs.h>
#include <random>

namespace liblinear{

// safe implementation of log(1 + exp(x))
double log1p_exp(double s);

// objective function for logistic regression with l1-reg
double lr_obj(const feature_node** data,
              const size_t l,
              const size_t n,
              const double* responses,
              double* w,
              double lambd,
              bool verbose = false);

// csl objective function
double csl_obj(const feature_node** data,
               const size_t l,
               const size_t n,
               const double* responses,
               double* w,
               double* igrad,
               double* ugrad,
               double lambd,
               bool verbose = false);

// csl objective function with shift regularization
double csl_obj(const feature_node** data,
               const size_t l,
               const size_t n,
               const double* responses,
               double* w,
               double* igrad,
               double* ugrad,
               double lambd,
               double alpha,
               double* w_prev,
               bool verbose = false);

// computes gradient of unregularized lr obj
void lr_gradient(const feature_node** data,
                 const size_t l,
                 const size_t n,
                 const double* responses,
                 double* w,
                 double* grad,
                 bool normalize = true,
                 size_t* active_dims = nullptr,
                 size_t active_size = -1);

// inner solver for proxCSL
void solve_prox_step(const feature_node** data,
                     const size_t l,
                     const size_t n,
                     const double* responses,
                     double lambda,
                     double* w,
                     double* grad,
                     double* vdiag,
                     double* Hdiag,
                     size_t* active_dims,
                     size_t active_size,
                     int max_iter,
                     double tol,
                     double* igrad,
                     double* ugrad,
                     double alpha,
                     double* w_orig);

// CSL update using OWLQN, used for sCSL and sDANE
void solve_csl_owlqn(const feature_node** data,
                     const size_t l,
                     const size_t n,
                     const double* responses,
                     double* w,
                     double* igrad,
                     double* ugrad,
                     double lambda,
                     size_t max_itr,
                     size_t* active_dims,
                     size_t active_size,
                     double alpha = 0.0,
                     bool verbose = false);

// CSL update using our solver for proxCSL update
void solve_csl_cd(const feature_node** data,
                   const size_t l,
                   const size_t n,
                   const double* responses,
                   double* w,
                   double* igrad,
                   double* ugrad,
                   double lambda,
                   size_t max_outer_iter,
                   size_t max_inner_iter,
                   size_t* active_dims,
                   size_t active_size,
                   double tol,
                   double alpha,
                   bool adaptive);
}

#include "csl_solvers_impl.hpp"

#endif
