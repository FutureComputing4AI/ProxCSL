#ifndef CSL_SOLVERS_IMPL_HPP
#define CSL_SOLVERS_IMPL_HPP

#include "csl_solvers.hpp"
#include <iostream>
#include <stdio.h>
#include <lbfgs.h>

namespace liblinear{


double log1p_exp(double s)
{
  double lls;
  if (s < 0)
  {
    lls = log(1 + exp(s));
  }
  else
  {
    // avoid overflow
    lls = s + log(1 + exp(-s));
  }
  return lls;
}

// logistic loss with L1 regularization
double lr_obj(const feature_node** data,
              const size_t l,
              const size_t n,
              const double* responses,
              double* w,
              double lambd,
              bool verbose)
{
  feature_node** X = (feature_node**) data;
  double* y = (double*) responses;
  feature_node* xj;

  double* Xtw = new double[l];

  for (size_t i = 0; i < l; ++i)
  {
     Xtw[i] = 0;
  }

  for (size_t j = 0; j < n; ++j)
  {
    if (w[j] == 0)
      continue;
    xj = X[j];
    while (xj->index != -1)
    {
      Xtw[xj->index-1] += w[j] * xj->value;
      xj++;
    }
  }

  double obj = 0;
  for (size_t i = 0; i < l; ++i)
  {
    obj += log1p_exp(Xtw[i]);
    if (y[i] == 1)
      obj += -1.0 * Xtw[i];
  }
  obj /= l;

  double wnorm = 0;
  for (size_t j = 0; j < n; ++j)
  {
    wnorm += fabs(w[j]);
  }

  if (verbose)
  {
    std::cout << "--> lr loss: " << obj << std::endl;
    std::cout << "--> reg loss: " << lambd * wnorm << std::endl;
  }

  delete [] Xtw;

  return obj + lambd * wnorm;
}

double csl_obj(const feature_node** data,
               const size_t l,
               const size_t n,
               const double* responses,
               double* w,
               double* igrad,
               double* ugrad,
               double lambd,
               bool verbose)
{
  double obj = lr_obj(data, l, n, responses, w, lambd, verbose);

  // compute gradient diff term here. Grads must be pre-normalized
  double gtw = 0;
  for (size_t j = 0; j < n; ++j)
  {
    gtw += (ugrad[j] - igrad[j]) * w[j];
  }

  if (verbose)
    std::cout << "--> csl loss: " << gtw << std::endl;

  return obj + gtw;
}

// overload for proximal regularized CSL
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
               bool verbose)
{
  double obj = lr_obj(data, l, n, responses, w, lambd, verbose);

  // compute gradient diff term here. Grads must be pre-normalized
  double gtw = 0;
  for (size_t j = 0; j < n; ++j)
  {
    gtw += (ugrad[j] - igrad[j]) * w[j];
  }

  double ad2 = 0;
  for (size_t j = 0; j < n; ++j)
  {
    ad2 += std::pow(w[j] - w_prev[j], 2);
  }
  ad2 *= 0.5 * alpha;

  if (verbose)
  {
    std::cout << "--> csl loss: " << gtw << std::endl;
    std::cout << "--> csl reg loss: " << ad2 << std::endl;
  }

  return obj + gtw + ad2;
}

struct lbfgsInfo {
    feature_node** X;
    double* y;
    size_t l;
    size_t n;
    double* igrad;
    double* ugrad;
    double lambda;
    size_t* active_dims;
    size_t active_size;
    double alpha;
    double* w_prev;
};

void lr_gradient(const feature_node** data,
                 const size_t l,
                 const size_t n,
                 const double* responses,
                 double* w,
                 double* grad,
                 bool normalize,
                 size_t* active_dims,
                 size_t active_size)
{
  double* resid = new double[l];
  feature_node** X = (feature_node**) data;
  double* y = (double*) responses;
  feature_node* xj;

  // handle feature subsetting, default to all if nullptr passed
  if (active_dims == nullptr)
    active_size = n;  

  // init residual vector to 0s
  for (size_t i = 0; i < l; ++i)
  {
     resid[i] = 0;
  }

  // init grad to 0 (since not all will be assigned)
  for (size_t j = 0; j < n; ++j)
  {
    grad[j] = 0;
  }

  // compute x dot w
  for (size_t s = 0; s < active_size; ++s)
  {
    size_t j = (active_dims == nullptr) ? s : active_dims[s];
    xj = X[j];
    // each linked list xj stores a feature column so we need transpose of standard dot fn
    while (xj->index != -1)
    {
      resid[xj->index-1] += w[j] * xj->value;
      xj++;
    }
  }

  for (size_t i = 0; i < l; ++i)
  {
    // convert x*w to logits
    resid[i] = 1 / (1 + exp(-resid[i]));
    
    // // we don't care if y is {0, 1} or {-1, 1}, just if y=1
    if (y[i] == 1)
      resid[i] -= 1;
  }

  // compute xj^t residual and normalize by sample size
  for (size_t s = 0; s < active_size; ++s)
  {
    size_t j = (active_dims == nullptr) ? s : active_dims[s];
    xj = X[j];
    grad[j] = sparse_operator::dot(resid, xj);
    if (normalize)
       grad[j] /= l;
  }

  delete [] resid;
}

// Obj + Gradient function for OWLQN CSL 
static lbfgsfloatval_t evaluate(void *instance,
                                const lbfgsfloatval_t *x,
                                lbfgsfloatval_t *g,
                                const int n,
                                const lbfgsfloatval_t step)
{
  // struct lbfgsInfo* info = static_cast<lbfgsInfo *>(instance);
  struct lbfgsInfo* info = (lbfgsInfo*) instance;

  double* w = new double[n];
  for (size_t j = 0; j < n; ++j)
  {
    w[j] = x[j];
  }

  double obj = csl_obj((const feature_node**) info->X,
                       (const size_t) info->l,
                       (const size_t) info->n,
                       (const double*) info->y,
                       w,
                       info->igrad,
                       info->ugrad,
                       info->lambda,
                       info->alpha,
                       info->w_prev);

  // all grads should be normalized. ugrad and igrad already normalized
  double* grad = new double[n];
  lr_gradient((const feature_node**) info->X,
              (const size_t) info->l,
              (const size_t) info->n,
              (const double*) info->y,
              w,
              grad,
              true,   // normalize
              info->active_dims,
              info->active_size);

  size_t active_size = (info->active_dims == nullptr) ? info->n : info->active_size;
  for (size_t s = 0; s < active_size; ++s)
  {
    size_t j = (info->active_dims == nullptr) ? s : info->active_dims[s];
    grad[j] += info->ugrad[j] - info->igrad[j];   // add CSL grad diff term

    // optional: alpha shift shrinkage
    grad[j] += info->alpha * (w[j] - info->w_prev[j]);
  }

  // probably unnecessary copying because i didn't assign directly into g
  for (size_t j = 0; j < n; ++j)
  {
    g[j] = grad[j];
  }

  delete [] w;
  delete [] grad;
  
  return obj;
}

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}

void lr_hess_diag(const feature_node** data,
                  const size_t l,
                  const size_t n,
                  const double* responses,
                  double* w,
                  double* vdiag,
                  double* Hdiag,
                  bool normalize,
                  size_t* active_dims,
                  size_t active_size)
{
  double* prob = new double[l];

  feature_node** X = (feature_node**) data;
  feature_node* xj;

  if (active_dims == nullptr)
    active_size = n;

  // init logit vector to 0s
  for (int i = 0; i < l; ++i)
  {
     prob[i] = 0;
  }

  // compute x dot w
  for (int s = 0; s < active_size; ++s)
  {
    int j = (active_dims == nullptr) ? s : active_dims[s];
    xj = X[j];
    // each linked list xj stores a feature column
    while (xj->index != -1)
    {
      prob[xj->index-1] += w[j] * xj->value;
      xj++;
    }
  }

  // convert x dot w to logits
  for (int i = 0; i < l; ++i)
  {
    prob[i] = 1 / (1 + exp(-prob[i]));
    vdiag[i] = prob[i] * (1 - prob[i]);
    if (normalize)
      vdiag[i] /= l;  // normalize by sample size
  }

  // init vector for Hessian diagonal to 0
  for (int j = 0; j < n; ++j)
  {
    Hdiag[j] = 0;
  }

  // compute diagonal of Hessian
  for (int s = 0; s < active_size; ++s)
  {
    int j = (active_dims == nullptr) ? s : active_dims[s];
    xj = X[j];
    while (xj->index != -1)
    {
      Hdiag[j] += xj->value * xj->value * vdiag[xj->index - 1];
      xj++;
    }
  }

  delete [] prob;
}

double linesearch(const feature_node** data,
                  const size_t l,
                  const size_t n,
                  const double* responses,
                  double* igrad,
                  double* ugrad,
                  double* w,
                  double* delta_w,
                  double lambd,
                  double alpha,
                  double* w_orig,
                  double gamma,
                  int max_steps)
{
  double min_loss = std::numeric_limits<double>::infinity();
  double scale = 1.0;
  double best_scale = 1.0;

  double* w_new = new double[n];
  double new_loss;

  for (int itr = 0; itr < max_steps; ++itr)
  {
    // compute candidate update
    for (int j = 0; j < n; ++j)
    {
      w_new[j] = w[j] + scale * delta_w[j];
    }
    new_loss = csl_obj(data, l, n, responses, w_new, igrad, ugrad, lambd, alpha, w_orig);

    // track if loss is improved
    if (new_loss < min_loss)
    {
      best_scale = scale;
      min_loss = new_loss;
    }
    scale *= gamma;
  }

  // compute best update
  for (int j = 0; j < n; ++j)
  {
    w[j] = w[j] + best_scale * delta_w[j];
  }
  
  return best_scale;
}


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
                     double* w_orig)
{
  // vdiag is length-l sample-weighted variance used for Hessian
  // Hdiag is length-n diagonal of local Hessian
  // tol is convergence tolerance
  // alpha is dampening/shift regularizer

  double nu = 1e-12;    // additional Hessian damping term
  double C = 1.0 / lambda;  // factor of n is accounted for already
  if (active_dims == nullptr)
    active_size = n;

  // compute delta in w already made since w_orig
  double* w_delt_orig = new double[n];
  for (size_t j = 0; j < n; ++j)
  {
    w_delt_orig[j] = w[j] - w_orig[j];
  }

  // initialize delta vector to be added to w
  double* w_delt = new double[n];
  for (int j = 0; j < n; ++j)
  {
    w_delt[j] = 0;
  }

  // X dot d (updated via eq 17)
  double* Xtd = new double[l];
  for (int i = 0; i < l; ++i)
  {
    Xtd[i] = 0;
  }

  feature_node** X = (feature_node**) data;
  feature_node* xj;

  int iter = 0;
  while (iter < max_iter)
  {
    // to track convergence
    double mbar = 0;

    for (int s = 0; s < active_size; ++s)
    {
      int j = (active_dims == nullptr) ? s : active_dims[s];
      xj = X[j];

      // grad_j L + (hess L * d)_j
      double G = grad[j];
      while (xj->index != -1)
      {
        int ind = xj->index - 1;
        G += xj->value * vdiag[ind] * Xtd[ind];
        xj++;
      }
      // dampening term for making Hessian PD
      G += w_delt[j] * nu;

      // proximal shift regularization
      G += (w_delt[j] + w_delt_orig[j]) * alpha;


      // Hess L_{jj}
      double H = Hdiag[j];
      H += (alpha + nu);   // dampening/proximal shift regularization
      
      // scale H and G by inverse lambda to match liblinear convention
      G *= C;
      H *= C;

      double wpd = w[j] + w_delt[j];
      double z;
      double Gp = G + 1;
      double Gn = G - 1;
      if (Gp < H * wpd)
        z = -Gp / H;
      else if (Gn > H * wpd)
        z = -Gn / H;
      else
        z = -wpd;

      // apply clamping avoid numerical issues
      if (abs(z) < 1.0e-12)
        continue;
      z = std::min(std::max(z, -10.0), 10.0);

      w_delt[j] += z;

      // update Xtd for fixed j
      xj = X[j];
      sparse_operator::axpy(z, xj, Xtd);

      // update convergence criterion (eq 27)
      if (wpd > 0)
      {
        mbar += abs(Gp);
      }
      else if (wpd < 0)
      {
        mbar += abs(Gn);
      }
      else
      {
        if (Gp < 0)
          mbar += -Gp;
        else if (Gn > 0)
          mbar += Gn;
      }
    }
    iter++;
    
    // check for convergence to break early
    if (mbar < tol)
      break;
  }

  // linesearch function finds best step size and applies it to w
  double best_scale = linesearch(data, l, n, responses,
                                 igrad,
                                 ugrad,
                                 w,
                                 w_delt,
                                 lambda,
                                 alpha,
                                 w_orig,
                                 0.5,
                                 20);

  std::cout << "Inner CD complete in " << iter << " cycles" << std::endl;

  // print ending objectives
  double cslLoss = csl_obj(data, l, n, responses, w, igrad, ugrad, lambda, alpha, w_orig);
  std::cout << "CSL ending objective: " << cslLoss << std::endl;

  double lrLoss = lr_obj(data, l, n, responses, w, lambda);
  std::cout << "local lr ending objective: " << lrLoss << std::endl;

  delete [] w_delt_orig;
  delete [] w_delt;
  delete [] Xtd;
}


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
                     double alpha,
                     bool verbose)
{
  if (verbose)
  {
    std::cout << "==== Solving CSL objective with OWL-QN ====" << std::endl;
  
    // print out starting objective info
    double cslLoss = csl_obj(data, l, n, responses, w, igrad, ugrad, lambda, alpha, w);
    std::cout << "CSL starting objective: " << cslLoss << std::endl;

    double lrLoss = lr_obj(data, l, n, responses, w, lambda);
    std::cout << "local lr starting objective: " << lrLoss << std::endl;
  }

  // make copy of w for proximal alpha regularization
  double* w_prev = new double[n];
  for (size_t j = 0; j < n; ++j)
  {
    w_prev[j] = w[j];
  }

  lbfgsfloatval_t fx;
  lbfgsfloatval_t *x = lbfgs_malloc(n);
  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  param.max_iterations = (int) max_itr;   // set max iter
  param.orthantwise_c = lambda;     // set L1 regularization
  param.linesearch = 2;     // OWL-QN requires Armijo backtracking
  param.orthantwise_start = 0;
  param.orthantwise_end = n;
  // param.max_linesearch = 40;
  // param.epsilon = 1e-6;
  // param.min_step = 1e-30;      // default 1e-20
  // param.ftol = 0.1;
  // param.gtol = 0.9;

  struct lbfgsInfo objInfo;
  objInfo.X = (feature_node**) data;
  objInfo.y = (double*) responses;
  objInfo.l = l;
  objInfo.n = n;
  objInfo.igrad = igrad;
  objInfo.ugrad = ugrad;
  objInfo.lambda = lambda;
  objInfo.active_dims = active_dims;
  objInfo.active_size = active_size;
  objInfo.alpha = alpha;
  objInfo.w_prev = w_prev;

  int status = lbfgs(n, x, &fx, evaluate, nullptr, &objInfo, &param);
  // uncomment to diagnose error
  // std::cout << "STATUS " << lbfgs_strerror(status) << std::endl;

  // copy parameters to w array
  for (size_t j = 0; j < n; ++j)
  {
    w[j] = x[j];
  }

  if (verbose)
  {
    // print ending objectives
    double cslLoss = csl_obj(data, l, n, responses, w, igrad, ugrad, lambda, alpha, w_prev);
    std::cout << "CSL ending objective: " << cslLoss << std::endl;

    double lrLoss = lr_obj(data, l, n, responses, w, lambda);
    std::cout << "local lr ending objective: " << lrLoss << std::endl;
  }


  delete [] w_prev;

  // std::cout << "==== Finishing CSL solve ====\n" << std::endl;
}


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
                   bool adaptive)
{
  std::cout << "==== Solving CSL objective with Prox-CD ====" << std::endl;

  // print starting objectives
  double cslLoss = csl_obj(data, l, n, responses, w, igrad, ugrad, lambda, alpha, w);
  std::cout << "CSL starting objective: " << cslLoss << std::endl;

  double lrLoss = lr_obj(data, l, n, responses, w, lambda);
  std::cout << "local lr starting objective: " << lrLoss << std::endl;

  double* grad = new double[n];
  double* vdiag = new double[l];
  double* Hdiag = new double[n];
  double* conv = new double[3];
  for (size_t k = 0; k < 3; ++k)
  {
    conv[k] = 1e10;    // just initialize with a large value
  }

  if (active_dims == nullptr)
    active_size = n;

  // make copy of w for alpha regularization
  double* w_orig = new double[n];
  for (size_t j = 0; j < n; ++j)
  {
    w_orig[j] = w[j];
  }

  for (size_t outer_itr = 0; outer_itr < max_outer_iter; ++outer_itr)
  {
    std::cout << "Outer step: " << outer_itr + 1 << std::endl;
    
    // compute CSL gradient
    lr_gradient(data, l, n, responses,
                w,
                grad,
                true,
                active_dims,
                active_size);
    for (size_t s = 0; s < active_size; ++s)
    {
      size_t j = (active_dims == nullptr) ? s : active_dims[s];
      grad[j] += ugrad[j] - igrad[j];
    }

    // update local hessian
    lr_hess_diag(data, l, n, responses,
                 w,
                 vdiag,
                 Hdiag,
                 true,
                 active_dims,
                 active_size);

    // find adaptive diagonal damping value to scale alpha
    // Step 1: disable adaptive mode if nonzeros < 1000 (when diverging starts)
    int nonzeros = 0;
    for (int j = 0; j < n; ++j)
    {
      if (std::abs(w[j]) > 1e-10)
        nonzeros += 1;
    }
    std::cout << "nonzeros: " << nonzeros << std::endl;
    if (nonzeros < 1000)
    {
      adaptive = false;
    }

    // Step 2: run adaptive search for alpha
    if (adaptive && (outer_itr == 0))
    {
      alpha = std::max(alpha, 1e-4);    // requires alpha > 0
      double curObj = lr_obj(data, l, n, responses, w, lambda);
      double curCslObj = csl_obj(data, l, n, responses, w, igrad, ugrad, lambda, alpha, w);

      double* w_tmp = new double[n];

      // permit alpha=1e-4 to scale up to 1e2
      for (size_t sc = 0; sc < 6; sc++)
      {
        // compute norm of w for bounding
        double wnorm = 0;
        for (size_t j = 0; j < n; ++j)
        {
          wnorm += w[j] * w[j];
        }

        // reset w_tmp to w
        for (size_t j = 0; j < n; ++j)
        {
          w_tmp[j] = w[j];
        }
        // run limited inner CD and check objective
        solve_prox_step(data, l, n, responses,
                        lambda,
                        w_tmp,
                        grad,
                        vdiag,
                        Hdiag,
                        active_dims,
                        active_size,
                        5,
                        tol,
                        igrad,
                        ugrad,
                        alpha,
                        w_orig);

        double diffNorm = 0;
        for (size_t j = 0; j < n; ++j)
        {
          diffNorm += (w_tmp[j] - w[j]) * (w_tmp[j] - w[j]);
        }

        double newObj = lr_obj(data, l, n, responses, w_tmp, lambda);
        double newCslObj = csl_obj(data, l, n, responses, w_tmp, igrad, ugrad, lambda, alpha, w);

        // std::cout << "alpha,curObj,newObj,curCslObj,newCslObj,wnorm,diffnorm: " << alpha << " " << curObj << " " << newObj << " " << curCslObj << " " << newCslObj << " " << wnorm << " " << diffNorm << std::endl;
        if (((newObj > curObj) && (newCslObj < curCslObj * 0.9)) || (newCslObj < curCslObj * 0.8))
        {
          alpha *= 10;
        }
        else
        {
          std::cout << "adaptive alpha: " << alpha << std::endl;
          break;
        }
      }
      delete [] w_tmp;
    }

    // run proximal Newton update with CD
    solve_prox_step(data, l, n, responses,
                    lambda,
                    w,
                    grad,
                    vdiag,
                    Hdiag,
                    active_dims,
                    active_size,
                    max_inner_iter,
                    tol,
                    igrad,
                    ugrad,
                    alpha,
                    w_orig);

    // check for outer convergence based on history
    double obj = csl_obj(data, l, n, responses,
                         w, igrad, ugrad, lambda, alpha, w_orig);    
    conv[2] = conv[1];
    conv[1] = conv[0];
    conv[0] = obj;
    if (abs(conv[0] - conv[1]) < 1e-8 && abs(conv[1] - conv[2]) < 1e-8)
    {
      std::cout << "Outer prox-Newton complete in " << outer_itr + 1 << " steps" << std::endl;
      break;
    }
    if (outer_itr == max_outer_iter - 1)
    {
      std::cout << "Max outer steps reached: " << max_outer_iter << std::endl;
    }
  }

  delete [] grad;
  delete [] vdiag;
  delete [] Hdiag;
  delete [] conv;
  delete [] w_orig;

  std::cout << "==== Finishing CSL solve ====\n" << std::endl;
}

}

#endif
