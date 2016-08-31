#include <RcppArmadillo.h>
#include <iostream>
#include "bigmemory/BigMatrix.h"
#include "bigmemory/MatrixAccessor.hpp"
#include "bigmemory/bigmemoryDefines.h"
#include <time.h>
#include <omp.h>

#include "utilities.h"
//#include "defines.h"
using namespace std;

// Coordinate descent for gaussian models
RcppExport SEXP cdfit_binomial_hsr(SEXP X_, SEXP y_, SEXP row_idx_, 
                                   SEXP lambda_, SEXP nlambda_,
                                   SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                                   SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                   SEXP ncore_, SEXP warn_,
                                   SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  // MatrixAccessor<double> xAcc(*xpMat);
  
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  // const char *xf_bin = CHAR(Rf_asChar(xf_bin_));
  // int nchunks = INTEGER(nchunks_)[0];
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int n_total = xMat->nrow(); // number of total observations
  int L = INTEGER(nlambda_)[0];
  
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  int warn = INTEGER(warn_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  
  // int chunk_cols = p / nchunks;

  NumericVector lambda(L);
  NumericVector Dev(L);
  IntegerVector iter(L);
  NumericVector beta0(L);
  NumericVector center(p);
  NumericVector scale(p);
  arma::sp_mat beta = arma::sp_mat(p, L); //beta
  double *a = Calloc(p, double); //Beta from previous iteration
  double a0 = 0.0; //beta0 from previousiteration

  double *w = Calloc(n, double);
//   for (int i = 0; i < n; i++) {
//     w[i] = 0.25;
//   }
  double *s = Calloc(n, double); //y_i - pi_i
  double *z = Calloc(p, double);
  double *eta = Calloc(n, double);
  int *e1 = Calloc(p, int); //ever-active set
  int *e2 = Calloc(p, int); //strong set
  int lstart = 0;
  int converged, violations;
  double xwr, xwx, pi, u, v, cutoff, l1, l2, shift, si;
  double max_update, update, thresh, delta; // for convergence check
  
  double lambda_max = 0.0;
  double *lambda_max_ptr = &lambda_max;
  int xmax_idx = 0;
  int *xmax_ptr = &xmax_idx;

  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("\nPreprocessing start: %s\n", buff1);
  }
  
  standardize_and_get_residual(center, scale, z, lambda_max_ptr, xmax_ptr, xMat, 
                               y, row_idx, lambda_min, alpha, n, p);
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  // Initialization
  double ybar = sum(y, n)/n;
  a0 = beta0[0] = log(ybar/(1-ybar));
  double nullDev = 0;
  double *r = Calloc(n, double);
  for (int i=0; i<n; i++) {
    r[i] = y[i];
    nullDev = nullDev - y[i]*log(ybar) - (1-y[i])*log(1-ybar);
    s[i] = y[i] - ybar;
    eta[i] = a0;
  }
  thresh = eps * nullDev;
  
  double sumS = sum(s, n); // temp result sum of s
  double sumResid = sum(r, n); // temp result sum of current residuals
  double sumWResid = 0.0; // temp result: sum of w * r

  if (!user) {
    // set up lambda, equally spaced on log scale
    double log_lambda_max = log(lambda_max);
    double log_lambda_min = log(lambda_min*lambda_max);
    double delta = (log_lambda_max - log_lambda_min) / (L-1);
    for (int l = 0; l < L; l++) {
      lambda[l] = exp(log_lambda_max - l * delta);
    }
    Dev[0] = nullDev;
    lstart = 1;
  } else {
    lambda = Rcpp::as<NumericVector>(lambda_);
  }

  // set up omp
  int useCores = INTEGER(ncore_)[0];
  int haveCores=omp_get_num_procs();
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);

  for (int l=lstart;l<L;l++) {
    
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
    
    if (l != 0) {
      // Assign a, a0 by previous b, b0
      a0 = beta0[l-1];
      for (int j=0;j<p;j++) {
        a[j] = beta(j, l-1);
      }
      // Check dfmax
      int nv = 0;
      for (int j=0; j<p; j++) {
        if (a[j] != 0) {
          nv++;
        }
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        free_memo_bin_hsr(s, w, a, r, z, e1, e2, eta);
        return List::create(beta0, beta, center, scale, lambda, Dev, iter);
      }
    
      // strong set
      cutoff = 2*lambda[l] - lambda[l-1];
      for (int j=0; j<p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[j])) e2[j] = 1;
      }
      
    } else {
      // strong set
      cutoff = 2*lambda[l] - lambda_max;
      for (int j=0; j<p; j++) if (fabs(z[j]) > (cutoff * alpha * m[j])) e2[j] = 1;
    }
    
    // path start
//     now = time (0);
//     strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//     Rprintf("Solution Path start: Now time: %s\n", buff);
    while (iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        while (iter[l] < max_iter) {
          iter[l]++;
          Dev[l] = 0.0;
          
          for (int i=0;i<n;i++) {
            if (eta[i] > 10) {
              pi = 1;
              w[i] = .0001;
            } else if (eta[i] < -10) {
              pi = 0;
              w[i] = .0001;
            } else {
              pi = exp(eta[i])/(1+exp(eta[i]));
              w[i] = pi*(1-pi);
            }
            s[i] = y[i] - pi;
            r[i] = s[i]/w[i];
            if (y[i]==1) {
              Dev[l] = Dev[l] - log(pi);
            } else {
              Dev[l] = Dev[l] - log(1-pi);
            }
          }
          
          if (Dev[l] / nullDev < .01) {
            if (warn) warning("Model saturated; exiting...");
            for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
            free_memo_bin_hsr(s, w, a, r, z, e1, e2, eta);
            return List::create(beta0, beta, center, scale, lambda, Dev, iter);
          }
          
          // Intercept
          xwr = crossprod(w, r, n, 0);
          xwx = sum(w, n);
          beta0[l] = xwr/xwx + a0;
          for (int i=0; i<n; i++) {
            si = beta0[l] - a0;
            r[i] -= si; //update r
            eta[i] += si; //update eta
          }
          // update temp result: sum of w * r, used for computing xwr;
          sumWResid = wsum(r, w, n);
         
          // Covariates
//           now = time (0);
//           strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//           Rprintf("Solve lasso start: Now time: %s\n", buff);
          // int sum_e1 = sum_discard(e1, p);
          // int sum_e2 = sum_discard(e2, p);
          // Rprintf("*********Ever-active set size: %d; strong set size: %d\n", sum_e1, sum_e2);
          max_update = 0.0;
          for (int j=0; j<p; j++) {
            if (e1[j]) {
              // Calculate u, v
              xwr = wcrossprod_resid(xMat, r, sumWResid, row_idx, center[j], scale[j], w, n, j);
              v = wsqsum_bm(xMat, w, row_idx, center[j], scale[j], n, j) / n;
              u = xwr/n + v*a[j];
              
              // Update b_j
              l1 = lambda[l] * m[j] * alpha;
              l2 = lambda[l] * m[j] * (1-alpha);
              beta(j, l) = lasso(u, l1, l2, v);
              
              // Update r
              shift = beta(j, l) - a[j];
              if (shift !=0) {
                update_resid_eta(r, eta, xMat, shift, row_idx, center[j], scale[j], n, j);
                // update temp result w * r, used for computing xwr;
                sumWResid = wsum(r, w, n);
                
                // update change of objective function
                update = (0.125 + n*l2) * pow(beta(j, l), 2) - pow(a[j], 2) + \
                  n * l1 * (abs(beta(j, l)) - abs(a[j])) - v* a[j] * shift;
                if (update > max_update) max_update = update;
              }
            }
          }

          // Check for convergence
          if (max_update < thresh) {
            converged = 1;
          } else {
            converged = 0;
          }
          // converged = checkConvergence(beta, a, eps, l, p);
          a0 = beta0[l];
          for (int j=0; j<p; j++) a[j] = beta(j, l);
          if (converged) break;
        }
//         now = time (0);
//         strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//         Rprintf("Solve lasso end: Now time: %s\n", buff);
        
        // Scan for violations in strong set
        sumS = sum(s, n);
        violations = check_strong_set_bin(e1, e2, z, xMat, row_idx, center, scale, lambda[l], 
                                          sumS, alpha, s, m, n, p);
        if (violations==0) break;
        // Rprintf("\tNumber of violations in strong set: %d\n", violations);
      }
      
      // Scan for violations in rest
//       now = time (0);
//       strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//       Rprintf("Scan rest set start: %s;   ", buff);
      violations = check_rest_set_bin(e1, e2, z, xMat, row_idx, center, scale, lambda[l], 
                                      sumS, alpha, s, m, n, p);
    
//       now = time (0);
//       strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
//       Rprintf("Scan rest set end: %s\n", buff);
      if (violations==0) break;
      // Rprintf("\tNumber of violations in rest set: %d\n", violations);
    }
  }

  free_memo_bin_hsr(s, w, a, r, z, e1, e2, eta);
  return List::create(beta0, beta, center, scale, lambda, Dev, iter);
  
}
