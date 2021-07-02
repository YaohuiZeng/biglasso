#include "utilities.h"

// dual function
double dual_bin(vector<double>& theta, double lambda, double lambda_max, int n) {
  double res = 0.0;
  double lam_ratio = lambda / lambda_max;
  for (int i = 0; i < n; i++) {
    res += (lam_ratio * theta[i]) * log(lam_ratio * theta[i]) + 
      (1 - lam_ratio * theta[i]) * log(1 - lam_ratio * theta[i]);
  }
  res = res / n;
  return res;
}

// Slores initialization
void slores_init(vector<double>& theta_lam, 
                 double *g_theta_lam_ptr, double *prod_deriv_theta_lam_ptr,
                 vector<double>& X_theta_lam_xi_pos,
                 vector<double>& prod_PX_Pxmax_xi_pos,
                 vector<double>& cutoff_xi_pos,
                 XPtr<BigMatrix> xMat, double *y, vector<double>& z, int xmax_idx,
                 int *row_idx, vector<int> &col_idx,
                 NumericVector &center, NumericVector &scale,
                 IntegerVector& ylab, int n_pos, int n, int p) {
  
  double n_pos_ratio = (double)n_pos / n;
  vector<double> deriv_theta_lam(n);
  double prod_deriv_theta_lam = 0.0;
  for (int i = 0; i < n; i++) {
    if (ylab[i] == 1) {
      theta_lam[i] = 1 - n_pos_ratio; 
    } else {
      theta_lam[i] = n_pos_ratio;
    }
    deriv_theta_lam[i] = log(theta_lam[i] / (1 - theta_lam[i])) / n;
    
    prod_deriv_theta_lam += deriv_theta_lam[i] * theta_lam[i];
  }
  *prod_deriv_theta_lam_ptr = prod_deriv_theta_lam;
  *g_theta_lam_ptr = dual_bin(theta_lam, 1.0, 1.0, n);
  
  // compute X_theta_lam_xi_pos (X^Ty) and prod_PX_Pxmax_xi_pos (<Px, Pxmax>)
  double sum_xmaxTy = crossprod_bm(xMat, y, row_idx, center[xmax_idx], scale[xmax_idx], n, xmax_idx);
  double sign_xmaxTy = sign(sum_xmaxTy);
  int j;
#pragma omp parallel for private(j) schedule(static) 
  for (j = 0; j < p; j++) {
    X_theta_lam_xi_pos[j] = -z[j] * n; // = -xTy
    prod_PX_Pxmax_xi_pos[j] = -sign_xmaxTy * crossprod_bm_Xj_Xk(xMat, row_idx, center, scale, n, col_idx[j], xmax_idx); 
    cutoff_xi_pos[j] = prod_PX_Pxmax_xi_pos[j] / n;
  }
}

// Slores screening
void slores_screen(int *slores_reject, vector<double>& theta_lam, 
                   double g_theta_lam, double prod_deriv_theta_lam,
                   vector<double>& X_theta_lam_xi_pos,
                   vector<double>& prod_PX_Pxmax_xi_pos,
                   vector<double>& cutoff_xi_pos,
                   int *row_idx, vector<int> &col_idx,
                   NumericVector &center, NumericVector &scale, int xmax_idx,
                   IntegerVector& ylab, double lambda, 
                   double lambda_max, int n_pos, int n, int p) {
  
  double TOLERANCE = 1e-8;
  double d, r, a2, a1_xi_pos, a0, Delta;
  double d_sq, one_minus_d_sq, n_sq, d_sq_times_n_sq;
  double u2star_xi_pos, u2star_xi_neg, T_temp_pos, T_temp_neg, tmp_pos, tmp_neg;
  NumericVector T_xi_pos(p);
  NumericVector T_xi_neg(p);
  
  if (fabs(lambda - lambda_max) <= TOLERANCE) {
    r = 0.0;
    d = 0.0;
  } else {
    r = sqrt(0.5 * n * (dual_bin(theta_lam, lambda, lambda_max, n) - g_theta_lam + 
      (1 - lambda / lambda_max) * prod_deriv_theta_lam));
    d = sqrt(n) * (lambda_max - lambda) / r;
  }
  d_sq = pow(d, 2);
  one_minus_d_sq = 1 - d_sq;
  n_sq  = pow(n, 2);
  d_sq_times_n_sq = pow(d * n, 2);
  a2 = n_sq * one_minus_d_sq;
  
  int j;
#pragma omp parallel for private(j, a1_xi_pos, a0, Delta, u2star_xi_pos, u2star_xi_neg, T_temp_pos, T_temp_neg, tmp_pos, tmp_neg) schedule(static)
  for (j = 0; j < p; j++) {
    // a1_xi_pos = 0.0;
    // a0 = 0.0; Delta = 0.0;
    // u2star_xi_pos = 0.0; u2star_xi_neg = 0.0;
    // T_temp_pos = 0.0; T_temp_neg = 0.0;
    
    a1_xi_pos = 2 * prod_PX_Pxmax_xi_pos[j] * n * one_minus_d_sq;
    a0 = pow(prod_PX_Pxmax_xi_pos[j], 2) - d_sq_times_n_sq;
    Delta = pow(a1_xi_pos, 2) - 4 * a2 * a0;
    if (Delta < 0.0) Delta = 0.0; // in case of -0.0 (at xmax)
    
    if (cutoff_xi_pos[j] >= d) {
      T_xi_pos[j] = r * sqrt(n) - X_theta_lam_xi_pos[j];
    } else {
      u2star_xi_pos = 0.5 * (-a1_xi_pos + sqrt(Delta)) / a2;
      tmp_pos = n + n * pow(u2star_xi_pos, 2) + 2 * u2star_xi_pos * prod_PX_Pxmax_xi_pos[j];
      if (tmp_pos < 0.0) tmp_pos = 0.0; // in case of -0.0 (at xmax)
      T_temp_pos = sqrt(tmp_pos);
      T_xi_pos[j] = r * T_temp_pos - u2star_xi_pos * n * (lambda_max - lambda) - X_theta_lam_xi_pos[j];
    }
    
    if (T_xi_pos[j] + TOLERANCE > n * lambda) { // cannot reject since T_xi_pos >= n * lambda, no need to compute T_xi_neg
      slores_reject[j] = 0;
    } else {
      // compute T_xi_neg: two cases
      // cutoff_xi_neg = -cutoff_xi_pos;
      if (-cutoff_xi_pos[j] >= d) {
        T_xi_neg[j] = r * sqrt(n) + X_theta_lam_xi_pos[j];
      } else {
        // a1_xi_neg = -a1_xi_pos;
        u2star_xi_neg = 0.5 * (a1_xi_pos + sqrt(Delta)) / a2;
        tmp_neg = n + n * pow(u2star_xi_neg, 2) + 2 * u2star_xi_neg * prod_PX_Pxmax_xi_pos[j];
        if (tmp_neg < 0.0) tmp_neg = 0.0; // in case of -0.0 (at xmax)
        T_temp_neg = sqrt(tmp_neg);
        T_xi_neg[j] = r * T_temp_neg - u2star_xi_neg * n * (lambda_max - lambda) + X_theta_lam_xi_pos[j];
      }
      if (T_xi_neg[j] + TOLERANCE > n * lambda) { // cannot reject since T_xi_neg > n * lambda
        slores_reject[j] = 0;
      } else {
        slores_reject[j] = 1; // both T_xi_pos and T_xi_pos are less than n * lambda
      }
    }
  }
}

// Slores update
void slores_update(vector<double>& theta_lam, vector<double> &z,
                   double sumResid, double *r,
                   double *g_theta_lam_ptr, double *prod_deriv_theta_lam_ptr,
                   vector<double>& X_theta_lam_xi_pos, double lambda_prev,
                   XPtr<BigMatrix> xMat, double *eta, int xmax_idx,
                   int *row_idx, vector<int> &col_idx,
                   NumericVector &center, NumericVector &scale,
                   IntegerVector& ylab, int n, int p) {
  
  vector<double> deriv_theta_lam(n);
  double prod_deriv_theta_lam = 0.0;
  for (int i = 0; i < n; i++) {
    
    theta_lam[i] = 1 / (1 + exp(ylab[i] * eta[i]));
    
    deriv_theta_lam[i] = log(theta_lam[i] / (1 - theta_lam[i])) / n;
    
    prod_deriv_theta_lam += deriv_theta_lam[i] * theta_lam[i];
  }
  *prod_deriv_theta_lam_ptr = prod_deriv_theta_lam;
  *g_theta_lam_ptr = dual_bin(theta_lam, 1.0, 1.0, n);
  
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol;
  int j, jj;
  double sum_xr;
#pragma omp parallel for private(j, jj, xCol, sum_xr) schedule(static)
  for (j = 0; j < p; j++) {
    jj = col_idx[j];
    xCol = xAcc[jj];
    sum_xr = 0.0;
    for(int i = 0; i < n; i++) {
      sum_xr = sum_xr + xCol[row_idx[i]] * r[i];
    }
    z[j] = (sum_xr - center[jj] * sumResid) / scale[jj] / n;
    X_theta_lam_xi_pos[j] = -z[j] * n; 
  }
  
}

// Slores update when updating xmax
void slores_update_xmax(vector<double>& prod_PX_Pxmax_xi_pos,
                        vector<double>& cutoff_xi_pos,
                        XPtr<BigMatrix> xMat, double *y, int xmax_idx,
                        int *row_idx,vector<int> &col_idx,
                        NumericVector &center, NumericVector &scale,
                        int n, int p) {
  double sum_xmaxTy = crossprod_bm(xMat, y, row_idx, center[xmax_idx], scale[xmax_idx], n, xmax_idx);
  double sign_xmaxTy = sign(sum_xmaxTy);
  int j;
#pragma omp parallel for private(j) schedule(static) 
  for (j = 0; j < p; j++) {
    prod_PX_Pxmax_xi_pos[j] = -sign_xmaxTy * crossprod_bm_Xj_Xk(xMat, row_idx, center, scale, n, col_idx[j], xmax_idx); 
    cutoff_xi_pos[j] = prod_PX_Pxmax_xi_pos[j] / n;
  }
}

// Coordinate descent for logistic models with ssr
RcppExport SEXP cdfit_binomial_ssr(SEXP X_, SEXP y_, SEXP row_idx_, 
                                   SEXP lambda_, SEXP nlambda_, SEXP lam_scale_,
                                   SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                                   SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                   SEXP ncore_, SEXP warn_, SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  int warn = INTEGER(warn_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  
  NumericVector lambda(L);
  NumericVector Dev(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L);
  NumericVector beta0(L);
  NumericVector center(p);
  NumericVector scale(p);
  int p_keep = 0; // keep columns whose scale > 1e-6
  int *p_keep_ptr = &p_keep;
  vector<int> col_idx;
  vector<double> z;
  double lambda_max = 0.0;
  double *lambda_max_ptr = &lambda_max;
  int xmax_idx = 0;
  int *xmax_ptr = &xmax_idx;
  
  // set up omp
  int useCores = INTEGER(ncore_)[0];
#ifdef BIGLASSO_OMP_H_
  int haveCores = omp_get_num_procs();
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);
#endif
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("\nPreprocessing start: %s\n", buff1);
  }
  
  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, lambda_max, xmax_idx;
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr,
                               xmax_ptr, xMat, y, row_idx, alpha, n, p);
  p = p_keep; // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  arma::sp_mat beta = arma::sp_mat(p, L); //beta
  double *a = R_Calloc(p, double); //Beta from previous iteration
  double a0 = 0.0; //beta0 from previousiteration
  double *w = R_Calloc(n, double);
  double *s = R_Calloc(n, double); //y_i - pi_i
  double *eta = R_Calloc(n, double);
  int *e1 = R_Calloc(p, int); //ever-active set
  int *e2 = R_Calloc(p, int); //strong set
  double xwr, xwx, pi, u, v, cutoff, l1, l2, shift, si;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart;
  
  double ybar = sum(y, n) / n;
  a0 = beta0[0] = log(ybar / (1-ybar));
  double nullDev = 0;
  double *r = R_Calloc(n, double);
  for (i = 0; i < n; i++) {
    r[i] = y[i];
    nullDev = nullDev - y[i]*log(ybar) - (1-y[i])*log(1-ybar);
    s[i] = y[i] - ybar;
    eta[i] = a0;
  }
  thresh = eps * nullDev / n;
  
  double sumS = sum(s, n); // temp result sum of s
  double sumWResid = 0.0; // temp result: sum of w * r
  
  // set up lambda
  if (user == 0) {
    if (lam_scale) { // set up lambda, equally spaced on log scale
      double log_lambda_max = log(lambda_max);
      double log_lambda_min = log(lambda_min*lambda_max);
      
      double delta = (log_lambda_max - log_lambda_min) / (L-1);
      for (l = 0; l < L; l++) {
        lambda[l] = exp(log_lambda_max - l * delta);
      }
    } else { // equally spaced on linear scale
      double delta = (lambda_max - lambda_min*lambda_max) / (L-1);
      for (l = 0; l < L; l++) {
        lambda[l] = lambda_max - l * delta;
      }
    }
    Dev[0] = nullDev;
    lstart = 1;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  }
  
  for (l = lstart; l < L; l++) {
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
    
    if (l != 0) {
      // Check dfmax
      int nv = 0;
      for (j = 0; j < p; j++) {
        if (a[j] != 0) {
          nv++;
        }
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
        return List::create(beta0, beta, center, scale, lambda, Dev, 
                            iter, n_reject, Rcpp::wrap(col_idx));
      }
      
      // strong set
      cutoff = 2*lambda[l] - lambda[l-1];
      for (j = 0; j < p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
      
    } else {
      // strong set
      cutoff = 2*lambda[l] - lambda_max;
      for (j = 0; j < p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    }
    
    n_reject[l] = p - sum(e2, p);
    while (iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        while (iter[l] < max_iter) {
          iter[l]++;
          Dev[l] = 0.0;
          
          for (i = 0; i < n; i++) {
            if (eta[i] > 10) {
              pi = 1;
              w[i] = .0001;
            } else if (eta[i] < -10) {
              pi = 0;
              w[i] = .0001;
            } else {
              pi = exp(eta[i]) / (1 + exp(eta[i]));
              w[i] = pi * (1 - pi);
            }
            s[i] = y[i] - pi;
            r[i] = s[i] / w[i];
            if (y[i] == 1) {
              Dev[l] = Dev[l] - log(pi);
            } else {
              Dev[l] = Dev[l] - log(1-pi);
            }
          }
          
          if (Dev[l] / nullDev < .01) {
            if (warn) warning("Model saturated; exiting...");
            for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
            R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
            return List::create(beta0, beta, center, scale, lambda, Dev,
                                iter, n_reject, Rcpp::wrap(col_idx));
          }
          
          // Intercept
          xwr = crossprod(w, r, n, 0);
          xwx = sum(w, n);
          beta0[l] = xwr / xwx + a0;
          si = beta0[l] - a0;
          if (si != 0) {
            a0 = beta0[l];
            for (i = 0; i < n; i++) {
              r[i] -= si; //update r
              eta[i] += si; //update eta
            }
          }
          sumWResid = wsum(r, w, n); // update temp result: sum of w * r, used for computing xwr;
          
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              xwr = wcrossprod_resid(xMat, r, sumWResid, row_idx, center[jj], scale[jj], w, n, jj);
              v = wsqsum_bm(xMat, w, row_idx, center[jj], scale[jj], n, jj) / n;
              u = xwr/n + v * a[j];
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(u, l1, l2, v);
              
              shift = beta(j, l) - a[j];
              if (shift !=0) {
                // update change of objective function
                // update = - u * shift + (0.5 * v + 0.5 * l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) + l1 * (fabs(beta(j, l)) - fabs(a[j]));
                
                update = pow(beta(j, l) - a[j], 2) * v;
                if (update > max_update) max_update = update;
                update_resid_eta(r, eta, xMat, shift, row_idx, center[jj], scale[jj], n, jj); // update r
                sumWResid = wsum(r, w, n); // update temp result w * r, used for computing xwr;
                a[j] = beta(j, l); // update a
              }
            }
          }
          // Check for convergence
          if (max_update < thresh)  break;
        }
        // Scan for violations in strong set
        sumS = sum(s, n);
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
        if (violations==0) break;
      }
      // Scan for violations in rest
      violations = check_rest_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
      if (violations==0) break;
    }
  }
  R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
  return List::create(beta0, beta, center, scale, lambda, Dev, iter, n_reject, Rcpp::wrap(col_idx));
  
}

// Coordinate descent for logistic models with ssr and approximate hessian
RcppExport SEXP cdfit_binomial_ssr_approx(SEXP X_, SEXP y_, SEXP row_idx_, 
                                          SEXP lambda_, SEXP nlambda_,
                                          SEXP lambda_min_, SEXP alpha_, SEXP user_, SEXP eps_, 
                                          SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                          SEXP ncore_, SEXP warn_, SEXP verbose_) {
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  int warn = INTEGER(warn_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  
  NumericVector lambda(L);
  NumericVector Dev(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L);
  NumericVector beta0(L);
  NumericVector center(p);
  NumericVector scale(p);
  int p_keep = 0; // keep columns whose scale > 1e-6
  int *p_keep_ptr = &p_keep;
  vector<int> col_idx;
  vector<double> z;
  double lambda_max = 0.0;
  double *lambda_max_ptr = &lambda_max;
  int xmax_idx = 0;
  int *xmax_ptr = &xmax_idx;
  
  // set up omp
  int useCores = INTEGER(ncore_)[0];
#ifdef BIGLASSO_OMP_H_
  int haveCores = omp_get_num_procs();
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);
#endif
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("\nPreprocessing start: %s\n", buff1);
  }
  
  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, lambda_max, xmax_idx;
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr,
                               xmax_ptr, xMat, y, row_idx, alpha, n, p);
  // set p = p_keep, only loop over columns whose scale > 1e-6
  p = p_keep;
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  arma::sp_mat beta = arma::sp_mat(p, L); //beta
  double *a = R_Calloc(p, double); //Beta from previous iteration
  double a0 = 0.0; //beta0 from previousiteration
  double *w = R_Calloc(n, double);
  double *s = R_Calloc(n, double); //y_i - pi_i
  double *eta = R_Calloc(n, double);
  int *e1 = R_Calloc(p, int); //ever-active set
  int *e2 = R_Calloc(p, int); //strong set
  double xwr, xwx, pi, u, v, cutoff, l1, l2, shift, si;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart; // temp index
  
  double ybar = sum(y, n)/n;
  a0 = beta0[0] = log(ybar/(1-ybar));
  double nullDev = 0;
  double *r = R_Calloc(n, double);
  for (i = 0; i < n; i++) {
    r[i] = y[i];
    nullDev = nullDev - y[i] * log(ybar) - (1 - y[i]) * log(1 - ybar);
    s[i] = y[i] - ybar;
    eta[i] = a0;
  }
  thresh = eps * nullDev / n; // threshold for convergence
  
  double sumS; // temp result sum of s
  double sumResid; // temp result sum of current residuals
  // double sumWResid = 0.0; // temp result: sum of w * r
  
  if (user == 0) {
    // set up lambda, equally spaced on log scale
    double log_lambda_max = log(lambda_max);
    double log_lambda_min = log(lambda_min*lambda_max);
    double delta = (log_lambda_max - log_lambda_min) / (L-1);
    for (l = 0; l < L; l++) {
      lambda[l] = exp(log_lambda_max - l * delta);
    }
    Dev[0] = nullDev;
    lstart = 1;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  }
  
  for (l = lstart; l < L; l++) {
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
    
    if (l != 0) {
      // Check dfmax
      int nv = 0;
      for (j = 0; j < p; j++) {
        if (a[j] != 0) {
          nv++;
        }
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
        return List::create(beta0, beta, center, scale, lambda, Dev, 
                            iter, n_reject, Rcpp::wrap(col_idx));
      }
      // strong set
      cutoff = 2*lambda[l] - lambda[l-1];
      for (j = 0; j < p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    } else {
      // strong set
      cutoff = 2*lambda[l] - lambda_max;
      for (int j=0; j<p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    }
    n_reject[l] = p - sum(e2, p);
    
    while (iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        while (iter[l] < max_iter) {
          iter[l]++;
          Dev[l] = 0.0;
          for (i = 0; i < n; i++) {
            if (eta[i] > 10) {
              pi = 1;
              //w[i] = .0001;
            } else if (eta[i] < -10) {
              pi = 0;
              //w[i] = .0001;
            } else {
              pi = exp(eta[i]) / (1 + exp(eta[i]));
              //w[i] = pi*(1-pi);
            }
            s[i] = y[i] - pi;
            r[i] = s[i] / 0.25;
            // r[i] = s[i]/w[i];
            if (y[i]==1) {
              Dev[l] = Dev[l] - log(pi);
            } else {
              Dev[l] = Dev[l] - log(1 - pi);
            }
          }
          if (Dev[l] / nullDev < .01) {
            if (warn) warning("Model saturated; exiting...");
            for (int ll = l; ll < L; ll++) iter[ll] = NA_INTEGER;
            R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
            return List::create(beta0, beta, center, scale, lambda, Dev, 
                                iter, n_reject, Rcpp::wrap(col_idx));
          }
          // Intercept
          xwr = 0.25 * sum(r, n);
          xwx = 0.25 * n;
          //           xwr = crossprod(w, r, n, 0);
          //           xwx = sum(w, n);
          beta0[l] = xwr / xwx + a0;
          si = beta0[l] - a0;
          if (si != 0) {
            a0 = beta0[l];
            for (i = 0; i < n; i++) {
              r[i] -= si; //update r
              eta[i] += si; //update eta
            }
          }
          sumResid = sum(r, n); // update temp result: sum of r, used for computing xwr;
          // sumWResid = wsum(r, w, n);
          
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              xwr = 0.25 * crossprod_resid(xMat, r, sumResid, row_idx, center[jj], scale[jj], n, jj);
              v = 0.25; // x^T * W * x / n = w = 0.25
              // xwr = wcrossprod_resid(xMat, r, sumWResid, row_idx, center[j], scale[j], w, n, j);
              // v = wsqsum_bm(xMat, w, row_idx, center[j], scale[j], n, j) / n;
              u = xwr / n + v * a[j];
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1 - alpha);
              beta(j, l) = lasso(u, l1, l2, v);
              
              shift = beta(j, l) - a[j];
              if (shift !=0) {
                // update change of objective function
                // update = - u * shift + (0.5 * v + 0.5 * l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) + l1 * (fabs(beta(j, l)) - fabs(a[j]));
                update = pow(beta(j, l) - a[j], 2) * v;
                if (update > max_update) max_update = update;
                update_resid_eta(r, eta, xMat, shift, row_idx, center[jj], scale[jj], n, jj); // update r
                sumResid = sum(r, n); // update temp result w * r, used for computing xwr;
                a[j] = beta(j, l); // update a
              }
            }
          }
          // Check for convergence
          if (max_update < thresh) break;
        }
        
        // Scan for violations in strong set
        sumS = sum(s, n);
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
        if (violations==0) break;
      }
      
      // Scan for violations in rest
      violations = check_rest_set(e1, e2, z, xMat, row_idx, col_idx,center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
      if (violations==0) break;
    }
  }
  
  R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
  return List::create(beta0, beta, center, scale, lambda, Dev, iter, n_reject, Rcpp::wrap(col_idx));
}

// Coordinate descent for logistic models with slores-ssr
RcppExport SEXP cdfit_binomial_slores_ssr(SEXP X_, SEXP y_, SEXP n_pos_, SEXP ylab_, 
                                          SEXP row_idx_, SEXP lambda_, SEXP nlambda_,
                                          SEXP lam_scale_, SEXP lambda_min_, 
                                          SEXP alpha_, SEXP user_, 
                                          SEXP eps_, SEXP max_iter_, SEXP multiplier_, 
                                          SEXP dfmax_, SEXP ncore_, SEXP warn_,
                                          SEXP safe_thresh_, SEXP verbose_) {
  //ProfilerStart("Slores.out");
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int n_pos = INTEGER(n_pos_)[0];
  IntegerVector ylabel = Rcpp::as<IntegerVector>(ylab_); // label vector of {-1, 1}
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  int warn = INTEGER(warn_)[0];
  int user = INTEGER(user_)[0];
  double slores_thresh = REAL(safe_thresh_)[0]; // threshold for safe test
  int verbose = INTEGER(verbose_)[0];
  
  NumericVector lambda(L);
  NumericVector Dev(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L); // number of total rejections;
  IntegerVector n_slores_reject(L); // number of safe rejections;
  NumericVector beta0(L);
  NumericVector center(p);
  NumericVector scale(p);
  int p_keep = 0; // keep columns whose scale > 1e-6
  int *p_keep_ptr = &p_keep;
  vector<int> col_idx;
  vector<double> z;
  double lambda_max = 0.0;
  double *lambda_max_ptr = &lambda_max;
  int xmax_idx = 0;
  int *xmax_ptr = &xmax_idx;
  
  // set up omp
  int useCores = INTEGER(ncore_)[0];
#ifdef BIGLASSO_OMP_H_
  int haveCores = omp_get_num_procs();
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);
#endif
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("\nPreprocessing start: %s\n", buff1);
  }
  
  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, lambda_max, xmax_idx;
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr,
                               xmax_ptr, xMat, y, row_idx, alpha, n, p);
  p = p_keep; // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  arma::sp_mat beta = arma::sp_mat(p, L); //beta
  double *a = R_Calloc(p, double); //Beta from previous iteration
  double a0 = 0.0; //beta0 from previousiteration
  double *w = R_Calloc(n, double);
  double *s = R_Calloc(n, double); //y_i - pi_i
  double *eta = R_Calloc(n, double);
  int *e1 = R_Calloc(p, int); //ever-active set
  int *e2 = R_Calloc(p, int); //strong set
  double xwr, xwx, pi, u, v, cutoff, l1, l2, shift, si;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart;
  
  double ybar = sum(y, n) / n;
  a0 = beta0[0] = log(ybar / (1-ybar));
  double nullDev = 0;
  double *r = R_Calloc(n, double);
  for (i = 0; i < n; i++) {
    r[i] = y[i];
    nullDev = nullDev - y[i]*log(ybar) - (1-y[i])*log(1-ybar);
    s[i] = y[i] - ybar;
    eta[i] = a0;
  }
  thresh = eps * nullDev / n;
  double sumS = sum(s, n); // temp result sum of s
  double sumWResid = 0.0; // temp result: sum of w * r
  
  // set up lambda
  if (user == 0) {
    if (lam_scale) { // set up lambda, equally spaced on log scale
      double log_lambda_max = log(lambda_max);
      double log_lambda_min = log(lambda_min*lambda_max);
      
      double delta = (log_lambda_max - log_lambda_min) / (L-1);
      for (l = 0; l < L; l++) {
        lambda[l] = exp(log_lambda_max - l * delta);
      }
    } else { // equally spaced on linear scale
      double delta = (lambda_max - lambda_min*lambda_max) / (L-1);
      for (l = 0; l < L; l++) {
        lambda[l] = lambda_max - l * delta;
      }
    }
    Dev[0] = nullDev;
    lstart = 1;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  }
  
  // Slores variables
  vector<double> theta_lam;
  double g_theta_lam = 0.0;
  double prod_deriv_theta_lam = 0.0;
  double *g_theta_lam_ptr = &g_theta_lam;
  double *prod_deriv_theta_lam_ptr = &prod_deriv_theta_lam;
  vector<double> X_theta_lam_xi_pos; 
  vector<double> prod_PX_Pxmax_xi_pos;
  vector<double> cutoff_xi_pos;
  int *slores_reject = R_Calloc(p, int);
  int *slores_reject_old = R_Calloc(p, int);
  for (int j = 0; j < p; j++) slores_reject_old[j] = 1;
  
  int slores; // if 0, don't perform Slores rule
  if (slores_thresh < 1) {
    slores = 1; // turn on slores
    theta_lam.resize(n);
    X_theta_lam_xi_pos.resize(p);
    prod_PX_Pxmax_xi_pos.resize(p);
    cutoff_xi_pos.resize(p);
    
    slores_init(theta_lam, g_theta_lam_ptr, prod_deriv_theta_lam_ptr, cutoff_xi_pos,
                X_theta_lam_xi_pos, prod_PX_Pxmax_xi_pos, 
                xMat, y, z, xmax_idx, row_idx, col_idx, 
                center, scale, ylabel, n_pos, n, p);
  } else {
    slores = 0;
  }
  
  if (slores == 1 && user == 0) n_slores_reject[0] = p;
  
  for (l = lstart; l < L; l++) {
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
    
    if (l != 0) {
      // Check dfmax
      int nv = 0;
      for (j = 0; j < p; j++) {
        if (a[j] != 0) {
          nv++;
        }
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        R_Free(slores_reject); R_Free(slores_reject_old);
        R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
        //ProfilerStop();
        return List::create(beta0, beta, center, scale, lambda, Dev, 
                            iter, n_reject, Rcpp::wrap(col_idx));
      }
      cutoff = 2*lambda[l] - lambda[l-1];
    } else {
      cutoff = 2*lambda[l] - lambda_max;
    }
    
    if (slores) {
      slores_screen(slores_reject, theta_lam, g_theta_lam, prod_deriv_theta_lam,
                    X_theta_lam_xi_pos, prod_PX_Pxmax_xi_pos, cutoff_xi_pos,
                    row_idx, col_idx, center, scale, xmax_idx, ylabel, 
                    lambda[l], lambda_max, n_pos, n, p);
      n_slores_reject[l] = sum(slores_reject, p);
      
      // update z[j] for features which are rejected at previous lambda but accepted at current one.
      update_zj(z, slores_reject, slores_reject_old, xMat, row_idx, col_idx, center, scale, sumS, s, m, n, p);
      
#pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        slores_reject_old[j] = slores_reject[j];
        // hsr screening
        // if (slores_reject[j] == 0 && (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]]))) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    } else {
      n_slores_reject[l] = 0; 
      // hsr screening over all
#pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    }
    n_reject[l] = p - sum(e2, p);
    
    while (iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        while (iter[l] < max_iter) {
          iter[l]++;
          Dev[l] = 0.0;
          
          for (i = 0; i < n; i++) {
            if (eta[i] > 10) {
              pi = 1;
              w[i] = .0001;
            } else if (eta[i] < -10) {
              pi = 0;
              w[i] = .0001;
            } else {
              pi = exp(eta[i]) / (1 + exp(eta[i]));
              w[i] = pi * (1 - pi);
            }
            s[i] = y[i] - pi;
            r[i] = s[i] / w[i];
            if (y[i] == 1) {
              Dev[l] = Dev[l] - log(pi);
            } else {
              Dev[l] = Dev[l] - log(1-pi);
            }
          }
          
          if (Dev[l] / nullDev < .01) {
            if (warn) warning("Model saturated; exiting...");
            for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
            R_Free(slores_reject); R_Free(slores_reject_old);
            R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
            return List::create(beta0, beta, center, scale, lambda, Dev, iter, n_reject, n_slores_reject, Rcpp::wrap(col_idx));
          }
          // Intercept
          xwr = crossprod(w, r, n, 0);
          xwx = sum(w, n);
          beta0[l] = xwr / xwx + a0;
          si = beta0[l] - a0;
          if (si != 0) {
            a0 = beta0[l];
            for (i = 0; i < n; i++) {
              r[i] -= si; //update r
              eta[i] += si; //update eta
            }
          }
          sumWResid = wsum(r, w, n); // update temp result: sum of w * r, used for computing xwr;
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              xwr = wcrossprod_resid(xMat, r, sumWResid, row_idx, center[jj], scale[jj], w, n, jj);
              v = wsqsum_bm(xMat, w, row_idx, center[jj], scale[jj], n, jj) / n;
              u = xwr/n + v * a[j];
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(u, l1, l2, v);
              
              shift = beta(j, l) - a[j];
              if (shift != 0) {
                // update change of objective function
                // update = - u * shift + (0.5 * v + 0.5 * l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) + l1 * (fabs(beta(j, l)) - fabs(a[j]));
                update = pow(beta(j, l) - a[j], 2) * v;
                if (update > max_update) max_update = update;
                update_resid_eta(r, eta, xMat, shift, row_idx, center[jj], scale[jj], n, jj); // update r
                sumWResid = wsum(r, w, n); // update temp result w * r, used for computing xwr;
                a[j] = beta(j, l); // update a
              }
            }
          }
          // Check for convergence
          if (max_update < thresh)  break;
        }
        // Scan for violations in strong set
        sumS = sum(s, n);
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
        if (violations == 0) break;
      }
      // Scan for violations in rest
      if (slores) {
        violations = check_rest_safe_set(e1, e2, slores_reject, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
      } else {
        violations = check_rest_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
      }
      if (violations == 0) break;
      if (n_slores_reject[l] <= p * slores_thresh) {
        slores = 0; // turn off slores screening for next iteration if not efficient
      }
    }
  }
  R_Free(slores_reject); R_Free(slores_reject_old);
  R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
  //ProfilerStop();
  return List::create(beta0, beta, center, scale, lambda, Dev, iter, n_reject, n_slores_reject, Rcpp::wrap(col_idx));
}

// Coordinate descent for logistic models with ada-slores_ssr
RcppExport SEXP cdfit_binomial_ada_slores_ssr(SEXP X_, SEXP y_, SEXP n_pos_, SEXP ylab_, 
                                              SEXP row_idx_, SEXP lambda_, SEXP nlambda_,
                                              SEXP lam_scale_, SEXP lambda_min_, 
                                              SEXP alpha_, SEXP user_, 
                                              SEXP eps_, SEXP max_iter_, SEXP multiplier_, 
                                              SEXP dfmax_, SEXP ncore_, SEXP warn_,
                                              SEXP safe_thresh_, SEXP update_thresh_, SEXP verbose_) {
  //ProfilerStart("Slores-ada.out");
  XPtr<BigMatrix> xMat(X_);
  double *y = REAL(y_);
  int n_pos = INTEGER(n_pos_)[0];
  IntegerVector ylabel = Rcpp::as<IntegerVector>(ylab_); // label vector of {-1, 1}
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  int warn = INTEGER(warn_)[0];
  int user = INTEGER(user_)[0];
  double slores_thresh = REAL(safe_thresh_)[0]; // threshold for safe test
  int verbose = INTEGER(verbose_)[0];
  double update_thresh = REAL(update_thresh_)[0];
  
  NumericVector lambda(L);
  NumericVector Dev(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L); // number of total rejections;
  IntegerVector n_slores_reject(L); // number of safe rejections;
  NumericVector beta0(L);
  NumericVector center(p);
  NumericVector scale(p);
  int p_keep = 0; // keep columns whose scale > 1e-6
  int *p_keep_ptr = &p_keep;
  vector<int> col_idx;
  vector<double> z;
  double lambda_max = 0.0;
  double *lambda_max_ptr = &lambda_max;
  int xmax_idx = 0;
  int *xmax_ptr = &xmax_idx;
  
  // set up omp
  int useCores = INTEGER(ncore_)[0];
#ifdef BIGLASSO_OMP_H_
  int haveCores = omp_get_num_procs();
  if(useCores < 1) {
    useCores = haveCores;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(useCores);
#endif
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("\nPreprocessing start: %s\n", buff1);
  }
  
  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, lambda_max, xmax_idx;
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, lambda_max_ptr,
                               xmax_ptr, xMat, y, row_idx, alpha, n, p);
  p = p_keep; // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  arma::sp_mat beta = arma::sp_mat(p, L); //beta
  double *a = R_Calloc(p, double); //Beta from previous iteration
  double a0 = 0.0; //beta0 from previousiteration
  double *w = R_Calloc(n, double);
  double *s = R_Calloc(n, double); //y_i - pi_i
  double *eta = R_Calloc(n, double);
  int *e1 = R_Calloc(p, int); //ever-active set
  int *e2 = R_Calloc(p, int); //strong set
  double xwr, xwx, pi, u, v, cutoff, l1, l2, shift, si;
  double max_update, update, thresh; // for convergence check
  int i, j, jj, l, violations, lstart;
  
  double ybar = sum(y, n) / n;
  a0 = beta0[0] = log(ybar / (1-ybar));
  double nullDev = 0;
  double *r = R_Calloc(n, double);
  for (i = 0; i < n; i++) {
    r[i] = y[i];
    nullDev = nullDev - y[i]*log(ybar) - (1-y[i])*log(1-ybar);
    s[i] = y[i] - ybar;
    eta[i] = a0;
  }
  thresh = eps * nullDev / n;
  double sumS = sum(s, n); // temp result sum of s
  double sumWResid = 0.0; // temp result: sum of w * r
  
  // set up lambda
  if (user == 0) {
    if (lam_scale) { // set up lambda, equally spaced on log scale
      double log_lambda_max = log(lambda_max);
      double log_lambda_min = log(lambda_min*lambda_max);
      
      double delta = (log_lambda_max - log_lambda_min) / (L-1);
      for (l = 0; l < L; l++) {
        lambda[l] = exp(log_lambda_max - l * delta);
      }
    } else { // equally spaced on linear scale
      double delta = (lambda_max - lambda_min*lambda_max) / (L-1);
      for (l = 0; l < L; l++) {
        lambda[l] = lambda_max - l * delta;
      }
    }
    Dev[0] = nullDev;
    lstart = 1;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  }
  
  // Slores variables
  vector<double> theta_lam;
  double g_theta_lam = 0.0;
  double prod_deriv_theta_lam = 0.0;
  double *g_theta_lam_ptr = &g_theta_lam;
  double *prod_deriv_theta_lam_ptr = &prod_deriv_theta_lam;
  vector<double> X_theta_lam_xi_pos; 
  vector<double> prod_PX_Pxmax_xi_pos;
  vector<double> cutoff_xi_pos;
  int *slores_reject = R_Calloc(p, int);
  int *slores_reject_old = R_Calloc(p, int);
  for (int j = 0; j < p; j++) slores_reject_old[j] = 1;
  int l_prev = lstart;
  double gain = 0.0;
  int xmax_invalid = 0;
  double beta_max = 0.0;
  int beta_max_idx = 0;
  
  int slores; // if 0, don't perform Slores rule
  if (slores_thresh < 1) {
    slores = 1; // turn on slores
    theta_lam.resize(n);
    X_theta_lam_xi_pos.resize(p);
    prod_PX_Pxmax_xi_pos.resize(p);
    cutoff_xi_pos.resize(p);
    
    slores_init(theta_lam, g_theta_lam_ptr, prod_deriv_theta_lam_ptr, cutoff_xi_pos,
                X_theta_lam_xi_pos, prod_PX_Pxmax_xi_pos, 
                xMat, y, z, xmax_idx, row_idx, col_idx, 
                center, scale, ylabel, n_pos, n, p);
  } else {
    slores = 0;
  }
  
  if (slores == 1 && user == 0) n_slores_reject[0] = p;
  
  for (l = lstart; l < L; l++) {
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
    
    if (l != 0) {
      // Check dfmax
      int nv = 0;
      for (j = 0; j < p; j++) {
        if (a[j] != 0) {
          nv++;
        }
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        R_Free(slores_reject); R_Free(slores_reject_old);
        R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
        //ProfilerStop();
        return List::create(beta0, beta, center, scale, lambda, Dev, 
                            iter, n_reject, Rcpp::wrap(col_idx));
      }
      cutoff = 2*lambda[l] - lambda[l-1];
    } else {
      cutoff = 2*lambda[l] - lambda_max;
    }
    
    if (slores) {
      // update rule if not discarding enough
      if(gain - n_slores_reject[l - 1] * (l - l_prev) > update_thresh * (xmax_invalid + 1) * p && l != L - 1) {
        l_prev = l - 1;
        // If xmax is not active, choose another x as xmax.
        if(xmax_invalid) {
          xmax_idx = beta_max_idx;
          xmax_invalid = 0;
          slores_update_xmax(prod_PX_Pxmax_xi_pos, cutoff_xi_pos, xMat, y, xmax_idx,
                             row_idx, col_idx, center, scale, n, p);
        }
        slores_update(theta_lam, z, sumS, s, g_theta_lam_ptr, prod_deriv_theta_lam_ptr,
                      X_theta_lam_xi_pos, lambda[l_prev], xMat, eta, xmax_idx, 
                      row_idx, col_idx, center, scale, ylabel, n, p);
        
        slores_screen(slores_reject, theta_lam, g_theta_lam, prod_deriv_theta_lam,
                      X_theta_lam_xi_pos, prod_PX_Pxmax_xi_pos, cutoff_xi_pos,
                      row_idx, col_idx, center, scale, xmax_idx, ylabel, 
                      lambda[l], lambda[l_prev], n_pos, n, p);
        n_slores_reject[l] = sum(slores_reject, p);
        gain = n_slores_reject[l];
        
      } else {
        slores_screen(slores_reject, theta_lam, g_theta_lam, prod_deriv_theta_lam,
                      X_theta_lam_xi_pos, prod_PX_Pxmax_xi_pos, cutoff_xi_pos,
                      row_idx, col_idx, center, scale, xmax_idx, ylabel, 
                      lambda[l], lambda[l_prev], n_pos, n, p);
        n_slores_reject[l] = sum(slores_reject, p);
        gain += n_slores_reject[l];
        
        // update z[j] for features which are rejected at previous lambda but accepted at current one.
        //update_zj(z, slores_reject, slores_reject_old, xMat, row_idx, col_idx, center, scale, sumS, s, m, n, p);
      }
      
      
#pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        slores_reject_old[j] = slores_reject[j];
        // hsr screening
        // if (slores_reject[j] == 0 && (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]]))) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    } else {
      n_slores_reject[l] = 0; 
      // hsr screening over all
#pragma omp parallel for private(j) schedule(static) 
      for (j = 0; j < p; j++) {
        if (fabs(z[j]) > (cutoff * alpha * m[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    }
    n_reject[l] = p - sum(e2, p);
    
    while (iter[l] < max_iter) {
      while (iter[l] < max_iter) {
        while (iter[l] < max_iter) {
          iter[l]++;
          Dev[l] = 0.0;
          
          for (i = 0; i < n; i++) {
            if (eta[i] > 10) {
              pi = 1;
              w[i] = .0001;
            } else if (eta[i] < -10) {
              pi = 0;
              w[i] = .0001;
            } else {
              pi = exp(eta[i]) / (1 + exp(eta[i]));
              w[i] = pi * (1 - pi);
            }
            s[i] = y[i] - pi;
            r[i] = s[i] / w[i];
            if (y[i] == 1) {
              Dev[l] = Dev[l] - log(pi);
            } else {
              Dev[l] = Dev[l] - log(1-pi);
            }
          }
          
          if (Dev[l] / nullDev < .01) {
            if (warn) warning("Model saturated; exiting...");
            for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
            R_Free(slores_reject); R_Free(slores_reject_old);
            R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
            return List::create(beta0, beta, center, scale, lambda, Dev, iter, n_reject, n_slores_reject, Rcpp::wrap(col_idx));
          }
          // Intercept
          xwr = crossprod(w, r, n, 0);
          xwx = sum(w, n);
          beta0[l] = xwr / xwx + a0;
          si = beta0[l] - a0;
          if (si != 0) {
            a0 = beta0[l];
            for (i = 0; i < n; i++) {
              r[i] -= si; //update r
              eta[i] += si; //update eta
            }
          }
          sumWResid = wsum(r, w, n); // update temp result: sum of w * r, used for computing xwr;
          max_update = 0.0;
          beta_max = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              xwr = wcrossprod_resid(xMat, r, sumWResid, row_idx, center[jj], scale[jj], w, n, jj);
              v = wsqsum_bm(xMat, w, row_idx, center[jj], scale[jj], n, jj) / n;
              u = xwr/n + v * a[j];
              l1 = lambda[l] * m[jj] * alpha;
              l2 = lambda[l] * m[jj] * (1-alpha);
              beta(j, l) = lasso(u, l1, l2, v);
              if(fabs(lasso(u, l1, l2, v)) > beta_max) {
                beta_max = fabs(lasso(u, l1, l2, v));
                beta_max_idx = jj;
              }
              if(jj == xmax_idx) {
                if(fabs(u) < l1) {
                  xmax_invalid = 1;
                } else {
                  xmax_invalid = 0;
                }
              }
              shift = beta(j, l) - a[j];
              if (shift != 0) {
                // update change of objective function
                // update = - u * shift + (0.5 * v + 0.5 * l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) + l1 * (fabs(beta(j, l)) - fabs(a[j]));
                update = pow(beta(j, l) - a[j], 2) * v;
                if (update > max_update) max_update = update;
                update_resid_eta(r, eta, xMat, shift, row_idx, center[jj], scale[jj], n, jj); // update r
                sumWResid = wsum(r, w, n); // update temp result w * r, used for computing xwr;
                a[j] = beta(j, l); // update a
              }
            }
          }
          // Check for convergence
          if (max_update < thresh)  break;
        }
        // Scan for violations in strong set
        sumS = sum(s, n);
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
        if (violations == 0) break;
      }
      // Scan for violations in rest
      if (slores) {
        violations = check_rest_safe_set(e1, e2, slores_reject, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
      } else {
        violations = check_rest_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumS, alpha, s, m, n, p);
      }
      if (violations == 0) break;
      /*
       if (n_slores_reject[l] <= p * slores_thresh) {
       slores = 0; // turn off slores screening for next iteration if not efficient
       }*/
    }
  }
  R_Free(slores_reject); R_Free(slores_reject_old);
  R_Free(s); R_Free(w); R_Free(a); R_Free(r); R_Free(e1); R_Free(e2); R_Free(eta);
  //ProfilerStop();
  return List::create(beta0, beta, center, scale, lambda, Dev, iter, n_reject, n_slores_reject, Rcpp::wrap(col_idx));
}
