#include "utilities.h"
//#include "gperftools/profiler.h"

// standardize for multiresponse
void standardize_and_get_residual(NumericVector &center, NumericVector &scale, 
                                  int *p_keep_ptr, vector<int> &col_idx, //columns to keep, removing columns whose scale < 1e-6
                                  vector<double> &z, double *lambda_max_ptr,
                                  int *xmax_ptr, XPtr<BigMatrix> xMat, 
                                  NumericMatrix &Y, int *row_idx,
                                  double alpha, int n, int p, int m) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol;
  double *sum_xy = R_Calloc(m, double);
  double *sum_y = R_Calloc(m, double);
  double zmax = 0.0, zj = 0.0;
  int i, j, k;
  
  for(k = 0; k < m; k++) {
    sum_y[k] = 0.0;
    for(i = 0; i < n; i++) {
      sum_y[k] += Y(k, i);
    }
  }
  
  for (j = 0; j < p; j++) {
    xCol = xAcc[j];
    for(k  = 0; k < m; k++) {
      sum_xy[k] = 0.0;
    }
    
    for (i = 0; i < n; i++) {
      center[j] += xCol[row_idx[i]];
      scale[j] += pow(xCol[row_idx[i]], 2);
      for(k  = 0; k < m; k++) {
        sum_xy[k] += xCol[row_idx[i]] * Y.at(k, i);
      }
    }
    
    center[j] = center[j] / n; //center
    scale[j] = sqrt(scale[j] / n - pow(center[j], 2)); //scale
    
    if (scale[j] > 1e-6) {
      col_idx.push_back(j);
      zj = 0.0;
      for(k = 0; k < m; k++) {
        zj += pow(sum_xy[k] - center[j] * sum_y[k], 2);
      }
      zj = sqrt(zj) / (scale[j] * n * sqrt(m)); //residual
      if (fabs(zj) > zmax) {
        zmax = fabs(zj);
        *xmax_ptr = j; // xmax_ptr is the index in the raw xMat, not index in col_idx!
      }
      z.push_back(zj);
    }
  }
  *p_keep_ptr = col_idx.size();
  *lambda_max_ptr = zmax / alpha;
  R_Free(sum_xy); R_Free(sum_y);
}

// standardize for multiresponse and store XtY
void standardize_and_get_residual(NumericVector &center, NumericVector &scale, 
                                  int *p_keep_ptr, vector<int> &col_idx, //columns to keep, removing columns whose scale < 1e-6
                                  vector<double> &z, vector<double> &XtY,
                                  double *lambda_max_ptr, int *xmax_ptr, 
                                  XPtr<BigMatrix> xMat, NumericMatrix &Y, 
                                  int *row_idx, double alpha,
                                  int n, int p, int m) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol;
  double *sum_xy = R_Calloc(m, double);
  double *sum_y = R_Calloc(m, double);
  double zmax = 0.0, zj = 0.0;
  int i, j, k;
  
  for(k = 0; k < m; k++) {
    sum_y[k] = 0.0;
    for(i = 0; i < n; i++) {
      sum_y[k] += Y(k, i);
    }
  }
  
  for (j = 0; j < p; j++) {
    xCol = xAcc[j];
    for(k  = 0; k < m; k++) {
      sum_xy[k] = 0.0;
    }
    
    for (i = 0; i < n; i++) {
      center[j] += xCol[row_idx[i]];
      scale[j] += pow(xCol[row_idx[i]], 2);
      for(k  = 0; k < m; k++) {
        sum_xy[k] += xCol[row_idx[i]] * Y.at(k, i);
      }
    }
    
    center[j] = center[j] / n; //center
    scale[j] = sqrt(scale[j] / n - pow(center[j], 2)); //scale
    
    if (scale[j] > 1e-6) {
      col_idx.push_back(j);
      zj = 0;
      for(k = 0; k < m; k++) {
        zj += pow(sum_xy[k] - center[j] * sum_y[k], 2);
        XtY.push_back((sum_xy[k] - center[j] * sum_y[k]) / scale[j]);
      }
      zj = sqrt(zj) / (scale[j] * n * sqrt(m)); //residual
      if (fabs(zj) > zmax) {
        zmax = fabs(zj);
        *xmax_ptr = j; // xmax_ptr is the index in the raw xMat, not index in col_idx!
      }
      z.push_back(zj);
    }
  }
  *p_keep_ptr = col_idx.size();
  *lambda_max_ptr = zmax / alpha;
  R_Free(sum_xy); R_Free(sum_y);
}

// Crossproduct xjTR
void crossprod_resid(double *xTR, XPtr<BigMatrix> xMat, double *R,
                     double *sumResid, int *row_idx,
                     double center, double scale, int n, int m, int j) {
  MatrixAccessor<double> xAcc(*xMat);
  double *xCol = xAcc[j];
  double xi;
  int i, k;
  for(k = 0; k < m; k++) xTR[k] = 0.0;
  for (i = 0; i < n; i++) {
    xi = xCol[row_idx[i]];
    for(k = 0; k < m; k++) {
      xTR[k] += xi * R[i*m+k];
    }
  }
  for(k = 0; k < m; k++){
    xTR[k] = (xTR[k] - center * sumResid[k]) / scale;
  } 
}

// Update beta
void lasso(arma::field<arma::sp_mat> &beta, double *xTR, double z, double l1, double l2, int j, int l, int m) {
  if(z <= l1) {
    for(int k = 0; k < m; k++) {
      beta.at(k).at(j, l) = 0;
    }
  } else {
    for(int k = 0; k < m; k++) {
      beta.at(k).at(j, l) = sqrt(m) * xTR[k] * (1 - l1 / z) / (1 + l2);
    }
  }
}

// update residul matrix
void update_resid(XPtr<BigMatrix> xpMat, double *R, double *sumResid, double *shift,
                  int *row_idx, double center, double scale, int n, int m, int j) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol = xAcc[j];
  double xi;
  int ik;
  for(int k = 0; k < m; k++) sumResid[k] = 0.0;
  for (int i =0; i < n; i++) {
    xi = (xCol[row_idx[i]] - center) / scale;
    for(int k = 0; k < m; k++) {
      ik = i * m + k;
      R[ik] -= shift[k] * xi;
      sumResid[k] += R[ik];
    }
  }
}

void bedpp_init(XPtr<BigMatrix> xMat, double *R, double *sumResid, vector<double> &XtY,
                double *lhs1, double *lhs2, double *lhs3, vector<double> &z, int xmax_idx,
                double lambda_max, int *row_idx, vector<int> &col_idx,
                NumericVector& center, NumericVector& scale, double alpha, 
                int n, int p, int m) {
  double xjtx; // xjtx_max
  // compute x_maxtY
  double *xTR = R_Calloc(m, double);
  int j, jj, k;
  crossprod_resid(xTR, xMat, R, sumResid, row_idx,
                  center[xmax_idx], scale[xmax_idx], n, m, xmax_idx);
#pragma omp parallel for private(j, jj, xjtx) schedule(static) 
  for(j = 0; j < p; j ++) {
    lhs3[j] = 0.0;
    jj = col_idx[j];
    xjtx = crossprod_bm_Xj_Xk(xMat, row_idx, center, scale, n, jj, xmax_idx);
    lhs1[j] = pow(z[j] * n, 2) * m;
    lhs2[j] = pow(xjtx * lambda_max * alpha, 2) * m;
    for(k = 0; k < m; k++) {
      lhs3[j] += XtY[j*m+k] * xTR[k];  
    }
    lhs3[j] *= xjtx / n;
  }
  R_Free(xTR);
}

// apply EDPP 
void edpp_screen(int *discard_beta, int n, int p, int m, double lambda_prev,
                 double rhs2, double *lhs1, double *lhs2, double*lhs3, double c,
                 double d, double *mp, double alpha, vector<int> &col_idx, bool EDPP) {
  int j;
  double rhs = n*lambda_prev*sqrt(m)*alpha - c*rhs2;
  if(rhs < 0) rhs = 0;
  for(j = 0; j < p; j ++) {
    if(EDPP) {
      if(pow(c*(1-d),2)*lhs1[j] + pow(1+c*d,2)*lhs2[j] + 2*c*(1+c*d)*(1-d)*lhs3[j] < pow(rhs,2)) discard_beta[j] = 1;
      else discard_beta[j] = 0;
    } else {
      if(pow(1+c,2)*lhs1[j] + pow(c,2)*lhs2[j] + 2*c*(1+c)*lhs3[j] < pow(rhs,2)) discard_beta[j] = 1;
      else discard_beta[j] = 0;
    }
  }
}

// Update EDPP rule
void edpp_update(XPtr<BigMatrix> xpMat, double *R, double *sumResid,
                 double *lhs2, double *lhs3, vector<double> &XtY,
                 int *row_idx, vector<int>& col_idx, NumericVector& center, 
                 NumericVector& scale, int n, int p, int m) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol;
  double *xTR;
  int i, k, j, jj;
  double sum2, sum3;
#pragma omp parallel for schedule(static) private(i, k, j, jj, xCol, xTR, sum2, sum3)
  for(j = 0; j < p; j++){
    jj = col_idx[j];
    xCol = xAcc[jj];
    xTR = R_Calloc(m, double);
    for(k = 0; k < m; k++) xTR[k] = 0;
    sum2 = 0.0;
    sum3 = 0.0;
    for(i = 0; i < n; i++) {
      for(k = 0; k < m; k++) {
        xTR[k] += xCol[row_idx[i]] * R[i*m+k];
      }
    }
    for(k = 0; k < m; k++) {
      sum2 += pow((xTR[k] - center[jj] * sumResid[k]) /  scale[jj], 2);
      sum3 += (xTR[k] - center[jj] * sumResid[k]) / scale[jj] * XtY[j*m+k];
    } 
    lhs2[j] = sum2;
    lhs3[j] = sum3;
    R_Free(xTR);
  }
}

// check KKT conditions over features in the strong set
int check_strong_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                     int *row_idx, vector<int> &col_idx, 
                     NumericVector &center, NumericVector &scale, double *a,
                     double lambda, double *sumResid, double alpha, 
                     double *R, double *mp, int n, int p, int m) {
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, *xTR, l1, l2, sum;
  int i, k, j, jj, violations = 0;
  
#pragma omp parallel for private(i, k, j, jj, xTR, l1, l2, sum) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e1[j] == 0 && e2[j] == 1) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      z[j] = 0.0;
      sum = 0.0;
      xTR = R_Calloc(m, double);
      for(k=0; k < m; k++) xTR[k] = 0.0;
      for (i=0; i < n; i++) {
        for (k=0; k < m; k++) {
          xTR[k] += xCol[row_idx[i]] * R[i*m+k];
        }
      }
      l1 = lambda * mp[jj] * alpha;
      l2 = lambda * mp[jj] * (1 - alpha);
      for(k=0; k < m; k++){
        xTR[k] = (xTR[k] - center[jj] * sumResid[k]) / scale[jj];
        z[j] += pow(xTR[k], 2);
        sum += pow(xTR[k] - n * sqrt(m) * l2 * a[j * m + k], 2);
      } 
      z[j] = sqrt(z[j]) / (n * sqrt(m));
      if(sum > m * pow(l1 * n, 2)) {
        e1[j] = 1;
        violations++;
      }
      R_Free(xTR);
    }
  }
  return violations;
}

// check KKT conditions over features in the rest set
int check_rest_set(int *e1, int *e2, vector<double> &z, XPtr<BigMatrix> xpMat, 
                   int *row_idx, vector<int> &col_idx, NumericVector &center,
                   NumericVector &scale, double *a, double lambda, double *sumResid,
                   double alpha, double *R, double *mp, int n, int p, int m) {
  
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, *xTR, l1, l2, sum;
  int i, k, j, jj, violations = 0;
#pragma omp parallel for private(i, k, j, xTR, l1, l2, sum) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e2[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      z[j] = 0.0;
      sum = 0.0;
      xTR = R_Calloc(m, double);
      for(k=0; k < m; k++) xTR[k] = 0.0;
      for (i=0; i < n; i++) {
        for (k=0; k < m; k++) {
          xTR[k] += xCol[row_idx[i]] * R[i*m+k];
        }
      }
      l1 = lambda * mp[jj] * alpha;
      l2 = lambda * mp[jj] * (1 - alpha);
      for(k=0; k < m; k++){
        xTR[k] = (xTR[k] - center[jj] * sumResid[k]) / scale[jj];
        z[j] += pow(xTR[k], 2);
        sum += pow(xTR[k] - n * sqrt(m) * l2 * a[j * m + k], 2);
      } 
      z[j] = sqrt(z[j]) / (n * sqrt(m));
      if(sum > m * pow(l1 * n, 2)) {
        e1[j] = e2[j] = 1;
        violations++;
      }
      R_Free(xTR);
    }
  }
  return violations;
}

// check KKT conditions over features in (the safe set - the strong set)
int check_rest_safe_set(int *e1, int *e2, int *discard_beta, vector<double> &z, 
                        XPtr<BigMatrix> xpMat, int *row_idx, vector<int> &col_idx,
                        NumericVector &center, NumericVector &scale, double *a,
                        double lambda, double *sumResid, double alpha, double *R,
                        double *mp, int n, int p, int m) {
  
  MatrixAccessor<double> xAcc(*xpMat);
  double *xCol, *xTR, l1, l2, sum;
  int i, k, j, jj, violations = 0;
#pragma omp parallel for private(i, k, j, xTR, l1, l2, sum) reduction(+:violations) schedule(static) 
  for (j = 0; j < p; j++) {
    if (e2[j] == 0 && discard_beta[j] == 0) {
      jj = col_idx[j];
      xCol = xAcc[jj];
      z[j] = 0.0;
      sum = 0.0;
      xTR = R_Calloc(m, double);
      for(k=0; k < m; k++) xTR[k] = 0.0;
      for (i=0; i < n; i++) {
        for (k=0; k < m; k++) {
          xTR[k] += xCol[row_idx[i]] * R[i*m+k];
        }
      }
      l1 = lambda * mp[jj] * alpha;
      l2 = lambda * mp[jj] * (1 - alpha);
      for(k=0; k < m; k++){
        xTR[k] = (xTR[k] - center[jj] * sumResid[k]) / scale[jj];
        z[j] += pow(xTR[k], 2);
        sum += pow(xTR[k] - n * sqrt(m) * l2 * a[j * m + k], 2);
      } 
      z[j] = sqrt(z[j]) / (n * sqrt(m));
      if(sum > m * pow(l1 * n, 2)) {
        e1[j] = e2[j] = 1;
        violations++;
      }
      R_Free(xTR);
    }
  }
  return violations;
}

// Coordinate descent for gaussian models with adaptive screening
RcppExport SEXP cdfit_mgaussian_ada(SEXP X_, SEXP y_, SEXP row_idx_, 
                                    SEXP lambda_, SEXP nlambda_, 
                                    SEXP lam_scale_, SEXP lambda_min_, 
                                    SEXP alpha_, SEXP user_, SEXP eps_, 
                                    SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                    SEXP ncore_, SEXP safe_thresh_, 
                                    SEXP update_thresh_, SEXP verbose_) {
  //ProfilerStart("mg_ada.out");
  XPtr<BigMatrix> xMat(X_);
  NumericMatrix Y(y_); // m responses * n samples matrix
  int m = Y.nrow(); // 
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *mp = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  double safe_thresh = REAL(safe_thresh_)[0]; // threshold for safe test
  double update_thresh = REAL(update_thresh_)[0];
  
  NumericVector lambda(L);
  NumericVector center(p);
  NumericVector scale(p);
  int p_keep = 0;
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
  
  // EDPP
  vector<double> XtY;
  
  // standardize: get center, scale; get p_keep_ptr, col_idx; get z, XtY, lambda_max, xmax_idx;
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, XtY,
                                 lambda_max_ptr, xmax_ptr, xMat, Y, row_idx,
                                 alpha, n, p, m);
  
  p = p_keep;   // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  // Objects to be returned to R
  arma::field<arma::sp_mat> beta(m);
  beta.for_each( [&](arma::sp_mat& beta_class) { beta_class.set_size(p, L); } );
  //arma::sp_mat beta = arma::sp_mat(m*p, L); // beta
  double *a = R_Calloc(m*p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L);
  
  double l1, l2, cutoff, cutoff0;
  double* shift = R_Calloc(m, double);
  double max_update, update, thresh; // for convergence check
  int i, j, k, jj, l, violations, lstart;
  
  int *e1 = R_Calloc(p, int); // ever active set
  int *e2 = R_Calloc(p, int); // strong set
  double *R = R_Calloc(m*n, double); // residual matrix
  double *sumResid = R_Calloc(m, double);
  loss[0] = 0.0;
  for(k = 0; k < m; k++) sumResid[k] = 0.0;
  for(i = 0; i < n; i++) {
    for(k = 0; k < m; k++){
      R[i*m+k] = Y.at(k, i);
      sumResid[k] += Y.at(k, i);
      loss[0] += pow(Y.at(k, i), 2);
    } 
  }
  double *xTR = R_Calloc(m, double);
  thresh = eps * loss[0] / n;
  
  // EDPP
  int *discard_beta = R_Calloc(p, int); // index set of discarded features;
  double c, d;
  double *lhs1 = R_Calloc(p, double); // 1st term on LHS, ||XTY||_2
  double *lhs2 = R_Calloc(p, double); // 2nd term on LHS, ||XTR||_2 if EDPP
  double *lhs3 = R_Calloc(p, double); // 3rd term on LHS, <XTY,XTR> if EDPP
  double rhs2 = 0.0; // second term on RHS
  double yhat;
  double Yhat_norm2 = 0.0;
  double YtYhat = 0.0;
  double Y_norm2 = loss[0]; // ||y||^2
  bool EDPP = false; // Whether using EDPP or BEDPP
  bool safe = true; // Whether using safe screen
  int gain = 0; // gain from updating EDPP
  IntegerVector n_safe_reject(L);
  
  
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
    lstart = 1;
    n_reject[0] = p;
    n_safe_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  }
  
  int l_prev = 0; // lambda index at previous update of EDPP
  
  // Path
  for (l = lstart; l < L; l++) {
    if(verbose) {
      // output time
      char buff[100];
      time_t now = time (0);
      strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
      Rprintf("Lambda %d. Now time: %s\n", l, buff);
    }
    c = (lambda[l_prev] - lambda[l]) / 2 / lambda[l];
    if (l != lstart) {
      // Check dfmax
      int nv = 0;
      for (j = 0; j < p; j++) {
        if (a[j*m] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        R_Free(a); R_Free(e1); R_Free(e2); R_Free(xTR); R_Free(shift); R_Free(R); R_Free(sumResid);
        R_Free(discard_beta); R_Free(lhs1); R_Free(lhs2); R_Free(lhs3);
        return List::create(beta, center, scale, lambda, loss, iter, 
                            n_reject, n_safe_reject, Rcpp::wrap(col_idx));
      }
      // strong set
      cutoff = 2 * lambda[l] - lambda[l_prev];
      cutoff0 = 2 * lambda[l] - lambda[l-1];
      for (j = 0; j < p; j++) {
        if (discard_beta[j] == 1) {
          if(z[j] > (cutoff * alpha * mp[col_idx[j]])) e2[j] = 1;
          else e2[j] = 0;
        } else {
          if(z[j] > (cutoff0 * alpha * mp[col_idx[j]])) e2[j] = 1;
          else e2[j] = 0;
        }
      }
      if(gain - n_safe_reject[l - 1] * (l - l_prev) > update_thresh * p && l != L - 1 && safe) { // Update EDPP if not discarding enough
        if(verbose) {
          // output time
          char buff[100];
          time_t now = time (0);
          strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
          Rprintf("Start updating EDPP rule at lambda %d. Now time: %s\n", l, buff);
        }
        EDPP = true;
        l_prev = l-1;
        c = (lambda[l_prev] - lambda[l]) / 2 / lambda[l];
        Yhat_norm2 = 0.0;
        YtYhat = 0.0;
        for(i = 0; i < n; i ++) {
          for(k = 0; k < m; k++) {
            yhat = Y.at(k, i) - R[i*m+k];
            Yhat_norm2 += pow(yhat, 2);
            YtYhat += Y.at(k, i) * yhat;
          }
        }
        d = YtYhat / Yhat_norm2;
        edpp_update(xMat, R, sumResid, lhs2, lhs3, XtY, row_idx, col_idx,
                    center, scale, n, p, m);
        rhs2 = sqrt(n * (Y_norm2 - YtYhat * YtYhat / Yhat_norm2));
        
        for(j = 0; j < p; j++) z[j] = sqrt(lhs2[j] / m) / n;
        if(verbose) {
          // output time
          char buff[100];
          time_t now = time (0);
          strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
          Rprintf("Done updating EDPP rule at lambda %d. Now time: %s\n", l, buff);
        }
        
        // Reapply EDPP
        edpp_screen(discard_beta, n, p, m, lambda[l_prev], rhs2, lhs1, lhs2, lhs3,
                    c, d, mp, alpha, col_idx, EDPP);
        n_safe_reject[l] = sum(discard_beta, p);
        if(n_safe_reject[l] <= safe_thresh * p) safe = false;
        gain = n_safe_reject[l];
      } else {
        // Apply EDPP to discard features
        if(EDPP) { // Apply EDPP check
          edpp_screen(discard_beta, n, p, m, lambda[l_prev], rhs2, lhs1, lhs2, lhs3,
                      c, d, mp, alpha, col_idx, EDPP);
        } else { // Apply BEDPP check
          c = (lambda_max - lambda[l]) / 2 / lambda[l];
          edpp_screen(discard_beta, n, p, m, lambda_max, rhs2, lhs1, lhs2, lhs3,
                      c, 0, mp, alpha, col_idx, EDPP);
        }
        n_safe_reject[l] = sum(discard_beta, p);
        gain += n_safe_reject[l];
      }
    } else {
      // strong set
      cutoff = 2*lambda[l] - lambda_max;
      for (j = 0; j < p; j++) {
        if (z[j] > (cutoff * alpha * mp[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
      // safe set with BEDPP
      if(verbose) {
        // output time
        char buff[100];
        time_t now = time (0);
        strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
        Rprintf("Start calculating BEDPP rule. Now time: %s\n", buff);
      }
      bedpp_init(xMat, R, sumResid, XtY, lhs1, lhs2, lhs3, z, xmax_idx, lambda_max, 
                 row_idx, col_idx, center, scale, alpha, n, p, m);
      rhs2 = sqrt(n * Y_norm2 - pow(n * lambda_max, 2) * m);
      if(verbose) {
        // output time
        char buff[100];
        time_t now = time (0);
        strftime (buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
        Rprintf("Done calculating BEDPP rule. Now time: %s\n", buff);
      }
      c = (lambda_max - lambda[l]) / 2 / lambda[l];
      edpp_screen(discard_beta, n, p, m, lambda_max, rhs2, lhs1, lhs2, lhs3,
                  c, 0, mp, alpha, col_idx, EDPP);
      n_safe_reject[l] = sum(discard_beta, p);
      gain = n_safe_reject[l];
      
    }
    n_reject[l] = p - sum(e2, p);
    
    while(iter[l] < max_iter) {
      while(iter[l] < max_iter){
        while(iter[l] < max_iter) {
          iter[l]++;
          
          //solve lasso over ever-active set
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              crossprod_resid(xTR, xMat, R, sumResid, row_idx, center[jj], scale[jj], n, m, jj);
              z[j] = 0;
              for(k = 0; k < m; k++) {
                xTR[k] = (xTR[k] / n + a[j * m + k]) / sqrt(m);
                z[j] += pow(xTR[k], 2);
              }
              z[j] = sqrt(z[j]);
              l1 = lambda[l] * mp[jj] * alpha;
              l2 = lambda[l] * mp[jj] * (1-alpha);
              lasso(beta, xTR, z[j], l1, l2, j, l, m);
              
              update = 0;
              for(k = 0; k < m; k++){
                shift[k] = beta.at(k).at(j, l) - a[j * m + k];
                update += pow(shift[k], 2);
              } 
              
              if (update !=0) {
                // compute objective update for checking convergence
                //update =  z[j] * shift - 0.5 * (1 + l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) - l1 * (fabs(beta(j, l)) -  fabs(a[j]));
                if (update > max_update) {
                  max_update = update;
                }
                update_resid(xMat, R, sumResid, shift, row_idx, center[jj], scale[jj], n, m, jj); // update R and sumResid
                for(k = 0; k < m; k++) a[j * m + k] = beta.at(k).at(j, l);
              }
            }
          }
          // Check for convergence
          if (max_update < thresh) break;
        }
        
        // Scan for violations in strong set
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx,
                                      center, scale, a, lambda[l], sumResid,
                                      alpha, R, mp, n, p, m); 
        if (violations==0) break;
      }
      
      // Scan for violations in rest safe set
      violations = check_rest_safe_set(e1, e2, discard_beta, z, xMat,
                                       row_idx, col_idx, center, scale, a,
                                       lambda[l], sumResid, alpha, R, mp, n, p, m);
      if (violations == 0) {
        loss[l] = 0;
        for(i = 0; i < n; i++) {
          for(k = 0; k < m; k++) loss[l] += pow(R[i*m+k], 2);  
        }
        break;
      }
    }
  }
  
  R_Free(a); R_Free(e1); R_Free(e2); R_Free(xTR); R_Free(shift); R_Free(R); R_Free(sumResid);
  R_Free(discard_beta); R_Free(lhs1); R_Free(lhs2); R_Free(lhs3);
  //ProfilerStop();
  return List::create(beta, center, scale, lambda, loss, iter,
                      n_reject, n_safe_reject, Rcpp::wrap(col_idx));
}

// Coordinate descent for gaussian models with ssr
RcppExport SEXP cdfit_mgaussian_ssr(SEXP X_, SEXP y_, SEXP row_idx_, 
                                    SEXP lambda_, SEXP nlambda_, 
                                    SEXP lam_scale_, SEXP lambda_min_, 
                                    SEXP alpha_, SEXP user_, SEXP eps_, 
                                    SEXP max_iter_, SEXP multiplier_, SEXP dfmax_, 
                                    SEXP ncore_, SEXP verbose_) {
  //ProfilerStart("mg.out");
  XPtr<BigMatrix> xMat(X_);
  NumericMatrix Y(y_); // m responses * n samples matrix
  int m = Y.nrow(); // 
  int *row_idx = INTEGER(row_idx_);
  double lambda_min = REAL(lambda_min_)[0];
  double alpha = REAL(alpha_)[0];
  int n = Rf_length(row_idx_); // number of observations used for fitting model
  
  int p = xMat->ncol();
  int L = INTEGER(nlambda_)[0];
  int lam_scale = INTEGER(lam_scale_)[0];
  int user = INTEGER(user_)[0];
  int verbose = INTEGER(verbose_)[0];
  double eps = REAL(eps_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *mp = REAL(multiplier_);
  int dfmax = INTEGER(dfmax_)[0];
  
  NumericVector lambda(L);
  NumericVector center(p);
  NumericVector scale(p);
  int p_keep = 0;
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
  standardize_and_get_residual(center, scale, p_keep_ptr, col_idx, z, 
                               lambda_max_ptr, xmax_ptr, xMat, Y, row_idx,
                               alpha, n, p, m);
  
  p = p_keep;   // set p = p_keep, only loop over columns whose scale > 1e-6
  
  if (verbose) {
    char buff1[100];
    time_t now1 = time (0);
    strftime (buff1, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now1));
    Rprintf("Preprocessing end: %s\n", buff1);
    Rprintf("\n-----------------------------------------------\n");
  }
  
  // Objects to be returned to R
  arma::field<arma::sp_mat> beta(m);
  beta.for_each( [&](arma::sp_mat& beta_class) { beta_class.set_size(p, L); } );
  //arma::sp_mat beta = arma::sp_mat(m*p, L); // beta
  double *a = R_Calloc(m*p, double); //Beta from previous iteration
  NumericVector loss(L);
  IntegerVector iter(L);
  IntegerVector n_reject(L);
  
  double l1, l2, cutoff;
  double* shift = R_Calloc(m, double);
  double max_update, update, thresh; // for convergence check
  int i, j, k, jj, l, violations, lstart;
  
  int *e1 = R_Calloc(p, int); // ever active set
  int *e2 = R_Calloc(p, int); // strong set
  double *R = R_Calloc(m*n, double); // residual matrix
  double *sumResid = R_Calloc(m, double);
  loss[0] = 0;
  for(k = 0; k < m; k++) sumResid[k] = 0;
  for(i = 0; i < n; i++) {
    for(k = 0; k < m; k++){
      R[i*m+k] = Y.at(k, i);
      sumResid[k] += Y.at(k, i);
      loss[0] += pow(Y.at(k, i), 2);
    } 
  }
  double *xTR = R_Calloc(m, double);
  thresh = eps * loss[0] / n;
  
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
    lstart = 1;
    n_reject[0] = p;
  } else {
    lstart = 0;
    lambda = Rcpp::as<NumericVector>(lambda_);
  }
  
  // Path
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
        if (a[j*m] != 0) nv++;
      }
      if (nv > dfmax) {
        for (int ll=l; ll<L; ll++) iter[ll] = NA_INTEGER;
        R_Free(a); R_Free(e1); R_Free(e2); R_Free(xTR); R_Free(shift); R_Free(R); R_Free(sumResid);
        return List::create(beta, center, scale, lambda, loss, iter, n_reject, Rcpp::wrap(col_idx));
      }
      // strong set
      cutoff = 2 * lambda[l] - lambda[l-1];
      for (j = 0; j < p; j++) {
        if (z[j] > (cutoff * alpha * mp[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      } 
    } else {
      // strong set
      cutoff = 2*lambda[l] - lambda_max;
      for (j = 0; j < p; j++) {
        if (z[j] > (cutoff * alpha * mp[col_idx[j]])) {
          e2[j] = 1;
        } else {
          e2[j] = 0;
        }
      }
    }
    n_reject[l] = p - sum(e2, p);
    
    while(iter[l] < max_iter) {
      while(iter[l] < max_iter){
        while(iter[l] < max_iter) {
          iter[l]++;
          
          //solve lasso over ever-active set
          max_update = 0.0;
          for (j = 0; j < p; j++) {
            if (e1[j]) {
              jj = col_idx[j];
              crossprod_resid(xTR, xMat, R, sumResid, row_idx, center[jj], scale[jj], n, m, jj);
              z[j] = 0;
              for(k = 0; k < m; k++) {
                xTR[k] = (xTR[k] / n + a[j * m + k]) / sqrt(m);
                z[j] += pow(xTR[k], 2);
              }
              z[j] = sqrt(z[j]);
              l1 = lambda[l] * mp[jj] * alpha;
              l2 = lambda[l] * mp[jj] * (1-alpha);
              lasso(beta, xTR, z[j], l1, l2, j, l, m);
              
              update = 0;
              for(k = 0; k < m; k++){
                shift[k] = beta.at(k).at(j, l) - a[j * m + k];
                update += pow(shift[k], 2);
              } 
              
              if (update !=0) {
                // compute objective update for checking convergence
                //update =  z[j] * shift - 0.5 * (1 + l2) * (pow(beta(j, l), 2) - pow(a[j], 2)) - l1 * (fabs(beta(j, l)) -  fabs(a[j]));
                if (update > max_update) {
                  max_update = update;
                }
                update_resid(xMat, R, sumResid, shift, row_idx, center[jj], scale[jj], n, m, jj); // update R and sumResid
                for(k = 0; k < m; k++) a[j * m + k] = beta.at(k).at(j, l);
              }
            }
          }
          // Check for convergence
          if (max_update < thresh) break;
        }
        
        // Scan for violations in strong set
        violations = check_strong_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, R, mp, n, p, m); 
        if (violations==0) break;
      }
      
      // Scan for violations in rest set
      violations = check_rest_set(e1, e2, z, xMat, row_idx, col_idx, center, scale, a, lambda[l], sumResid, alpha, R, mp, n, p, m);
      if (violations == 0) {
        loss[l] = 0;
        for(i = 0; i < n; i++) {
          for(k = 0; k < m; k++) loss[l] += pow(R[i*m+k], 2);  
        }
        break;
      }
    }
  }
  
  R_Free(a); R_Free(e1); R_Free(e2); R_Free(xTR); R_Free(shift); R_Free(R); R_Free(sumResid);
  //ProfilerStop();
  return List::create(beta, center, scale, lambda, loss, iter, n_reject, Rcpp::wrap(col_idx));
}
