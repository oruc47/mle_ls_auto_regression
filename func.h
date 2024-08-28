/*
Code written by Ikbal Oruc
moruc@ku.edu.tr
*/

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <vector> 
#include <functional>

using namespace std;

vector<double> transform(const vector<double>& p);

vector<double> transform_ar(const vector<double>& p_);

double log_likelihood(const vector<double>& p, const vector<double>& y, const vector<int>& t);

vector<double> gen_gradient(function<double(vector<double>)> function, const vector<double>& p, double epsilon);

vector<vector<double>> invert(const vector<vector<double>>& matrix);

vector<double> gradient_descent(function<double(vector<double>)> function, vector<double>& p, double learningRate, int iterations);

vector<vector<double>> gen_hessian(function<double(vector<double>)> function, const vector<double>& p, double epsilon);


void solve_mle();


template <typename T>
double sum_squared_resid(const vector<double>& p, const vector<double>& y, const vector<T>& t);

vector<double> gen_gradient_ar(function<double(vector<double>)> function, const vector<double>& p, double epsilon);

template <typename T>
vector<double> gradient_descent_ar(function<double(vector<double>)> function, vector<double>& p, const vector<double>& y, const vector<T>& t, double learning_rate, int iterations);

vector<double> residuals(const vector<double>& p, const vector<double>& y, const vector<int>& t);

double sum_squared_ar_1(const vector<double>& p, const vector<double>& resid, const vector<double>& resid_lag);

double sum_squared_ar_2(const vector<double>& p, const vector<double>& resid, const vector<double>& resid_lag1, const vector<double>& resid_lag2);

vector<double> residuals_ar_1(const vector<double>& p, const vector<double>& resid, const vector<double>& resid_lag);

template <typename T>
T mean(const vector<T>& v);

template <typename T>
vector<T> demean(const vector<T>& v, const T m);

vector<vector<double>> gen_identity(const int& n);

template <typename T>
vector<double> transform_gls(vector<vector<double>> psi, const vector<T>& data);

function<double(vector<double>)> wrap(const vector<double>& a, const vector<double>& b, const vector<double>& c);

template <typename T>
double sum_squared_ar_2_nls(const vector<double>& p_, const vector<double>& y, const vector<T>& t);

vector<double> compute_ols(const vector<double>& optimized_p, const vector<double>& y, const vector<int>& t);

vector<double> compute_ols_mc(const vector<double>& optimized_p, const vector<double>& y, const vector<int>& t);

double log_ll_ar(const vector<double>& p, const vector<double>& y, const vector<int>& t);

vector<double> compute_ar_2_mle(const vector<double>& ols, double sigma, const vector<double>& y, const vector<int>& t);

vector<double> compute_ar_2(const vector<double>& optimized_p, const vector<double>& y, const vector<int>& t);


void solve_ar();


void to_csv(const vector<double>& series, const string& name);

vector<double> gen_ar1_series(double phi, int t, double sigma, double y_0);

void monte_carlo();

void compute_summary_stats(const vector<double>& data, double& mean, double& stddev, double& mean_bias, double& rmse, double& q5, double& median, double& q95, double true_value);

void simulation(int sample, int replications);

#endif