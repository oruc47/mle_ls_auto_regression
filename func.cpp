/*
Code written by Ikbal Oruc
moruc@ku.edu.tr
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <limits>
#include <random>
#include <fstream>
#include "func.h"

/*
Transformation function 
specified by the assignment 
requirements
*/
vector<double> transform(const vector<double>& p){
    vector<double> p_ = p;
    p_[2] = (exp(p[2]) - 1) / (exp(p[2]) + 1);
    p_[3] = exp(p[3]);
    return p_;
}


/*
Modified transformation function for AR(2)
*/

vector<double> transform_ar(const vector<double>& p_){

    vector<double> p = p_;

    p[2] = (exp(p[2]) - 1) / (exp(p[2]) + 1);
    p[3] = (exp(p[3]) - 1) / (exp(p[3]) + 1);
    return p;
}

/*
Basic Log likelihood function
*/

double log_likelihood(const vector<double>& p, const vector<double>& y, const vector<int>& t) {
    double alpha = p[0];
    double beta = p[1];
    double phi = (exp(p[2]) - 1) / (exp(p[2]) + 1);
    double sigma = exp(p[3]);

    double result = 0;
    int T = y.size();

    double u_0 = (1 - phi * phi) * pow(y[0] - alpha - beta * t[0], 2);
    double u_sum = 0;
    for (int i = 1; i < T; i++) {
        double yt = y[i];
        double yt_1 = y[i - 1];
        double u_2 = pow(yt - alpha - beta * t[i] - phi * yt_1 + phi * alpha + phi * beta * t[i - 1], 2);
        u_sum += u_2;
    }

    result = (-log(2 * M_PI) * T / 2) - (T * log(sigma) / 2) + (log(sqrt(1 - phi * phi))) - ((1 / (2 * sigma)) * (u_0 + u_sum));

    return -result;
}

/*
Gradient function using finite differences. 
parameters are adjusted by epsiolon amount 
each time
*/


vector<double> gen_gradient(function<double(vector<double>)> function, const vector<double>& p, double epsilon = 1e-6) {
    vector<double> gradient(p.size());
    double ll = function(p);

    for (int i = 0; i < p.size(); i++) {
        vector<double> pe = p;
        pe[i] += epsilon;
        double llp = function(pe);
        pe[i] -= 2 * epsilon;
        double llm = function(pe);
        gradient[i] = (llp - llm) / (2 * epsilon);
    }

    return gradient;
}


/*
Mechincal invert function 
to invert matricies
*/

vector<vector<double>> invert(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<vector<double>> inv(n, vector<double>(n, 0));
    vector<vector<double>> aug = matrix;

    for (int i = 0; i < n; i++) {
        aug[i].resize(2 * n, 0);
        aug[i][n + i] = 1;
    }

    for (int i = 0; i < n; i++) {
        double d = aug[i][i];
        if (d == 0) {
            continue; 
        }
        for (int j = 0; j < 2 * n; j++) {
            aug[i][j] /= d;
        }
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double f = aug[k][i];
                for (int l = 0; l < 2 * n; l++) {
                    aug[k][l] -= f * aug[i][l];
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv[i][j] = aug[i][j + n];
        }
    }

    return inv;

}


/* 
Simple implementation of gradient descent
done simply by taking the the vec norm 
of the results and lopps breaks once
convergence is reached
checks included to make sure 
parameters dont overshoot

*/
vector<double> gradient_descent(function<double(vector<double>)> function, vector<double>& p, double learning_rate, int iterations) {
    double limit_zero = 1e-6;
    for (int i = 0; i < iterations; i++) {
        vector<double> gradient = gen_gradient(function, p);

        double vec_norm = 0;
        for (int j = 0; j < p.size(); j++) {
            vec_norm += pow(gradient[j], 2);
        }
        vec_norm = sqrt(vec_norm);

        if (vec_norm > 1.0) {
            for (int j = 0; j < p.size(); j++) {
                gradient[j] = gradient[j] * (1.0 / vec_norm);
            }
        }

        for (int j = 0; j < p.size(); j++) {
            p[j] -= gradient[j] * learning_rate;
        }

        if (p[2] < -20) p[2] = -10;  
        if (p[2] > 20) p[2] = 10; 

        if (vec_norm < limit_zero) {
            break;
        }
    }

    return p;
}

/* 
solves the first part of the assginment

*/

void solve_mle(){
    vector<double> y = {1, 2, 4, 6, 4, 12, 11, 13, 11, 14, 16, 17};
    vector<int> T(y.size());
    iota(T.begin(), T.end(), 1);

    vector<double> p = {0.0, 0.0, 0.0, 0.0};

    double learning_rate = 0.0001;
    int max_iterations = 100000;

    vector<double> optimized_p = gradient_descent(
        [&y, &T](const vector<double>& parameters) { return log_likelihood(parameters, y, T); },
        p, learning_rate, max_iterations
    );



    vector<double> final = transform(optimized_p);

    auto function = [&y, &T](const vector<double>& parameters) { return log_likelihood(parameters, y, T); };   
    vector<vector<double>> hessian = gen_hessian(function, optimized_p, 1e-5);

    vector<vector<double>> hessianInvert = invert(hessian);


    vector<double> std(final.size());
    vector<double> t(final.size());
    for (int i = 0; i < final.size(); i++) {
        std[i] = sqrt(hessianInvert[i][i]);
        t[i] = final[i] / std[i];
    }

    cout << "Alpha: " << final[0] << " Standard Error: " << std[0] << " T-Statistic: " << t[0] << endl;
    cout << "Beta: " << final[1] << " Standard Error: " << std[1] << " T-Statistic: " << t[1] << endl;
    cout << "Phi: " << final[2] << " Standard Error: " << std[2] << " T-Statistic: " << t[2] << endl;
    cout << "Sigma: " << final[3] << " Standard Error: " << std[3] << " T-Statistic: " << t[3] << endl;
}

/* 

generates hessian, again simplified 
calculations

*/

vector<vector<double>> gen_hessian(function<double(vector<double>)> function, const vector<double>& p, double epsilon) {
    vector<vector<double>> hessian(p.size(), vector<double>(p.size(), 0));
    for (int i = 0; i < p.size(); i++) {
        for (int j = 0; j < p.size(); j++) {
            vector<double> pe = p;
            pe[i] += epsilon;
            pe[j] += epsilon;
            double pp = function(pe);
            pe[j] -= 2 * epsilon;
            double pm = function(pe);
            pe[i] -= 2 * epsilon;
            pe[j] += 2 * epsilon;
            double mp = function(pe);
            pe[j] -= 2 * epsilon;
            double mm = function(pe);
            hessian[i][j] = (pp - pm - mp + mm) / (4 * pow(epsilon, 2));
        }
    }

    for (int i = 0; i < p.size(); i++) {
        hessian[i][i] += epsilon;
    }

    return hessian;
}

//simple ssr function for ols optimization

template <typename T>
double sum_squared_resid(const vector<double>&p, const vector<double>& y, const vector<T>& t){
    
    double ssr = 0.0;

    for(int i = 0; i < y.size(); i++){
        double r = y[i] - (p[0] + p[1] * t[i]);
        ssr += r * r  ;
    }
    
    return ssr; 
}


//gradient function modified for ar2

vector<double> gen_gradient_ar(function<double(vector<double>)> function, const vector<double>& p, double epsilon = 1e-6){
    vector<double> gradient(p.size());
    double ll = function(p);

    for (int i = 0; i < p.size(); i++){
        vector<double> pe = p;
        pe[i] += epsilon;
        double llp = function(pe);

        gradient[i] = (llp - ll) / epsilon;
    }

    return gradient;

}

//grad. descent modified for ar2

template <typename T>
vector<double> gradient_descent_ar(function<double(vector<double>)> function, vector<double>& p, const vector<double>& y, const vector<T>& t, double learning_rate, int iterations){
    double limit_zero = 1e-6;
    for(int i = 0; i < iterations; i++){
        vector<double> gradient = gen_gradient_ar(function, p);

        double vec_norm = 0;
        for(int j = 0; j < p.size(); j++){
            p[j] -= gradient[j] * learning_rate;
            vec_norm += pow(gradient[j], 2);
        }
        vec_norm = sqrt(vec_norm);

        if(vec_norm < limit_zero){
            break;
        }

    }

    return p;

}

//simple residual function to find error

vector<double> residuals(const vector<double>&p, const vector<double>& y, const vector<int>& t){
    vector<double> r(y.size());
    for(int i = 0; i < y.size(); i++){
        r[i] = y[i] - (p[0] + (p[1] * t[i]));
    }
    return r; 
}

//ssr for ar1, only uses lags for t - 1

double sum_squared_ar_1(const vector<double>&p, const vector<double>&resid, const vector<double>&resid_lag){
    double ssr = 0.0;

    for(int i = 0; i < resid_lag.size(); i++){
        double r = resid_lag[i] - p[0] * resid[i];
        ssr += r * r;
    }
    return ssr; 

}


//ssr for ar1,  uses lags for t - 1 and t - 2

double sum_squared_ar_2(const vector<double>&p, const vector<double>&resid, const vector<double>&resid_lag1, const vector<double>&resid_lag2){
    double ssr = 0.0;

    for(int i = 0; i < resid.size(); i++){
        double r = resid[i] - p[0] * resid_lag1[i] - p[1] *resid_lag2[i];
        ssr += r * r;
    }
    return ssr; 

}


//residuals for t - 1


vector<double> residuals_ar_1(const vector<double>&p, const vector<double>& resid, const vector<double>& resid_lag){

    vector<double> r(resid_lag.size());
    for(int i = 0; i < resid_lag.size(); i++){
        r[i] = resid[i] - (p[0] * resid_lag[i]);
    }
    return r; 
}

//generic mean calc function

template <typename T>
T mean(const vector<T>& v){
    double sum = 0.0;
    for(auto& i: v){
        sum += i;
    }
    return sum / v.size();
}

//generic demean function

template <typename T>
vector<T> demean(const vector<T>&v, const T m){
    vector<double> d(v.size());
    for(int i = 0; i < v.size(); i++){
        d[i] = v[i] - m;
    }
    return d;
}

//basic identity matrix generator

vector<vector<double>> gen_identity(const int& n){
    vector<vector<double>> i(n, vector<double>(n, 0.0));
    for(int j = 0; j < n; j++){
        i[j][j] = 1.0;
    }
    return i;
}

//since we have psi now we can compute gls using this func.

template <typename T>
vector<double> transform_gls(vector<vector<double>> psi, const vector<T>& data){
    int n = data.size();
    vector<double> result(n);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            result[i] += psi[i][j] * data[j];
        }
    }
    return result;

}

//function wrapper, just helps code be more clean

function<double(vector<double>)> wrap(const vector<double>& a,const vector<double>& b,const vector<double>& c) {
    return [&a, &b, &c](const vector<double>& parameters) {
        return sum_squared_ar_2(parameters, a, b, c);
    };
}

//ssr for ar2

template <typename T>
double sum_squared_ar_2_nls(const vector<double>&p_, const vector<double>&y, const vector<T>&t){



    vector<double> x(y.size() -2, 0.0);

    vector<double> p = transform_ar(p_);

    for(int i = 2; i < y.size(); i++){
        double a = p[0] * (1 - p[2] - p[3]) + p[1] * (p[2] + p[3]);
        double b = p[1] * (1 - p[2] - p[3]) * t[i];
        double c = p[2] * y[i - 1];
        double d = p[3] * y[i - 2];
        x[i - 2] = y[i] - (a + b + c + d);
    }

    

    double ssr = inner_product(x.begin(), x.end(), x.begin(), 0.0);
    return ssr; 

}

//computes ols, also computes gls in the process for ar2


vector<double> compute_ols(const vector<double>&optimized_p, const vector<double>& y, const vector<int>& t){
    vector<double> residual = residuals(optimized_p, y, t);

    vector<double> resid(residual.begin() + 1, residual.end());

    vector<double> resid_lag (residual.begin(), residual.end() - 1);

    vector<double> p = {0};

    double learning_rate = 0.0001;
    int max_iterations = 100000;

    vector<double> phi = gradient_descent_ar(
        [&resid, &resid_lag](const vector<double>& parameters) { return sum_squared_ar_1(parameters, resid, resid_lag); },
        p, resid, resid_lag, learning_rate, max_iterations
    );

    vector<double> error = residuals_ar_1(phi, resid, resid_lag);

    double errorMean = mean(error);

    vector<double> demeaned = demean(error, errorMean);

    double mean_sq = 0.0;

    for(auto& d: demeaned){
        mean_sq += pow(d,2);

    }

    double sigma = mean_sq/(t.size() - 2);

    vector<vector<double>> identity = gen_identity(t.size());
    
    identity[0][0] = sqrt((1 - pow(phi[0], 2)));

    for(int i = 1; i < t.size() - 1; i++){
        identity[i][i+1] = -p[0];
    }


    vector<double> y_gls = transform_gls(identity, y);
    vector<double> t_gls = transform_gls(identity, t);

    
    vector<double> p_gls = {0,0};

    vector<double> optimized_gls = gradient_descent_ar(
        [&y_gls, &t_gls](const vector<double>& parameters) { return sum_squared_resid(parameters, y_gls, t_gls); },
        p_gls, y_gls, t_gls, learning_rate, max_iterations
    );

    optimized_gls.push_back(phi[0]);

    optimized_gls.push_back(sigma);

    return optimized_gls;


}

//loglikelihood adjusted for ar2

double log_ll_ar(const vector<double>& p, const vector<double>& y, const vector<int>& t) {
    double alpha = p[0];
    double beta = p[1];
    double phi1 = p[2];
    double phi2 = p[3];
    double sigma = p[4];


    phi1 = (exp(phi1) - 1) / (exp(phi1) + 1);
    phi2 = (exp(phi2) - 1) / (exp(phi2) + 1);
    sigma = exp(sigma);
    double d = (1 + phi2) * (1 - phi1 - phi2) * (1 - phi2 + phi1);

    int T = y.size();


    double d1 = -log(sigma)
                + 0.5 * log(d / (1 - phi2))
                - (d / (2 * sigma * (1 - phi2))) * pow(y[0] - alpha - beta * t[0], 2)
                + 0.5 * log(1 - phi2 * phi2)
                - ((1 - phi2 * phi2) / (2 * sigma)) * pow(y[1] - alpha - beta * t[1] - (phi1 / (1 - phi2)) * (y[1] - alpha - beta * t[1]), 2);


    double sum_d2 = 0;
    for (int i = 2; i < T; i++) {
        double d2 = (y[i] - phi1 * y[i - 1] - phi2 * y[i - 2]
                     - beta * (t[i] - phi1 * t[i - 1] - phi2 * t[i - 2])
                     + alpha * (phi1 + phi2 - 1)) / (2 * sigma);
        sum_d2 += pow(d2, 2);
    }

    double result = d1 - sum_d2 - (T - 3) * 0.5 * log(sigma);

    return -result;
}

//mle calculatoins adjusted for ar2

vector<double> compute_ar_2_mle(const vector<double>& ols, double sigma, const vector<double>& y, const vector<int>& t){
    double learning_rate = 0.0001;
    int max_iterations = 1000000;
    vector<double> p = {ols[0], ols[1], 0.0, 0.0, sigma};
    vector<double> optimized_p = gradient_descent_ar(
        [&y, &t](const vector<double>& parameters) { return log_ll_ar(parameters, y, t); },
        p, y, t, learning_rate, max_iterations
    );

   

    optimized_p[2] = (exp(optimized_p[2]) - 1) / (exp(optimized_p[2]) + 1);
    optimized_p[3] = (exp(optimized_p[3]) - 1) / (exp(optimized_p[3]) + 1);

    return optimized_p;

}

//compute ols(nls) for ar2

vector<double> compute_ar_2(const vector<double>&optimized_p, const vector<double>& y, const vector<int>& t){
    vector<double> residual = residuals(optimized_p, y, t);

    vector<double> resid(residual.begin() + 2, residual.end());

    vector<double> resid_lag1(residual.begin() + 1, residual.end() - 1);

    vector<double> resid_lag2 (residual.begin(), residual.end() - 2);

    vector<double> p = {0, 0};

    double learning_rate = 0.0001;
    int max_iterations = 100000;

    auto ar_2 = wrap(resid, resid_lag1, resid_lag2);

    vector<double> phi = gradient_descent_ar(ar_2, p, resid, resid_lag1, learning_rate, max_iterations);

    double ar_2_resid = sum_squared_ar_2(phi, resid, resid_lag1, resid_lag2);

    double sigma = ar_2_resid / (resid.size() - 2);

    cout << "OLS ar_2: " << endl;
    cout << "Alpha :" << optimized_p[0] << endl;
    cout << "Beta :" << optimized_p[1] << endl;
    cout << "Phi 1:" << phi[0] << endl;
    cout << "Phi 2:" << phi[1] << endl;
    cout << "Sigma :" << sigma << endl;

    vector<double> pNLS = {0.0, 0.0, 0.0, 0.0};

    vector<double> p_nls_o = gradient_descent_ar(
        [&y, &t](const vector<double>& parameters) { return sum_squared_ar_2_nls(parameters, y, t); },
        pNLS, y, t, learning_rate, max_iterations
    );

    p_nls_o[2] = (exp(p_nls_o[2]) - 1) / (exp(p_nls_o[2]) + 1);
    p_nls_o[3] = (exp(p_nls_o[3]) - 1) / (exp(p_nls_o[3]) + 1);



    double resid_nls_2 = sum_squared_ar_2_nls(p_nls_o, y, t);


    double sigma_nls = resid_nls_2 / (resid.size() - 2);
    
    cout << "NLS ar_2: " << endl;
    cout << "Alpha :" << p_nls_o[0] << endl;
    cout << "Beta :" << p_nls_o[1] << endl;
    cout << "Phi 1:" << p_nls_o[2] << endl;
    cout << "Phi 2:" << p_nls_o[3] << endl;
    cout << "Sigma :" << sigma_nls << endl;


    vector<double> ar_2_mle = compute_ar_2_mle(optimized_p, sigma, y, t);

    cout << "MLE ar_2: " << endl;
    cout << "Alpha :" << ar_2_mle[0] << endl;
    cout << "Beta :" << ar_2_mle[1] << endl;
    cout << "Phi 1:" << ar_2_mle[2] << endl;
    cout << "Phi 2:" << ar_2_mle[3] << endl;
    cout << "Sigma :" << ar_2_mle[4] << endl;



    return phi;

}

//solves second part of the first section
void solve_ar(){
    vector<double> y = {1, 2, 4, 6, 4, 12, 11, 13, 11, 14, 16, 17};
    vector<int> T(y.size());
    iota(T.begin(), T.end(), 1);

    vector<double> p = {0.0, 0.0};

    double learning_rate = 0.0001;
    int max_iterations = 100000;

    vector<double> optimizedP = gradient_descent_ar(
        [&y, &T](const vector<double>& parameters) { return sum_squared_resid(parameters, y, T); },
        p, y, T, learning_rate, max_iterations
    );

   


    vector<double> ols = compute_ols(optimizedP, y, T);

    cout << "NLS AR1: " << endl;
    cout << "Alpha :" << ols[0] << endl;
    cout << "Beta :" << ols[1] << endl;
    cout << "Phi :" << ols[2] << endl;
    cout << "Sigma :" << ols[3] << endl;

    vector<double> olsAR2 = compute_ar_2(optimizedP, y, T);
}


//not important, just to save output to plot later
void to_csv(const vector<double>& series, const string& name){
    ofstream file(name);
    for(int i = 0; i < series.size(); i++){
        file << i << "," << fixed << setprecision(6) << series[i] << endl;
    }
    file.close();
}

//generate series using standard libraries for randomness
vector<double> gen_ar1_series(double phi, int t, double sigma, double y_0 = 0){
    vector<double> series(t);

    random_device random;

    mt19937 gen(random());

    normal_distribution<> normal(0, sigma);

    series[0] = y_0;

    for(int i = 1; i < t; i++){
        series[i] = phi * series[i - 1] + normal(gen);
    }

    return series;
}


//ols compuation adjusted for mc (basically doesnt include gls transform)

vector<double> compute_ols_mc(const vector<double>&optimized_p, const vector<double>& y, const vector<int>& t){
    vector<double> residual = residuals(optimized_p, y, t);

    vector<double> resid(residual.begin() + 1, residual.end());

    vector<double> resid_lag (residual.begin(), residual.end() - 1);

    vector<double> p = {0};

    double learning_rate = 0.0001;
    int max_iterations = 100000;

    vector<double> phi = gradient_descent_ar(
        [&resid, &resid_lag](const vector<double>& parameters) { return sum_squared_ar_1(parameters, resid, resid_lag); },
        p, resid, resid_lag, learning_rate, max_iterations
    );

    vector<double> results(2);
    results[0] = phi[0];

    vector<double> error = residuals_ar_1(phi, resid, resid_lag);

    double errorMean = mean(error);

    vector<double> demeaned = demean(error, errorMean);

    double mean_sq = 0.0;

    for(auto& d: demeaned){
        mean_sq += pow(d,2);

    }

    double sigma = mean_sq/(t.size() - 2);

    results[1] = sigma;

    return results;

}

//monte carlo simulation function

void monte_carlo(){
    int dgp = 1;
    int t = 100;    
    double sigma = 3.0;
    vector<double> phi_values = {0.5, 0.9, 0.99, 1.0};

    for(double phi: phi_values){
        vector<double> series1 =  gen_ar1_series(phi, t, sigma, 0.0);
        vector<double> series2 =  gen_ar1_series(phi, t, sigma, 0.0);
        to_csv(series1, "series1_phi_" + to_string(phi) + ".csv");
        to_csv(series2, "series1_phi_" + to_string(phi) + ".csv");
        vector<double> y = series1;
        vector<int> T(y.size());
        iota(T.begin(), T.end(), 1);

        vector<double> p = {0.0, 0.0, 0.0, 0.0};

        double learning_rate = 0.0001;
        int max_iterations = 100000;

        vector<double> optimized_p = gradient_descent(
            [&y, &T](const vector<double>& parameters) { return log_likelihood(parameters, y, T); },
            p, learning_rate, max_iterations
        );



        vector<double> final = transform(optimized_p);

        auto function = [&y, &T](const vector<double>& parameters) { return log_likelihood(parameters, y, T); };   
        vector<vector<double>> hessian = gen_hessian(function, optimized_p, 1e-6);

        vector<vector<double>> hessian_invert = invert(hessian);


        vector<double> std(final.size());
        vector<double> t(final.size());
        for (int i = 0; i < final.size(); i++) {
            std[i] = sqrt(hessian_invert[i][i]);
            t[i] = final[i] / std[i];
        }

        

        cout << "DGP " << dgp << endl;

        cout << "Alpha: " << final[0] << " Standard Error: " << std[0] << endl;
        cout << "Beta: " << final[1] << " Standard Error: " << std[1] << endl;
        cout << "Phi: " << final[2] << " Standard Error: " << std[2] << endl;

        vector<double> results = compute_ols_mc(optimized_p, y, T);

        cout << "OLS " << results[0] << endl;
        cout << "OLS Sigma " << results[1] << endl;

        dgp += 2;

        }
}

//summary stats for replication
void compute_summary_stats(const vector<double>& data, double& mean, double& stddev, double& mean_bias, double& rmse, double& q5, double& median, double& q95, double true_value) {
    mean = accumulate(data.begin(), data.end(), 0.0) / data.size();
    stddev = sqrt(inner_product(data.begin(), data.end(), data.begin(), 0.0) / data.size() - mean * mean);
    mean_bias = mean - true_value;
    rmse = sqrt(inner_product(data.begin(), data.end(), data.begin(), 0.0) / data.size() - true_value * true_value);
    
    vector<double> sorted_data = data;
    sort(sorted_data.begin(), sorted_data.end());
    q5 = sorted_data[data.size() * 0.05];
    median = sorted_data[data.size() * 0.5];
    q95 = sorted_data[data.size() * 0.95];
}

//simulation function, can determine sample size and replications

void simulation(int sample, int replications){
    int T = 200;
    int r = 500;
    double true_alpha = 0;
    double true_beta = 0;
    double true_phi = 0;
    double sigma = 3.0;
    vector<double> phi_values = {0.5, 0.9, 0.99, 1.0};

    vector<vector<double>> alpha_ml(r, vector<double>(4));
    vector<vector<double>> beta_ml(r, vector<double>(4));
    vector<vector<double>> phi_ml(r, vector<double>(4));
    vector<vector<double>> phi_ols(r, vector<double>(4));

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < phi_values.size(); j++) {
            double phi = phi_values[j];
            vector<double> series = gen_ar1_series(phi, T, sigma, 0.0);

            vector<int> t(T);
            iota(t.begin(), t.end(), 1);

            vector<double> p = {0.0, 0.0, 0.0, 0.0};
            double learning_rate = 0.00001;  // Smaller learning rate
            int max_iterations = 100;

            vector<double> optimized_p = gradient_descent(
                [&series, &t](const vector<double>& parameters) { return log_likelihood(parameters, series, t); },
                p, learning_rate, max_iterations
            );

            vector<double> final = transform(optimized_p);

            alpha_ml[i][j] = final[0];
            beta_ml[i][j] = final[1];
            phi_ml[i][j] = final[2];

            vector<double> ols_results = compute_ols_mc(optimized_p, series, t);
            phi_ols[i][j] = ols_results[0];
        }
    }

    for (int j = 0; j < phi_values.size(); j++) {
        double mean, stddev, mean_bias, rmse, q5, median, q95;

        compute_summary_stats(alpha_ml[j], mean, stddev, mean_bias, rmse, q5, median, q95, true_alpha);
        cout << "Alpha (phi = " << phi_values[j] << "): " << "Mean = " << mean << ", StdDev = " << stddev << ", Mean Bias = " << mean_bias << ", RMSE = " << rmse << ", 5% Quantile = " << q5 << ", Median = " << median << ", 95% Quantile = " << q95 << endl;

        compute_summary_stats(beta_ml[j], mean, stddev, mean_bias, rmse, q5, median, q95, true_beta);
        cout << "Beta (phi = " << phi_values[j] << "): " << "Mean = " << mean << ", StdDev = " << stddev << ", Mean Bias = " << mean_bias << ", RMSE = " << rmse << ", 5% Quantile = " << q5 << ", Median = " << median << ", 95% Quantile = " << q95 << endl;

        compute_summary_stats(phi_ml[j], mean, stddev, mean_bias, rmse, q5, median, q95, true_phi);
        cout << "Phi_ML (phi = " << phi_values[j] << "): " << "Mean = " << mean << ", StdDev = " << stddev << ", Mean Bias = " << mean_bias << ", RMSE = " << rmse << ", 5% Quantile = " << q5 << ", Median = " << median << ", 95% Quantile = " << q95 << endl;

        compute_summary_stats(phi_ols[j], mean, stddev, mean_bias, rmse, q5, median, q95, true_phi);
        cout << "Phi_OLS (phi = " << phi_values[j] << "): " << "Mean = " << mean << ", StdDev = " << stddev << ", Mean Bias = " << mean_bias << ", RMSE = " << rmse << ", 5% Quantile = " << q5 << ", Median = " << median << ", 95% Quantile = " << q95 << endl;
    }
}
