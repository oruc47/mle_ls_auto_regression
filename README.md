# mle_ls_auto_regression
Analysis of the finite sample behavior of estimators and tests of regression models with AR disturbances, where Least Squares (LS) and Maximum Likelihood (ML) estimation and inference are examined. Empirical behavior of the statistics in finite samples are analyzed using Monte Carlo simulation under standard and nonstandard conditions.

# Major Assignment: Autoregression Disturbances for MLE and LS

**Author: Muhammed İkbal Oruç**

## Introduction

This report presents the results and analysis of MLE and LS estimates of a simple autoregressive model. C++ was used to write and optimize both the log-likelihood and the LS functions. I first compute estimates for a simple version of the model and then later on perform Monte Carlo simulations to observe more complicated versions of the model and see if the estimates converge over time. I also hard-coded all the optimization functions to not use any packages. I implemented a simple version to [generate gradients](https://github.com/oruc47/mle_ls_auto_regression/blob/a50b911b1a742f933baa170d412d19fdf1187049/func.cpp#L78C1-L92C2), [function to invert matricies](https://github.com/oruc47/mle_ls_auto_regression/blob/a50b911b1a742f933baa170d412d19fdf1187049/func.cpp#L100C1-L136C2), 
[gradient descent](https://github.com/oruc47/mle_ls_auto_regression/blob/a50b911b1a742f933baa170d412d19fdf1187049/func.cpp#L148C1-L178C2), and to [generate hessian matricies](https://github.com/oruc47/mle_ls_auto_regression/blob/a50b911b1a742f933baa170d412d19fdf1187049/func.cpp#L230C1-L255C1)


## Model

This is an AR(2) model where the errors of the model are influenced by both the $t-1$ period and the $t-2$ period. $\phi$ is an important variable of interest as it is the coefficient of the errors for each period. There will be cases where the model will be reduced to AR(1) by setting $\phi_2 = 0$.

$$
y_t = \alpha + \beta t + u_t
$$

and

$$
u_t = \phi_1 u_{t-1} + \phi_2 u_{t-2} + \varepsilon_t \quad \varepsilon_t \sim \text{NIID}(0, \sigma^2_\varepsilon) \quad \text{for} \quad t = 1, \ldots, T
$$

### Maximum Likelihood and Non-Linear Least Squares

|              | ML        | Standard Errors | t-statistics |
|--------------|-----------|-----------------|--------------|
| $\alpha$   | -0.373115 | 0.749832        | -0.497598    |
| $\beta$    | 1.48195   | 0.102728        | 14.426       |
| $\phi$    | -0.28469  | 0.573275        | -0.496604    |
| $\sigma$   | 2.22299   | 0.408475        | 5.44216      |

*Table 1: MLE Estimation Results*

The results of the initial estimation with $phi_2 = 0$ are shown above. We see that $\beta$ is statistically significant, and with a coefficient of 1.48195, we can confirm there is a clear relationship between $y$ and $t$, which makes sense in this context as we are neglecting the $t-2$ part of the error. This is also further confirmed by the $\phi$ value being insignificant, thus not impacting the AR(1) model much. However, $\sigma$ also being significant means that there are parts of the estimation that are not captured by the model.

|              | NLS       |
|--------------|-----------|
| $\alpha$   | -0.41569  |
| $\beta$    | 1.47656   |
| $\phi\$     | -0.306232 |
| $\sigma$   | 2.66469   |

*Table 2: NLS AR(1) Estimation Results*

We can confirm that MLE is an accurate estimator for NLS as the parameter values are relatively similar. It is important to note that in both cases the log-likelihood function and the sum of squared residuals functions are both optimized using the same optimization function present in `func.cpp`. The function `gradient_descent` primarily uses the finite difference method to calculate gradients. I acknowledge that this is not the most robust manner to optimize, but since I wrote the optimization functions from scratch, I chose this method.

|              | OLS       | NLS       | MLE       |
|--------------|-----------|-----------|-----------|
| $\alpha$   | -0.363624 | -0.146065 | -0.608293 |
| $\beta\$   | 1.47902   | 1.44913   | 1.50779   |
| $\phi_1\$   | -0.328802 | -0.327517 | -0.249232 |
| $\phi_2\$   | -0.0609359| -0.0586215| -0.033152 |
| $\sigma^2$ | 3.26954   | 3.33666   | 0.452973  |

*Table 3: Estimation Results for OLS, NLS, and MLE (AR(2))*

We further see this pattern continue when we move to AR(2). There is no significant difference among the results of the parameters. We also consistently see that $\phi_2$ has relatively small parameter values, meaning that the $t-2$ errors cause fewer disturbances. However, we do notice that the $\sigma^2$ value is smaller for MLE, which means that MLE captures the nonlinear model more consistently than OLS. Finally, we cannot use both $t$ and $t-1$ as regressors, as it would introduce multicollinearity, since $t-1$ is perfectly correlated with $t$.

**[Full Report Can be Found Here](https://github.com/oruc47/mle_ls_auto_regression/blob/4930f36af5823a3283c9dedf7a89c6ba1f3566eb/major_assignment_report.pdf)** 
