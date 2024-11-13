# Linear Regression

## What is Linear Regression?

Linear regression can be used to fit a model to an observed data set of values of the resonse (dependent) variable $y$ at different values of the predictor (independent) variable $x$. The model takes the form of a linear equation. The goal is to find the best-fitting straight line through the points.

## What is the nomenclature for input variables, output variables, training example, set?

- Input variables: $x^{(i)}$ is the vector of input variables for the i-th training example, $x^{(i)} = [x_1^{(i)}, x_2^{(i)}, ..., x_n^{(i)}]_{(n + 1) \times 1}$, where $x_0^{(i)} = 1$ and $n$ is the number of features.
- Output variables: $y^{(i)}$ is the output variable for the i-th training example.
- Training example: $(x^{(i)}, y^{(i)})$ is the i-th training example.
- Training set: $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)})\}$ is the training set of m examples.

## What is the goal in linear regression?

The goal is to learn a function $h(x) : x \rightarrow y$ so that $h(x)$ is a "good" predictor for the corresponding value of $y$.

## What is the hypothesis function in linear regression? (for the $i^{th}$ training example)

The hypothesis function $$ y^{(i)} = h(x^{(i)}) = \theta_0 + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + ... + \theta_n x_n^{(i)} = \sum_{j=0}^n \theta_j x_j^{(i)} = \theta^T x^{(i)} $$

## What is the type of regression called for $n=1$ and $n>1$?

- For $n=1$: Simple linear regression $$ y = \theta_0 + \theta_1 x $$
- For $n>1$: Multiple linear regression $$ y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n $$

## What is multivariate linear regression?

Multivariate linear regression refers to the case where there are multiple dependent variables and multiple independent variables. $$ y_1, y_2, ..., y_m = f(x_1, x_2, ..., x_n) $$

## What is the cost function in linear regression? What is it called?

The cost function for linear regression is the Mean Squared Error (MSE) or the sum of squared errors (SSE) over the training set. $$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h(x^{(i)}) - y^{(i)})^2 $$
This is the Ordinary Least Squares (OLS) cost function, working to minimize the mean squared error.

## Rewriting the linear equation in matrix form, what would it look like?

In matrix form, the hypothesis function can be written:

$$X = \begin{bmatrix} - \left( x^{(1)} \right)^T - \\ - \left( x^{(2)} \right)^T - \\ \vdots \\ - \left( x^{(m)} \right)^T - \end{bmatrix}_{(m \times (n+1))} , \qquad \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{bmatrix}_{((n+1) \times 1)} \qquad and \qquad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix} _{(m \times 1)}$$

Then the vector of predictions, 

$$\hat{y} = X\theta = \begin{bmatrix} - \left( x^{(1)} \right)^T\theta - \\ - \left( x^{(2)} \right)^T\theta - \\ \vdots \\ - \left( x^{(m)} \right)^T\theta - \end{bmatrix}_{(m \times 1)}$$

## Rewriting the cost function in matrix form, what would it look like?

In matrix form, the cost function can be written:

$$J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y)$$

## How do you derive the Normal Equation for linear regression?

The normal equation is an analytical solution to the linear regression problem with a ordinary least square cost function. That is, to find the value of $\theta$ that minimizes $J(\theta)$, take the gradient of $J(\theta)$ with respect to $\theta$ and equate to $0$, i.e. $\delta_{\theta} J(\theta) = 0$.

## What is the formula for the Normal Equation?

$$\theta = (X^T X)^{-1} X^T y$$

## How does gradient descent work for linear regression?

Gradient descent is based on the observation that if the function $J({\theta})$ is differentiable in a neighborhood of a point $\theta$, then $J({\theta})$ decreases fastest if one goes from $\theta$ in the direction of the negative gradient of $J({\theta})$ at $\theta$. 

Thus if we repeatedly apply the following update rule, ${\theta := \theta - \alpha \nabla J(\theta)}$ for a sufficiently small value of **learning rate**, $\alpha$, we will eventually converge to a value of $\theta$ that minimizes $J({\theta})$.

## What is the update rule for a specific parameter $\theta_j$?

For a specific paramter $\theta_j$, the update rule is 

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J({\theta}) $$

Using the definition of $J({\theta})$, we get

$$\frac{\partial}{\partial \theta_j} J({\theta}) = \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

Therefore, we repeatedly apply the following update rule (batch gradient descent) for all parameters $\theta_j$:

$$
\begin{align*}
& \text{Loop:} \\
& \quad \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)} \\
& \quad \text{simultaneously update } \theta_j \text{ for all } j \\
\end{align*}
$$

## What is the alternative to batch gradient descent?

An alternative to batch gradient descent is stochastic gradient descent (SGD), where we update the parameters using the gradient of the error for each training example.

$$
\begin{align*}
& \text{Loop:} \\
& \quad \text{for } i=1 \text{ to } m \text{ \{} \\
& \quad \quad \theta_j := \theta_j - \alpha \left( h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)} \\
& \quad \quad \text{simultaneously update } \theta_j \text{ for all } j \\
& \quad \text{\}} \\
\end{align*}
$$

## What is the learning rate in gradient descent?

The learning rate, $\alpha$, is a hyperparameter that controls how much we are adjusting the weights of our network with respect to the loss gradient. It is a scalar value, typically between $0$ and $1$. If the learning rate is too small, the model will take a long time to converge; if it is too large, the model may overshoot the minimum.

## What are the different types of gradient descent?

The method that looks at every example in the entire training set on every step is called **batch gradient descent (BGD)**. 

When the cost function $J$ is convex, all local minima are also global minima, so in this case gradient descent can converge to the global solution.

There is also **stochastic gradient descent (SGD)** (also incremental gradient descent), where we repeatedly run through the training set, and for each training example, we update the parameters using gradient of the error for that training example only.

## What are the trade-offs between batch and stochastic gradient descent?

Whereas BGD has to scan the entire training set before taking a single step, SGD can start making progress right away with each example it looks at. 

Often, SGD gets $\theta$ *close* to the minimum much faster than BGD. However it may never *converge* to the minimum, and $\theta$ will keep oscillating around the minimum of $J(\theta)$; but in practice these values are reasonably good approximations. Also, by slowly decreasing $\alpha$ to $0$ as the algorithm runs, $\theta$ converges to the global minimum rather than oscillating around it.

## How is regularization used in linear regression? Types of regularization?

Regularization is a technique to reduce overfitting in machine learning. This technique discourages learning a more complex or flexible model, by shrinking the parameters towards $0$.

We can regularize machine learning methods through the cost function using $L1$ regularization or $L2$ regularization. $L1$ regularization adds an absolute penalty term to the cost function, while $L2$ regularization adds a squared penalty term to the cost function. A model with $L1$ norm for regularisation is called **lasso regression**, and one with (squared) $L2$ norm for regularisation is called **ridge regression**. [Link](https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261)

## What is the formula for the cost function with regularization (L1 & L2)?

$$J(\theta)_{L1} = \frac{1}{2m} \left( \sum_{i=1}^m \left( h_\theta\left( x^{(i)} \right) - y^{(i)} \right)^2 \right) + \frac{\lambda}{2m} \left( \sum_{j=1}^n |\theta_j| \right)$$

$$J(\theta)_{L2} = \frac{1}{2m} \left( \sum_{i=1}^m \left( h_\theta\left( x^{(i)} \right) - y^{(i)} \right)^2 \right) + \frac{\lambda}{2m} \left( \sum_{j=1}^n \theta_j^2 \right)$$

## What is the partial derivative of the cost function with respect to $\theta$ for L1 regularization?

The partial derivative of the cost function for lasso linear regression is:

$$
\begin{align}
& \frac{\partial J(\theta)_{L1}}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta \left(x^{(i)} \right) - y^{(i)} \right) x_j^{(i)} 
& \qquad \text{for } j = 0 \\
& \frac{\partial J(\theta)_{L1}}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta \left( x^{(i)} \right) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{2m} signum (\theta_j)
& \qquad \text{for } j \ge 1
\end{align}
$$

These equations can be substituted into the general gradient descent update rule to get the specific lasso / ridge update rules.

## What is the partial derivative of the cost function with respect to $\theta$ for L2 regularization?

The partial derivative of the cost function for ridge linear regression is:

$$
\begin{align}
& \frac{\partial J(\theta)_{L2}}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta \left(x^{(i)} \right) - y^{(i)} \right) x_j^{(i)} 
& \qquad \text{for } j = 0 \\
& \frac{\partial J(\theta)_{L2}}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta \left( x^{(i)} \right) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} \theta_j 
& \qquad \text{for } j \ge 1
\end{align}
$$

These equations can be substituted into the general gradient descent update rule to get the specific lasso / ridge update rules.

## How does Lasso regression differ from Ridge regression?

Lasso regression uses the $L1$ regularization term, which adds an absolute penalty term to the cost function. This leads to sparsity in the model, as it can shrink some coefficients to zero, effectively performing feature selection.
Ridge regression uses the $L2$ regularization term, which adds a squared penalty term to the cost function. This leads to all coefficients being shrunk by the same factor, but none are eliminated.

## When to use Lasso and when to use Ridge regression?

Use Lasso regression when you believe that many features are irrelevant or when you want to perform feature selection. Use Ridge regression when you believe that most features are important and you want to avoid overfitting.

## What is Elastic Net regularization?

Elastic Net regularization is a combination of $L1$ and $L2$ regularization. It adds both the absolute and squared penalty terms to the cost function, with two hyperparameters $\lambda_1$ and $\lambda_2$ to control the strength of each regularization term.

## What are the assumptions behind linear regression?

1. **Linearity**: The relationship between the independent and dependent variables is linear.
2. **Independence**: The residuals are independent of each other.
3. **Normality**: The residuals follow a normal distribution.
4. **Equal Variance (Homoscedasticity)**: The residuals have constant variance.
5. **No Multicollinearity**: The independent variables are not highly correlated.
6. **No Endogeneity**: The independent variables are not correlated with the residuals.

## How do you check for linearity in linear regression? Fixes?

You can check for linearity by plotting the residuals against the fitted values. If the residuals show a pattern (e.g., a curve), the relationship may not be linear. You can include polynomial terms (e.g., \(X^2\), \(X^3\)) or transform variables (log, square root) to capture non-linear relationships.

## How do you check for independence in linear regression? Fixes?

You can check for independence using the Durbin-Watson test or by plotting the residuals against time. If the residuals show a pattern over time, there may be autocorrelation. You can add lag terms / use time series models like ARIMA to account for autocorrelation.

## How do you check for normality in linear regression? Fixes?

You can check for normality using a Q-Q plot, the Shapiro-Wilk test, or by plotting a histogram of the residuals. If the Q-Q plot has a S-shaped curve or a bow-shaped curve, the residuals are not normally distributed. If the residuals are not normally distributed, you can transform the variables or consider robust regression techniques.

## What is a Q-Q plot?

A Q-Q (quantile-quantile) plot is a graphical tool to help us assess if a set of data plausibly came from some theoretical distribution such as a Normal or exponential. The data is plotted against a theoretical distribution in such a way that the points should form approximately a straight line if the data is approximately normally distributed.

## How do you check for homoscedasticity in linear regression? Fixes?

You can check for homoscedasticity by plotting the residuals against the fitted values or using the Breusch-Pagan test. If the residuals show a pattern (e.g., a cone shape), you can transform the response variable or use Weighted Least Squares (WLS). Transforming the response variable (e.g., log, square root) can stabilize the variance.

## How do you check for multicollinearity in linear regression? Fixes?

You can check for multicollinearity using the Variance Inflation Factor (VIF) or by looking at the correlation matrix. If the VIF is greater than 5-10 or if there are high correlations between predictors, you can remove or combine correlated variables, or use techniques like Principal Component Analysis (PCA).

## How do you check for endogeneity in linear regression? Fixes?

Endogeneity occurs when the independent variables are correlated with the residuals. This can happen when:
1. Omitted variable bias: Important variables are missing from the model.
2. Simultaneity: The relationship between variables is bidirectional. For example, the prices of goods and the demand for those goods.

You can check for endogeneity by plotting the residuals against the predictors. If there is a pattern, you can use instrumental variables or 2SLS (Two-Stage Least Squares) to address endogeneity.

## What will happen (challenges) if we try to predict categorical values using linear regression? (Using a threshold)

1. Violates assumptions of linear regression.
2. Predictions may fall outside valid range (e.g., probabilities < 0 or > 1).
3. Threshold choice affects results significantly.
4. Poor performance compared to logistic regression for binary outcomes.

## Why we divide the value by 2 in loss function?

Dividing by 2 simplifies the derivative calculation, making optimization easier without changing the optimal parameters.

## Why we use square & root under in loss function why not take the absolute value?

Squaring makes the function differentiable everywhere, unlike absolute value. It also penalizes larger errors more heavily.

## How do we optimize the loss function?

Gradient descent: Iteratively update parameters in the direction of steepest decrease in the loss function.

## What is $R^2$ and importance of it?

R-squared (R²) measures the proportion of variance in the dependent variable explained by the independent variables. It ranges from 0 to 1, with higher values indicating better fit.

## What is the formula for $R^2$?

$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} $$

Where $SS_{res} = \sum_{i=1}^m (y^{(i)} - h(x^{(i)}))^2$ is the sum of squared residuals and $SS_{tot} = \sum_{i=1}^m (y^{(i)} - \bar{y})^2$ is the total sum of squares.

## What is VIF?

Variance Inflation Factor (VIF) measures multicollinearity in regression analysis. High VIF (>5-10) indicates high correlation between predictors.

## What is the difference between $R^2$ & Adjusted $R^2$?

Adjusted R² penalizes the addition of unnecessary predictors, while R² always increases with more predictors. Adjusted R² is more suitable for comparing models with different numbers of predictors.

## What is the formula for Adjusted $R^2$?

$$ \text{Adjusted } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - k - 1} $$

Where $n$ is the number of observations and $k$ is the number of predictors.

## How do we handle over fitting problem?

1. Regularization
2. Feature selection
3. Cross-validation
4. Increasing training data
5. Reducing model complexity

## What are the different metrics for the model evaluation?

1. MSE/RMSE
2. MAE
3. R-squared
4. Adjusted R-squared
5. F-statistic
6. AIC/BIC

## What is the formula for MSE?

$$ MSE = \frac{1}{m} \sum_{i=1}^m (y^{(i)} - h(x^{(i)}))^2 $$

## What is the formula for RMSE?

$$ RMSE = \sqrt{MSE} $$

## What is the formula for MAE?

$$ MAE = \frac{1}{m} \sum_{i=1}^m |y^{(i)} - h(x^{(i)})| $$

## What is the formula for F-statistic?

The F-statistic is used to test the overall significance of the model as compared to a model with no predictors. It is calculated as:

$$ F = \frac{(TSS - RSS) / p}{RSS / (n - p - 1)} $$

Where $TSS = \sum_{i=1}^m (y^{(i)} - \bar{y})^2$ is the total sum of squares, $RSS = \sum_{i=1}^m (y^{(i)} - h(x^{(i)}))^2$ is the residual sum of squares, $p$ is the number of predictors, and $n$ is the number of observations. Here $n - p - 1$ is the degrees of freedom.

It's range is from 0 to infinity. Higher values indicate a better fit.

## What is Prob(F-statistic)?

The probability associated with the F-statistic is the probability of observing an F-statistic as extreme as the one computed from the data, assuming the null hypothesis is true. It is used to test the overall significance of the model.

In this case, the null hypothesis is that all coefficients are zero, i.e., the model has no predictive power.

A smaller values, typically less than 0.05, indicates that the model is significant.

## What is the formula for AIC?

AIC (Akaike Information Criterion) is a measure of the relative quality of a statistical model for a given set of data. It is calculated as:

$$ AIC = 2k - 2\ln(L) $$

Where $k$ is the number of parameters and $L$ is the likelihood function.

## What is the formula for BIC?

BIC (Bayesian Information Criterion) is a criterion for model selection among a finite set of models. It is calculated as:

$$ BIC = k\ln(n) - 2\ln(L) $$

Where $n$ is the number of observations.

## What do the coef, std. error, t-statistic, p-value, and 95% confidence intervals represent in the summary output?

1. **Coef**: Estimated coefficients for the predictors. Represents the change in the dependent variable for a one-unit change in the predictor.
2. **Std. Error**: Standard error of the coefficient estimate. Measures the variability in the estimate. A lower value indicates a more precise estimate.
3. **t-statistic**: The coefficient divided by its standard error. Measures the significance of the coefficient estimate. Value greater than 2 indicates significance.
4. **P-value**: Probability of observing the t-statistic if the null hypothesis is true. Typically, p < 0.05 is considered significant.
5. **95% Confidence Intervals**: Range of values within which the true coefficient is likely to fall with 95% confidence.

## What do the model diagnostics (Omnibus, Skew, Kurtosis, Jarque-Bera, Durbin-Watson, Condition Number) represent?

1. **Omnibus**: Tests the skewness and kurtosis of the residuals. A value close to zero indicates normal distribution. 
2. **Skew**: Measures the symmetry of the residuals. A value close to zero indicates normal distribution. $ < 0$ indicates negative skew and $ > 0$ indicates positive skew.
3. **Kurtosis**: Measures the heaviness of the tails of the residuals. A value close to zero indicates normal distribution. $ < 3$ indicates light tails and $ > 3$ indicates heavy tails.
4. **Jarque-Bera**: Tests the skewness and kurtosis of the residuals. A value close to zero indicates normal distribution.
5. **Durbin-Watson**: Tests for autocorrelation in the residuals. Value of 2 indicates no autocorrelation, while values < 2 or > 2 indicate positive or negative autocorrelation.
6. **Condition Number**: Measures multicollinearity in the model. Values > 30 indicate multicollinearity.