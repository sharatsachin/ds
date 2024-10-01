# Linear Regression

## What is Linear Regression?

Linear regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables. In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. It is mostly used for finding out the relationship between variables and the target variable.

## What are the assumptions behind linear regression?

1. Linearity: The relationship between X and Y is linear.
2. Independence: Observations are independent of each other.
3. Homoscedasticity: Constant variance of residuals.
4. Normality: Residuals are normally distributed.
5. No multicollinearity: Independent variables are not highly correlated.

## What is Homoscedasticity in the assumption?

Homoscedasticity means the variance of residuals is constant across all levels of the independent variables.

## What will happen (challenges) if we try to predict categorical values using linear regression? (Using a threshold)

1. Violates assumptions of linear regression.
2. Predictions may fall outside valid range (e.g., probabilities < 0 or > 1).
3. Threshold choice affects results significantly.
4. Poor performance compared to logistic regression for binary outcomes.

## What is the loss function of linear regression?

Mean Squared Error (MSE):

$MSE = \frac{1}{2n} \sum_{i=1}^n (y_i - \hat{y_i})^2$

## Why we divide the value by 2 in loss function?

Dividing by 2 simplifies the derivative calculation, making optimization easier without changing the optimal parameters.

## Why we use square & root under in loss function why not take the absolute value?

Squaring makes the function differentiable everywhere, unlike absolute value. It also penalizes larger errors more heavily.

## How do we optimize the loss function?

Gradient descent: Iteratively update parameters in the direction of steepest decrease in the loss function.

## What is the formula for new theta value (old theta – lambda* d/dx)?

$\theta_{new} = \theta_{old} - \alpha \frac{\partial J}{\partial \theta}$

Where $\alpha$ is the learning rate and $\frac{\partial J}{\partial \theta}$ is the partial derivative of the cost function with respect to $\theta$.

## What is d/dx here?

$\frac{\partial J}{\partial \theta}$ represents the partial derivative of the cost function J with respect to the parameter $\theta$.

## What is R**2 and importance of it?

R-squared (R²) measures the proportion of variance in the dependent variable explained by the independent variables. It ranges from 0 to 1, with higher values indicating better fit.

## What is VIF?

Variance Inflation Factor (VIF) measures multicollinearity in regression analysis. High VIF (>5-10) indicates high correlation between predictors.

## What is the difference between R**2 & Adjusted R**2?

Adjusted R² penalizes the addition of unnecessary predictors, while R² always increases with more predictors. Adjusted R² is more suitable for comparing models with different numbers of predictors.

## What are the different metrics for the model evaluation?

1. MSE/RMSE
2. MAE
3. R-squared
4. Adjusted R-squared
5. F-statistic
6. AIC/BIC

## How do we handle over fitting problem?

1. Regularization
2. Feature selection
3. Cross-validation
4. Increasing training data
5. Reducing model complexity

## What are the different techniques of Regularization?

1. Lasso (L1)
2. Ridge (L2)
3. Elastic Net

## What is the math behind Lasso & Ridge & Elastic Net?

Lasso (L1): $\min_{\beta} \sum_{i=1}^n (y_i - x_i^T\beta)^2 + \lambda \sum_{j=1}^p |\beta_j|$

Ridge (L2): $\min_{\beta} \sum_{i=1}^n (y_i - x_i^T\beta)^2 + \lambda \sum_{j=1}^p \beta_j^2$

Elastic Net: $\min_{\beta} \sum_{i=1}^n (y_i - x_i^T\beta)^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2$

## What is the difference between Lasso & Ridge & Elastic Net?

Lasso (L1) can lead to feature selection by setting some coefficients to zero. Ridge (L2) shrinks all coefficients but doesn't set them to zero. Elastic Net combines both L1 and L2 penalties, balancing between feature selection and coefficient shrinkage.