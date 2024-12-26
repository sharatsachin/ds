# Logistic Regression

## What is Logistic Regression?

Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes). The goal of logistic regression is to find the best fitting model to describe the relationship between the feature of interest and the independent variables.

## What are the assumptions behind logistic regression?

1. The dependent variable must be dichotomous in nature.
2. There should be no outliers in the data.
3. There should be no high correlation between the independent variables.
4. There should be a linear relationship between the independent and dependent variables.
5. The sample size should be sufficiently large.

## What is the sigmoid function?

The sigmoid function is:

$\sigma(z) = \frac{1}{1 + e^{-z}}$

It maps any real-valued number to a value between 0 and 1, useful for converting linear predictions to probabilities.

## Is there any distribution related assumption in LR?

Logistic Regression assumes the log-odds of the outcome are linearly related to the predictors. It doesn't assume normality of predictors, but assumes the binary outcome follows a Bernoulli distribution.

## What is the interpretation of weights in logistic regression?

Weights represent the change in log-odds of the outcome for a one-unit increase in the corresponding predictor, holding other predictors constant. Exponentiating gives the odds ratio.

## What is the loss function?

The loss function for logistic regression is the negative log-likelihood:

$J(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$

Where $h_\theta(x^{(i)})$ is the predicted probability for the i-th example.

## Explain any method through which we get this loss function?

This loss function is derived from the principle of maximum likelihood estimation. We maximize the likelihood of observing the data given our model parameters, which is equivalent to minimizing the negative log-likelihood.

## What are the different evaluation metrics for LR?

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. ROC-AUC
6. Log loss
7. Confusion matrix

## Why accuracy is not a good metric here?

Accuracy can be misleading for imbalanced datasets. It doesn't distinguish between types of errors (false positives vs. false negatives), which can be crucial in many applications.

## What is precision & recall?

Precision: $\frac{TP}{TP + FP}$ (proportion of positive predictions that are correct)
Recall: $\frac{TP}{TP + FN}$ (proportion of actual positives that are correctly identified)

Where TP = True Positives, FP = False Positives, FN = False Negatives

## Which is more important? Precision or recall?

It depends on the problem context. High precision is important when the cost of false positives is high. High recall is crucial when missing positive cases is costly.

## What is F1 & F2 score?

F1-score: Harmonic mean of precision and recall
$F1 = 2 * \frac{precision * recall}{precision + recall}$

F2-score: Similar to F1, but weighs recall higher than precision
$F2 = 5 * \frac{precision * recall}{4 * precision + recall}$

## What is ROC & AUC?

ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs False Positive Rate at various threshold settings.

AUC (Area Under the Curve) is the area under the ROC curve, measuring the model's ability to distinguish between classes.

## What is on x-axis & y-axis in AUC?

X-axis: False Positive Rate (1 - Specificity)
Y-axis: True Positive Rate (Sensitivity or Recall)

## How do we calculate the area under curve (is there any formula for area?)

AUC can be calculated using the trapezoidal rule:

$AUC \approx \sum_{i=1}^{n-1} (x_{i+1} - x_i) \frac{y_i + y_{i+1}}{2}$

Where $(x_i, y_i)$ are points on the ROC curve.

## Given a scenario which each will be more important, precision or recall?

Scenario-dependent. For cancer detection, high recall is crucial to avoid missing cases. For spam filtering, high precision might be preferred to avoid marking important emails as spam.

## How can we increase precision by not reducing recall much?

1. Feature engineering to create more informative predictors
2. Ensemble methods like stacking or boosting
3. Adjust classification threshold
4. Use more sophisticated algorithms (e.g., gradient boosting)
5. Collect more relevant training data

Remember, there's often a trade-off between precision and recall, so significant improvements in both simultaneously can be challenging.