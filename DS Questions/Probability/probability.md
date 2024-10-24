# Statistics

## What is statistics? Descriptive and Inferential?

A field that focuses on learning from data. It involves collecting, organizing, analyzing, interpreting, and presenting data. It is used to understand and make decisions about a population based on a sample drawn from that population.
1. **Descriptive statistics**: Describes basic features of data in a study, provides simple summaries about the sample and the measures. It is used to summarize and describe the characteristics of a data set.
2. **Inferential statistics**: Makes inferences and predictions about a population based on a sample of data taken from the population. It is used to draw conclusions about a population based on the sample data.

## Quantitative vs Qualitative variables

- **Qualitative variables**: Variables that are not measured on a numeric scale. They can be nominal (like gender, color) or ordinal (like rating, satisfaction level).
- **Quantitative variables**: Variables that are measured on a numeric scale. They can be discrete or continuous. Examples include age, height, weight, income, etc.

## What is the difference between ratio and interval variables?

- **Interval variables**: Variables that have a natural order and a consistent difference between values, but do not have a true zero point. Examples include temperature (Celsius, Fahrenheit), dates, and times.
- **Ratio variables**: Variables that have a natural order, a consistent difference between values, and a true zero point. Examples include height, weight, distance, and time.

## Measures of central tendency
- answer the question "where is the center of the data?"
- **Mean**: The average of the data points. It is calculated as $\mu = \bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$.
- **Median**: The middle value of the data points when they are sorted in ascending order. $median = \begin{cases} x_{(n+1)/2} & \text{if } n \text{ is odd} \\ \frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ is even} \end{cases}$.
- **Mode**: The value that appears most frequently in the data set.

## Skewness

A measure of the asymmetry of the probability distribution of a real-valued random variable about its mean:
1. positive skew: data skewed to higher values (long tail on the right, majority of data points on the left, mean > median > mode)
2. negative skew: data skewed to lower values (long tail on the left, majority of data points on the right, mean < median < mode)
3. zero skew: data is symmetric (mean = median = mode)

## Measures of dispersion (variability)

- answer the question "how spread out are the data points?"
- **Range**: The difference between the maximum and minimum values in a data set.
- **Variance**: The average of the squared differences from the mean. It is calculated as $\sigma^2 = \frac{\sum_{i=1}^{n} (x_i - \mu)^2}{n}$.
- **Standard deviation**: The square root of the variance. It is calculated as $\sigma = \sqrt{\sigma^2}$.
- **Coefficient of variation**: The ratio of the standard deviation to the mean. It is used to compare the variability of data sets with different units.
- **Interquartile range (IQR)**: The difference between the third quartile (Q3) and the first quartile (Q1). It is more robust to outliers than the range.

## Why use (n-1) in the denominator for sample variance?

- We use (n-1) in the denominator instead of n when calculating for a sample instead of a population
- to provide an unbiased estimation of the population variance, since adjustment accounts for the loss of one degree of freedom when estimating the sample mean (helps avoid underestimation of true population variance/std. dev)
- using n-1, we provide a more conservative estimate of the population variance, ensuring our stat. inference is more accurate

## 5 point summary

- Consists of the minimum, first quartile (Q1), median, third quartile (Q3), and maximum of a data set.

## What is the Z-score?

The number of standard deviations a data point is from the mean. It is used to understand the position of a data point in a distribution. It is calculated as $z = \frac{x-\mu}{\sigma}$.

## Box plot, where is the median, Q1, Q3, whiskers, and outliers?

- Median : center line of the box
- Q1, Q3 : lower and upper bounds of the box
- Whiskers : lines extending from the box, representing the range of the data (usually Q1-1.5*IQR and Q3+1.5*IQR)
- Outliers : data points beyond the whiskers

## What is the expirical rule?

If a dataset has a approximately bell shaped relative frequency distribution, then:
- 68% of the data falls within 1 standard deviation of the mean ($\bar{x} \pm s$) for samples, ($\mu \pm \sigma$) for population
- 95% of the data falls within 2 standard deviations of the mean
- 99.7% of the data falls within 3 standard deviations of the mean

## What is Chebyshev's theorem?

For any data set, the proportion of observations that lie within $k$ standard deviations of the mean is at least $1 - \frac{1}{k^2}$, where $k$ is any positive number larger than 1.
- at least 75% of the data lie within two standard deviations, 89% within three standard deviations, etc.

## What is hypothesis testing?

A method of statistical inference. It is used to determine whether there is enough evidence in a sample of data to infer that a certain condition is true for the entire population.
Allows us to make probabilistic statements about a population parameter based on a statistic computed from a sample randomly drawn from that population.

## Explain the null hypothesis and alternate hypothesis

**Null hypothesis ($H_0$)**:

- status quo, says that there is no change or difference from what you already know
- the hypothesis that there is no significant difference between specified populations, any observed difference being due to sampling or experimental error.

**Alternative hypothesis ($H_a$)**:

- the hypothesis that we are trying to prove
- the hypothesis that sample observations are influenced by some non-random cause, and that the observed difference between the sample and population reflects this cause

## What is a type I error?

Occurs when the null hypothesis is true but is rejected. It is asserting something that is absent, a false hit. A type I error is often referred to as a false positive (a result that indicates that a given condition is present when it actually is not present).

## What is a type II error?

Occurs when the null hypothesis is false but erroneously fails to be rejected. It is failing to assert what is present, a miss. A type II error is often called a false negative (where an actual hit was disregarded by the test and is seen as a miss).

## Null Hypothesis (True vs False) vs Decision (Reject vs Fail to Reject) - Fill in the table

| | Null Hypothesis True | Null Hypothesis False |
| --- | --- | --- |
| **Reject** | Type I Error (FP) p = $\alpha$ | Correct Decision (TP) p = $1-\beta$ |
| **Fail to Reject** | Correct Decision (TN) p = $1-\alpha$ | Type II Error (FN) p = $\beta$ |

## What is the level of significance ($\alpha$)?

The probability of rejecting the null hypothesis when it is true. It is the probability of making a type I error. It is usually set to 0.05 or 0.01.

## What is the power of a test?

It is the probability of rejecting the null hypothesis when it is false. It is the probability of not making a type II error, hence it is equal to $1-\beta$.

## Graph of relation between Type I and Type II errors

<img alt="picture 0" src="https://cdn.jsdelivr.net/gh/sharatsachin/images-cdn@master/images/ef13633fcd2944eb7836a0202410da64de5ac27e44faa3f77221d0d1d98dc060.png" width="500" />  

## How is the power of a test affected by the standard deviation, the sample size and the effect size?

- Larger the standard deviation, lower the power
- Larger the sample size, higher the power
- Larger the effect size, higher the power

## What is a t-test? When would you use it?

Statistical test used to compare the means of two groups. It is often used in hypothesis testing to determine whether a process or treatment actually has an effect on the population of interest, or whether two groups are different from one another.

[Link](https://en.wikipedia.org/wiki/Student%27s_t-test)

## What is a f-test? When would you use it?

Family of statistical tests in which the test statistic has an F-distribution under the null hypothesis. It is used to compare statistical models that have been fitted to a data set, in order to identify the model that best fits the population from which the data were sampled.

[Link](https://en.wikipedia.org/wiki/F-test)

## What is a p-value? What is its importance?

The probability of obtaining test results at least as extreme as the results actually observed during the test, assuming that the null hypothesis is correct. The smaller the p-value, the greater the statistical significance of the observed difference.
The p-value is used in the context of null hypothesis testing in order to quantify the statistical significance of the obtained results, the result being the observed value of the test statistic $T$. The null hypothesis is rejected if the p-value is less than a predetermined level $\alpha$.

[Link](https://en.wikipedia.org/wiki/P-value)

## What is the Normal distribution?

A continuous probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. In graph form, normal distribution will appear as a bell curve.

$$f(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
where $\mu$ is the mean and $\sigma^2$ is the variance. $f(x \mid \mu, \sigma^2)$ is the probability density function.

[Link](https://en.wikipedia.org/wiki/Normal_distribution)

## What is the Bernoulli distribution?

The discrete probability distribution of a random variable which takes the value 1 with probability $p$ and the value 0 with probability $q=1-p$. It can be used to represent a coin toss where 1 and 0 would represent "head" and "tail" (or vice versa), respectively. It is a special case of a Binomial distribution where $n=1$.

$$ f(k;p) = \begin{cases} p & \text{if } k=1 \\ q=1-p & \text{if } k=0 \end{cases} $$

[Link](https://en.wikipedia.org/wiki/Bernoulli_distribution)

## What is the Binomial distribution?

The discrete probability distribution of the number of successes in a sequence of $n$ independent experiments, each asking a yes–no question, and each with its own boolean-valued outcome: success/yes/true/one (with probability $p$) or failure/no/false/zero (with probability $q=1-p$). A single success/failure experiment is also called a Bernoulli trial or Bernoulli experiment and a sequence of outcomes is called a Bernoulli process.

$$ f(k;n,p) = \Pr(X = k) = \Pr(k \text{ successes in } n \text{ trials}) = \binom{n}{k} p^k (1-p)^{n-k} $$

[Link](https://en.wikipedia.org/wiki/Binomial_distribution)

## What is the Poisson distribution?

A discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event.

$$ f(k; \lambda) = \Pr(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} $$

[Link](https://en.wikipedia.org/wiki/Poisson_distribution)

## What is the Beta distribution, and how is it different from the Binomial distribution?

A family of continuous probability distributions defined on the interval $[0,1]$ parametrized by two positive shape parameters, denoted by $\alpha$ and $\beta$, that appear as exponents of the random variable and control the shape of the distribution. It is often used to model the uncertainty about the probability of success of an experiment.

The Beta can be used to analyze probabilistic experiments that have only two possible outcomes:

- "success", with probability $X$
- "failure", with probability $1-X$

Steps to use the Beta distribution:

1. Assign uniform prior to probability of success $X$
2. Perform experiment and observe outcome $k$ successes out of $n$ trials
3. Update prior using the observed data to obtain posterior distribution
4. Posterior distribution is a Beta distribution with parameters $\alpha = k+1$ and $\beta = n-k+1$

$$ f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$$
where $B(\alpha, \beta)$ is the Beta function.

The expected value of a Beta distribution is $\frac{\alpha}{\alpha+\beta}$.

[Link](https://www.statlect.com/probability-distributions/beta-distribution)

## Different ways to sample from a dataset

- Simple random sampling : each item in the population has an equal chance of being selected
- Stratified sampling : the population is divided into strata and a random sample is taken from each stratum
- Cluster sampling : the population is divided into clusters and a random sample of clusters are selected
- Systematic sampling : the sample is chosen by selecting a random starting point and then picking every $k$th element in the population
- Multistage sampling : a combination of two or more of the above methods

[Link](https://www.simplilearn.com/types-of-sampling-techniques-article)

## What is the Central Limit Theorem?

In probability theory, the central limit theorem (CLT) establishes that, in some situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution (informally a bell curve) even if the original variables themselves are not normally distributed.

[Link](https://www.probabilitycourse.com/chapter7/7_1_2_central_limit_theorem.php)

## What is the Law of Large Numbers?

A theorem that describes the result of performing the same experiment a large number of times. According to the law, the average of the results obtained from a large number of trials should be close to the expected value, and will tend to become closer as more trials are performed.

[Link](https://en.wikipedia.org/wiki/Law_of_large_numbers)

## How can we identify skewness in a distribution using mean, median and mode?

- If mean < median < mode, the distribution is negatively skewed
- If mean > median > mode, the distribution is positively skewed
- If mean = median = mode, the distribution is symmetric

## Hypothesis of normal distribution

The hypothesis of normality is a hypothesis that the data is normally distributed. It is often tested using the Shapiro-Wilk test, the Kolmogorov-Smirnov test, or the Anderson-Darling test.
Normal distribution is characterized by the mean and the standard deviation. The mean is the center of the distribution and the standard deviation is the measure of the spread of the distribution, and normality implies that the data is symmetric around the mean, with most of the data points lying close to the mean.

## What is correlation? What is covariance? What is the difference between the two?

**Correlation** is a statistical measure that describes the size and direction of a relationship between two or more variables. A positive correlation indicates the extent to which those variables increase or decrease in parallel; a negative correlation indicates the extent to which one variable increases as the other decreases.
Correlation is dimensionless and lies between -1 and 1.
**Covariance** is a measure of the joint variability of two random variables. If the greater values of one variable mainly correspond with the greater values of the other variable, and the same holds for the lesser values, i.e., the variables tend to show similar behavior, the covariance is positive. In the opposite case, when the greater values of one variable mainly correspond to the lesser values of the other, i.e., the variables tend to show opposite behavior, the covariance is negative.
Covaliance is in the units of the product of the units of the two variables.

## How to find correlation between categorical & numerical columns

- Cramer's V for two categorical variables
- Point biserial correlation for one categorical and one numerical variable
- $\eta$-squared for one categorical and one numerical variable

## What is the confidence score?

The confidence score is the probability that the value of a parameter falls within a specified range of values. It is a measure of the reliability of the estimate. It is usually set to 95% or 99%.

## What is the confidence interval?

A range of values, derived from the sample statistics, that is likely to contain the value of an unknown population parameter. The interval has an associated confidence level that the true parameter is in the proposed range.








## Chebyshev’s Theorem
For any data set, the proportion of observations that lie within $k$ standard deviations of the mean is at least $1 - \frac{1}{k^2}$, where $k$ is any positive number larger than 1.
- at least 75% of the data lie within two standard deviations, 89% within three standard deviations, etc.

%% [markdown]
# Probability

- A population is any specific collection of objects of interest. A sample is any subset or subcollection of the population, including the case that the sample consists of the whole population, in which case it is termed a census.
- A measurement is a number or attribute computed for each member of a population or of a sample. The measurements of sample elements are collectively called the sample data.
- A parameter is a number that summarizes some aspect of the population as a whole. A statistic is a number computed from the sample data.
- Statistics computed from samples vary randomly from sample to sample. Conclusions made about population parameters are statements of probability.

## Random experiments
- random experiments are actions that occur by chance, and their outcomes are not predictable
- sample space: the set of all possible outcomes of a random event
    - discrete sample space: a sample space with a finite number of outcomes
    - continuous sample space: a sample space with an infinite number of outcomes
- event: a subset of the sample space of a random experiment
- probability: a numerical measure of the likelihood that an event will occur
$$ P(A) = \frac{\text{number of outcomes in A}}{\text{number of outcomes in S}} = \frac{\text{number of favorable outcomes}}{\text{number of possible outcomes}}$$
- empirical probability: the relative frequency of an event occurring in a series of trials
$$ P_{empirical}(A) = \frac{\text{number of times A occurs}}{\text{number of observations}}$$
- theoretical probability: the probability of an event occurring based on mathematical reasoning
- law of large numbers: as the number of trials increases, the empirical probability of an event will converge to the theoretical probability of that event

## Events
- an event is a subset of the sample space of a random experiment
- an event $A$ occurs on a particular trial of a random experiment if the outcome of that trial is in $A$
- complement of an event: the set of all outcomes in the sample space that are not in the event
  - the complement of an event $A$ is denoted by $A^c$
    - $A^c = S - A, A \cup A^c = S, A \cap A^c = \emptyset$
    - $P(A^c) = 1 - P(A)$
- union of two events: the set of all outcomes that are in either event
  - the union of two events $A$ and $B$ is denoted by $A \cup B$
- intersection of two events: the set of all outcomes that are in both events
  - the intersection of two events $A$ and $B$ is denoted by $A \cap B$
- mutually exclusive / disjoint events: events that have no outcomes in common
  - if $A$ and $B$ are mutually exclusive, then $A \cap B = \emptyset$
- independent events: events that have no effect on each other
- addition rule: the probability of the union of two events is equal to the sum of the probabilities of the individual events minus the probability of their intersection $$ P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
  - if mutually exclusive, $P(A \cap B) = 0$, then $P(A \cup B) = P(A) + P(B)$

## Probability
- the probability of an outcome $e$ in a sample space $S$ is a number $P$ between $0$ and $1$ that measures the likelihood that $e$ will occur on a single trial of a random experiment. The probability of an event $E$ is the sum of the probabilities of the outcomes in $E$.
- a number assigned to each member of the sample space of a random experiment that satisfies the following axioms:
    1. $0 \leq P(A) \leq 1$
    2. $P(S) = 1$
    3. For two events $A$ and $B$, if $A$ and $B$ are mutually exclusive, then $P(A \cup B) = P(A) + P(B)$

%% [markdown]
# Conditional probability

The conditional probability of $A$ given $B$, denoted $P(A|B)$, is the probability that event $A$ has occurred in a trial of a random experiment for which it is known that event $B$ has definitely occurred.
For any two events $A$ and $B$ with $P(B) > 0$, the conditional probability of $A$ given $B$ is defined as: $$ P(A|B) = \frac{P(A \cap B)}{P(B)}$$
Here, $P(A \cap B)$ is equal to both $P(A)P(B|A)$ and $P(B)P(A|B)$.

Conditional probability relation for three events $A$, $B$, and $C$:
$$ P(A \cap B \cap C) = P(A|B \cap C)P(B \cap C) = P(A|B \cap C)P(B|C)P(C)$$

This is called the multiplication rule, which can be extended to any number of events.

## Independent events

- We expect $P(A | B)$ to be different from $P(A)$, but it does not always happen. If $P(A | B) = P(A)$, then $A$ and $B$ are independent events and the occurrence of $B$ has no effect on the likelihood of $A$.
  - $P(A|B) = P(A)$ if and only if $P(A \cap B) = P(A)P(B)$, that is, the probability of $A$ and $B$ occurring together is equal to the product of their individual probabilities
  - if A and B are not independent, then they are dependent and $P(A \cap B) \neq P(A)P(B)$
  - independence intuitively means that the occurrence of one event does not affect the probability of the other event
    - independence does not imply disjointness

## Law of total probability

If $A_1, A_2, ..., A_n$ are mutually exclusive and exhaustive events, then for any event $B$, $$ P(B) = P(B|A_1)P(A_1) + P(B|A_2)P(A_2) + ... + P(B|A_n)P(A_n)$$

%% [markdown]
# Bayes Theorem

Let $P = {A_1, A_2, ..., A_n}$ be a partition of the sample space $S$ of a random experiment, and let $B$ be an event such that $P(B) > 0$. Then for any $i = 1, 2, ..., n$, $$ P(A_i|B) = \frac{P(A_i \cap B)}{P(B)} = \frac{P(A_i)P(B|A_i)}{P(A_1)P(B|A_1) + P(A_2)P(B|A_2) + ... + P(A_n)P(B|A_n)}$$

## Bayesian learning

- Bayesian learning is a method of statistical inference in which Bayes' theorem is used to update the probability for a hypothesis as more evidence or information becomes available.
- Naive Bayes classifier: a simple probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.
    - assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
    - is a linear classifier, which means it assumes that the data is linearly separable.
    - is a fast, simple classification algorithm that performs well on large datasets.   
    - is a good choice when the dimensionality of the inputs is high.
    - is often used for text classification, spam filtering, sentiment analysis, and recommender systems.

## Bayes' theorem
- in the context of machine learning, Bayes' theorem can be used to calculate the probability of a hypothesis given our prior knowledge. $$ P(h|D) = \frac{P(D|h)P(h)}{P(D)}$$
  - $P(h)$ is the prior probability of $h$ being true
  - $P(D)$ is the prior probability of $D$ being true
  - $P(h|D)$ is the posterior probability of $h$ being true given $D$
  - $P(D|h)$ is the likelihood of $D$ given $h$
- maximum a posteriori (MAP) estimation: a method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the posterior probability. $$h_{MAP} = \arg\max\limits_{h \in H} P(h|D) = \arg\max\limits_{h \in H} \frac{P(D|h)P(h)}{P(D)} = \arg\max\limits_{h \in H} P(D|h)P(h)$$
  - $P(D)$ is a constant, so we can ignore it
  - $h_{MAP}$ is called the maximum a posteriori hypothesis
  - we find the parameter values that make the observed data most probable given our prior knowledge about the parameters
- if we assume that all hypotheses are equally likely, then $P(h)$ is a constant, so we can ignore it. $$h_{ML} = \arg\max\limits_{h \in H} P(D|h)$$
  - $h_{ML}$ is called the maximum likelihood hypothesis
  - we find the parameter values that make the observed data most probable
  - used when we have no prior knowledge about the parameters

## Conditional independence
Conditional independence, given C, is defined as independence under the probability law P(·|C). That is, A and B are conditionally independent given C if and only if $$ P(A \cap B|C) = P(A|C)P(B|C)$$
- independence does not imply conditional independence [(link)](https://www.youtube.com/watch?v%253DTAyA-rjmesQ%2526list%253DPLUl4u3cNGP60hI9ATjSFgLZpbNJ7myAg6%2526index%253D35)
- means that $$ P(A|B,C) = P(A|C)$$ and $$ P(A \cap B|C) = P(A|C)P(B|C)$$
<img src="https://i.imgur.com/O7op64y.jpg" width="300" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">
- Naive Bayes assumes that all features are conditionally independent given the class. $$ P(X_1, X_2 | Y) = P(X_1 | X_2, Y)P(X_2 | Y) = P(X_1 | Y)P(X_2 | Y)$$
- General form : $$ P(X_1, X_2, ..., X_n | Y) = \prod\limits_{i=1}^n P(X_i | Y)$$
  - how many parameters do we need to estimate? (??)
  - how many parameters do we need to estimate if we assume that all features are conditionally independent given the class? (??)

## Naive Bayes classifier

- Naive Bayes classifier is a simple probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.
- We have a set of features $X = {X_1, X_2, ..., X_n}$ and a class variable $Y$.
- We want to find the class $Y$ that maximizes the posterior probability $P(Y|X)$.
- Then $$ P(Y = y_k|X_1, X_2, ..., X_n) = \frac{P(Y = y_k)P(X_1, X_2, ..., X_n|Y = y_k)}{\sum\limits_{j} P(Y = y_j)P(X_1, X_2, ..., X_n|Y = y_j)} $$
- Assuming conditional independence, we have $P(X_1, X_2, ..., X_n|Y = y_k) = \prod\limits_{i=1}^n P(X_i|Y = y_k)$. Therefore, $$P(Y = y_k|X_1, X_2, ..., X_n) = \frac{P(Y = y_k)\prod\limits_{i=1}^n P(X_i|Y = y_k)}{\sum\limits_{j} P(Y = y_j)\prod\limits_{i=1}^n P(X_i|Y = y_j)}$$
- Pick the most probable class: $$\hat{y} = \arg\max\limits_{y_k} P(Y = y_k)\prod\limits_{i} P(X_i|Y = y_k)$$

Steps to apply Naive Bayes classifier, given a table like this:

|Weather|Play|
|---|---|
|Sunny|No|
|...|...|
|Rainy|Yes|

We convert it into a frequency table like this:

|Weather|No|Yes|Total|Prob|
|---|---|---|---|---|
|Sunny|2|3|5| P(Sunny) = $\frac{5}{14}$|
|Overcast|0|4|4| P(Overcast) = $\frac{4}{14}$|
|Rainy|3|2|5| P(Rainy) = $\frac{5}{14}$|
|Total|5|9|14|1|
|Prob|P(No) = $\frac{5}{14}$|P(Yes) = $\frac{9}{14}$|1|

Then we can calculate the posterior probability of each class, given the evidence (weather), for example, $P(Yes|Sunny)$:
$$ P(Yes|Sunny) = \frac{P(Sunny|Yes)P(Yes)}{P(Sunny)} = \frac{\frac{3}{9}\frac{9}{14}}{\frac{5}{14}} = \frac{3}{5}$$

If there are multiple features, we can calculate the posterior probability of each class, given the evidence (weather and temperature), for example, $P(Yes|Sunny, Cool)$:
$$ P(Yes|Sunny, Cool) = \frac{P(Sunny, Cool|Yes)P(Yes)}{P(Sunny, Cool)} = \frac{P(Sunny|Yes)P(Cool|Yes)P(Yes)}{P(Sunny)P(Cool)} $$

## Naive Bayes for text classification
You need a document $d$, a set of classes $C = {c_1, c_2, ..., c_n}$, and a set of $m$ hand-labelled documents $(d_1, c_1), (d_2, c_2), ..., (d_m, c_m)$. The for a document $d$, we want to find the class $c$ that maximizes the posterior probability $P(c|d)$.
$$ P(c|d) = \frac{P(c)P(d|c)}{P(d)} = \frac{P(c)\prod\limits_{i=1}^n P(w_i|c)}{P(d)}$$
Here, there are two assumptions : bag of words (position doesn't matter) and conditional independence.
Then, we pick the most probable class: $$c_{MAP} = \arg\max\limits_{c} P(c)\prod\limits_{i=1}^n P(w_i|c)$$
Here, $$ P(c_j) = \frac{docCount(C = c_j)}{N_{doc}} $$ and $$ P(w_i|c_j) = \frac{wordCount(w_i, C = c_j)}{\sum\limits_{w \in V} wordCount(w, C = c_j)} $$, where $V$ is the vocabulary.
This has a problem of zero probability, so we use Laplace smoothing: $$ P(w_i|c_j) = \frac{wordCount(w_i, C = c_j) + 1}{\sum\limits_{w \in V} wordCount(w, C = c_j) + |V|} $$ where $|V|$ is the number of distinct words in the dataset.

[Example](https://www.fi.muni.cz/~sojka/PV211/p13bayes.pdf):
<img src="https://i.imgur.com/p3nZUNM.png" width="500" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">
<img src="https://i.imgur.com/kcNsCro.png" width="500" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">

Therefore, $$P(C|d_5) = \frac{3}{4} {(\frac{3}{7})}^3 \frac{1}{14} \frac{1}{14} \frac{1}{P(d_5)}$$
and $$P(\bar{C} | d_5) = \frac{1}{4} {(\frac{2}{9})}^3 \frac{2}{9} \frac{2}{9} \frac{1}{P(d_5)}$$

$P(d_5)$ is the same for both classes, so we can ignore it.




%% [markdown]
# Random variables
- random variables are variables that take on numerical values based on the outcome of a random experiment
- discrete random variables: random variables that can take on a finite number of values
- continuous random variables: random variables that can take on an infinite number of values

## Probability distributions of discrete random variables
The probability distribution of a discrete random variable $X$ is a list of each possible value of $X$ together with the probability that $X$ takes that value in one trial of the experiment.
The probabilities in the probability distribution of a discrete random variable $X$ must satisfy the following two conditions:
1. $0 \leq P(X = x) \leq 1$ for each possible value $x$ of $X$
2. $\sum_{\text{all } x} P(X = x) = 1$

Example : probability distribution of $X$, the sum of the two dice, is given by:

$$\begin{array}{c|ccccccccccc} x &2 &3 &4 &5 &6 &7 &8 &9 &10 &11 &12 \\ \hline P(x) &\dfrac{1}{36} &\dfrac{2}{36} &\dfrac{3}{36} &\dfrac{4}{36} &\dfrac{5}{36} &\dfrac{6}{36} &\dfrac{5}{36} &\dfrac{4}{36} &\dfrac{3}{36} &\dfrac{2}{36} &\dfrac{1}{36} \\ \end{array}$$

- $P(X \geq 9) = P(X = 9) + P(X = 10) + P(X = 11) + P(X = 12) = \dfrac{10}{36} = \dfrac{5}{18}$
- $P(\text{X is even}) = P(X = 2) + P(X = 4) + P(X = 6) + P(X = 8) + P(X = 10) + P(X = 12) = \dfrac{18}{36} = \dfrac{1}{2}$

### Mean of a discrete random variable
The mean (expected value / expectation) of a discrete random variable $X$ is the weighted average of the possible values of $X$, where the weights are the probabilities of the values of $X$.
$$ \mu = E(X) = \sum_{\text{all } x} xP(x)$$

The mean of a discrete random variable is the long-run average value of the variable.

Rules of expected value:
1. $E(aX + b) = aE(X) + b$
2. $E(X + Y) = E(X) + E(Y)$
3. $E(XY) = E(X)E(Y)$ iff $X$ and $Y$ are independent

### Variance and standard deviation of a discrete random variable
The variance of a discrete random variable $X$ is the weighted average of the squared deviations of the possible values of $X$ from the mean of $X$, where the weights are the probabilities of the values of $X$.
$$ \sigma^2 = Var(X) = \sum(x - \mu)^2 P(x) = [\sum x^2 P(x)] - \mu^2 = E(X^2) - [E(X)]^2$$

The standard deviation of a discrete random variable $X$ is the square root of the variance of $X$.
$$ \sigma = \sqrt{Var(X)}$$

Rules of variance:
1. $Var(aX + b) = a^2Var(X)$
2. $Var(X + Y) = Var(X) + Var(Y)$ if $X$ and $Y$ are independent

## Probability distribution of a continuous random variable
With continuous random variables one is concerned not with the event that the variable assumes a single particular value, but with the event that the random variable assumes a value in a particular interval.

The probability distribution of a continuous random variable $X$ is an assignment of probabilities to intervals of decimal numbers using a function $f(x)$, called a density function, in the following way: the probability that $X$ assumes a value in the interval $[a, b]$ is equal to the area of the region that is bounded above by the graph of the equation $y=f(x)$, bounded below by the x-axis, and bounded on the left and right by the vertical lines through $a$ and $b$. The probability density function $f(x)$ must satisfy the following two conditions:
1. $f(x) \geq 0$ for all $x$
2. $\int_{-\infty}^{\infty} f(x) dx = 1$
3. $P(a \leq X \leq b) = \int_a^b f(x) dx$

### Cumulative probability distribution function
The cumulative probability distribution function of a continuous random variable $X$ is the function $F(x)$ defined by $F(x) = P(X \leq x)$ for all $x$.
$$ F(x) = \int_{-\infty}^x f(u) du$$

To get the probability density function $f(x)$ from the cumulative probability distribution function $F(x)$, we differentiate $F(x)$ with respect to $x$.
$$ f(x) = \frac{d}{dx} F(x)$$

### Mean of a continuous random variable
The mean (expected value / expectation) of a continuous random variable $X$ is the weighted average of the possible values of $X$, where the weights are the probabilities of the values of $X$.
$$ \mu = E(X) = \int_{-\infty}^{\infty} xf(x) dx$$

### Variance and standard deviation of a continuous random variable
The variance of a continuous random variable $X$ is the weighted average of the squared deviations of the possible values of $X$ from the mean of $X$, where the weights are the probabilities of the values of $X$.
$$ \sigma^2 = Var(X) = \int_{-\infty}^{\infty} (x - \mu)^2 f(x) dx = \int_{-\infty}^{\infty} x^2 f(x) dx - \mu^2 = E(X^2) - [E(X)]^2$$

The standard deviation of a continuous random variable $X$ is the square root of the variance of $X$.

%% [markdown]
# Joint probability distributions

## Joint probability distribution of two discrete random variables
The joint probability distribution of two discrete random variables $X$ and $Y$ is a list of each possible pair of values of $X$ and $Y$ together with the probability that $X$ takes the first value and $Y$ takes the second value in one trial of the experiment. It's also called the joint probability mass function $f(x, y) = P(X = x, Y = y)$.
The probabilities in the joint probability distribution of two discrete random variables $X$ and $Y$ must satisfy the following two conditions:
1. $0 \leq f(x,y) \leq 1$ for each possible pair of values $(x, y)$ of $X$ and $Y$
2. $\sum_{\text{all } x} \sum_{\text{all } y} f(x,y) = 1$

## Marginal probability distribution of a discrete random variable
The marginal probability distribution of a discrete random variable $X$ is the probability distribution of $X$ alone, regardless of the value of $Y$. It's also called the marginal probability mass function $f_X(x) = P(X = x)$.
$$ f_X(x) = \sum_{\text{all } y} f(x,y)$$
$$ f_Y(y) = \sum_{\text{all } x} f(x,y)$$

## Joint probability distribution of two continuous random variables
The joint probability distribution of two continuous random variables $X$ and $Y$ is an assignment of probabilities to regions of the $xy$-plane using a function $f(x,y)$, called a joint density function, in the following way: the probability that $(X, Y)$ assumes a value in the region $R$ is equal to the volume of the region that is bounded above by the graph of the equation $z=f(x,y)$, bounded below by the $xy$-plane, and bounded on the left and right by region $R$. 
The joint density function $f(x,y)$ must satisfy the following two conditions:
1. $f(x,y) \geq 0$ for all $(x,y)$
2. $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x,y) dx dy = 1$
3. $P((X,Y) \in R) = \int \int_R f(x,y) dx dy$

## Marginal probability distribution of a continuous random variable
The marginal probability distribution of a continuous random variable $X$ is the probability distribution of $X$ alone, regardless of the value of $Y$. It's also called the marginal probability density function $f_X(x)$.
$$ f_X(x) = \int_{-\infty}^{\infty} f(x,y) dy$$
$$ f_Y(y) = \int_{-\infty}^{\infty} f(x,y) dx$$

Cumulative probability distribution tables, when available, facilitate computation of probabilities encountered in typical practical situations.
- In place of $P(X = x)$, we can use $P(X \leq x) = P(X = 0) + P(X = 1) + \cdots + P(X = x)$
- Here, $P(X \geq x) = 1 - P(X < x) = 1 - P(X \leq x - 1)$
- and $P(x) = P(X \leq x) - P(X \leq x - 1)$

%% [markdown]
# Special probability distributions

## Uniform distribution
- Discrete uniform distribution: the discrete random variable $X$ that has a probability distribution given by the formula $$ P(X = x) = \frac{1}{n}$$ for $x = 1, 2, ..., n$ is said to have the discrete uniform distribution with parameter $n$.
- In the continuous case, the uniform distribution is a probability distribution wherein all intervals of the same length on the distribution's support are equally probable. The support is defined by the two parameters, $a$ and $b$, which are its minimum and maximum values. The distribution is often abbreviated $U(a, b)$.
$$ f(x) = \frac{1}{b - a} \text{ for } a \leq x \leq b $$
- The mean, variance of the uniform random variable $X$ with parameters $a$ and $b$ are given by the formulas:
    $$ \mu = E(X) = \frac{a + b}{2}$$
    $$ \sigma^2 = Var(X) = \frac{(b - a)^2}{12}$$


## Bernoulli distribution
$$ X \sim Bern(p)$$
- Models a single trial of a random experiment that has two possible outcomes, `success` or `failure`.
- The probability of `success` is $p$ and the probability of `failure` is $1 - p$.
- The discrete random variable $X$ that has a probability distribution given by the formula $$ P(X = x) = p^x (1 - p)^{1 - x} \text{ for } x = 0, 1$$ is said to have the Bernoulli distribution with parameter $p$.
- The mean, variance of the Bernoulli random variable $X$ with parameter $p$ are given by the formulas:
    $$ \mu = E(X) = p$$
    $$ \sigma^2 = Var(X) = p(1 - p)$$

## Binomial distribution
$$ X \sim Bin(n, p)$$
- The discrete random variable $X$ that counts the number of successes in $n$ identical, independent trials of a procedure that always results in either of two outcomes, `success` or `failure` and in which the probability of success on each trial is the same number $p$, is called the binomial random variable with parameters $n$ and $p$.
- There is a formula for the probability that the binomial random variable with parameters $n$ and $p$ will take a particular value $x$.
    $$ P(X = x) = \binom{n}{x} p^x (1 - p)^{n - x} = \frac{n!}{x!(n - x)!} p^x (1 - p)^{n - x} \text{ for } x = 0, 1, ..., n$$
- The mean, variance of the binomial random variable $X$ with parameters $n$ and $p$ are given by the formulas: (derivation in slides)
    $$ \mu = E(X) = np$$
    $$ \sigma^2 = Var(X) = np(1 - p)$$

## Poisson distribution
$$ X \sim Poisson(\lambda)$$
- The discrete random variable $X$ that counts the number of occurrences of an event over a specified interval of time or space is said to have the Poisson distribution with parameter $\lambda$.
- The probability that the Poisson random variable $X$ with parameter $\lambda$ will take a particular value $x$ is given by the formula: $$ P(X = x) = \frac{e^{-\lambda} \lambda^x}{x!} \text{ for } x = 0, 1, 2, ... \infty $$ where $e \approx 2.718$ and $\lambda$ is the average number of occurrences per interval.
- The mean, variance of the Poisson random variable $X$ with parameter $\lambda$ are given by the formulas:
    $$ \mu = E(X) = \lambda$$
    $$ \sigma^2 = Var(X) = \lambda$$
- The Poisson distribution is a limiting case of the binomial distribution when the number of trials $n$ is large and the probability of success $p$ is small.

## Normal distribution
$$ X \sim N(\mu, \sigma)$$
- The probability distribution corresponding to the density function for the bell curve with parameters $\mu$ and $\sigma$ is called the normal distribution with mean $\mu$ and standard deviation $\sigma$. A continuous random variable whose probabilities are described by the normal distribution with mean $\mu$ and standard deviation $\sigma$ is called a normally distributed random variable, or a normal random variable for short, with mean $\mu$ and standard deviation $\sigma$.
- The density curve for the normal distribution is symmetric about the mean $\mu$.
- The density curve for the normal distribution with mean $\mu$ and standard deviation $\sigma$ is given by the equation:
    $$ y = f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x - \mu}{\sigma})^2}$$
    where $\pi \approx 3.14159$ and $e \approx 2.71828$ 
- Standard normal distribution is the normal distribution with mean $\mu = 0$ and standard deviation $\sigma = 1$, denoted by $Z = N(0, 1)$.
    $$ y = f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}x^2}$$
    - rules for help computing $P(Z)$:
      - $P(Z \leq z) = P(Z < z) + P(Z = z) = P(Z < z)$
      - $P(Z \geq z) = 1 - P(Z < z)$
      - $P(z_1 \leq Z \leq z_2) = P(Z \leq z_2) - P(Z < z_1)$
- 68-95-99.7 rule: for any normal distribution, approximately 68% of the observations fall within one standard deviation of the mean, approximately 95% of the observations fall within two standard deviations of the mean, and approximately 99.7% of the observations fall within three standard deviations of the mean.
- If $X$ is a normally distributed random variable with mean $\mu$ and standard deviation $\sigma$, then
    $$P(X \leq a) = P\left(\frac{X - \mu}{\sigma} \leq \frac{a - \mu}{\sigma}\right) = P\left(Z \leq \frac{a - \mu}{\sigma}\right)$$
    $$P(a < X < b) = P\left( \frac{a - \mu}{\sigma} < Z < \frac{b - \mu}{\sigma} \right)$$
    where $Z$ is a standard normal random variable, and $a$ and $b$ are any two real numbers with $a < b$.
    - The new endpoints $\frac{a - \mu}{\sigma}$ and $\frac{b - \mu}{\sigma}$ are called the standard score or z-score of the original endpoints $a$ and $b$.
- normal approximation to the binomial distribution: if $X$ is a binomial random variable with parameters $n$ and $p$, then for large $n$, the distribution of $X$ is approximately normal with mean $\mu = np$ and standard deviation $\sigma = \sqrt{np(1 - p)}$ (if $n > 30$, $np > 15$, and $n(1 - p) > 15$
    - continuity correction: when approximating a discrete distribution with a continuous distribution, we can add or subtract $0.5$ to the endpoints of the interval to account for the fact that the continuous distribution is continuous and the discrete distribution is not.
    - $P(X \leq a) \approx P\left(Z \leq \frac{a + 0.5 - \mu}{\sigma}\right)$
    - $P(a < X < b) \approx P\left( \frac{a - 0.5 - \mu}{\sigma} < Z < \frac{b + 0.5 - \mu}{\sigma} \right)$

## t-distribution
$$ X \sim t(n)$$
- The t-distribution has the probability density function:
    $$ f(x) = \frac{\Gamma(\frac{n + 1}{2})}{\sqrt{n\pi} \Gamma(\frac{n}{2})} \left(1 + \frac{x^2}{n}\right)^{-\frac{n + 1}{2}}$$
    where $\Gamma$ is the gamma function and $n$ is the degrees of freedom.
- Used when working with small samples (less than 30) and when the population standard deviation is unknown.
- Similar to the standard normal distribution, but with heavier tails, accounting for the extra variability in small samples.
- Often used in hypothesis testing and confidence intervals for the mean of a population.
- The t-distibution arises as the sampling distribution of the t-statistic.
- Let $x_1, x_2, ..., x_n$ be a random sample from a normal distribution with mean $\mu$ and standard deviation $\sigma$. Then the random variable $$ t = \frac{\bar{x} - \mu}{s / \sqrt{n}}$$ has a t-distribution with $n - 1$ degrees of freedom, where $\bar{x}$ is the sample mean and $s$ is the unbiased sample standard deviation.

## Chi-square distribution
$$ X \sim \chi^2(n)$$
- The chi-square distribution with $n$ degrees of freedom is the distribution of the sum of the squares of $n$ independent standard normal random variables.
- The chi-square distribution has the probability density function:
    $$ f(x) = \frac{1}{2^{\frac{n}{2}} \Gamma(\frac{n}{2})} x^{\frac{n}{2} - 1} e^{-\frac{x}{2}}$$
    where $\Gamma$ is the gamma function and $n$ is the degrees of freedom.
- Not symmetric, skewed to the right, varies from $0$ to $\infty$.
- Depends on the degrees of freedom $n$.
- Used in hypothesis testing and confidence intervals for the variance of a population, measure of goodness of fit, and test of independence.

## F-distribution
$$ X \sim F(n_1, n_2)$$
- The F-distribution with $n_1$ and $n_2$ degrees of freedom is the distribution of the ratio of two independent chi-square random variables divided by their respective degrees of freedom.
- The F-distribution has the probability density function:
    $$ f(x) = \frac{\Gamma(\frac{n_1 + n_2}{2})}{\Gamma(\frac{n_1}{2}) \Gamma(\frac{n_2}{2})} \left(\frac{n_1}{n_2}\right)^{\frac{n_1}{2}} x^{\frac{n_1}{2} - 1} \left(1 + \frac{n_1}{n_2}x\right)^{-\frac{n_1 + n_2}{2}}$$
    where $\Gamma$ is the gamma function and $n_1$ and $n_2$ are the degrees of freedom.
- Not symmetric, skewed to the right, varies from $0$ to $\infty$.
- Depends on the degrees of freedom $n_1$ and $n_2$, and also the order in which they are written.

%% [markdown]
# Sampling
- population: the entire group of individuals or instances about whom we hope to learn
- sample: a subset of the population, examined in hope of learning about the population
- two types of sampling:
    1. random sampling: each individual is chosen randomly and entirely by chance
        1. simple random sampling: each individual has the same chance of being chosen
        2. stratified sampling: the population is divided into groups, called strata, and a random sample is taken from each stratum
            - potential to match the overall population's demographics better than simple random sampling
    2. non-random sampling: individuals are chosen by some non-random mechanism, and not by chance
        1. convenience sampling: individuals are chosen based on the ease of access
        2. snowball sampling: individuals are chosen based on referrals from other individuals
        3. quota sampling: individuals are chosen based on pre-specified quotas regarding demographics, etc.
        4. judgement sampling: individuals are chosen based on the judgement of the researcher
- sampling variability: the value of a statistic varies in repeated random sampling, it decreases as the sample size increases

# Sampling distributions

- The sampling distribution of a statistic is the probability distribution of the statistic when the statistic is computed from samples of the same size from the same population.
- There are formulas that relate the mean and standard deviation of the sample mean to the mean and standard deviation of the population from which the sample is drawn.
- For example, consider random variable $\bar{X}$, the sampling distribution of the sample mean, when the sample size is $n$. The mean of this r.v. is $\mu_{\bar{X}}$ and the standard deviation is $\sigma_{\bar{X}}$. Then:
    $$ \mu_{\bar{X}} = \mu$$
    $$ \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$$ 
    where $\mu$ and $\sigma$ are the mean and standard deviation of the population.
- The shape of the sampling distribution of $\bar{X}$ is approximately normal if the sample size is large enough.
- As $n$ increases, the shape of the sampling distribution of $\bar{X}$ becomes more and more like the shape of the normal distribution. The probabilities on the lower and upper ends shrink, and the probabilities in the middle become larger in relation. 
<img src="https://i.imgur.com/9lItK6A.jpg" width="400" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">

## Central limit theorem
- In general, one may start with any distribution and the sampling distribution of the sample mean will increasingly resemble the bell-shaped normal curve as the sample size increases. This is the content of the Central Limit Theorem.
- For sample sizes of 30 or more, the sampling distribution of the sample mean is approximately normal, regardless of the shape of the population distribution, with mean $\mu_{\bar{X}} = \mu$ and standard deviation $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$. The larger the sample size, the better the approximation.

<img src="https://i.imgur.com/0vzPLFF.jpg" width="600" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">

- The importance of CLT is that it allows us to make probability statements about the sample mean, in relation to its value in comparison to the population mean. 
Realize there are two distributions involved: 
1. $X$, the population distribution, mean $\mu$, standard deviation $\sigma$
2. $\bar{X}$, the sampling distribution of the sample mean, mean $\mu_{\bar{X}} = \mu$, standard deviation $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$

- For samples of any size drawn from a normal population, the sampling distribution of the sample mean is normal. For samples of any size drawn from a non-normal population, the sampling distribution of the sample mean is approximately normal if the sample size is 30 or more.
- If the size of the population is finite, then we apply a correction factor to the standard deviation of the sample mean:
    $$ \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}} \sqrt{\frac{N - n}{N - 1}}$$
    where $N$ is the population size and $n$ is the sample size.
    - thumb rule : if sample size is less than 5% of the population size, then we can ignore the correction factor.

## Sample proportion
- There are formulas that relate the mean and standard deviation of the sample proportion to the mean and standard deviation of the population from which the sample is drawn.
- sample proportion is the percentage of the sample that has a certain characteristic $\hat{p}$, as opposed to the population proportion $p$.
- viewed as a random variable, $\hat{P}$ has a mean $\mu_{\hat{P}}$ and a standard deviation $\sigma_{\hat{P}}$.
- if $np > 15$ and $n(1 - p) > 15$, then the relation to population proportion $p$:
    $$ \mu_{\hat{P}} = p$$
    $$ \sigma_{\hat{P}} = \sqrt{\frac{p(1 - p)}{n}}$$
- CLT applies to sample proportion as well, but the condition is more complex:
  - for large samples, sample proportion is normally distributed with mean $p$ and standard deviation $\sqrt{\frac{p(1 - p)}{n}}$.
  - to check if sample size is large enough, we need to check if $\left[ p - 3 \sigma_{\hat{P}}, p + 3 \sigma_{\hat{P}} \right]$ lies wholly within the interval $[0, 1]$.
    - since $p$ is unknown, we use $\hat{p}$ instead.
    - since $\sigma_{\hat{P}}$ is unknown, we use $\sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$ instead.

%% [markdown]
# Estimation

- The goal of estimation is to use sample data to estimate the value of an unknown population parameter.
- Point estimation : use a single value to estimate a population parameter
  - e.g. use sample mean $\bar{x}$ to estimate population mean $\mu$
  - problem: we don't know how reliable the estimate is
- Interval estimation : use an interval of values to estimate a population parameter
  - we use the data to compute $E$, such that $[\bar{x} - E, \bar{x} + E]$ has a certain probability of containing the population parameter $\mu$.
  - we do this in such a way that, $95\%$ of the all the intervals constructed from sample data will contain the population parameter $\mu$. 
  - $E$ is called the margin of error, and the interval is called the $95\%$ confidence interval for $\mu$.

The empirical rule states that you must go about 2 standard deviations in either direction from the mean to capture $95\%$ of the values of $\bar{X}$.

The key idea is that, in sample after sample $95\%$ of the values of $\bar{X}$ lie in the interval $[\mu - E, \mu + E]$. So if we adjoin to each side of the point estimate $x$ a wing of length E, $95\%$ of the time the wing will contain the population mean $\mu$.
- $95\%$ confidence interval is thus $\hat{x} \pm 1.960 \frac{\sigma}{\sqrt{n}}$
  - for a different confidence level, use a different multiplier instead of $1.960$.
  - Here, $1.960$ is the value of $z$ such that $P(-1.960 < Z < 1.960) = 0.95$, and is given by $z_{\alpha/2} = z_{0.025} = 1.960$, where $\alpha = 0.05$

<img src="https://i.imgur.com/u3ME5Qj.png" width="600" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">

In selecting the correct formula for construction of a confidence interval for a population mean ask two questions: is the population standard deviation $\sigma$ known or unknown, and is the sample large or small?


## Large sample $100(1 - \alpha)\%$ confidence interval for $\mu$
- if $\sigma$ is known:
    $$ \bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$
- if $\sigma$ is unknown:
    $$ \bar{x} \pm z_{\alpha/2} \frac{s}{\sqrt{n}}$$

A sample of size $n$ is large if $n \geq 30$ or if the population from which the sample is drawn is normal or approximately normal.
The number $E = z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$ or $E = z_{\alpha/2} \frac{s}{\sqrt{n}}$ is called the margin of error for the estimate $\bar{x}$ of $\mu$.

## Small sample $100(1 - \alpha)\%$ confidence interval for $\mu$

We use the Student's $t$ distribution instead of the $z$ distribution. The $t$ distribution is similar to the $z$ distribution, but it is more spread out. The spread increases as the degrees of freedom decrease. The $t$ distribution is symmetric and bell-shaped, but it has more area in the tails than the $z$ distribution.

- if $\sigma$ is known:
    $$ \bar{x} \pm t_{\alpha/2, n-1} \frac{\sigma}{\sqrt{n}}$$
- if $\sigma$ is unknown:
    $$ \bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$$ 
    with the degrees of freedom $df = n - 1$

The population must be normal or approximately normal. 

## Large sample estimation for population proportion $p$

- $100(1 - \alpha)\%$ confidence interval for $p$:
    $$ \hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$$

## Finding the minimum sample size

We have a population with mean $\mu$ and standard deviation $\sigma$. We want to estimate the population mean $\mu$ to within $E$ with $100(1 - \alpha)\%$ confidence. What is the minimum sample size $n$ required?
$$ n = \left( \frac{z_{\alpha/2} \sigma}{E} \right)^2 \text{rounded up}$$

For population proportion $p$, we have:
$$ n = \left( \frac{z_{\alpha/2} \sqrt{p(1 - p)}}{E} \right)^2 \text{rounded up}$$