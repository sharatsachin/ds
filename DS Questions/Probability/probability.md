# Statistics

## What is statistics? Descriptive and Inferential?
- study of learning from data
- involves collecting, organizing, analyzing, interpreting, and presenting data to make decisions about a population based on a sample.
  1. **descriptive**: summarizes and describes the characteristics of a data set.
  2. **inferential**: makes predictions and inferences about a population based on sample data.

## What are quantitative and qualitative variables?
- **qualitative**: variables not measured on a numeric scale
  - can be nominal (like gender, color) or ordinal (like rating, satisfaction level)
- **quantitative**: variables that are measured on a numeric scale
  - discrete or continuous
  - examples include age, height, weight, income, etc.

## What is the difference between ratio and interval variables?
- **interval**: have a true zero point
  - examples include temperature (Celsius, Fahrenheit), dates, and times
- **ratio**: have a true zero point
  - examples include height, weight, distance, and time

## What are the measures of central tendency?
- answer the question "where is the center of the data?"
- **mean**: average of the data points, calculated as $\mu = \bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$
- **median**: middle value of the data points when sorted in ascending order, $median = \begin{cases} x_{(n+1)/2} & \text{if } n \text{ is odd} \\ \frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ is even} \end{cases}$
- **mode**: value that appears most frequently in the data set

## What is skewness?
- measure of the asymmetry of the probability distribution of a real-valued random variable about its mean
- positive skew: data skewed to higher values (long tail on the right, mean > median > mode)
- negative skew: data skewed to lower values (long tail on the left, mean < median < mode)
- zero skew: data is symmetric (mean = median = mode)

## What are the measures of dispersion (variability)?
- answer the question "how spread out are the data points?"
- **range**: difference between the maximum and minimum values in a data set
- **variance**: average of the squared differences from the mean, calculated as $\sigma^2 = \frac{\sum_{i=1}^{n} (x_i - \mu)^2}{n}$
- **standard deviation**: square root of the variance
- **coefficient of variation**: ratio of the standard deviation to the mean
  - used to compare variability of data sets with different units
- **IQR**: Q3 - Q1, more robust to outliers than the range

## Why use ($n-1$) in the denominator for sample variance?
- use $n-1$ in the denominator instead of $n$ when calculating for a sample instead of a population
- for an unbiased estimation of the population variance, since adjustment accounts for the loss of one degree of freedom when estimating the sample mean (helps avoid underestimation of true population variance/std. dev)
- provides a more conservative estimate of the population variance, ensuring our stat. inference is more accurate

## Which are the 5 points of the 5-number summary?
- minimum, first quartile (Q1), median, third quartile (Q3), and maximum

## What is the Z-score?
- number of standard deviations a data point is from the mean
- used to understand the position of a data point in a distribution
- $z = \frac{x-\mu}{\sigma}$.

## Box plot, where is the median, Q1, Q3, whiskers, and outliers?
- median : center line of the box
- Q1, Q3 : lower and upper bounds of the box
- whiskers : lines extending from the box, representing the range of the data (usually Q1-1.5*IQR and Q3+1.5*IQR)
- outliers : data points beyond the whiskers

## What is the expirical rule?
- for dataset with approximately bell shaped relative frequency distribution, then:
  - $\bar{x} \pm \sigma$ : $68\%$
  - $\bar{x} \pm 2\sigma$ : $95\%$
  - $\bar{x} \pm 3\sigma$ : $99.7\%$

## What is the Chebyshev's theorem?
- percentage of data within $\bar{x} \pm k\sigma$ is at least $1 - \frac{1}{k^2}$, where $k > 1$
  - $\bar{x} \pm \sigma$ : at least $75\%$
  - $\bar{x} \pm 2\sigma$ : at least $89\%$

## What is hypothesis testing?
- method of statistical inference
- to check if evidence in a sample is enough to infer a condition is true for the entire population
- allows us to make probabilistic statements about a population parameter based on a statistic computed from a sample

## What is the null hypothesis?
- $H_0$
- status quo, says there is no change or difference from what you already know
- hypothesis of "no difference between specified populations, any observed difference being due to sampling or experimental error"

## What is the alternate hypothesis?
- $H_a$
- hypothesis that we are trying to prove
- hypothesis that "sample observations are influenced by some non-random cause, and that the observed difference between the sample and population reflects this cause"

## What is a type I error?
- occurs when the null hypothesis is true but is rejected
- FP, asserting something that is absent
- probability of making a type I error is $\alpha$

## What is a type II error?
- occurs when the null hypothesis is false but erroneously fails to be rejected
- FN, failing to assert what is present
- probability of making a type II error is $\beta$

## Null Hypothesis (True vs False) vs Decision (Reject vs Fail to Reject) - Fill in the table
| | Null Hypothesis True | Null Hypothesis False |
| --- | --- | --- |
| **Reject** | Type I Error (FP) p = $\alpha$ | Correct Decision (TP) p = $1-\beta$ |
| **Fail to Reject** | Correct Decision (TN) p = $1-\alpha$ | Type II Error (FN) p = $\beta$ |

## What is the level of significance ($\alpha$)?
- probability of rejecting the null hypothesis when it is true (type I error $\alpha$, usually $0.05$ or $0.01$)

## What is the power of a test?
- probability of rejecting the null hypothesis when it is false ($ 1 - \beta $)
- probability of not making a type II error

## What is the graph of the relation between Type I and Type II errors?

<img alt="picture 0" src="https://cdn.jsdelivr.net/gh/sharatsachin/images-cdn@master/images/ef13633fcd2944eb7836a0202410da64de5ac27e44faa3f77221d0d1d98dc060.png" width="500" />  

## How is the power of a test affected by the standard deviation, the sample size and the effect size?
- Larger the standard deviation, lower the power
- Larger the sample size, higher the power
- Larger the effect size, higher the power

## What is a t-test? When would you use it?
- [link](https://en.wikipedia.org/wiki/Student%27s_t-test)
- used in hypothesis testing to determine whether:
  - a process or treatment actually has an effect on the population of interest
  - two groups are different from one another

## What is a f-test? When would you use it?
- [link](https://en.wikipedia.org/wiki/F-test)
- used to compare statistical models that have been fitted to a data set
- used to identify the model that best fits the population from which the data were sampled

## What is a p-value? What is its importance?
- $P(\text{observing test results at least as extreme as the results observed} | H_0)$
- smaller the $p$-value, greater the statistical significance of the observed difference
- used to quantify the statistical significance of the obtained results
- if $p < \alpha$, reject $H_0$

## Different ways to sample from a dataset
- **simple random** : each item in the population has an equal chance of being selected
- **stratified** : the population is divided into strata and a random sample is taken from each stratum
- **cluster** : the population is divided into clusters and a random sample of clusters are selected
- **systematic** : the sample is chosen by selecting a random starting point and then picking every $k$th element in the population
- **multistage** : a combination of two or more of the above methods

## What is the Central Limit Theorem?
- [link](https://www.probabilitycourse.com/chapter7/7_1_2_central_limit_theorem.php)
- establishes that, in some situations, when independent random variables are added, their properly normalized sum tends toward a normal distribution
- even if the original variables themselves are not normally distributed

## What is the Law of Large Numbers?
- [link](https://en.wikipedia.org/wiki/Law_of_large_numbers)
- describes the result of performing the same experiment a large number of times
- the average of the results obtained from a large number of trials should be close to the expected value

## How can we identify skewness in a distribution using mean, median and mode?
- if mean < median < mode, the distribution is negatively skewed
- if mean > median > mode, the distribution is positively skewed
- if mean = median = mode, the distribution is symmetric

## What is correlation and covariance, and how are they different?
- **correlation** : a measure of the strength and direction of the linear relationship between two variables
  - ranges from -1 to 1
  - positive correlation: both variables move in the same direction
  - negative correlation: variables move in opposite directions
- **covariance** : a measure of the joint variability of two random variables
  - positive covariance: variables tend to show similar behavior
  - negative covariance: variables tend to show opposite behavior
  - in the units of the product of the units of the two variables

## What is the pearson correlation coefficient?
$$ r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}} $$
ranges from -1 to 1, where:
- $r = 1$ indicates a perfect positive linear relationship
- $r = -1$ indicates a perfect negative linear relationship
- $r = 0$ indicates no linear relationship

## How can you find correlation between categorical and numerical columns?
- **Cramer's V** for two categorical variables
- **Point biserial correlation or $\eta$-squared** for one categorical and one numerical variable

## What is the confidence score and confidence interval?
- **confidence score** : the probability that the value of a parameter falls within a specified range of values
- **confidence interval** : a range of values, derived from the sample statistics, that is likely to contain the value of an unknown population parameter

## What is a probability? What are empirical and theoretical probabilities?
- numerical measure of the likelihood that an event will occur
- $P(A) = \frac{\text{number of favorable outcomes}}{\text{number of possible outcomes}}$
- **empirical probability** : the relative frequency of an event occurring in a series of trials
- **theoretical probability** : the probability of an event occurring based on mathematical reasoning

## What is a population and a sample?
- **population** : any specific collection of objects of interest
- **sample** : any subset or subcollection of the population, including the case that the sample consists of the whole population

## What is a measurement?
- a number or attribute computed for each member of a population or of a sample

## What is a parameter and a statistic?
- **parameter** : a number that summarizes some aspect of the population as a whole (fixed and unknown)
- **statistic** : a number computed from the sample data (varies randomly from sample to sample)
  - conclusions made about population parameters are statements of probability

## What is a random experiment?
- actions that occur by chance, and their outcomes are not predictable

## What is a sample space?
- the set of all possible outcomes of a random event
- discrete sample space: a sample space with a finite number of outcomes

## What is an event? What is the complement of an event?
- a subset of the sample space of a random experiment
- the complement of an event is the set of all outcomes in the sample space that are not in the event

## What is the union and intersection of two events?
- **union** : the set of all outcomes that are in either event
- **intersection** : the set of all outcomes that are in both events

## What are mutually exclusive and independent events?
- **mutually exclusive** : events that have no outcomes in common
- **independent** : events that have no effect on each other

## What is the addition rule?
- the probability of the union of two events is equal to the sum of the probabilities of the individual events minus the probability of their intersection
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

## What are the conditions of probability values?
- $0 \leq P(A) \leq 1$
- $P(S) = 1$
- For two events $A$ and $B$, if $A$ and $B$ are mutually exclusive, then $P(A \cup B) = P(A) + P(B)$

## What is conditional probability?
- the probability of an event occurring given that another event has already occurred
- $P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{P(A)P(B|A)}{P(B)}$
- Also, $P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$

## What is the multiplication rule?
- $P(A \cap B \cap C) = P(A|B \cap C)P(B \cap C) = P(A|B \cap C)P(B|C)P(C)$
- can be extended to any number of events

## What are independent events?
- if $P(A|B) = P(A)$, then $A$ and $B$ are independent events, and the occurrence of $B$ has no effect on the likelihood of $A$
- $P(A|B) = P(A)$ if and only if $P(A \cap B) = P(A)P(B)$
- independence does not imply disjointness

## What is the law of total probability?
- if $A_1, A_2, ..., A_n$ are mutually exclusive and exhaustive events, then for any event $B$, $$P(B) = P(B|A_1)P(A_1) + P(B|A_2)P(A_2) + ... + P(B|A_n)P(A_n)$$

## What is Bayes' theorem?
- used to update the probability for a hypothesis as more evidence or information becomes available
- let $P = {A_1, A_2, ..., A_n}$ be a partition of the sample space $S$ of a random experiment, and let $B$ be an event such that $P(B) > 0$, then for any $i = 1, 2, ..., n$, $$P(A_i|B) = \frac{P(A_i)P(B|A_i)}{P(A_1)P(B|A_1) + P(A_2)P(B|A_2) + ... + P(A_n)P(B|A_n)}$$

## What is Bayesian learning?
- a method of statistical inference in which Bayes' theorem is used to update the probability for a hypothesis as more evidence or information becomes available
- Naive Bayes classifier is a simple probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features

## What is the Naive Bayes classifier?
- a simple probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features
- used for text classification, spam filtering, sentiment analysis, and recommender systems
- we have a set of features $X = {X_1, X_2, ..., X_n}$ and a class variable $Y$
- we want to find the class $Y$ that maximizes the posterior probability $P(Y|X)$
- assuming conditional independence, we have $P(X_1, X_2, ..., X_n|Y = y_k) = \prod_{i=1}^n P(X_i|Y = y_k)$
- then we pick the most probable class: $\hat{y} = \arg\max_{y_k} P(Y = y_k)\prod_{i} P(X_i|Y = y_k)$

## What are the advantages of the Naive Bayes classifier?
- simple and easy to implement, fast and efficient
- performs well on large datasets, good choice when the dimensionality of the inputs is high
- often used for text classification, spam filtering, sentiment analysis, and recommender systems

## What are the disadvantages of the Naive Bayes classifier?
- assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature
- is a linear classifier, which means it assumes that the data is linearly separable

## How can you describe Bayes theorem in the context of machine learning?
- used to calculate the probability of a hypothesis given our prior knowledge
- $P(h|D) = \frac{P(D|h)P(h)}{P(D)}$
  - $P(h)$ is the prior probability of $h$ being true
  - $P(D)$ is the prior probability of $D$ being true
  - $P(h|D)$ is the posterior probability of $h$ being true given $D$
  - $P(D|h)$ is the likelihood of $D$ given $h$

## What is maximum a posteriori (MAP) estimation?
- a method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the posterior probability
- $h_{MAP} = \arg\max\limits_{h \in H} P(D|h)P(h) = \arg\max\limits_{h \in H} P(D|h)P(h)$
- we find the parameter values that make the observed data most probable given our prior knowledge about the parameters

## What is maximum likelihood (ML) estimation?
- a method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the likelihood of the observed data
- $h_{ML} = \arg\max\limits_{h \in H} P(D|h)$
- we find the parameter values that make the observed data most probable
- used when we have no prior knowledge about the parameters

## What is conditional independence?
- independence under the probability law $P(·|C)$
- $A$ and $B$ are conditionally independent given $C$ if and only if $P(A \cap B|C) = P(A|C)P(B|C)$
- Naive Bayes assumes that all features are conditionally independent given the class

## Does independence imply conditional independence?
- independence does not imply conditional independence [(link)](https://www.youtube.com/watch?v%253DTAyA-rjmesQ%2526list%253DPLUl4u3cNGP60hI9ATjSFgLZpbNJ7myAg6%2526index%253D35)
- means that $P(A|B,C) = P(A|C)$ and $P(A \cap B|C) = P(A|C)P(B|C)$

<img src="https://i.imgur.com/O7op64y.jpg" width="300" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">

## How can we use naive Bayes for text classification?
- given a document $d$, a set of classes $C = {c_1, c_2, ..., c_n}$, and a set of $m$ hand-labeled documents $(d_1, c_1), (d_2, c_2), ..., (d_m, c_m)$
- we want to find the class $c$ that maximizes the posterior probability $P(c|d)$
- $P(c|d) = \frac{P(c)P(d|c)}{P(d)} = \frac{P(c)\prod_{i=1}^n P(w_i|c)}{P(d)}$
- we pick the most probable class: $c_{MAP} = \arg\max_{c} P(c)\prod_{i=1}^n P(w_i|c)$
- $P(c_j) = \frac{docCount(C = c_j)}{N_{doc}}$
- $P(w_i|c_j) = \frac{wordCount(w_i, C = c_j)}{\sum_{w \in V} wordCount(w, C = c_j)}$

## What is the Laplace smoothing technique?
- used to handle the problem of zero probability in Naive Bayes
- Laplace smoothing: $P(w_i|c_j) = \frac{wordCount(w_i, C = c_j) + 1}{\sum_{w \in V} wordCount(w, C = c_j) + |V|}$

## Solved example of Naive Bayes for text classification

[Example](https://www.fi.muni.cz/~sojka/PV211/p13bayes.pdf):
<img src="https://i.imgur.com/p3nZUNM.png" width="500" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">
<img src="https://i.imgur.com/kcNsCro.png" width="500" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">

Therefore, $$P(C|d_5) = \frac{3}{4} {(\frac{3}{7})}^3 \frac{1}{14} \frac{1}{14} \frac{1}{P(d_5)}$$
and $$P(\bar{C} | d_5) = \frac{1}{4} {(\frac{2}{9})}^3 \frac{2}{9} \frac{2}{9} \frac{1}{P(d_5)}$$

$P(d_5)$ is the same for both classes, so we can ignore it.

## What are random variables?
- random variables are variables that take on numerical values based on the outcome of a random experiment
- discrete random variables: random variables that can take on a finite number of values
- continuous random variables: random variables that can take on an infinite number of values

## What is the probability distribution of discrete random variables?
- the probability distribution of a discrete random variable is a list of each possible value of the random variable together with the probability that the random variable takes that value in one trial of the experiment
- the probabilities in the probability distribution of a discrete random variable must satisfy the following two conditions:
  1. $0 \leq P(X = x) \leq 1$ for each possible value $x$ of $X$
  2. $\sum_{\text{all } x} P(X = x) = 1$

## How would you describe the probability distribution of the sum of two dice?
- the probability distribution of the sum of two dice is given by:
$$\begin{array}{c|ccccccccccc} x &2 &3 &4 &5 &6 &7 &8 &9 &10 &11 &12 \\ \hline P(x) &\dfrac{1}{36} &\dfrac{2}{36} &\dfrac{3}{36} &\dfrac{4}{36} &\dfrac{5}{36} &\dfrac{6}{36} &\dfrac{5}{36} &\dfrac{4}{36} &\dfrac{3}{36} &\dfrac{2}{36} &\dfrac{1}{36} \\ \end{array}$$
- $P(X \geq 9) = P(X = 9) + P(X = 10) + P(X = 11) + P(X = 12) = \dfrac{10}{36} = \dfrac{5}{18}$
- $P(\text{X is even}) = P(X = 2) + P(X = 4) + P(X = 6) + P(X = 8) + P(X = 10) + P(X = 12) = \dfrac{18}{36} = \dfrac{1}{2}$

## What is the mean of a discrete random variable?
- the mean / expected value $E(X)$ of a discrete random variable $X$ is the weighted average of the possible values of $X$, where the weights are the probabilities of the values of $X$
- $E(X) = \sum_{\text{all } x} xP(x)$

## What are the properties of the mean of a discrete random variable?
- $E(aX + b) = aE(X) + b$
- $E(X + Y) = E(X) + E(Y)$
- $E(XY) = E(X)E(Y)$ if $X$ and $Y$ are independent

## What is the variance of a discrete random variable?
- the variance $Var(X)$ of a discrete random variable $X$ is the weighted average of the squared deviations of the possible values of $X$ from the mean of $X$, where the weights are the probabilities of the values of $X$
- $Var(X) = E(X^2) - [E(X)]^2 = \sum(x - \mu)^2 P(x)$

## What are the properties of the variance of a discrete random variable?
- $Var(aX + b) = a^2Var(X)$
- $Var(X + Y) = Var(X) + Var(Y)$ if $X$ and $Y$ are independent

## What is the mean and variance of a R.V. for the sum of two dice?
- the mean of the sum of two dice is $E(X) = \sum_{\text{all } x} xP(x) = 7$
- the variance of the sum of two dice is $Var(X) = E(X^2) - [E(X)]^2 = 5.83$

## How is the probability distribution of a continuous random variable defined?
- the probability distribution of a continuous random variable is an assignment of probabilities to intervals of decimal numbers using a function $f(x)$, called a density function
- this means that the probability that the random variable assumes a value in the interval $[a, b]$ is equal to the area of the region that is bounded above by the graph of the equation $y=f(x)$, bounded below by the x-axis, and bounded on the left and right by the vertical lines through $a$ and $b$

## What are the conditions of the probability density function of a continuous random variable?
- $f(x) \geq 0$ for all $x$
- $\int_{-\infty}^{\infty} f(x) dx = 1$
- $P(a \leq X \leq b) = \int_a^b f(x) dx$

## What is the cumulative probability distribution function of a continuous random variable?
- the cumulative probability distribution function of a continuous random variable is the function $F(x)$ defined by $F(x) = P(X \leq x)$ for all $x$
- $F(x) = \int_{-\infty}^x f(u) du$

## How do we get the probability density function from the cumulative probability distribution function?
- to get the probability density function $f(x)$ from the cumulative probability distribution function $F(x)$, we differentiate $F(x)$ with respect to $x$
- $f(x) = \frac{d}{dx} F(x)$

## What is the mean of a continuous random variable?
- the mean / expected value $E(X)$ of a continuous random variable $X$ is the weighted average of the possible values of $X$, where the weights are the probabilities of the values of $X$
- $$ \mu = E(X) = \int_{-\infty}^{\infty} xf(x) dx$$

## What is the variance and standard deviation of a continuous random variable?
- the variance $Var(X)$ of a continuous random variable $X$ is the weighted average of the squared deviations of the possible values of $X$ from the mean of $X$, where the weights are the probabilities of the values of $X$
- $$ \sigma^2 = Var(X) = \int_{-\infty}^{\infty} (x - \mu)^2 f(x) dx = \int_{-\infty}^{\infty} x^2 f(x) dx - \mu^2 = E(X^2) - [E(X)]^2$$

## What is the joint probability distribution of two discrete random variables?
- the joint probability distribution of two discrete random variables $X$ and $Y$ is a list of each possible pair of values of $X$ and $Y$ together with the probability that $X$ takes the first value and $Y$ takes the second value in one trial of the experiment
- the probabilities in the joint probability distribution of two discrete random variables $X$ and $Y$ must satisfy the following two conditions:
  1. $0 \leq f(x,y) \leq 1$ for each possible pair of values $(x, y)$ of $X$ and $Y$
  2. $\sum_{\text{all } x} \sum_{\text{all } y} f(x,y) = 1$

## What is the marginal probability distribution of a discrete random variable?
- the marginal probability distribution of a discrete random variable $X$ is the probability distribution of $X$ alone, regardless of the value of $Y$
- $$ f_X(x) = \sum_{\text{all } y} f(x,y)$$
- $$ f_Y(y) = \sum_{\text{all } x} f(x,y)$$

## What is the joint probability distribution of two continuous random variables?
- the joint probability distribution of two continuous random variables $X$ and $Y$ is an assignment of probabilities to regions of the $xy$-plane using a function $f(x,y)$, called a joint density function
- the joint density function $f(x,y)$ must satisfy the following two conditions:
  1. $f(x,y) \geq 0$ for all $(x,y)$
  2. $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x,y) dx dy = 1$
  3. $P((X,Y) \in R) = \int \int_R f(x,y) dx dy$

## What is the marginal probability distribution of a continuous random variable?
- the marginal probability distribution of a continuous random variable $X$ is the probability distribution of $X$ alone, regardless of the value of $Y$
- $$ f_X(x) = \int_{-\infty}^{\infty} f(x,y) dy$$
- $$ f_Y(y) = \int_{-\infty}^{\infty} f(x,y) dx$$

## What are cumulative probability distribution tables?
- facilitate computation of probabilities encountered in typical practical situations
- in place of $P(X = x)$, we can use $P(X \leq x) = P(X = 0) + P(X = 1) + \cdots + P(X = x)$
- $P(X \geq x) = 1 - P(X < x) = 1 - P(X \leq x - 1)$
- $P(x) = P(X \leq x) - P(X \leq x - 1)$

## What is the uniform probability distribution?
- discrete uniform distribution: the discrete random variable $X$ that has a probability distribution given by the formula $P(X = x) = \frac{1}{n}$ for $x = 1, 2, ..., n$ is said to have the discrete uniform distribution with parameter $n$
- in the continuous case, the uniform distribution is a probability distribution wherein all intervals of the same length on the distribution's support are equally probable
- $f(x) = \frac{1}{b - a}$ for $a \leq x \leq b$

## What is the mean of the uniform random variable?
- the mean of the uniform random variable $X$ with parameters $a$ and $b$ is given by the formula:
  $$ \mu = E(X) = \frac{a + b}{2}$$

## What is the variance of the uniform random variable?
- the variance of the uniform random variable $X$ with parameters $a$ and $b$ is given by the formula:
  $$ \sigma^2 = Var(X) = \frac{(b - a)^2}{12}$$

## What is the Bernoulli distribution?
- models a single trial of a random experiment that has two possible outcomes, `success` or `failure`
- the probability of `success` is $p$ and the probability of `failure` is $1 - p$
- the discrete random variable $X$ that has a probability distribution given by the formula $P(X = x) = p^x (1 - p)^{1 - x}$ for $x = 0, 1$ is said to have the Bernoulli distribution with parameter $p$

## What is the mean of the Bernoulli random variable?
- the mean of the Bernoulli random variable $X$ with parameter $p$ is given by the formula:
  $$ \mu = E(X) = p$$

## What is the variance of the Bernoulli random variable?
- the variance of the Bernoulli random variable $X$ with parameter $p$ is given by the formula:
  $$ \sigma^2 = Var(X) = p(1 - p)$$

## What is the Binomial distribution?
- discrete probability distribution of the number of successes in a sequence of $n$ independent experiments
- each asking a yes–no question, and each with its own boolean-valued outcome: $P(\text{success}) = p$ and $P(\text{failure}) = q=1-p$
- $f(k;n,p) = \Pr(X = k) = \Pr(k \text{ successes in } n \text{ trials}) = \binom{n}{k} p^k (1-p)^{n-k}$

## What is the mean of the Binomial random variable?
- the mean of the Binomial random variable $X$ with parameters $n$ and $p$ is given by the formula:
  $$ \mu = E(X) = np$$

## What is the variance of the Binomial random variable?
- the variance of the Binomial random variable $X$ with parameters $n$ and $p$ is given by the formula:
  $$ \sigma^2 = Var(X) = np(1 - p)$$

## What is the Poisson distribution?
- discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time
- $f(k; \lambda) = \Pr(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$

## What is the mean of the Poisson random variable?
- the mean of the Poisson random variable $X$ with parameter $\lambda$ is given by the formula:
  $$ \mu = E(X) = \lambda$$

## What is the variance of the Poisson random variable?
- the variance of the Poisson random variable $X$ with parameter $\lambda$ is given by the formula:
  $$ \sigma^2 = Var(X) = \lambda$$

## What is the Beta distribution?
- family of continuous probability distributions defined on the interval $[0,1]$
- parametrized by two positive shape parameters, denoted by $\alpha$ and $\beta$
- used to model the uncertainty about the probability of success of an experiment

## What is the expected value of a Beta random variable?
- the expected value of a Beta random variable $X$ with parameters $\alpha$ and $\beta$ is given by the formula:
  $$ \mu = E(X) = \frac{\alpha}{\alpha+\beta}$$

## What is the variance of a Beta random variable?
- the variance of a Beta random variable $X$ with parameters $\alpha$ and $\beta$ is given by the formula:
  $$ \sigma^2 = Var(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

## What is the normal distribution?
- the probability distribution corresponding to the density function for the bell curve with parameters $\mu$ and $\sigma$ is called the normal distribution with mean $\mu$ and standard deviation $\sigma$
- the density curve for the normal distribution is symmetric about the mean $\mu$
- the density curve for the normal distribution with mean $\mu$ and standard deviation $\sigma$ is given by the equation:
  $$ y = f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x - \mu}{\sigma})^2}$$

## What is the standard normal distribution?
- the standard normal distribution is the normal distribution with mean $\mu = 0$ and standard deviation $\sigma = 1$, denoted by $Z = N(0, 1)$
- the density curve for the standard normal distribution is given by the equation:
  $$ y = f(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}x^2}$$

## What is the mean of the standard normal random variable?
- the mean of the standard normal random variable $Z$ is given by the formula:
  $$ \mu = E(Z) = 0$$

## What is the variance of the standard normal random variable?
- the variance of the standard normal random variable $Z$ is given by the formula:
  $$ \sigma^2 = Var(Z) = 1$$

## What are some rules for help computing $P(Z)$?
- $P(Z \leq z) = P(Z < z) + P(Z = z) = P(Z < z)$
- $P(Z \geq z) = 1 - P(Z < z)$
- $P(z_1 \leq Z \leq z_2) = P(Z \leq z_2) - P(Z < z_1)$

## If $X$ is a normally distributed random variable with mean $\mu$ and standard deviation $\sigma$, then a) $P(X \leq a)$ and b) $P(a < X < b)$ can be computed as:
- $P(X \leq a) = P\left(\frac{X - \mu}{\sigma} \leq \frac{a - \mu}{\sigma}\right) = P\left(Z \leq \frac{a - \mu}{\sigma}\right)$
- $P(a < X < b) = P\left( \frac{a - \mu}{\sigma} < Z < \frac{b - \mu}{\sigma} \right)$
- the new endpoints $\frac{a - \mu}{\sigma}$ and $\frac{b - \mu}{\sigma}$ are called the standard score or z-score of the original endpoints $a$ and $b$

## What is the normal approximation to the binomial distribution?
- if $X$ is a binomial random variable with parameters $n$ and $p$, then for large $n$, the distribution of $X$ is approximately normal with mean $\mu = np$ and standard deviation $\sigma = \sqrt{np(1 - p)}$ (if $n > 30$, $np > 15$, and $n(1 - p) > 15$

## What is the continuity correction?
- when approximating a discrete distribution with a continuous distribution, we can add or subtract $0.5$ to the endpoints of the interval to account for the fact that the continuous distribution is continuous and the discrete distribution is not
- $P(X \leq a) \approx P\left(Z \leq \frac{a + 0.5 - \mu}{\sigma}\right)$
- $P(a < X < b) \approx P\left( \frac{a - 0.5 - \mu}{\sigma} < Z < \frac{b + 0.5 - \mu}{\sigma} \right)$

## What is the t-distribution?
- the t-distribution is a family of distributions that arise when estimating the mean of a normally distributed population in situations where the sample size is small and the population standard deviation is unknown
- similar to the standard normal distribution, but with heavier tails, accounting for the extra variability in small samples
- often used in hypothesis testing and confidence intervals for the mean of a population

## When does the t-distribution arise?
- arises as the sampling distribution of the t-statistic
- let $x_1, x_2, ..., x_n$ be a random sample from a normal distribution with mean $\mu$ and standard deviation $\sigma$. Then the random variable $$ t = \frac{\bar{x} - \mu}{s / \sqrt{n}}$$ has a t-distribution with $n - 1$ degrees of freedom

## What is the chi-square distribution?
- the chi-square distribution with $n$ degrees of freedom is the distribution of the sum of the squares of $n$ independent standard normal random variables
- not symmetric, skewed to the right, varies from $0$ to $\infty$
- depends on the degrees of freedom $n$
- used in hypothesis testing and confidence intervals for the variance of a population, measure of goodness of fit, and test of independence

## What is the F-distribution?
- the F-distribution with $n_1$ and $n_2$ degrees of freedom is the distribution of the ratio of two independent chi-square random variables divided by their respective degrees of freedom
- not symmetric, skewed to the right, varies from $0$ to $\infty$
- depends on the degrees of freedom $n_1$ and $n_2$, and also the order in which they are written

## What are some ways of random sampling?
- random sampling: each individual is chosen randomly and entirely by chance
  - simple random sampling: each individual has the same chance of being chosen
  - stratified sampling: the population is divided into groups, called strata, and a random sample is taken from each stratum

## What are some ways of non-random sampling?
- non-random sampling: individuals are chosen by some non-random mechanism, and not by chance
  - convenience sampling: individuals are chosen based on the ease of access
  - snowball sampling: individuals are chosen based on referrals from other individuals
  - quota sampling: individuals are chosen based on pre-specified quotas regarding demographics, etc.
  - judgement sampling: individuals are chosen based on the judgement of the researcher

## What is sampling variability?
- the value of a statistic varies in repeated random sampling, it decreases as the sample size increases

## What is the sampling distribution of a statistic?
- the probability distribution of a statistic when the statistic is computed from samples of the same size from the same population

## What are the formulas that relate the mean and standard deviation of the sample mean to the mean and standard deviation of the population?
- for random variable $\bar{X}$, the sampling distribution of the sample mean, when the sample size is $n$, the mean and standard deviation are:
  $$ \mu_{\bar{X}} = \mu$$
  $$ \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$$

## What is the shape of the sampling distribution of the sample mean?
- the shape of the sampling distribution of the sample mean is approximately normal if the sample size is large enough

## What is the sampling distribution of the t-statistic?
- arises when estimating the mean of a normally distributed population in situations where the sample size is small and the population standard deviation is unknown

## What is the importance of the Central Limit Theorem?
- the importance of the Central Limit Theorem is that it allows us to make probability statements about the sample mean, in relation to its value in comparison to the population mean
- there are two distributions involved: 
  1. $X$, the population distribution, mean $\mu$, standard deviation $\sigma$
  2. $\bar{X}$, the sampling distribution of the sample mean, mean $\mu_{\bar{X}} = \mu$, standard deviation $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$

## What are the conditions under which the Central Limit Theorem applies?
- for samples of any size drawn from a normal population, the sampling distribution of the sample mean is normal
- for samples of any size drawn from a non-normal population, the sampling distribution of the sample mean is approximately normal if the sample size is $30$ or more

## What is the formula for the standard deviation of the sample mean when the size of the population is finite?
- if the size of the population is finite, then we apply a correction factor to the standard deviation of the sample mean:
  $$ \sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}} \sqrt{\frac{N - n}{N - 1}}$$
  where $N$ is the population size and $n$ is the sample size
  - thumb rule: if the sample size is less than $5\%$ of the population size, then we can ignore the correction factor

## What is the sampling distribution of the sample proportion?
- the sample proportion is the percentage of the sample that has a certain characteristic $\hat{p}$, as opposed to the population proportion $p$
- viewed as a random variable, $\hat{P}$ has a mean $\mu_{\hat{P}}$ and a standard deviation $\sigma_{\hat{P}}$
- if $np > 15$ and $n(1 - p) > 15$, then the relation to the population proportion $p$ is:
  $$ \mu_{\hat{P}} = p$$
  $$ \sigma_{\hat{P}} = \sqrt{\frac{p(1 - p)}{n}}$$

## What are the conditions under which the Central Limit Theorem applies to the sample proportion?
- the Central Limit Theorem applies to the sample proportion as well, but the condition is more complex
- for large samples, the sample proportion is normally distributed with mean $p$ and standard deviation $\sqrt{\frac{p(1 - p)}{n}}$
- to check if the sample size is large enough, we need to check if $\left[ p - 3 \sigma_{\hat{P}}, p + 3 \sigma_{\hat{P}} \right]$ lies wholly within the interval $[0, 1]$
  - since $p$ is unknown, we use $\hat{p}$ instead
  - since $\sigma_{\hat{P}}$ is unknown, we use $\sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$ instead

## What is estimation?
- the goal of estimation is to use sample data to estimate the value of an unknown population parameter
- point estimation: use a single value to estimate a population parameter
  - e.g. use sample mean $\bar{x}$ to estimate population mean $\mu$
  - problem: we don't know how reliable the estimate is

## What is interval estimation?
- interval estimation: use an interval of values to estimate a population parameter
- we use the data to compute $E$, such that $[\bar{x} - E, \bar{x} + E]$ has a certain probability of containing the population parameter $\mu$
- we do this in such a way that $95\%$ of all the intervals constructed from sample data will contain the population parameter $\mu$
- $E$ is called the margin of error, and the interval is called the $95\%$ confidence interval for $\mu$

## What is the empirical rule?
- the empirical rule states that you must go about $2$ standard deviations in either direction from the mean to capture $95\%$ of the values of $\bar{X$

## What is the formula for the $95\%$ confidence interval for the population mean?
- the $95\%$ confidence interval is $\hat{x} \pm 1.960 \frac{\sigma}{\sqrt{n}}$
  - for a different confidence level, use a different multiplier instead of $1.960$
  - here, $1.960$ is the value of $z$ such that $P(-1.960 < Z < 1.960) = 0.95$, and is given by $z_{\alpha/2} = z_{0.025} = 1.960$, where $\alpha = 0.05$

<img src="https://i.imgur.com/u3ME5Qj.png" width="600" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">

## What is the formula for the large sample $100(1 - \alpha)\%$ confidence interval for the population mean?
- if $\sigma$ is known:
  $$ \bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$
- if $\sigma$ is unknown:
  $$ \bar{x} \pm z_{\alpha/2} \frac{s}{\sqrt{n}}$$

## When is a sample considered large?
- a sample of size $n$ is large if $n \geq 30$ or if the population from which the sample is drawn is normal or approximately normal

## What is the margin of error?
- the number $E = z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$ or $E = z_{\alpha/2} \frac{s}{\sqrt{n}}$ is called the margin of error for the estimate $\bar{x}$ of $\mu$

## What is the formula for the small sample $100(1 - \alpha)\%$ confidence interval for the population mean?
- if $\sigma$ is known:
  $$ \bar{x} \pm t_{\alpha/2, n-1} \frac{\sigma}{\sqrt{n}}$$
- if $\sigma$ is unknown:
  $$ \bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$$
  with the degrees of freedom $df = n - 1$

## What is the formula for the large sample estimation for the population proportion?
- $100(1 - \alpha)\%$ confidence interval for $p$:
  $$ \hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$$

## What is the formula for finding the minimum sample size for the population mean?
- we have a population with mean $\mu$ and standard deviation $\sigma$. We want to estimate the population mean $\mu$ to within $E$ with $100(1 - \alpha)\%$ confidence. What is the minimum sample size $n$ required?
  $$ n = \left( \frac{z_{\alpha/2} \sigma}{E} \right)^2 \text{rounded up}$$

## What is the formula for finding the minimum sample size for the population proportion?
- for population proportion $p$, we have:
  $$ n = \left( \frac{z_{\alpha/2} \sqrt{p(1 - p)}}{E} \right)^2 \text{rounded up}$$
- the population must be normal or approximately normal