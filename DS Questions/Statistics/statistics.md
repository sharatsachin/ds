# Statistics

## Hypothesis of normal distribution

The hypothesis of normality is a hypothesis that the data is normally distributed. It is often tested using the Shapiro-Wilk test, the Kolmogorov-Smirnov test, or the Anderson-Darling test.
Normal distribution is characterized by the mean and the standard deviation. The mean is the center of the distribution and the standard deviation is the measure of the spread of the distribution, and normality implies that the data is symmetric around the mean, with most of the data points lying close to the mean.

## What is hypothesis testing?

A method of statistical inference. It is used to determine whether there is enough evidence in a sample of data to decide between two competing claims (hypotheses) about a population parameter. It allows us to make probabilistic statements about a population parameter based on a statistic computed from a sample randomly drawn from that population.

## What is the null hypothesis?

The null hypothesis is a statement about the value of a population parameter that is assumed to be true until there is convincing evidence to the contrary. It is the status quo, and it is the hypothesis that is tested. It is denoted by $H_0$.

## What is the alternative hypothesis?

The alternative hypothesis is a statement that is accepted if the sample data provide sufficient evidence that the null hypothesis is false. It is the hypothesis that we are trying to prove. It is denoted by $H_a$.

### What are the errors in hypothesis testing?

|  | $H_0$ is true | $H_0$ is false |
| --- | --- | --- |
| Reject $H_0$ | Type I error | Correct decision |
| Do not reject $H_0$ | Correct decision | Type II error |

## What is the logic of hypothesis testing? Initial assumption, decision criteria, and conclusion

The test procedure is based on the initial assumption that $H_0$ is true.

The criterion for judging between $H_0$ and $H_a$ based on the sample data is: if the value of $\bar{X}$ would be highly unlikely to occur if $H_0$ were true, but favors the truth of $H_a$, then we reject $H_0$ in favor of $H_a$. Otherwise, we do not reject $H_0$.

Supposing for now that $\bar{X}$ follows a normal distribution, when the null hypothesis is true, the density function for the sample mean $\bar{X}$ must be a bell curve centered at $\mu_0$. Thus, if $H_0$ is true, then $\bar{X}$ is likely to take a value near $\mu_0$ and is unlikely to take values far away. Our decision procedure, therefore, reduces simply to:

- If $H_a$ has the form $H_a: \mu < \mu_0$, then reject $H_0$ if $\bar{x}$ is far to the left of $\mu_0$ (rejection region is $[\infty, C]$, left-tailed test)
- If $H_a$ has the form $H_a: \mu > \mu_0$, then reject $H_0$ if $\bar{x}$ is far to the right of $\mu_0$ (rejection region is $[C, \infty]$, right-tailed test)
- If $H_a$ has the form $H_a: \mu \neq \mu_0$, then reject $H_0$ if $\bar{x}$ is far away from $\mu_0$ in either direction (rejection region is $(-\infty, C] \cup [C', \infty)$, two-tailed test)

## What is the rejection region, and critical value in hypothesis testing? What are the steps to select the critical value?

Rejection region is therefore the set of values of $\bar{x}$ that are far away from $\mu_0$ in the direction indicated by $H_a$. The critical value or critical values of a test of hypotheses are the number or numbers that determine the rejection region.

Procedure for selecting $C$:
- define a rare event : an event is rare if it has a probability of occurring that is less than or equal to $\alpha$. (say $\alpha = 0.01$)
- then critical value $C$ is the value of $\bar{x}$ that cuts off a tail of area $\alpha$ in the appropriate direction.
  - when the rejection region is in two tails, the critical values are the values of $\bar{x}$ that cut off a tail of area $\alpha/2$ in each direction.

<img src="https://i.imgur.com/4xBkgrW.png" width="600" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">

For example, $z_{0.005} = 2.58$ is the critical value for a test of $H_0: \mu = 100$ against $H_a: \mu \neq 100$ at the $\alpha = 0.01$ level of significance. The critical value will be $$ C = 100 \pm 2.58 \cdot \sigma_{\bar{x}} = 100 \pm 2.58 \cdot \frac{\sigma}{\sqrt{n}}$$

## How can we compare two populations $P_1(\mu_1, \sigma_1)$ with sample $(n_1, \bar{x}_1, s_1)$ and $P_2(\mu_2, \sigma_2)$ with sample $(n_2, \bar{x}_2, s_2)$ where $n1, n2 > 30$?

- A point estimate for the difference in two population means is simply the difference in the corresponding sample means.
- The $100(1-\alpha)$% confidence interval for the difference in two population means is given by
$$ (\bar{x}_1 - \bar{x}_2) \pm z_{\alpha/2} \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$$

The same five-step procedure used to test hypotheses concerning a single population mean is used to test hypotheses concerning the difference between two population means. The only difference is in the formula for the standardized test statistic.

$$ H_0: \mu_1 - \mu_2 = D_0$$
$$ H_a: \mu_1 - \mu_2 < D_0 \text{ or } \mu_1 - \mu_2 > D_0 \text{ or } \mu_1 - \mu_2 \neq D_0$$

Standardized test statistic:
$$ Z = \frac{(\bar{x}_1 - \bar{x}_2) - D_0}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
The samples must be independent and the population distributions must be normal or the sample sizes must be large.

## How can we compare two populations $P_1(\mu_1, \sigma_1)$ with sample $(n_1, \bar{x}_1, s_1)$ and $P_2(\mu_2, \sigma_2)$ with sample $(n_2, \bar{x}_2, s_2)$ where $n1, n2 < 30$?

The $100(1-\alpha)$% confidence interval for the difference in two population means is given by
$$ (\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2} \sqrt{s_p^2 \left( \frac{1}{n_1} + \frac{1}{n_2} \right)}$$
where $s_p^2$ is the pooled sample variance, defined as
$$ s_p^2 = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}$$
and the number of degrees of freedom is $n_1 + n_2 - 2$.

Standardized test statistic:
$$ T = \frac{(\bar{x}_1 - \bar{x}_2) - D_0}{s_p^2 \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$
The test statistic has Student's t distribution with $n_1 + n_2 - 2$ degrees of freedom.

## What does it mean for two variables to have a linear relationship?

Two variables $x$ and $y$ have a deterministic linear relationship if points plotted from $(x, y)$ pairs lie exactly along a single straight line. In practice it is common for two variables to exhibit a relationship that is close to linear but which contains an element, possibly large, of randomness.

## What is the covariance of two variables?

The covariance of two variables $x$ and $y$ is a measure of the linear association between the two variables. The covariance is denoted by $S_{xy}$ and is defined by the following formula:
$$ S_{xy} = \frac{\sum{(x_i - \bar{x})(y_i - \bar{y})}}{n - 1}$$
where $\bar{x}$ and $\bar{y}$ are the sample means of the $x$ and $y$ values, respectively.

## How would you calculate the covariance matrix for multiple variables?

If $X_1, X_2, ..., X_n$ are $n$ variables, then the covariance matrix is defined as:
   $$ S = \begin{bmatrix}
   S_{11} & S_{12} & \cdots & S_{1n} \\
   S_{21} & S_{22} & \cdots & S_{2n} \\
   \vdots & \vdots & \ddots & \vdots \\
   S_{n1} & S_{n2} & \cdots & S_{nn}
   \end{bmatrix}$$ 
where $S_{ij} = S_{ji}$ and $S_{ii} = Var(X_i)$ .

$S_{xy} = E[(X - \mu_x)(Y - \mu_y)] = E[XY] - \mu_x \mu_y$

## What is the linear correlation coefficient?

The linear correlation coefficient is a number computed directly from the data that measures the strength of the linear relationship between the two variables $x$ and $y$. The linear correlation coefficient is denoted by $r$ and is defined by the following formula:
$$ r = \frac{\sum{(x_i - \bar{x})(y_i - \bar{y})}}{\sqrt{\sum{(x_i - \bar{x})^2} \sum{(y_i - \bar{y})^2}}} = \frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}}$$
where $S_{xy}$, $S_{xx}$, and $S_{yy}$ are the sums of squares, and $\bar{x}$ and $\bar{y}$ are the sample means of the $x$ and $y$ values, respectively.

## What are the properties of the linear correlation coefficient, with respect to its value and sign?

1. value of $r$ is always between -1 and 1, inclusive
2. sign of $r$ indicates the direction of the linear relationship between $x$ and $y$
3. size of $|r|$ indicates the strength of the linear relationship between $x$ and $y$
   - $|r|$ close to 1 indicates a strong linear relationship, $|r|$ close to 0 indicates a weak linear relationship

## What is the regression line?

The regression line is the line that best fits the data / minimizes the sum of the squared errors $SSE = \sum{(y_i - \hat{y}_i)^2}$.

The equation of the regression line is $$ y = a + bx$$ where $a$ is the y-intercept and $b$ is the slope of the line.
$$ b = \frac{S_{xy}}{S_{xx}} = \frac{\sum{(x_i - \bar{x})(y_i - \bar{y})}}{\sum{(x_i - \bar{x})^2}}$$
$$ a = \bar{y} - b\bar{x}$$

## What is the coefficient of determination?
The coefficient of determination is a number computed directly from the data that measures the proportion of the total variation in the $y$ values that is explained by the regression line. The coefficient of determination is denoted by $r^2$ and is defined by:
$$ r^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum{(y_i - \hat{y}_i)^2}}{\sum{(y_i - \bar{y})^2}}$$ 
where $RSS$ is the residual sum of squares and $TSS$ is the total sum of squares.

## What does the value of the coefficient of determination tell us?

A value of $r^2$ close to 1 indicates that the regression line explains a large proportion of the variation in the $y$ values, while a value of $r^2$ close to 0 indicates that the regression line explains very little of the variation in the $y$ values.

## What does the $\chi^2$ family of distributions represent?

All the $\chi^2$ distributions form a family, each specified by a parameter called the number of degrees of freedom. The number of degrees of freedom for a $\chi^2$ distribution is equal to the number of independent standard normal random variables that are squared and summed to obtain the $\chi^2$ random variable.

<img src="https://i.imgur.com/T4Ow1S0.jpg" width="400" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">

The value of the $\chi^2$ random variable with $df = k$ that cuts off a right tail with an area of $c$ is denoted by $\chi^2_c$, and is called a critical value.

## What does the $\chi^2$ test for independence test?

The $\chi^2$ test for independence is used to test the null hypothesis that two categorical variables are independent. For example:
- $H_0$: Baby gender and baby heart rate (high/low) are independent (two factors are independent)
- $H_1$: Baby gender and baby heart rate are not independent (two factors are not independent)

## How do you perform the $\chi^2$ test for independence?

Steps : 
1. Create the contingency table (rows denoting Factor 1, columns denoting Factor 2) and compute the row and column totals
2. Compute the expected counts for each cell, $$E_{ij} = \frac{row\ total \times column\ total}{grand\ total}$$
3. Compute the $\chi^2$ test statistic, $$\chi^2 = \sum{\frac{(O_{ij} - E_{ij})^2}{E_{ij}}}$$, where $O_{ij}$ is the observed count for cell $(i, j)$ and $E_{ij}$ is the expected count for cell $(i, j)$
4. Based on decided $\alpha$ level and degrees of freedom ($df = (n_{rows} - 1) \times (n_{columns} - 1)$, find the critical value $\chi^2_\alpha$
5. Compare the test statistic and critical value, if $\chi^2 > \chi^2_\alpha$, reject $H_0$
    - If $H_0$ is rejected, conclude that the two categorical variables are not independent

## What is teh $\chi^2$ goodness-of-fit test used for?

The $\chi^2$ goodness-of-fit test is used to test the null hypothesis that a population distribution follows a specified distribution. For example:
- $H_0$: The 6-sided die is fair
- $H_1$: The 6-sided die is not fair

We wish to determine that every face of the die has the same probability of appearing. We roll the die 60 times and record the number of times each face appears. We then compare the observed counts with the expected counts.

## How do you perform the $\chi^2$ goodness-of-fit test?

Steps:
1. Create the contingency table (rows denoting each face of the die, columns denoting observed counts and expected counts)
2. Compute the expected counts for each face, $$E_{i} = \frac{\text{total number of rolls}}{\text{number of faces}}$$
3. Compute the $\chi^2$ test statistic, $$\chi^2 = \sum{\frac{(O_{i} - E_{i})^2}{E_{i}}}$$, where $O_{i}$ is the observed count for face $i$ and $E_{i}$ is the expected count for face $i$
4. Based on decided $\alpha$ level and degrees of freedom ($df = n_{faces} - 1$, find the critical value $\chi^2_\alpha$
5. Compare the test statistic and critical value, if $\chi^2 > \chi^2_\alpha$, reject $H_0$
    - If $H_0$ is rejected, conclude that the die is not fair

## What is the F-distribution family used for?

- Family of F-distributions, each specified by two parameters called the degrees of freedom ($df_1$ and $df_2$), also called the numerator and denominator degrees of freedom. They are not interchangeable.

<img src="https://i.imgur.com/pfMjAvT.png" width="400" style="display: block; margin-left: auto; margin-right: auto; padding-top: 10px; padding-bottom: 10px;">

The value of the F random variable with $df_1$ and $df_2$ that cuts off a right tail with an area of $c$ is denoted by $F_{df_1, df_2, \alpha}$, and is called a critical value.
Also, $$F_{df_1, df_2, \alpha} = \frac{1}{F_{df_2, df_1, 1 - \alpha}}$$

## What is the F-test for equality of two variances used for?

The F-test for equality of two variances is used to test the null hypothesis that two populations have equal variances. For example:
- $H_0$: $\sigma_1^2 = \sigma_2^2$
- Three forms of $H_1$:
    - $\sigma_1^2 \neq \sigma_2^2$ (two-tailed test)
    - $\sigma_1^2 > \sigma_2^2$ (right-tailed test)
    - $\sigma_1^2 < \sigma_2^2$ (left-tailed test)

## How do you perform the F-test for equality of two variances?

Steps:
1. We take two samples from the two populations of size $n_1$ and $n_2$ and compute the sample variances $s_1^2$ and $s_2^2$
    - the samples are independent, and the populations are normally distributed
2. Compute the F test statistic, $$F = \frac{s_1^2}{s_2^2}$$
3. Based on decided $\alpha$ level and degrees of freedom ($df_1 = n_1 - 1$ and $df_2 = n_2 - 1$), find the critical value $F_{df_1, df_2, \alpha}$
4. There are different rejection regions for the three forms of $H_1$:
    - $\sigma_1^2 \neq \sigma_2^2$: reject $H_0$ if $F \le F_{df_1, df_2, 1 - \alpha/2}$ or $F \ge F_{df_1, df_2, \alpha/2}$
    - $\sigma_1^2 > \sigma_2^2$: reject $H_0$ if $F \ge F_{df_1, df_2, \alpha}$
    - $\sigma_1^2 < \sigma_2^2$: reject $H_0$ if $F \le F_{df_1, df_2, 1 - \alpha}$

## What is the F-test for equality of means of several populations used for?

Given population 1 with sample size $n_1$, sample mean $\bar{x}_1$ and sample variance $s_1^2$, population 2 with sample size $n_2$, sample mean $\bar{x}_2$ and sample variance $s_2^2$, and so on, the F-test for equality of means of several populations is used to test the null hypothesis that the means of all populations are equal. 

## How do you perform the F-test for equality of means of several populations?

We calculate the following quantities:
- $n = n_1 + n_2 + \dots + n_k$
- $\bar{x} = \frac{n_1\bar{x}_1 + n_2\bar{x}_2 + \dots + n_k\bar{x}_k}{n}$
- Mean square treatment (MST), $MST = \frac{n_1(\bar{x}_1 - \bar{x})^2 + n_2(\bar{x}_2 - \bar{x})^2 + \dots + n_k(\bar{x}_k - \bar{x})^2}{k - 1}$
- Mean square error (MSE), $MSE = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2 + \dots + (n_k - 1)s_k^2}{n - k}$

Finally we have the F-test statistic, $$F = \frac{MST}{MSE}$$
If the $k$ populations are normally distributed, then $F$ has an F-distribution with $k - 1$ and $n - k$ degrees of freedom.
The test is right tailed, so we reject $H_0$ if $F \ge F_{k - 1, n - k, \alpha}$

## What is the ANOVA test used for?

Analysis of variance (ANOVA) is a collection of statistical models and their associated estimation procedures used to analyze the differences among group means in a sample. ANOVA was developed by Ronald Fisher in 1918 and is the extension of the t and the z test.

## What is the One-way ANOVA test used for?

- ANOVA stands for analysis of variance, and is a generalization of the F-test for equality of means of several populations.
- ANOVA is used to test the null hypothesis that the means of $k$ populations are equal.
- $k$ is the number of populations, $n$ is the total number of observations, $n_i$ is the number of observations in population $i$, $\bar{x}_i$ is the sample mean of population $i$, $s_i^2$ is the sample variance of population $i$.
- $df_{treatment} = k - 1$, $df_{error} = n - k$, $df_{total} = n - 1$

## How do you perform the One-way ANOVA test?

Steps:
1. We take $k$ samples from the $k$ populations of size $n_1, n_2, \dots, n_k$ and compute the sample means $\bar{x}_1, \bar{x}_2, \dots, \bar{x}_k$
    - the samples are independent, and the populations are normally distributed
2. Calculate the following quantities:
    - $n = n_1 + n_2 + \dots + n_k$
    - $$\bar{x} = \frac{n_1\bar{x}_1 + n_2\bar{x}_2 + \dots + n_k\bar{x}_k}{n}$$
3. Calculate the following quantities:
    - Sum of Squares Regression (SSR), $$ SSR = n_1(\bar{x}_1 - \bar{x})^2 + n_2(\bar{x}_2 - \bar{x})^2 + \dots + n_k(\bar{x}_k - \bar{x})^2 $$
    - Sum of Squares Error (SSE), $$ SSE = \sum_{i = 1}^{k} SSE_i $$ where $$ SSE_i = \sum_{j = 1}^{n_i} (x_{ij} - \bar{x}_i)^2 $$
    - Then we have the Sum of Squares Total (SST), $$ SST = SSR + SSE $$
4. Fill in the ANOVA table:

| Source of variation | Sum of squares | Degrees of freedom | Mean square | F |
| --- | --- | --- | --- | --- |
| Treatment | $SSR$ | $k - 1$ | $MST = \frac{SSR}{k - 1}$ | $F = \frac{MST}{MSE}$ |
| Error | $SSE$ | $n - k$ | $MSE = \frac{SSE}{n - k}$ | |
| Total | $SST$ | $n - 1$ | | |

5. If the $k$ populations are normally distributed, then $F$ has an F-distribution with $k - 1$ and $n - k$ degrees of freedom. The critical value is $F_{k - 1, n - k, \alpha}$, and if $F \ge F_{k - 1, n - k, \alpha}$, we reject $H_0$.

### What is teh two-way ANOVA test used for?

| | 1 | 2 | ... | $m$ | Total $x_{i.}$ | Total $x_{i.}^2$ | 
| --- | --- | --- | --- | --- | --- | --- |
| 1 | $x_{11}$ | $x_{12}$ | ... | $x_{1m}$ | $x_{1.}$ | $x_{1.}^2$ |
| 2 | $x_{21}$ | $x_{22}$ | ... | $x_{2m}$ | $x_{2.}$ | $x_{2.}^2$ |
| ... | ... | ... | ... | ... | ... | ... |
| $k$ | $x_{k1}$ | $x_{k2}$ | ... | $x_{km}$ | $x_{k.}$ | $x_{k.}^2$ |
| Total $x_{.j}$ | $x_{.1}$ | $x_{.2}$ | ... | $x_{.m}$ | G | |
| Total $x_{.j}^2$ | $x_{.1}^2$ | $x_{.2}^2$ | ... | $x_{.m}^2$ | | |

- $k$ is the number of treatments, $m$ is the number of blocks, $n$ is the total number of observations.
- ANOVA is used to test the null hypothesis that the means of $k$ populations are equal.
- Two way ANOVA is used to test the null hypothesis that the means of $k$ populations are equal, but the populations are divided into groups based on a factor.
- Two pairs of hypotheses:
    - $H_{01}$: There is no significant difference in the means of different groups
    - $H_{02}$: There is no significant difference in the means of different blocks
    - $H_{11}$: Atleast one pair of means of different groups is significantly different
    - $H_{12}$: Atleast one pair of means of different blocks is significantly different

## How do you perform the Two-way ANOVA test?

Steps:
1. Find the correction factor $$ CF = \frac{(\sum_{j = 1}^{m} \sum_{i = 1}^{n} x_{ij})^2}{n} $$
2. Find the total sum of squares $$ TSS = \sum_{j = 1}^{k} \sum_{i = 1}^{m} x_{ij}^2 - CF $$
3. Find the sum of squares for treatments $$ SST = \sum_{j = 1}^{k} \frac{x_{j.}^2}{m} - CF $$
4. Find the sum of squares for blocks $$ SSB = \sum_{i = 1}^{m} \frac{x_{.i}^2}{k} - CF $$
5. Find the sum of squares for error $$ SSE = TSS - SST - SSB $$
6. Anova table (two-way):

| Source of variation | Sum of squares | Degrees of freedom | Mean square | F |
| --- | --- | --- | --- | --- |
| Treatments | $SST$ | $k - 1$ | $MST = \frac{SST}{k - 1}$ | $F_t = \frac{MST}{MSE}$ |
| Blocks | $SSB$ | $m - 1$ | $MSB = \frac{SSB}{m - 1}$ | $F_b = \frac{MSB}{MSE}$ |
| Error | $SSE$ | $(k - 1)(m - 1)$ | $MSE = \frac{SSE}{(k - 1)(m - 1)}$ | |
| Total | $TSS$ | $km - 1$ | | |

7. The critical value for $F_t$ is $F_{k - 1, (k - 1)(m - 1), \alpha}$, and the critical value for $F_b$ is $F_{m - 1, (k - 1)(m - 1), \alpha}$. If $F_t \ge F_{k - 1, (k - 1)(m - 1), \alpha}$ or $F_b \ge F_{m - 1, (k - 1)(m - 1), \alpha}$, we reject $H_{01}$ or $H_{02}$ respectively.

## What are the characteristics of time series data?

- a set of observations on a variable measured at successive points in time ($y_0$, $y_1$, $y_2$, ..., $y_n$)
- the measurement of the variables may be made continuously or at discrete points in time
- often a variable continuous in time is measured at discrete points in time
- discrete time series data may be generated from an accumulation of data over a period of time
    - monthly sales, daily rainfall, annual production

## What are the 4 components of a time series?

- a time series may be decomposed into four components
    - trend (long term progression of the series, secular variation)
        - exists when there is a persistent, long term increase or decrease in the data
        - may be linear or nonlinear
    - seasonal
        - exists when a series is influenced by seasonal factors
        - seasonal factors are cyclical and repeat over a fixed period
        - seasonal factors are usually multiplicative
    - cyclical
        - exists when data exhibit rises and falls that are not of fixed period
        - cyclical variation is usually due to economic conditions
        - usually of at least 2 years duration (longer and more erratic than seasonal)
    - irregular
        - exists when data are influenced by factors not considered in the analysis
        - may be due to unusual events, one time occurrences, or other sources of variation
        - also called residual or error
- sometimes trend and cyclical are combined into a single component called trend-cycle component

## What are the different types of time series, in terms of the components?

- these components are additive or multiplicative
    - additive: $y_t = T_t + S_t + C_t + I_t$
        - the magnitude of the seasonal variation does not depend on the magnitude of the time series
    - multiplicative: $y_t = T_t \times S_t \times C_t \times I_t$
        - the magnitude of the seasonal variation depends on the magnitude of the time series
    - or a combination of the two

## What are the factors that affect the prediction of future events or quantities?

- the prediction of future events or a quantity depends on several factors including:
    1. how well we understand the factors that contribute to the quantity
    2. how much data is available
    3. whether the forecasts can affect the thing we are trying to forecast

## What is an example of forecasts affecting the thing we are trying to forecast?

- if a company forecasts a large increase in demand for its product, it may increase production to meet the demand
- the increased production may lead to an increase in demand for raw materials, which may lead to an increase in the price of raw materials
- the increase in the price of raw materials may lead to an increase in the price of the product, which may lead to a decrease in demand for the product

## What are the different types of forecasts in terms of the horizon?

- short term: up to 6 months, or more frequently
    - needed for the scheduling of production, inventory, and personnel
    - forecasts of demand for individual products are needed for production scheduling
- medium term: 6 months to 2 years
    - needed for sales and production planning, budgeting, and cost control
    - to determine future resource requirements, in order to purchase raw materials and hire personnel, buy machinery and equipment
    - forecasts of total demand are needed for sales planning
- long term: more than 3 years

## What is the linear trend model?

- $y_t = a + b t + e_t$
- $a$ and $b$ are the intercept and slope of the trend line, $e_t$ is the error term, $t$ is the time period (1, 2, 3, ..., $n$)
- the trend line is a straight line that best fits the data
- First calculate $x$ based on $t$ so that it is centered around $0$:
    - say for $5$ data points [2016, 2017, 2018, 2019, 2020], $x = [-2, -1, 0, 1, 2]$
    - for $6$ data points [2015, 2016, 2017, 2018, 2019, 2020], $x = [-5, -3, -1, 1, 3, 5]$
- create table with:
    - $t$ (time period), $y_t$ (data), $x_t$ (centered time period), $x_t^2$, $x_t y_t$
- then calculate $$b = \frac{\sum x_t y_t}{\sum {x_t}^2}$$ $$a = \frac{\sum y_t}{n}$$
- then forcasted value is $$\hat{y}_{n+1} = a + b (x_{n+1})$$

## What is the quadratic trend model?

- $y_t = a + b t + c t^2 + e_t$
- We can create 3 equations with 3 unknowns ($a$, $b$, $c$) and solve them to get the values of $a$, $b$, $c$:
    - $\sum y_t = a n + b \sum x + c \sum x^2$
    - $\sum x_t y_t = a \sum x + b \sum x^2 + c \sum x^3$
    - $\sum x_t^2 y_t = a \sum x^2 + b \sum x^3 + c \sum x^4$
- $x$ is centered around $0$ as in the linear trend model
- create table with:
    - $t$ (time period), $y_t$ (data), $x_t$ (centered time period), $x_t^2$, $x_t^3$, $x_t^4$, $x_t y_t$, $x_t^2 y_t$
- then forcasted value is $$\hat{y}_{n+1} = a + b (x_{n+1}) + c (x_{n+1})^2$$

## What is the moving average model?

- appropriate for data with horizontal pattern (stationary data)
- $y_t = \frac{1}{k} \sum_{i=1}^k y_{t-i}$

## What is the centered moving average model?

- appropriate for data with trend pattern
- by default, moving average values are placed at the period in which they are calculated
- when you center the moving averages, they are placed at the center of the range rather than the end of it
- if $k$ is odd:
    - say $k = 3$, then the first numeric moving average value is placed at period $2$, the next at period $3$, and so on
    - in this case, the moving average value for the first and last periods is missing
- if $k$ is even:
    - say $k = 4$, then because you cannot place a moving average value at period $2.5$, calculate the average of the first four values and name it $ma_1$
    - then calculate the average of the next four values and name it $ma_2$
    - the average of those two values is the number and place at period $3$

## What is the exponential smoothing model?

- Calculates exponentially smoothed time series $f_t$ from the original time series $y_t$ as follows:
    - $f_1 = y_1$
    - $f_{t+1} = \alpha y_t + (1 - \alpha) f_t$ where $0 < \alpha < 1$

## What is the components form of exponential smoothing?

1. Forecast equation
    - $\hat{y}_{t+h} = l_t$
2. Smoothing equation
    - $l_t = \alpha y_t + (1 - \alpha) l_{t-1}$ where $l_t$ is the smoothed value of $y_t$ and $h$ is the forecast horizon

## What are some general metrics used to evaluate the performance of a time series model?

1. Mean Absolute Error (MAE)
    - gives equal weight to all errors
    - more robust to outliers, but less sensitive to large errors
    - MAE optimizes the median, but may not be unique
    $$MAE = \frac{\sum_{t=1}^n |y_t - \hat{y}_t|}{n}$$
1. MAE %
    - gives a percentage of the MAE
    $$MAE\% = \frac{\sum_{t=1}^n |y_t - \hat{y}_t|}{\sum_{t=1}^n y_t} \times 100$$
2. Mean Squared Error (MSE)
    - gives more weight to large errors, but more sensitive to outliers
    - optimizes for the mean
    $$MSE = \frac{\sum_{t=1}^n (y_t - \hat{y}_t)^2}{n}$$
3. Mean Absolute Percentage Error (MAPE)
    - gives less overall weight to large errors if the time series values are large
    - puts a heavier penalty on negative errors than positive ones
    - can give misleading results when actual values are very small, biased towards forecasts that are too low
    $$MAPE = \frac{\sum_{t=1}^n \frac{|y_t - \hat{y}_t|}{y_t}}{n} \times 100$$
4. Largest Absolute Deviation (LAD)
    - tells us that all deviations fall below a certain threshold value
    $$LAD = \max_{1 \leq t \leq n} |y_t - \hat{y}_t|$$
5. SMAPE (Symmetric Mean Absolute Percentage Error)
    - gives less overall weight to large errors if the time series values are small
    $$SMAPE = \frac{2}{n} \sum_{t=1}^n \frac{|y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|} \times 100$$
6. Accuracy
    $$Accuracy = 100 - \frac{\sum_{t=1}^n |y_t - \hat{y}_t|}{\sum_{t=1}^n y_t} \times 100 = 100 - MAE\%$$
    - ranges from $-\infty$ to $100$, with $100$ indicating a perfect forecast
7. Bias
    $$Bias = \frac{\sum_{t=1}^n (y_t - \hat{y}_t)}{\sum_{t=1}^n y_t} \times 100$$
    - ranges from $-100$ to $\infty$, with $0$ indicating no bias
8. Ideal forecast metric
    - the ideal forecast $$|MAE \%| + |Bias|$$
    - ranges from $0$ to $\infty$, with $0$ indicating a perfect forecast

## What is the Holt-Winters method for forecasting?

Holt-Winters method is a triple exponential smoothing method that takes into account the trend and seasonality of a time series. It is used for time series data that exhibit trend and seasonality.

## What is Holt's linear trend method  / double exponential smoothing?

1. Forecast equation
    - $\hat{y}_{t+h} = l_t + h b_t$
2. Level equation
    - $l_t = \alpha y_t + (1 - \alpha) (l_{t-1} + b_{t-1})$
3. Trend equation
    - $b_t = \beta^* (l_t - l_{t-1}) + (1 - \beta^*) b_{t-1}$
    
Where $l_t$ denotes an estimate of the level of the series at time $t$, $b_t$ denotes an estimate of the trend (slope) of the series at time $t$, $\alpha$ and $\beta^*$ are smoothing parameters, and $0 \leq \alpha \leq 1$ and $0 \leq \beta^* \leq 1$
- $\alpha$ is the level smoothing parameter
- $\beta^*$ is the trend smoothing parameter

With yearly data:

|Year|#Sold|Level|Trend|Forecast|Error|
|---|---|---|---|---|---|
|1|$y_1$|$l_1$|$b_1$|$\hat{f}_1$|$y_1 - \hat{f}_1$|
|2|$y_2$|$l_2$|$b_2$|$\hat{f}_2$|$y_2 - \hat{f}_2$|
|...|...|...|...|...|...|
|10|$y_{10}$|$l_{10}$|$b_{10}$|$\hat{f}_{10}$|$y_{10} - \hat{f}_{10}$|

First calculate $l_1$ by setting $l_1 = y_1$ and $b_1 = 0$

Then calculate $l_2$, $b_2$, $\hat{f}_{2}$ onwards using the following formula:
- $l_t = \alpha y_t + (1 - \alpha) (l_{t-1} + b_{t-1})$
- $b_t = \beta^* (l_t - l_{t-1}) + (1 - \beta^*) b_{t-1}$
- $\hat{f}_{t+1} = l_t + b_t$

Finally, to predict into the future, use the following formula:
- $\hat{f}_{t+k} = l_t + k b_t$

## What is Holt-Winters seasonal method / triple exponential smoothing? 

[Link](https://youtu.be/4_ciGzvrQl8)

1. Forecast equation
    - $\hat{y}_{t+h} = l_t + h b_t + s_{t+h-m(k+1)}$
2. Level equation
    - $l_t = \alpha (y_t - s_{t-m}) + (1 - \alpha) (l_{t-1} + b_{t-1})$
3. Trend equation
    - $b_t = \beta^* (l_t - l_{t-1}) + (1 - \beta^*) b_{t-1}$
4. Seasonal equation
    - $s_t = \gamma (y_t - l_{t-1} - b_{t-1}) + (1 - \gamma) s_{t-m}$

where $k$ is the integer part of $\frac{h-1}{m}$, $m$ is the number of seasons in a year and $\gamma$ is the seasonal smoothing parameter

With monthly data, $m = 12$ : 

|Index|Month|#Sold|Level|Trend|Seasonal|Forecast|Error|
|---|---|---|---|---|---|---|---|
|1|Jan|$y_1$|$l_1$|$b_1$|$s_1$|$\hat{f}_1$|$y_1 - \hat{f}_1$|
|2|Feb|$y_2$|$l_2$|$b_2$|$s_2$|$\hat{f}_2$|$y_2 - \hat{f}_2$|
|...|...|...|...|...|...|...|...|
|12|Dec|$y_{12}$|$l_{12}$|$b_{12}$|$s_{12}$|$\hat{f}_{12}$|$y_{12} - \hat{f}_{12}$|
|13|Jan|$y_{13}$|$l_{13}$|$b_{13}$|$s_{13}$|$\hat{f}_{13}$|$y_{13} - \hat{f}_{13}$|
|14|Feb|$y_{14}$|$l_{14}$|$b_{14}$|$s_{14}$|$\hat{f}_{14}$|$y_{14} - \hat{f}_{14}$|

First calculate $s_1$ to $s_{12}$ using the following formula:
- $s_t = \frac{1}{12} \frac{y_t}{\sum_{k=1}^{12} y_k}$

Then calculate $l_{13}$, $b_{13}$ using the following formula:
- $l_{13} = \frac{y_{13}}{s_1}$
- $b_{13} = \frac{y_{13}}{s_1} - \frac{y_{12}}{s_12}$

Then calculate $s_{13}$ using the following formula:
- $s_{13} = \gamma \frac{y_{13}}{l_{13}} + (1 - \gamma) s_{(13 - 12)}$

Then calculate $l_{14}$, $b_{14}$, $s_{14}$, $\hat{f}_{14}$ onwards using the following formula: (forecast within the data)
- $l_{t} = \alpha (y_t - s_{t-m}) + (1 - \alpha) (l_{t-1} + b_{t-1})$
- $b_{t} = \beta^* (l_{t} - l_{t-1}) + (1 - \beta^*) b_{t-1}$
- $s_{t} = \gamma \frac{y_{t}}{l_{t}} + (1 - \gamma) s_{(t-m)}$
- $\hat{f}_{t+1} = (l_t + b_t) s_{t-m+1}$

Finally, to predict into the future, use the following formula: (forecast beyond the last data point)
- $\hat{f}_{t+k} = (l_t + k b_t) s_{t-m+k}$

## What is a stationary stochastic process in time series analysis?

A stochastic process is a collection of random variables indexed by time. A time series is a realization of a stochastic process. A time series is said to be stationary if its statistical properties do not change over time. In particular, its mean, variance, autocorrelation remain constant over time. A stationary series has no trend, its variations around its mean have a constant amplitude, and it wiggles in a consistent fashion, i.e., its short-term random time patterns always look the same in a statistical sense.

## What is the Autocoavariance Function (ACVF)?
The autocovariance function (ACVF) of a stationary time series $y_t$ is defined as
    - $\gamma(h) = Cov(y_t, y_{t+h}) = E[(y_t - \bar{y})(y_{t+h} - \bar{y})] = \frac{1}{n} \sum_{t=1}^{n-h} (y_t - \bar{y})(y_{t+h} - \bar{y})$

## What is the Autocorrelation Function (ACF)?
- the autocorrelation function (ACF) of a stationary time series $y_t$ is defined as
    - $\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \frac{Cov(y_t, y_{t+h})}{Var(y_t)}$

## What is auto correlation, and how is it calculated?

Auto correlation is the correlation of a time series with the same time series lagged by $k$ time units. It is a measure of how well the present value of a time series is related with its past values. If auto correlation is high, it means that the present value is well correlated with the immediate past value. The value of auto correlation coefficient can range from $-1$ to $1$.
$$ r_h = \rho(h) = \frac{\sum_{t=1}^{n-h} (y_t - \bar{y})(y_{t+h} - \bar{y})}{\sum_{t=1}^{n} (y_t - \bar{y})^2}$$
- $r_1$ measures the correlation between $y_t$ and $y_{t-1}$
- $r_2$ measures the correlation between $y_t$ and $y_{t-2}$ and so on
- autocorrelation for small lags tends to be large and positive because observations nearby in time are also nearby in size
- autocorrelation will be larger for smaller seasonal lags
- time series that show no autocorrelation are called white noise (i.i.d. random variables with zero mean and constant variance)
    - for white noise, we expect $95\%$ of the sample autocorrelations to lie in the interval $(-2/\sqrt{n}, 2/\sqrt{n})$ where $n$ is the sample size

## What is the partial autocorrelation function (PACF)?

- the partial autocorrelation function (PACF) of a stationary time series $y_t$ is defined as $$\phi_{hh} = \rho(h) = \frac{Cov(y_t, y_{t+h} | y_{t+1}, y_{t+2}, ..., y_{t+h-1})}{\sqrt{Var(y_t | y_{t+1}, y_{t+2}, ..., y_{t+h-1}) Var(y_{t+h} | y_{t+1}, y_{t+2}, ..., y_{t+h-1})}}$$

#### What does the ACF plot tell you about a time series?

- shows the correlation between a series and its lagged values
- helps in identifying the presence of autocorrelation in the data, which is a measure of how each data point in the series is related to its previous values
- significant autocorrelation at a particular lag indicates that past values of the series are useful in predicting future values

## What does the PACF plot tell you about a time series?

- shows the correlation between a series and its lagged values after removing the contributions of the intermediate lags.
- helps in identifying the direct and isolated relationships between the current value and its past values, excluding the influence of other lags.
- helps in determining the order of an autoregressive (AR) model. If there is a significant spike at a specific lag, it suggests that this lag is a potential candidate for inclusion in the AR model.

## What is an autoregressive (AR) model?

- an autoregressive (AR) model is when the value of a variable in one period is related to its values in previous periods
- an AR model of order $p$ is denoted by $AR(p)$
- $AR(1)$ model: $y_t = c + \phi_1 y_{t-1} + e_t$
- $AR(p)$ model: $y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + e_t$ where $e_t$ is white noise and $\phi_1, \phi_2, ..., \phi_p$ are parameters of the model
- this is like a multiple regression model with lagged values of $y_t$ as predictors
- each partial correlation coefficient can be estimated as the last coefficient in an AR model
    - specifically, $\alpha_{k}$ the partial autocorrelation coefficient at lag $k$ is the estimate of $\phi_k$ in an $AR(k)$ model

## What is a moving average (MA) model?

- a moving average (MA) model is when the value of a variable in one period is related to the error term in the previous period
- an MA model of order $q$ is denoted by $MA(q)$
- $MA(1)$ model: $y_t = c + e_t + \theta_1 e_{t-1}$
- $MA(q)$ model: $y_t = c + e_t + \theta_1 e_{t-1} + \theta_2 e_{t-2} + ... + \theta_q e_{t-q}$ where $e_t$ is white noise and $\theta_1, \theta_2, ..., \theta_q$ are parameters of the model

## What is an autoregressive moving average (ARMA) model?

- an autoregressive moving average (ARMA) model is a combination of autoregressive and moving average models
- an ARMA model of order $(p, q)$ is denoted by $ARMA(p, q)$
- $ARMA(1, 1)$ model: $y_t = c + \phi_1 y_{t-1} + e_t + \theta_1 e_{t-1}$
- $ARMA(p, q)$ model: $y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + e_t + \theta_1 e_{t-1} + \theta_2 e_{t-2} + ... + \theta_q e_{t-q}$ where $e_t$ is white noise and $\phi_1, \phi_2, ..., \phi_p, \theta_1, \theta_2, ..., \theta_q$ are parameters of the model

## What is an autoregressive integrated moving average (ARIMA) model?

- an autoregressive integrated moving average (ARIMA) model is a generalization of an autoregressive moving average (ARMA) model that includes an additional integrated component
- the integrated component of an ARIMA model is the differencing of raw observations to allow for the time series to become stationary
- an ARIMA model is characterized by 3 terms: $p$, $d$, $q$ where
    - $p$ is the order of the autoregressive model (AR)
    - $d$ is the degree of differencing (the number of times the data have had past values subtracted)
    - $q$ is the order of the moving average model (MA)
- $ARIMA(p, d, q)$ model: $$ y'_t = c + \phi_1 y'_{t-1} + \phi_2 y'_{t-2} + ... + \phi_p y'_{t-p} + e_t + \theta_1 e_{t-1} + \theta_2 e_{t-2} + ... + \theta_q e_{t-q} $$ where $e_t$ is white noise and $\phi_1, \phi_2, ..., \phi_p, \theta_1, \theta_2, ..., \theta_q$ are parameters of the model and $y'_t$ is the differenced series

## What is the seasonal autoregressive integrated moving-average (SARIMA) model?

- a seasonal ARIMA (SARIMA) model is an extension of the ARIMA model that explicitly supports univariate time series data with a seasonal component
- it has additional hyperparameters to specify the autoregression (AR), differencing (I), and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality
- $\text{SARIMA}(p, d, q)(P, D, Q)_m$ model: $$ y'_t = c + \phi_1 y'_{t-1} + \phi_2 y'_{t-2} + ... + \phi_p y'_{t-p} + e_t + \theta_1 e_{t-1} + \theta_2 e_{t-2} + ... + \theta_q e_{t-q} + \phi_1 y'_{t-m} + \phi_2 y'_{t-2m} + ... + \phi_P y'_{t-Pm} + e_t + \theta_1 e_{t-m} + \theta_2 e_{t-2m} + ... + \theta_Q e_{t-Qm} $$ where $e_t$ is white noise and $\phi_1, \phi_2, ..., \phi_p, \theta_1, \theta_2, ..., \theta_q$ are parameters of the model and $y'_t$ is the differenced series

## What is the seasonal autoregressive integrated moving-average with exogenous regressors (SARIMAX) model?

- a seasonal autoregressive integrated moving-average with exogenous regressors (SARIMAX) is an extension of the SARIMA model that also includes the modeling of exogenous variables
- it adds to the SARIMA model a linear regression model that is used to model the exogenous variables
- $\text{SARIMAX}(p, d, q)(P, D, Q)_m$ model: $$ y'_t = c + \phi_1 y'_{t-1} + \phi_2 y'_{t-2} + ... + \phi_p y'_{t-p} + e_t + \theta_1 e_{t-1} + \theta_2 e_{t-2} + ... + \theta_q e_{t-q} + \phi_1 y'_{t-m} + \phi_2 y'_{t-2m} + ... + \phi_P y'_{t-Pm} + e_t + \theta_1 e_{t-m} + \theta_2 e_{t-2m} + ... + \theta_Q e_{t-Qm} + \beta_1 x_{1t} + \beta_2 x_{2t} + ... + \beta_k x_{kt} $$ where $e_t$ is white noise and $\phi_1, \phi_2, ..., \phi_p, \theta_1, \theta_2, ..., \theta_q$ are parameters of the model and $y'_t$ is the differenced series and $x_{1t}, x_{2t}, ..., x_{kt}$ are the exogenous variables

## What is maximum likelihood estimation (MLE) in time series analysis?

Estimation of the parameters of a model is often done by maximum likelihood estimation (MLE). The likelihood function is defined as the probability of the observed data as a function of the parameters of the model. The maximum likelihood estimate of the parameters is the value of the parameters that maximize the likelihood function.
Likelikood function for a time series model is the joint probability distribution of the observed data. The likelihood function is maximized with respect to the parameters of the model to obtain the maximum likelihood estimates of the parameters.

Suppose we have random variables $X_1, X_2, ..., X_n$ that are independent and identically distributed (i.i.d.) with probability density function $f(x; \theta)$ where $\theta$ is the parameter of the distribution. The likelihood function is defined as $$L(\theta) = \prod_{i=1}^n f(x_i; \theta)$$ The maximum likelihood estimate of $\theta$ is the value of $\theta$ that maximizes the likelihood function $L(\theta)$.

The maximum likelihood estimate of $\theta$, $$\hat{\theta} = \arg \max_{\theta} L(\theta) = \arg \max_{\theta} \log L(\theta)$$
For maximization, we have $\frac{\partial \log L(\theta)}{\partial \theta} = 0$ and $\frac{\partial^2 \log L(\theta)}{\partial \theta^2} < 0$

## What is the likelihood function and maximum likelihood estimate for binomial distribution?

- the likelihood function is $$L(p) = \prod_{i=1}^N {}^n C_{x_i} p^{x_i} (1 - p)^{n - x_i}$$
- the maximum likelihood estimate of $p$ is $$\hat{p} = \frac{\sum_{i=1}^N x_i}{n N}$$ where $n$ is the number of trials and $N$ is the number of experiments

## What is the likelihood function and maximum likelihood estimate for Poisson distribution?

- the likelihood function is $$L(\lambda) = \prod_{i=1}^N \frac{e^{-\lambda} \lambda^{x_i}}{x_i!}$$
- the maximum likelihood estimate of $\lambda$ is $$\hat{\lambda} = \frac{\sum_{i=1}^N x_i}{N}$$

## What is the likelihood function and maximum likelihood estimate for normal distribution?

- the likelihood function is $$L(\mu, \sigma^2) = \prod_{i=1}^N \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x_i - \mu)^2}{2 \sigma^2}}$$
- the maximum likelihood estimate is $$\hat{\mu} = \frac{\sum_{i=1}^N x_i}{N}, \hat{\sigma}^2 = \frac{\sum_{i=1}^N (x_i - \hat{\mu})^2}{N}$$

## What is the likelihood function and maximum likelihood estimate for uniform distribution?

- the likelihood function is $$L(a, b) = \prod_{i=1}^N \frac{1}{b - a}$$
- the maximum likelihood estimate is $$\hat{a} = \min_{i=1}^N x_i, \hat{b} = \max_{i=1}^N x_i$$