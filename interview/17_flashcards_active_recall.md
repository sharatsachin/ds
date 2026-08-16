# Active Recall Flashcards

This file is different from the rest of the kit. Files 01–14 and 16 teach the
material — derivations, worked examples, "Interview angle" Q&A. This file
does not teach anything new. It is a **spaced-repetition-style flashcard
deck** for testing whether you actually retained it.

**Rule of the deck: try to answer before you click.** Read the question under
each `<summary>`, say your answer out loud or in your head, *then* expand the
`<details>` block to check yourself. If you read straight through without
attempting an answer first, you are wasting the format — you'll recognize
the answer and feel like you know it, which is a much weaker signal than
being able to produce it cold.

Every card links back to its companion file by name/number. If a card stumps
you, that's your cue to go re-read that section in depth — this deck tells
you *what* to re-study, not the full *why*.

## How to Use This Deck

1. **Pass 1 (today):** Go section by section. For every card, attempt an
   answer, then expand. Keep a running list (paper, notes app, whatever) of
   the questions you missed or hedged on.
2. **Re-drill +1 day:** Redo only the missed cards. Anything you get right
   twice in a row can be retired for now; anything still shaky goes back on
   the list.
3. **Re-drill +3 days:** Same process — only the cards still on the list.
   By now the deck you're drilling should be small.
4. **Re-drill +7 days:** Final pass on whatever survived. If a card is still
   missed at this point, it usually means you're missing the underlying
   intuition, not just the fact — go back to the companion file and re-read
   the full derivation, not just the recall sheet.
5. **Night-before-interview:** Do one fast full pass of the whole deck (all
   ~180+ cards) as a confidence check, not a learning session. You should be
   able to get through it quickly at this point.

Treat "I remembered the shape of the answer but not the exact formula" as a
miss, not a hit — interviewers notice the difference between "roughly" and
"precisely," and the gap only shows up in your recall drilling if you're
honest with yourself about it.

---

## 1. Statistics & Probability (→ `01_statistics_and_probability.md`)

<details>
<summary><b>Q: Formula for sample variance, and why divide by (n-1) instead of n?</b></summary>

$s^2 = \frac{1}{n-1}\sum(x_i-\bar x)^2$. Dividing by $n-1$ (Bessel's correction) removes the bias introduced by estimating $\mu$ with $\bar x$ — one degree of freedom is "used up," since $E[\sum(x_i-\bar x)^2] = (n-1)\sigma^2$.

</details>

<details>
<summary><b>Q: What does positive vs. negative skewness tell you about mean vs. median?</b></summary>

Positive skew = long right tail, mean > median. Negative skew = long left tail, mean < median. Formula: $\gamma_1 = E[(X-\mu)^3]/\sigma^3$.

</details>

<details>
<summary><b>Q: What is excess kurtosis, and what do positive/negative values indicate?</b></summary>

Excess kurtosis $= E[(X-\mu)^4]/\sigma^4 - 3$. Positive = fatter tails than normal (leptokurtic, more extreme outliers); negative = thinner tails (platykurtic).

</details>

<details>
<summary><b>Q: Mean and variance of a Bernoulli(p) random variable?</b></summary>

Mean $= p$, variance $= p(1-p)$.

</details>

<details>
<summary><b>Q: Mean and variance of Binomial(n, p)? What does it model?</b></summary>

Mean $= np$, variance $= np(1-p)$ — the count of successes across $n$ independent trials.

</details>

<details>
<summary><b>Q: Mean and variance of Poisson(λ)? What's distinctive about it?</b></summary>

Mean $=$ variance $= \lambda$ — models counts of rare/independent events per fixed interval.

</details>

<details>
<summary><b>Q: What distribution models waiting time between Poisson events, and what unique property does it have?</b></summary>

Exponential($\lambda$): mean $=1/\lambda$, variance $=1/\lambda^2$. It's **memoryless**: $P(X>s+t \mid X>s) = P(X>t)$.

</details>

<details>
<summary><b>Q: When would you reach for a log-normal distribution rather than normal?</b></summary>

For multiplicative/compounding processes where the *log* of the variable is normally distributed — e.g., stock prices, income. Mean $= e^{\mu+\sigma^2/2}$.

</details>

<details>
<summary><b>Q: State the Central Limit Theorem, and name a condition under which it fails in practice.</b></summary>

$\frac{\bar X_n - \mu}{\sigma/\sqrt n} \to N(0,1)$ regardless of the population's underlying shape, provided variance is finite. It breaks down (needs much larger $n$ than the "$n\geq30$" rule of thumb) under heavy skew, heavy tails, or infinite variance.

</details>

<details>
<summary><b>Q: LLN vs. CLT — what does each one actually tell you?</b></summary>

LLN says *where* the sample mean converges (to $\mu$, weakly in probability or strongly almost surely). CLT describes the *shape of the fluctuations* around that convergence — asymptotically normal, scaled by $1/\sqrt n$.

</details>

<details>
<summary><b>Q: State Bayes' theorem and explain the base-rate fallacy with a concrete flavor of example.</b></summary>

$P(A\mid B) = \dfrac{P(B\mid A)P(A)}{P(B)}$. Base-rate fallacy: when the prior is small (rare condition), even a highly "accurate" test yields a surprisingly low posterior — a 99%-sensitive test with a 5% false-positive rate on a 1%-prevalence condition gives only ~16.7% posterior probability of actually having it.

</details>

<details>
<summary><b>Q: Frequentist vs. Bayesian inference — what's the core philosophical difference?</b></summary>

Frequentist: the parameter is fixed and unknown; a 95% CI is a statement about the long-run behavior of the *procedure*. Bayesian: the parameter is itself a random variable with a prior; the posterior/credible interval is a direct probability statement about the parameter given the data observed.

</details>

<details>
<summary><b>Q: Pearson vs. Spearman correlation — when do you use each?</b></summary>

Pearson measures linear association on raw values, sensitive to outliers, range $[-1,1]$. Spearman = Pearson computed on ranks — captures monotonic (including non-linear) relationships and is robust to outliers.

</details>

<details>
<summary><b>Q: What is Simpson's Paradox? Give the classic example structure.</b></summary>

A trend that holds in every subgroup reverses when subgroups are aggregated, due to a lurking confounder. Classic example: kidney stone treatment A beats treatment B within both the small-stone and large-stone subgroups, but B appears better overall because of how patients were distributed across subgroups.

</details>

<details>
<summary><b>Q: Distinguish survivorship bias, selection bias, and confirmation bias.</b></summary>

Survivorship bias: drawing conclusions only from the "survivors," ignoring failures (WWII bomber armor placement). Selection bias: a non-representative sampling mechanism correlated with the outcome. Confirmation bias: cherry-picking evidence that supports a pre-existing belief.

</details>

<details>
<summary><b>Q: What is "regression to the mean," and why is it commonly mistaken for a treatment effect?</b></summary>

Extreme observations tend to be followed by more average ones purely due to natural random variance — not because of any intervention. If you select on an extreme value and then measure again, you'll usually see improvement even with no real effect.

</details>

<details>
<summary><b>Q: Monty Hall problem — should you switch doors, and why?</b></summary>

Yes, switching wins with probability $2/3$ (vs $1/3$ for staying). The host's non-random act of revealing a goat (never the car, never your door) concentrates the original 2/3 probability mass onto the one remaining unopened door.

</details>

<details>
<summary><b>Q: Birthday paradox — how many people are needed for a >50% chance two share a birthday?</b></summary>

23 people: $1-\prod_{k=0}^{22}\frac{365-k}{365}\approx 0.507$.

</details>

---

## 2. Hypothesis Testing & A/B Testing (→ `02_hypothesis_testing_and_ab_testing.md`)

<details>
<summary><b>Q: Define Type I and Type II error, and relate them to power.</b></summary>

Type I error ($\alpha$) = rejecting a true $H_0$ (false positive). Type II error ($\beta$) = failing to reject a false $H_0$ (false negative). Power $= 1-\beta = P(\text{reject } H_0 \mid H_1 \text{ true})$.

</details>

<details>
<summary><b>Q: What does a p-value actually mean, and what is the most common misinterpretation?</b></summary>

$P(\text{data at least as extreme as observed} \mid H_0 \text{ true})$. It is **not** $P(H_0 \mid \text{data})$ — that reversal requires Bayes' rule and a prior the p-value never uses.

</details>

<details>
<summary><b>Q: Name the four levers that increase statistical power.</b></summary>

Larger effect size ↑, larger sample size ↑, larger $\alpha$ ↑, smaller variance ↑ power (equivalently, larger variance decreases power).

</details>

<details>
<summary><b>Q: Formula for Welch's t-test, and when do you prefer it over the pooled two-sample t-test?</b></summary>

$t = \dfrac{\bar x_1 - \bar x_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$, with Welch–Satterthwaite degrees of freedom. Prefer it whenever you can't assume equal variances — it's the safer default two-sample test in practice.

</details>

<details>
<summary><b>Q: Paired t-test vs. independent two-sample t-test — when does each apply?</b></summary>

Paired: same units measured twice (before/after), $t=\bar d/(s_d/\sqrt n)$ — removes between-subject variance by differencing. Independent: two separate, unrelated groups.

</details>

<details>
<summary><b>Q: Chi-square test statistic formula, and what two use cases does it cover?</b></summary>

$\chi^2 = \sum \dfrac{(O_i-E_i)^2}{E_i}$ — used for goodness-of-fit (one categorical variable vs. expected distribution) and independence (two categorical variables in a contingency table).

</details>

<details>
<summary><b>Q: One-way ANOVA F-statistic — what does it test, and why not just run pairwise t-tests?</b></summary>

Tests whether ≥3 group means differ: $F = \dfrac{MS_{between}}{MS_{within}} = \dfrac{SS_{between}/(k-1)}{SS_{within}/(N-k)}$. Running many pairwise t-tests instead inflates the family-wise Type I error rate — ANOVA controls it in one test.

</details>

<details>
<summary><b>Q: What is the key scaling relationship in the two-proportion sample-size formula?</b></summary>

$n \propto 1/\delta^2$ where $\delta$ is the minimum detectable effect (MDE). Halving the MDE roughly **quadruples** the required sample size per arm.

</details>

<details>
<summary><b>Q: FWER vs. FDR — what does each control, and name the standard correction for each.</b></summary>

FWER = probability of *any* false positive across $m$ tests, $FWER = 1-(1-\alpha)^m$ — controlled via **Bonferroni** ($\alpha_{Bonf}=\alpha/m$, conservative, best for few tests). FDR = expected *fraction* of false positives among rejections — controlled via **Benjamini-Hochberg**, more powerful, better for many tests.

</details>

<details>
<summary><b>Q: What is CUPED and what problem does it solve?</b></summary>

Variance-reduction technique using a pre-experiment covariate: $Y_{CUPED} = Y - \theta(X-\bar X)$, $\theta = Cov(X,Y)/Var(X)$. Reduces variance to $Var(Y)(1-\rho^2)$, letting you detect the same effect with less traffic/time.

</details>

<details>
<summary><b>Q: Why does "peeking" at an A/B test's results early and often inflate false positives?</b></summary>

The fixed-horizon $\alpha$ guarantee is a "one look at the planned end" promise, not a "look whenever you want" promise — repeated interim checks compound the chance of crossing the significance threshold by chance alone. Fix: group sequential testing, alpha-spending functions, or mSPRT.

</details>

<details>
<summary><b>Q: How do you check for Sample Ratio Mismatch (SRM), and why does it matter so much?</b></summary>

Chi-square test of observed vs. expected split, e.g. $\chi^2=\frac{(O_1-N/2)^2}{N/2}+\frac{(O_2-N/2)^2}{N/2}$, using a strict threshold like $p<0.001$. An SRM means the arms may not be comparable at all — it can invalidate every downstream result, so it's checked *before* trusting anything else.

</details>

<details>
<summary><b>Q: How would you handle a heavily right-skewed metric like revenue-per-user in an A/B test?</b></summary>

Bootstrap the difference in means, log-transform (careful with back-transform bias), trim/winsorize extreme values, or apply CUPED for variance reduction — don't trust a raw t-test's p-value on raw skewed data.

</details>

<details>
<summary><b>Q: Novelty effect vs. interference/network effect — how do you detect and handle each?</b></summary>

Novelty/primacy effect: temporary reaction to something new that fades — judge on the trend over time, not day-one results. Interference/network effect: one arm's behavior spills over to affect the other, violating SUTVA — fix with cluster randomization or switchback designs, not per-user randomization.

</details>

<details>
<summary><b>Q: What's the non-parametric alternative to a two-sample t-test, and when would you use it?</b></summary>

Mann-Whitney U test — rank-based, no normality assumption; use for heavily skewed, ordinal, or outlier-heavy data.

</details>

<details>
<summary><b>Q: What does a 95% confidence interval actually guarantee?</b></summary>

Over repeated sampling/experiment replications, 95% of intervals *constructed this way* would contain the true parameter. It is a property of the procedure — not a 95% probability statement about this one specific already-computed interval.

</details>

---

## 3. ML Fundamentals (→ `03_ml_fundamentals.md`)

<details>
<summary><b>Q: State the bias-variance decomposition of expected test error.</b></summary>

$\text{Err}(x_0) = \text{Bias}(\hat f(x_0))^2 + \text{Var}(\hat f(x_0)) + \sigma^2$. Bias² typically decreases and variance increases with model complexity, so total error is U-shaped.

</details>

<details>
<summary><b>Q: How do you tell high bias (underfitting) apart from high variance (overfitting) using learning curves?</b></summary>

High bias: train and validation error both high, close together (adding data won't help much). High variance: train error low, validation error high, and the gap widens as complexity grows (regularization or more data helps).

</details>

<details>
<summary><b>Q: Lasso vs. Ridge — penalty terms and their qualitative effect on coefficients?</b></summary>

Lasso: $\text{RSS}+\lambda\sum|w_j|$ — diamond-shaped constraint region, corners drive some coefficients exactly to zero (sparsity/feature selection), no closed form. Ridge: $\text{RSS}+\lambda\sum w_j^2$ — circular constraint, smooth boundary, shrinks all coefficients toward zero but rarely to exactly zero.

</details>

<details>
<summary><b>Q: ElasticNet formula, and when would you choose it over pure Lasso or Ridge?</b></summary>

$\text{RSS} + \lambda[\alpha\sum|w_j| + (1-\alpha)\sum w_j^2]$. Choose it when features are correlated — it groups/shares weight among correlated features (Ridge-like) while still allowing partial sparsity (Lasso-like), unlike Lasso which arbitrarily picks one from a correlated group.

</details>

<details>
<summary><b>Q: Ridge regression's closed-form solution — why does it always exist even when plain OLS doesn't?</b></summary>

$\hat w = (X^\top X + \lambda I)^{-1}X^\top y$. Adding $\lambda I$ shifts every eigenvalue of $X^\top X$ up by $\lambda$, guaranteeing invertibility and improving numerical conditioning even under multicollinearity.

</details>

<details>
<summary><b>Q: Normal equation for OLS, and its computational cost?</b></summary>

$\hat\beta = (X^\top X)^{-1}X^\top y$, cost $O(p^3)$ from the matrix inversion — use gradient descent instead when $p$ (or $n$) is huge or $X^\top X$ is singular/ill-conditioned.

</details>

<details>
<summary><b>Q: Batch, stochastic, and mini-batch gradient descent — what's the difference?</b></summary>

Batch GD: uses the full dataset per update (stable but slow/memory-heavy). SGD: one sample per update (noisy, fast, escapes shallow local minima). Mini-batch: small batch per update — the practical default, balancing gradient-estimate stability and compute.

</details>

<details>
<summary><b>Q: List the core linear regression assumptions, and what breaks if they're violated.</b></summary>

Linearity, independent errors, homoscedasticity (constant error variance), no perfect multicollinearity, normally-distributed residuals (only needed for inference, not point estimates under Gauss-Markov). Violations cause biased/misleading coefficients, invalid p-values/CIs, or inefficient (high-variance) estimates.

</details>

<details>
<summary><b>Q: VIF formula and the typical thresholds for concern?</b></summary>

$\text{VIF}_j = \dfrac{1}{1-R_j^2}$ (from regressing feature $j$ on all other features). Flag around VIF > 5, strong concern above 10. Fixes: drop/combine correlated features, regularize, or use PCA.

</details>

<details>
<summary><b>Q: What happens to OLS coefficients under perfect multicollinearity?</b></summary>

$X^\top X$ becomes singular, so there's no unique $\hat\beta$ — infinitely many solutions fit equally well. Predictions can still be stable even though individual coefficients become meaningless/unstable.

</details>

<details>
<summary><b>Q: Sigmoid function and its derivative?</b></summary>

$\sigma(z) = \dfrac{1}{1+e^{-z}}$ (inverse-logit); derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ — this clean self-referential form is why logistic regression's gradient is so simple.

</details>

<details>
<summary><b>Q: Derive log-loss's origin — what likelihood is it derived from?</b></summary>

Log-loss $= -\sum_i[y_i\ln p_i + (1-y_i)\ln(1-p_i)]$ is the negative log-likelihood of a Bernoulli model fit via MLE — minimizing it is equivalent to maximizing the likelihood of the observed binary labels.

</details>

<details>
<summary><b>Q: Gradient of the logistic regression loss w.r.t. weights?</b></summary>

$\nabla_w \mathcal L = X^\top(\hat y - y)$ — the same clean "predicted minus actual, times inputs" form as linear regression's gradient.

</details>

<details>
<summary><b>Q: How do you interpret a logistic regression coefficient in terms of odds?</b></summary>

Odds $= p/(1-p)$, logit $= \ln(\text{odds}) = x^\top w$. A one-unit increase in $x_j$ multiplies the odds by $e^{w_j}$ (not the probability directly).

</details>

<details>
<summary><b>Q: Softmax formula for multiclass classification, and when would you prefer One-vs-Rest instead?</b></summary>

$P(y=k\mid x) = \dfrac{e^{z_k}}{\sum_j e^{z_j}}$, trained via joint cross-entropy. Prefer OvR (independent binary classifiers per class) for very large or incrementally-growing label spaces, or for multilabel settings where classes aren't mutually exclusive.

</details>

---

## 4. Trees, Ensembles & Boosting (→ `04_trees_ensembles_boosting.md`)

<details>
<summary><b>Q: Gini impurity and entropy formulas — how do they compare in practice?</b></summary>

Gini $= 1-\sum p_k^2$; Entropy $= -\sum p_k\log_2 p_k$. They almost always agree on the best split; Gini is preferred computationally since it avoids the log calculation.

</details>

<details>
<summary><b>Q: What do regression trees split on, and what's the leaf prediction?</b></summary>

They split to maximize variance/SSE reduction; the leaf prediction is simply the mean of the training targets that land in that leaf.

</details>

<details>
<summary><b>Q: Pre-pruning vs. post-pruning (cost-complexity pruning) — formula and tradeoff?</b></summary>

Pre-pruning: `max_depth`, `min_samples_split`, `min_samples_leaf` — greedy, cheap, risks stopping too early. Post-pruning (CCP): $R_\alpha(T)=R(T)+\alpha|T|$ — grow the full tree, generate a nested sequence of pruned subtrees across $\alpha$, select via cross-validation.

</details>

<details>
<summary><b>Q: Bagging vs. boosting vs. stacking — one-line distinction for each.</b></summary>

Bagging: parallel, independent learners, reduces variance via unweighted averaging (→ Random Forest). Boosting: sequential, each learner corrects the last, reduces bias via weighted additive combination (→ XGBoost/LightGBM/AdaBoost). Stacking: parallel *heterogeneous* base models combined by a learned meta-model.

</details>

<details>
<summary><b>Q: Derive the ~63.2% "in-bag" rate for a bootstrap sample.</b></summary>

Probability a given row is excluded from one bootstrap draw of size $n$: $(1-1/n)^n \to e^{-1} \approx 0.368$ as $n\to\infty$. So ~63.2% of unique rows appear in each bootstrap sample, and the excluded ~36.8% form the out-of-bag set.

</details>

<details>
<summary><b>Q: Why does feature subsampling at each split help Random Forest reduce variance?</b></summary>

It decorrelates the trees — without it, all trees would tend to split on the same dominant feature and their errors would be highly correlated, keeping the ensemble variance ($\rho\sigma^2$-driven) high. Random feature subsets force diversity, lowering that variance floor.

</details>

<details>
<summary><b>Q: What is Out-of-Bag (OOB) error, and why is it useful?</b></summary>

Each tree is evaluated on the ~36.8% of rows it didn't see during its bootstrap sampling — averaging these gives a free, built-in validation estimate without needing a separate held-out set.

</details>

<details>
<summary><b>Q: What is a "pseudo-residual" in gradient boosting, and what does it equal for squared-error loss?</b></summary>

$r_i = -\partial L/\partial F(x_i)$ — the negative gradient of the loss w.r.t. the current prediction. For squared error, this works out exactly to $y_i - F(x_i)$, the ordinary residual.

</details>

<details>
<summary><b>Q: What does the shrinkage/learning-rate parameter ν do in the additive boosting update?</b></summary>

$F_m = F_{m-1} + \nu h_m$ — smaller $\nu$ slows convergence but generally improves generalization (needs more trees); it's the core "shrinkage" regularization tradeoff in boosting.

</details>

<details>
<summary><b>Q: XGBoost's regularized objective — what does $\Omega(f)$ penalize?</b></summary>

$\sum_i l(y_i,\hat y_i) + \sum_k \Omega(f_k)$, where $\Omega(f) = \gamma T + \tfrac12\lambda\sum_j w_j^2$ — $T$ is the number of leaves (penalizes tree size, via $\gamma$) and $\sum w_j^2$ penalizes large leaf weights (via $\lambda$).

</details>

<details>
<summary><b>Q: Optimal leaf weight formula in XGBoost, derived from the second-order Taylor expansion?</b></summary>

$w_j^* = -G_j/(H_j+\lambda)$, where $G_j,H_j$ are the summed first/second-order gradients of samples in leaf $j$. This falls out of minimizing the Taylor-approximated per-leaf objective.

</details>

<details>
<summary><b>Q: XGBoost's split gain formula — what does it measure?</b></summary>

$Gain = \tfrac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$ — the reduction in the regularized objective from splitting a node into left/right children, net of the per-leaf complexity penalty $\gamma$.

</details>

<details>
<summary><b>Q: `depthwise` vs. `lossguide` tree growth policy — what controls each, and what's the difference?</b></summary>

`depthwise` (level-wise): expands all nodes at the current depth before going deeper, controlled by `max_depth`. `lossguide` (leaf-wise): always splits the single leaf with the highest gain globally, controlled by `max_leaves` — this is also LightGBM's default growth strategy.

</details>

<details>
<summary><b>Q: How does LightGBM's leaf-wise growth differ from level-wise, and what's the risk?</b></summary>

Leaf-wise always expands the globally highest-gain leaf rather than every leaf at the current depth — more efficient per leaf added, but can grow deep, unbalanced trees that overfit if `num_leaves` isn't controlled.

</details>

<details>
<summary><b>Q: What is GOSS in LightGBM, and how does it stay (approximately) unbiased?</b></summary>

Gradient-based One-Side Sampling: keep all large-gradient (under-trained) rows, randomly sample the small-gradient rows, then rescale the sampled small-gradient rows' contribution by a compensation factor $(1-a)/b$ so the gradient sum estimate stays approximately unbiased.

</details>

<details>
<summary><b>Q: What is Exclusive Feature Bundling (EFB), and what kind of features does it target?</b></summary>

It bundles mutually-exclusive sparse features (e.g., one-hot encoded columns that are never simultaneously non-zero) into a single feature, cutting dimensionality near-losslessly and speeding up histogram construction.

</details>

<details>
<summary><b>Q: How does XGBoost handle missing values during split finding?</b></summary>

Sparsity-aware split finding: at each node, it learns the best *default direction* (left or right) for missing values by comparing the resulting gain both ways and picking whichever is better — no imputation needed.

</details>

<details>
<summary><b>Q: For XGBoost/LightGBM, which hyperparameters increase overfitting risk if turned up, and which reduce it?</b></summary>

↑ overfitting risk: `learning_rate`, `max_depth`, `num_leaves`, `n_estimators`. ↓ overfitting risk (more regularization/stochasticity): ↑ `subsample`, `colsample_bytree`, `min_child_weight`/`min_data_in_leaf`, `lambda`/`alpha`/`gamma`.

</details>

---

## 5. Other ML Algorithms (→ `05_other_ml_algorithms.md`)

<details>
<summary><b>Q: SVM hard-margin objective, and what does adding slack variables ξ give you?</b></summary>

Minimize $\frac12\|w\|^2$ subject to $y_i(w\cdot x_i+b)\geq 1$. Adding slack $\xi_i$ with penalty $C\sum\xi_i$ gives the soft-margin SVM, which tolerates some misclassification/margin violation for non-separable data.

</details>

<details>
<summary><b>Q: What is the kernel trick, and what's the formula for the RBF kernel?</b></summary>

$K(x,x') = \phi(x)\cdot\phi(x')$ — compute a high-dimensional dot product without ever forming $\phi(x)$ explicitly. RBF kernel: $K(x,x') = \exp(-\gamma\|x-x'\|^2)$.

</details>

<details>
<summary><b>Q: In SVM, what does a high vs. low value of C do?</b></summary>

High C: low bias/high variance — narrow margin, penalizes misclassification heavily, risks overfitting. Low C: high bias/low variance — wide margin, tolerates more violations, risks underfitting.

</details>

<details>
<summary><b>Q: In an RBF-kernel SVM, what does a high vs. low gamma do?</b></summary>

High gamma: small influence radius per support vector → jagged, tightly-fit decision boundary, overfitting risk. Low gamma: large influence radius → smooth boundary, underfitting risk.

</details>

<details>
<summary><b>Q: What distance metric would you use for text/sparse high-dimensional data in k-NN, and why not Euclidean?</b></summary>

Cosine similarity/distance — it measures angle, not magnitude, so it's robust to differing document lengths and sparse high-dimensional vectors, unlike Euclidean distance which is dominated by magnitude differences.

</details>

<details>
<summary><b>Q: What is the "curse of dimensionality" for distance-based methods like k-NN?</b></summary>

As dimensionality grows, pairwise distances between points concentrate — nearly all points become roughly equidistant from each other, so "nearest neighbor" loses discriminative meaning. Also, the ratio of hypersphere-to-hypercube volume shrinks toward zero.

</details>

<details>
<summary><b>Q: How does choice of k in k-NN trade off bias and variance?</b></summary>

Small k: low bias, high variance (very sensitive to local noise). Large k: high bias, low variance (oversmoothed, blurs decision boundary). Tune via cross-validation.

</details>

<details>
<summary><b>Q: Name three ways to speed up k-NN at inference time.</b></summary>

KD-tree (good for low dimensions), ball tree (moderate dimensions), or approximate nearest neighbor methods like LSH/HNSW (large-scale, high-dimensional, accepts approximate results for speed).

</details>

<details>
<summary><b>Q: Naive Bayes decision rule, and why does it work despite an unrealistic assumption?</b></summary>

$\hat y = \arg\max_y P(y)\prod_i P(x_i\mid y)$, from Bayes' theorem assuming conditional independence of features given the class. It works well in practice even when independence is violated because it only needs to get the *ranking* of class probabilities right, not their exact calibrated values.

</details>

<details>
<summary><b>Q: Laplace smoothing formula — what problem does it solve?</b></summary>

$P(x_i\mid y) = \dfrac{\text{count}+\alpha}{\text{count}(y)+\alpha|V|}$ — prevents any single unseen feature value from zeroing out an entire class's posterior probability.

</details>

<details>
<summary><b>Q: K-means objective function, and what's k-means++ trying to fix?</b></summary>

$J = \sum_k\sum_{x_i\in C_k}\|x_i-\mu_k\|^2$, minimized via Lloyd's algorithm (assign → update → repeat). k-means++ improves on random initialization by seeding centroids spread apart, sampled proportional to $D(x)^2$ — reduces the chance of poor local optima.

</details>

<details>
<summary><b>Q: Elbow method vs. silhouette score for choosing K in k-means?</b></summary>

Elbow: plot within-cluster sum of squares (WCSS) vs. K, look for the bend where adding clusters stops helping much — subjective. Silhouette: $s(i) = \dfrac{b(i)-a(i)}{\max(a(i),b(i))}$, range $[-1,1]$, higher is better — more quantitative, accounts for both cohesion and separation.

</details>

<details>
<summary><b>Q: What are k-means' core limitations?</b></summary>

Assumes spherical, equal-sized, similar-density clusters; sensitive to feature scale and outliers; requires choosing K upfront.

</details>

<details>
<summary><b>Q: Name the four linkage criteria in hierarchical clustering and one property of each.</b></summary>

Single (nearest point — prone to chaining), complete (farthest point — compact but outlier-sensitive), average (mean distance — a compromise), Ward (minimizes within-cluster variance increase — behaves like k-means).

</details>

<details>
<summary><b>Q: DBSCAN's two key parameters, and its main strength vs. weakness?</b></summary>

`eps` (neighborhood radius) and `min_samples` (density threshold) define core/border/noise points. Strength: finds arbitrarily-shaped clusters and naturally handles noise/outliers. Weakness: struggles with clusters of varying density (HDBSCAN addresses this).

</details>

<details>
<summary><b>Q: How does a Gaussian Mixture Model generalize k-means?</b></summary>

GMM does soft, probabilistic clustering: $p(x) = \sum_k \pi_k \mathcal N(x\mid\mu_k,\Sigma_k)$, fit via EM (E-step: compute responsibilities; M-step: update $\mu_k,\Sigma_k,\pi_k$). It generalizes k-means by allowing elliptical (not just spherical) cluster shapes and fractional/soft cluster assignment.

</details>

<details>
<summary><b>Q: PCA — walk through the steps and how you interpret the first principal component.</b></summary>

Center the data → compute covariance matrix $\Sigma=\frac{1}{n-1}X_c^\top X_c$ → eigen-decompose ($\Sigma v_i = \lambda_i v_i$) → sort eigenvectors by eigenvalue descending. PC1 is the direction of maximum variance in the data; explained variance ratio $= \lambda_i/\sum_j\lambda_j$.

</details>

<details>
<summary><b>Q: t-SNE vs. UMAP — key practical differences?</b></summary>

t-SNE: preserves local structure well but inter-cluster distances aren't meaningful, no transform for new points, slower. UMAP: faster, better preserves global structure, has a topological foundation, and supports transforming new/unseen points.

</details>

<details>
<summary><b>Q: Feature selection vs. feature extraction — what's the tradeoff?</b></summary>

Feature selection (filter/wrapper/embedded methods — correlation, RFE, Lasso, tree importance) keeps the original, interpretable features. Feature extraction (PCA, autoencoders) creates new compressed features that are less interpretable but can capture more signal per dimension.

</details>

---

## 6. Model Evaluation & Feature Engineering (→ `06_model_evaluation_feature_engineering.md`)

<details>
<summary><b>Q: Precision and recall formulas — what question does each answer?</b></summary>

Precision $= TP/(TP+FP)$: of everything predicted positive, how much was actually correct. Recall $=TP/(TP+FN)$: of everything actually positive, how much did we find.

</details>

<details>
<summary><b>Q: Why is F1 the harmonic mean of precision and recall rather than the arithmetic mean?</b></summary>

The harmonic mean punishes imbalance between precision and recall much more heavily than the arithmetic mean — a model with 100% precision and 1% recall would still score well on an arithmetic mean but scores near-zero on F1, which is the desired behavior.

</details>

<details>
<summary><b>Q: What does ROC-AUC actually measure, probabilistically?</b></summary>

$P(\text{a random positive example is ranked above a random negative example})$ by the model's score.

</details>

<details>
<summary><b>Q: Why is PR-AUC often preferred over ROC-AUC on imbalanced datasets?</b></summary>

ROC-AUC's false-positive-rate denominator ($FP+TN$) is dominated by the large negative class, so it can look deceptively good even with many false positives relative to the (small) positive class. PR-AUC's precision denominator ($TP+FP$) directly reflects minority-class pollution.

</details>

<details>
<summary><b>Q: RMSE vs. MAE vs. MAPE — key behavioral differences?</b></summary>

RMSE squares errors — outlier-sensitive, penalizes large errors disproportionately. MAE is linear — outlier-robust. MAPE is a percentage error — blows up or is undefined near $y=0$, and is asymmetric (penalizes over- vs. under-forecasts differently).

</details>

<details>
<summary><b>Q: Why does R² always increase (or stay flat) as you add features, and what fixes that?</b></summary>

R² is non-decreasing with added predictors by construction, even useless/noise ones. Adjusted R² penalizes extra predictors via a $(n-1)/(n-p-1)$ factor, only increasing if a new feature adds more explanatory power than expected by chance.

</details>

<details>
<summary><b>Q: Why are tree ensembles/boosted models often poorly calibrated, and what are the two standard fixes?</b></summary>

Their raw scores reflect leaf-vote proportions or additive margins, not true probabilities. Fixes: Platt scaling (fit a sigmoid on top of the scores — parametric, works with small data) or isotonic regression (fit a flexible monotonic step function — needs more data, can overfit if data is scarce).

</details>

<details>
<summary><b>Q: K-fold cross-validation — how does the choice of k trade off bias and variance of the CV estimate?</b></summary>

Small k (e.g. 5): higher bias, lower variance, cheaper. Large k (approaching LOOCV, k=n): lower bias, higher variance, and much more expensive computationally.

</details>

<details>
<summary><b>Q: What is stratified k-fold, and when is it essential?</b></summary>

It preserves each fold's class ratio to match the overall dataset. Essential under class imbalance — plain k-fold could leave a fold with too few (or zero) minority-class examples.

</details>

<details>
<summary><b>Q: Why is standard k-fold cross-validation invalid for time series data?</b></summary>

It causes temporal leakage (training on "future" data to predict the "past") and violates the assumption of independent folds, since time series data is autocorrelated. Use walk-forward validation instead — expanding window (training set grows each fold) or sliding window (fixed size, moves forward), with an embargo gap if features use rolling windows.

</details>

<details>
<summary><b>Q: How does SMOTE synthesize new minority-class samples?</b></summary>

It interpolates between a minority-class point and one of its nearest minority-class neighbors, creating new synthetic points along that line segment — rather than simply duplicating existing minority rows.

</details>

<details>
<summary><b>Q: Class weighting vs. resampling (SMOTE/undersampling) for imbalance — what's the mechanism difference?</b></summary>

Class weighting reweights the loss function directly (e.g., inversely proportional to class frequency) without touching the data. Resampling methods (SMOTE, undersampling, oversampling) change the training data distribution itself — undersampling loses information, oversampling risks overfitting to duplicated/synthetic points.

</details>

<details>
<summary><b>Q: What's the risk with target encoding for categorical features, and how do you mitigate it?</b></summary>

Leakage — encoding a category using the target itself can leak label information directly into the feature. Mitigate with smoothing (blend with the global mean) and computing the encoding out-of-fold (never using a row's own fold to encode itself).

</details>

<details>
<summary><b>Q: Standardization vs. normalization (min-max scaling) — when do you need either, and when doesn't it matter?</b></summary>

Standardization: $(x-\mu)/\sigma$. Min-max: $(x-x_{\min})/(x_{\max}-x_{\min})$. Both matter for distance-based (k-NN, k-means, SVM) and gradient-based (linear/logistic regression, neural nets) models — but irrelevant for tree-based models, whose splits are scale-invariant.

</details>

<details>
<summary><b>Q: What extra column should you always add when imputing missing data, and why?</b></summary>

A `was_missing` binary indicator flag — "missingness" itself can be informative (e.g., a missing income field might correlate with the target), and the flag preserves that signal even after imputing a value.

</details>

<details>
<summary><b>Q: Give the Shapley value formula for SHAP, and name its three axiomatic guarantees.</b></summary>

$\phi_i = \sum_{S\subseteq F\setminus\{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S\cup\{i\})-f(S)]$ — the feature's average marginal contribution across all possible orderings of the other features. Guarantees: local accuracy (contributions sum to the prediction), consistency (a feature's attribution can't decrease if its marginal contribution increases), and missingness (a feature absent from the model gets zero attribution).

</details>

<details>
<summary><b>Q: Why is SHAP considered superior to plain tree gain/split-count feature importance?</b></summary>

Gain/split-count importance has no theoretical guarantees, is biased toward high-cardinality features, and is global-only (no per-prediction explanation). SHAP is axiomatically justified (proven unique under its three guarantees) and gives both local (per-prediction) and global explanations.

</details>

<details>
<summary><b>Q: LIME vs. SHAP — what's the core methodological difference?</b></summary>

LIME perturbs the input and fits a local weighted linear surrogate model around one prediction — a heuristic approach with no theoretical guarantees and instability across runs. SHAP is grounded in cooperative game theory with proven-unique attribution properties.

</details>

<details>
<summary><b>Q: PDP vs. ICE plots — what does each show, and what's PDP's blind spot?</b></summary>

PDP (Partial Dependence Plot) shows the *averaged* marginal effect of a feature across all instances — assumes feature independence and can extrapolate into implausible feature combinations. ICE (Individual Conditional Expectation) shows one curve per instance, revealing heterogeneous effects that PDP's averaging would hide.

</details>

---

## 7. Time Series Forecasting (→ `07_time_series_forecasting.md`)

<details>
<summary><b>Q: Define weak (covariance) stationarity.</b></summary>

Constant mean over time, constant variance over time, and autocovariance that depends only on the lag between two points (not on absolute time).

</details>

<details>
<summary><b>Q: ADF vs. KPSS test — what's each one's null hypothesis, and why do you run both?</b></summary>

ADF's $H_0$: a unit root is present (series is non-stationary). KPSS's $H_0$ is the *opposite*: the series is stationary. Running both and checking for agreement resolves the ambiguity a single test can leave.

</details>

<details>
<summary><b>Q: Formulas for first-order differencing and seasonal differencing?</b></summary>

First-order: $\nabla X_t = X_t - X_{t-1}$. Seasonal (period $m$): $\nabla_m X_t = X_t - X_{t-m}$.

</details>

<details>
<summary><b>Q: How do you read ACF/PACF plots to identify AR(p) vs. MA(q) order?</b></summary>

AR(p): ACF tails off gradually, PACF cuts off sharply after lag $p$. MA(q): ACF cuts off sharply after lag $q$, PACF tails off gradually. ARMA: both tail off.

</details>

<details>
<summary><b>Q: What do the p, d, q components of ARIMA(p,d,q) represent?</b></summary>

AR(p): regress on the series' own past $p$ values. I(d): order of differencing applied to induce stationarity. MA(q): regress on the past $q$ forecast errors.

</details>

<details>
<summary><b>Q: SARIMA vs. SARIMAX — what does each add on top of ARIMA?</b></summary>

SARIMA adds a seasonal $(P,D,Q)_m$ component operating at lag multiples of the seasonal period $m$. SARIMAX further adds exogenous linear regressors (e.g., promotions, price, holidays, weather).

</details>

<details>
<summary><b>Q: Prophet's additive model formula, and what does `changepoint_prior_scale` control?</b></summary>

$y(t) = g(t) + s(t) + h(t) + \epsilon$ — piecewise linear/logistic trend, Fourier-series seasonality, holiday effects. Higher `changepoint_prior_scale` = more flexible trend (overfitting risk); lower = more rigid trend (underfitting risk).

</details>

<details>
<summary><b>Q: When is Prophet a strong choice, and when does it underperform?</b></summary>

Strong for business time series with clear seasonality, holiday effects, and missing data. Weaker on series with complex autocorrelation structure or very short/high-frequency data, where ARIMA-family or ML models often do better.

</details>

<details>
<summary><b>Q: Simple exponential smoothing (SES) update formula, and what's its limitation?</b></summary>

$\hat y_{t+1} = \alpha y_t + (1-\alpha)\hat y_t$ — a weighted average of the latest observation and the prior forecast. It has no trend or seasonal component, so it forecasts a flat line for any series with trend/seasonality (Holt/Holt-Winters extend it).

</details>

<details>
<summary><b>Q: Holt-Winters — additive vs. multiplicative seasonality, how do you choose?</b></summary>

Additive: seasonal fluctuations have constant absolute amplitude regardless of the series' level. Multiplicative: seasonal amplitude scales proportionally with the level (bigger swings when the series is bigger).

</details>

<details>
<summary><b>Q: Why use Fourier terms (sin/cos) rather than raw integer or one-hot calendar features for seasonality?</b></summary>

Fourier terms $\sin/\cos(2\pi d/m)$ correctly encode cyclical closeness (e.g., December is "close to" January) for both linear and tree-based models — a raw integer or one-hot encoding treats month 12 and month 1 as maximally distant.

</details>

<details>
<summary><b>Q: Why can't tree-based models extrapolate a trend, and what's the fix?</b></summary>

Leaf predictions are bounded averages of training targets, so a tree literally cannot output a value outside the range it saw in training — it can't extrapolate a rising trend into the future. Fix: detrend/difference the target before modeling, or feed an explicit (possibly monotonic-constrained) trend feature.

</details>

<details>
<summary><b>Q: Global vs. local time series models — what's the tradeoff?</b></summary>

Global: one model trained across all series (e.g., all SKUs) — shares statistical strength, scales operationally, handles cold-start naturally, but may underfit idiosyncratic individual series. Local: a separate model per series — captures series-specific patterns well but doesn't scale and fails on short-history series.

</details>

<details>
<summary><b>Q: Name four deep learning architectures for forecasting and one distinguishing feature of each.</b></summary>

LSTM: sequential, gated long-range memory. TFT: attention + variable-selection networks, interpretable, quantile output. N-BEATS: pure residual backcast/forecast stacks, no recurrence/attention, has an interpretable trend+seasonality variant. DeepAR: global autoregressive RNN with likelihood-based probabilistic output.

</details>

<details>
<summary><b>Q: Weighted averaging vs. stacking for ensembling forecasts — what must stacking respect that other ML stacking doesn't need to worry about as much?</b></summary>

Weighted averaging: combine base forecasts via inverse-error weights, NNLS/constrained optimization, or grid search on a validation set. Stacking: a meta-model trained on out-of-fold base predictions — critically, it must respect time order (train the meta-model only on past folds) to avoid temporal leakage.

</details>

<details>
<summary><b>Q: Why is MASE generally recommended as the best general-purpose forecast accuracy metric?</b></summary>

It's scale-independent (comparable across series with different units/magnitudes), well-defined even when actuals include zeros (unlike MAPE), and benchmarks the model against a naive seasonal forecast baseline — so a MASE < 1 means you're beating the naive baseline.

</details>

<details>
<summary><b>Q: Name the four hierarchical forecast reconciliation approaches.</b></summary>

Top-down (forecast the total, disaggregate by historical share), bottom-up (forecast bottom level, sum up), middle-out (forecast a middle level, aggregate up and disaggregate down), MinT (optimal reconciliation using the base-forecast error covariance across all levels simultaneously).

</details>

<details>
<summary><b>Q: How do you forecast for a brand-new product/SKU with no history (cold start)?</b></summary>

Attribute-based analog matching, a global model that can score the new item immediately via shared attribute features, hierarchical/pooled category-share forecasts, or launch-curve templates from historical analogs — blended toward data-driven forecasts as real sales accumulate.

</details>

<details>
<summary><b>Q: Croston's method — what two quantities does it smooth separately, and what does SBA correct?</b></summary>

It separately exponentially smooths the non-zero demand size $\hat z$ and the inter-demand interval $\hat q$, forecasting a rate $=\hat z/\hat q$ held constant between demand occasions. The Syntetos-Boylan Approximation (SBA) multiplies this by $(1-\alpha/2)$ to correct Croston's known over-forecasting bias.

</details>

---

## 8. SQL, PySpark, dbt & Data Engineering (→ `09_sql_pyspark_dbt_data_engineering.md`)

<details>
<summary><b>Q: What's the "fan-out" bug in SQL joins, and how do you fix it?</b></summary>

Joining a "one" side to an un-aggregated "many" side duplicates the one side's rows once per match, inflating any subsequent SUM/COUNT on it. Fix: aggregate the "many" side down to one row per key *before* joining to the "one" side.

</details>

<details>
<summary><b>Q: ROW_NUMBER vs. RANK vs. DENSE_RANK — how do they differ on ties?</b></summary>

ROW_NUMBER: always unique, sequential, arbitrary tiebreak. RANK: ties share the same rank, then skips subsequent rank numbers (a gap). DENSE_RANK: ties share the same rank, no gap in the following numbers.

</details>

<details>
<summary><b>Q: Window function syntax for a running total and a moving average?</b></summary>

Running total: `SUM(x) OVER (ORDER BY x ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`. Moving average: `AVG(x) OVER (ORDER BY x ROWS BETWEEN N PRECEDING AND CURRENT ROW)`.

</details>

<details>
<summary><b>Q: CTE vs. subquery — what's the real difference, and is a CTE always materialized separately?</b></summary>

CTEs primarily improve readability, reusability (referencing the same CTE multiple times), and enable recursion. Materialization behavior is engine/version-specific — e.g., Postgres 12+ inlines CTEs by default unless you force `MATERIALIZED`, so don't assume a CTE is always computed once and cached.

</details>

<details>
<summary><b>Q: General syntax pattern for a recursive CTE, and name two classic use cases.</b></summary>

`WITH RECURSIVE name AS (anchor-query UNION ALL recursive-term-referencing-name) SELECT ...`. Classic uses: generating a date series, and traversing an employee-manager hierarchy.

</details>

<details>
<summary><b>Q: WHERE vs. HAVING — when does each apply, and can HAVING reference aggregates?</b></summary>

WHERE filters individual rows *before* aggregation (can't reference aggregate functions). HAVING filters *groups* after aggregation and can reference aggregate functions like `COUNT(*)` or `SUM(x)`.

</details>

<details>
<summary><b>Q: Why does an index sometimes get ignored by the query planner?</b></summary>

Indexes help most on high-cardinality, selective filters on the *leading* column(s) of a composite index. Low-cardinality columns (e.g., a boolean) or filtering on a non-leading column of a composite index are often not selective enough for the planner to prefer the index over a sequential scan.

</details>

<details>
<summary><b>Q: Why avoid `SELECT *` in production queries?</b></summary>

Reduces unnecessary I/O and network transfer, avoids silent breakage when the underlying schema changes, and enables covering-index / index-only scans that can skip touching the base table entirely.

</details>

<details>
<summary><b>Q: What is partition pruning, and how do you enable it?</b></summary>

The query engine skips scanning irrelevant partitions entirely when you filter on the physical partition key (commonly a date column) — dramatically reducing I/O on large tables.

</details>

<details>
<summary><b>Q: SQL pattern for "top-N rows per group"?</b></summary>

`ROW_NUMBER() OVER (PARTITION BY grp ORDER BY metric DESC)` in a CTE/subquery, then filter `WHERE rn <= N` in the outer query.

</details>

<details>
<summary><b>Q: SQL pattern for "gaps and islands" (finding consecutive-day runs)?</b></summary>

`date - ROW_NUMBER() OVER (PARTITION BY key ORDER BY date)` produces a constant value within each consecutive run — group by that constant to identify islands.

</details>

<details>
<summary><b>Q: SQL pattern for deduplication, keeping only the latest row per key?</b></summary>

`ROW_NUMBER() OVER (PARTITION BY dedup_keys ORDER BY updated_at DESC)` then filter `WHERE rn = 1`; Postgres shortcut: `SELECT DISTINCT ON (dedup_keys) ... ORDER BY dedup_keys, updated_at DESC`.

</details>

<details>
<summary><b>Q: RDD vs. DataFrame API in Spark — which should you default to, and why?</b></summary>

RDD: low-level, no query optimizer, full manual control. DataFrame: has a schema and benefits from the Catalyst optimizer (predicate pushdown, projection pruning, join reordering) — default choice for almost all workloads.

</details>

<details>
<summary><b>Q: `repartition()` vs. `coalesce()` in Spark — what's the tradeoff?</b></summary>

`repartition()`: full shuffle, can increase or decrease partition count, redistributes data evenly (fixes skew) — but costly (network + disk I/O + serialization). `coalesce()`: no full shuffle, can only decrease partition count, much cheaper, but can leave uneven partition sizes since it just merges adjacent partitions.

</details>

<details>
<summary><b>Q: Name three Spark operations that trigger a shuffle.</b></summary>

`groupBy` aggregations across partitions, joins on a key that isn't co-partitioned between the two DataFrames, and `repartition()`/global `sort`/`distinct`.

</details>

<details>
<summary><b>Q: Broadcast join vs. shuffle (sort-merge) join — when does Spark choose each, and how do you force it?</b></summary>

Broadcast join: the small table is copied in full to every executor, avoiding a shuffle of the large table — used when one side is below `spark.sql.autoBroadcastJoinThreshold`. Force explicitly with `broadcast(df)` when Spark's size estimate is wrong (e.g., after filtering).

</details>

<details>
<summary><b>Q: What does Spark's lazy evaluation mean in practice, and what triggers actual execution?</b></summary>

Transformations (`filter`, `select`, `join`, etc.) just build up a logical plan (DAG) — nothing runs yet. Only actions (`collect`, `count`, `write`) trigger execution, at which point the Catalyst optimizer applies predicate pushdown, projection pruning, and join reordering to the physical plan.

</details>

<details>
<summary><b>Q: dbt's four materializations — what does each produce?</b></summary>

View (no storage, always fresh query), table (full rebuild each run), incremental (only processes new/changed rows), ephemeral (inlined as a CTE into downstream models, no warehouse object created).

</details>

<details>
<summary><b>Q: dbt's three incremental strategies — when do you use each?</b></summary>

Append: for immutable event data (no updates, just new rows). Merge: upsert by a unique key (handles updates to existing rows). Delete+insert: partition-level replace, used when the warehouse doesn't support native MERGE.

</details>

<details>
<summary><b>Q: How do you fix a Spark join suffering from data skew (one key has way more rows than others)?</b></summary>

Salt the skewed side: append a random bucket number to the join key so the heavy key's rows spread across N hash-partitioned buckets instead of overwhelming one task; explode the small/dimension side so every salt bucket has a matching row to join against.

</details>

---

## 9. MLOps & Cloud Deployment (→ `10_mlops_cloud_deployment.md`)

<details>
<summary><b>Q: MLflow Tracking vs. Model Registry — what does each manage?</b></summary>

Tracking: logs params, metrics, and artifacts per experiment run. Model Registry: manages model *versions* through lifecycle stages (None → Staging → Production → Archived, or named aliases like `@champion`).

</details>

<details>
<summary><b>Q: What three things make an ML experiment reproducible in MLflow-style workflows?</b></summary>

A git commit tag (code version), an auto-captured environment/dependency file, and a manually-tagged data version — together these let you reconstruct exactly what produced a given model.

</details>

<details>
<summary><b>Q: Flask vs. FastAPI for model serving — what's the real difference, and does async actually speed up CPU-bound inference?</b></summary>

Flask: WSGI, synchronous. FastAPI: ASGI, native `async def`, built-in Pydantic request validation, auto-generated Swagger docs. Async helps with I/O-bound concurrent serving (e.g., waiting on a downstream call) but does *not* speed up raw CPU-bound prediction compute.

</details>

<details>
<summary><b>Q: Docker image vs. container — what's the distinction?</b></summary>

Image: an immutable build artifact/blueprint. Container: a running (or stopped) instance of that image.

</details>

<details>
<summary><b>Q: Why use multi-stage Docker builds for ML services?</b></summary>

The builder stage compiles/installs dependencies (may need heavy build tools like gcc); the runtime stage copies over only the finished artifacts — resulting in a much smaller final image with a smaller attack surface.

</details>

<details>
<summary><b>Q: Name two ML-specific testing practices that go beyond standard software CI/CD.</b></summary>

Data schema validation (e.g., pandera, Great Expectations) to catch upstream data changes, and champion-challenger quality gates that compare a new model's offline/online metrics against the current production model before promoting it.

</details>

<details>
<summary><b>Q: Data drift vs. concept drift — what changes in each, and how do you detect them?</b></summary>

Data drift: $P(x)$ (the input feature distribution) changes — detect via PSI (>0.25 typically flags a major shift) or KS-test, no labels needed. Concept drift: $P(y\mid x)$ changes even if $P(x)$ doesn't — detect via live performance vs. delayed ground truth, or proxy signals like prediction-distribution shift when labels are delayed/sparse.

</details>

<details>
<summary><b>Q: What are the three general triggers for model retraining?</b></summary>

Scheduled (fixed cadence), data-volume-based (retrain once N new labeled examples accumulate), and drift-triggered (retrain when monitoring detects data or concept drift). Mature systems typically combine all three.

</details>

<details>
<summary><b>Q: In AWS, what's the difference between ECS, Fargate, Step Functions, and SageMaker Pipelines?</b></summary>

ECS: container orchestration without managing a Kubernetes control plane. Fargate: serverless per-task compute, ideal for bursty/periodic batch jobs. Step Functions: general-purpose state-machine orchestration with built-in retry/branching. SageMaker Pipelines: ML-native DAG orchestration with step caching and model registry integration.

</details>

<details>
<summary><b>Q: How does GCP's Vertex AI compare to AWS's SageMaker at a high level?</b></summary>

Vertex AI is a more unified platform (pipelines, registry, endpoints, and model monitoring bundled into one product); SageMaker is more modular, stitching together several separate AWS services for the equivalent functionality.

</details>

<details>
<summary><b>Q: Why does BigQuery pricing push you toward partitioning and avoiding SELECT *?</b></summary>

BigQuery is a serverless columnar warehouse billed by bytes scanned — partitioning, clustering, avoiding `SELECT *`, and using approximate aggregation functions all directly reduce the bytes scanned and therefore the cost.

</details>

<details>
<summary><b>Q: Git-flow vs. trunk-based development — which fits continuous ML deployment better?</b></summary>

Git-flow: long-lived develop/release branches, suited to scheduled/versioned releases. Trunk-based: short-lived feature branches merged frequently into main — fits continuous deployment (and therefore ML CI/CD) better.

</details>

<details>
<summary><b>Q: Rebase vs. merge — when is each safe to use?</b></summary>

Rebase rewrites commit history into a linear sequence — safe only on local/private branches nobody else has pulled. Merge preserves true history via a two-parent merge commit — safe (and required) on shared/public branches; never rebase a branch others are already working from.

</details>

---

## 10. NLP & Deep Learning Fundamentals (→ `11_nlp_and_deep_learning_fundamentals.md`)

<details>
<summary><b>Q: Stemming vs. lemmatization — give an example that shows the difference.</b></summary>

Stemming: crude rule-based suffix stripping, fast, can produce non-words (e.g., "better" stays "better" under a simple stemmer). Lemmatization: dictionary/POS-aware reduction to a valid base word (e.g., "better" → "good").

</details>

<details>
<summary><b>Q: What's the danger of blindly removing stopwords before sentiment analysis?</b></summary>

Standard stopword lists often include negators like "not," "no," "never" — removing them can flip or destroy the sentiment/meaning of a sentence ("not good" → "good").

</details>

<details>
<summary><b>Q: TF-IDF formula, and what does the IDF term accomplish?</b></summary>

$tfidf(t,d) = tf(t,d)\times\log(N/df(t))$. The IDF factor zeroes out (or heavily down-weights) words appearing in nearly every document, boosting words that are rare and therefore discriminative.

</details>

<details>
<summary><b>Q: Main limitations of Bag-of-Words / TF-IDF representations?</b></summary>

No semantic meaning captured, no word order/context, and the resulting vectors are sparse and very high-dimensional.

</details>

<details>
<summary><b>Q: CBOW vs. Skip-gram in Word2Vec — what does each predict, and when does each work better?</b></summary>

CBOW predicts the center word from its surrounding context — faster, works well for frequent words. Skip-gram predicts the surrounding context from the center word — better for rare words and smaller datasets.

</details>

<details>
<summary><b>Q: GloVe vs. Word2Vec — what's the fundamental difference in how they learn embeddings?</b></summary>

GloVe factorizes a global word co-occurrence matrix via weighted least squares (count-based, uses global corpus statistics). Word2Vec predicts words from local context windows online (local, prediction-based).

</details>

<details>
<summary><b>Q: What problem does FastText solve that Word2Vec/GloVe can't?</b></summary>

FastText represents words as bags of character n-grams, so it can produce reasonable embeddings for out-of-vocabulary, rare, or morphologically complex words — Word2Vec/GloVe have no representation at all for words unseen during training.

</details>

<details>
<summary><b>Q: Vanilla RNN hidden-state update formula, and why do vanishing gradients occur during training?</b></summary>

$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$. Backpropagation through time multiplies many Jacobians together (each involving $\tanh'$ times $W_{hh}$, typically <1 in magnitude), so the gradient shrinks exponentially as sequence length grows.

</details>

<details>
<summary><b>Q: Why does LSTM's cell-state mechanism mitigate vanishing gradients better than a vanilla RNN?</b></summary>

The cell state update $C_t = f_t\odot C_{t-1} + i_t\odot\tilde C_t$ is *additive* rather than purely multiplicative, so gradients can flow through the cell state across many timesteps with far less exponential decay.

</details>

<details>
<summary><b>Q: GRU vs. LSTM — what's structurally different, and what's the practical tradeoff?</b></summary>

GRU merges the forget/input gates into a single update gate and a reset gate, with no separate cell state — fewer parameters, faster to train, comparable performance to LSTM in most tasks.

</details>

<details>
<summary><b>Q: What problem in seq2seq models does the attention mechanism fix?</b></summary>

The fixed-length context vector bottleneck — a vanilla encoder-decoder compresses the entire input sequence into one fixed-size vector, losing information for long sequences. Attention lets the decoder compute alignment scores → softmax weights → a weighted sum over *all* encoder hidden states at each decoding step.

</details>

<details>
<summary><b>Q: How does the attention mechanism act as a conceptual bridge to Transformer self-attention?</b></summary>

The score → softmax → weighted-sum pattern used in seq2seq attention is the direct ancestor of Transformer self-attention's Q/K/V mechanics — the same "compute relevance, normalize, aggregate" idea, generalized and made fully parallelizable.

</details>

<details>
<summary><b>Q: Batch normalization formula, and what effect does it have on training?</b></summary>

$\hat x = (x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$, then $y=\gamma\hat x+\beta$. It smooths the loss landscape, enables higher learning rates, and provides mild regularization; at test time it uses running statistics accumulated during training, not the current batch's.

</details>

<details>
<summary><b>Q: How does dropout work, and why doesn't test-time inference need any changes (inverted dropout)?</b></summary>

During training, activations are zeroed with probability $p$ and survivors scaled up by $1/(1-p)$ ("inverted dropout") — this scaling means the expected activation magnitude matches what test time sees, so no rescaling is needed at inference. Prevents co-adaptation of neurons, approximating an ensemble of many sub-networks.

</details>

<details>
<summary><b>Q: What's the single most critical hyperparameter to tune in deep learning, and why?</b></summary>

Learning rate — too high causes divergence/instability, too low causes stalled/slow training. Learning rate schedules and warmup are standard mitigations. Random search or Bayesian optimization generally beat grid search for the same compute budget.

</details>

---

## 11. GenAI, LLMs & Transformers (→ `12_genai_llms_transformers.md`)

<details>
<summary><b>Q: Full formula for scaled dot-product attention, and why scale by $\sqrt{d_k}$?</b></summary>

$\text{softmax}(QK^\top/\sqrt{d_k})V$. Dot-product magnitude grows with $d_k$; without scaling, large scores would push softmax into a saturated regime with near-zero gradients — dividing by $\sqrt{d_k}$ keeps the pre-softmax variance roughly constant.

</details>

<details>
<summary><b>Q: Multi-head attention formula, and why use multiple heads instead of one large attention operation?</b></summary>

$\text{Concat}(\text{head}_1,\dots,\text{head}_h)W_O$. Multiple heads let the model specialize in capturing different kinds of relationships (e.g., syntactic vs. coreference) in parallel subspaces, instead of averaging everything into a single attention pattern.

</details>

<details>
<summary><b>Q: Why do Transformers need explicit positional encoding, and sinusoidal vs. learned — what's the tradeoff?</b></summary>

Self-attention is permutation-invariant by construction — without positional information, "the cat sat" and "sat cat the" would look identical to the attention mechanism. Sinusoidal encodings support relative-position linear algebra and can be evaluated at any position (even beyond training length); learned positional embeddings are more flexible but don't extrapolate past the trained max length.

</details>

<details>
<summary><b>Q: Encoder-only vs. decoder-only vs. encoder-decoder architectures — give an example model and typical use case for each.</b></summary>

Encoder-only (bidirectional, e.g. BERT): understanding tasks (classification, embeddings). Decoder-only (causal, e.g. GPT): generation tasks. Encoder-decoder (both + cross-attention, e.g. T5): sequence-to-sequence tasks like translation/summarization.

</details>

<details>
<summary><b>Q: LayerNorm — what axis does it normalize over, and pre-norm vs. post-norm?</b></summary>

Normalizes per-token across the feature dimension (not across the batch, unlike BatchNorm) — suits variable-length sequences. Pre-norm (normalize before the sublayer) is more stable for training very deep/large-scale Transformers than post-norm.

</details>

<details>
<summary><b>Q: Residual connection formula, and why does it matter in deep Transformer stacks?</b></summary>

$x + \text{Sublayer}(x)$ — the identity shortcut keeps gradients flowing directly backward through many layers, preventing vanishing gradients in very deep stacks.

</details>

<details>
<summary><b>Q: BPE vs. WordPiece vs. SentencePiece — what's the core difference in how each builds its vocabulary?</b></summary>

BPE: iteratively merges the most frequent adjacent byte/character pair. WordPiece: merges pairs that maximize training-corpus likelihood (not just frequency). SentencePiece: operates directly on raw text without assuming whitespace pre-tokenization — language-agnostic, important for languages like Chinese/Japanese with no spaces.

</details>

<details>
<summary><b>Q: MLM vs. CLM pretraining objectives — which model type uses each, and why does the choice matter for downstream use?</b></summary>

MLM (BERT): predicts randomly masked tokens using bidirectional context — good for understanding/embedding tasks. CLM (GPT): predicts the next token autoregressively, $P(x)=\prod_t P(x_t\mid x_{<t})$ — matches how the model is actually used at inference (generation), which is why GPT-style models dominate generative use cases.

</details>

<details>
<summary><b>Q: Fine-tuning vs. prompting vs. RAG — when would you reach for each?</b></summary>

Fine-tuning: narrow, well-defined tasks with ample labeled data, or when you need to change model behavior/format/style. Prompting: fast, no-training-needed tasks the base model can already mostly do. RAG: when you need fresh or proprietary/grounded knowledge without retraining the model.

</details>

<details>
<summary><b>Q: LoRA formula, and why is it so efficient?</b></summary>

$W = W_0 + BA$ — freeze the pretrained weights $W_0$, train only low-rank matrices $A,B$. Exploits the empirical finding that fine-tuning updates have low intrinsic rank; at inference, $BA$ can be merged into $W_0$ with zero added latency.

</details>

<details>
<summary><b>Q: What does QLoRA add on top of LoRA?</b></summary>

4-bit quantization of the frozen base model, double quantization (quantizing the quantization constants themselves), and paged optimizers — together enabling fine-tuning of very large models on limited GPU memory.

</details>

<details>
<summary><b>Q: Adapters vs. prefix-tuning vs. prompt-tuning — how do they differ structurally?</b></summary>

Adapters: small bottleneck feed-forward modules inserted between Transformer layers (adds inference latency, can't be merged away). Prefix-tuning: trainable "virtual token" vectors injected at every layer. Prompt-tuning: trainable virtual tokens only at the embedding/input layer (lighter-weight than prefix-tuning).

</details>

<details>
<summary><b>Q: Walk through the RLHF pipeline end to end.</b></summary>

(1) Supervised fine-tuning (SFT) on demonstration data. (2) Train a reward model on human preference pairs using a Bradley-Terry pairwise loss. (3) Optimize the policy with PPO against the reward model, with a KL penalty against the SFT model to prevent reward hacking/degeneration.

</details>

<details>
<summary><b>Q: How does DPO avoid needing a separate reward model or an RL loop?</b></summary>

DPO derives a direct classification-style loss on preference pairs mathematically equivalent to the RLHF objective, using the ratio of the policy's log-probabilities to a frozen reference model's as an implicit reward — no separate reward model training, no PPO/RL loop, generally more stable to train.

</details>

<details>
<summary><b>Q: How is instruction tuning different from RLHF/DPO?</b></summary>

Instruction tuning is supervised fine-tuning on a diverse set of (instruction, response) pairs to teach general instruction-following. RLHF/DPO are separate stages layered on top that optimize for human *preference* between candidate responses, not just imitation of a single reference response.

</details>

<details>
<summary><b>Q: What does the Chinchilla scaling law say, and what was "wrong" about GPT-3-era model training?</b></summary>

Compute-optimal training scales model parameters and training tokens roughly proportionally (~20 tokens per parameter). GPT-3-era models were significantly undertrained relative to their parameter count — a smaller model trained on proportionally more tokens would have performed better for the same compute budget.

</details>

<details>
<summary><b>Q: RoPE vs. ALiBi — how does each let a model extrapolate beyond its trained context length?</b></summary>

RoPE: rotates Q/K vectors by a position-dependent angle so the dot product depends only on the *relative* offset between tokens. ALiBi: subtracts a distance-proportional penalty directly from the raw attention scores. Both extrapolate better than absolute positional encodings; ALiBi is often the strongest raw extrapolator.

</details>

<details>
<summary><b>Q: Order the prompting techniques from simplest to most sophisticated: zero-shot, few-shot, CoT, self-consistency, Tree-of-Thought, ReAct.</b></summary>

Zero-shot (instruction only) → few-shot (add in-context examples) → CoT (elicit intermediate reasoning tokens as scratch space) → self-consistency (majority vote across multiple sampled CoT paths) → Tree-of-Thought (branching search with backtracking over reasoning paths) → ReAct (interleave reasoning steps with tool actions/observations).

</details>

<details>
<summary><b>Q: What's the underlying purpose of separating system prompts from user prompts?</b></summary>

System prompts set persistent behavior/persona/constraints defined by the application (higher trust/priority); user prompts are per-turn task input (lower trust). This separation underlies both instruction-priority handling and prompt-injection defenses.

</details>

<details>
<summary><b>Q: Name three concrete defenses against prompt injection.</b></summary>

Delimit trusted instructions from untrusted content (so retrieved/user text can't masquerade as a system instruction), validate/sanitize model outputs before acting on them, and enforce least-privilege access for any tools the model can call.

</details>

---

## 12. RAG, Agents & LLM Systems (→ `13_rag_agents_llm_systems.md`)

<details>
<summary><b>Q: RAG vs. fine-tuning — how do you decide, and when would you use both?</b></summary>

RAG injects facts at query time — cheap, keeps knowledge fresh, and answers are grounded/citable. Fine-tuning changes the model's behavior/style/reasoning patterns, needed when the task requires a different output format or domain-specific reasoning the base model can't be prompted into. Many production systems use both together.

</details>

<details>
<summary><b>Q: Fixed-size vs. semantic chunking — what's the tradeoff, and what's a typical overlap percentage?</b></summary>

Fixed-size: simple to implement, but risks splitting a coherent idea across chunk boundaries. Semantic chunking: splits along natural or embedding-similarity boundaries, variable chunk size, better idea coherence but more complex. Typical overlap: ~10–20% to preserve context across boundaries.

</details>

<details>
<summary><b>Q: HNSW vs. IVF indexing — what's each one's structure, and the practical tradeoff?</b></summary>

HNSW: graph-based navigable small-world index — generally the best recall/speed default. IVF: cluster-based (Voronoi cell) index — faster to build and lower memory, but can miss neighbors near cluster boundaries.

</details>

<details>
<summary><b>Q: Cosine similarity vs. dot product vs. L2 distance for vector search — when are cosine and dot product equivalent?</b></summary>

Cosine measures angle only (normalization-invariant); dot product is magnitude-sensitive and computationally fastest; L2 is straight-line Euclidean distance. Dot product equals cosine similarity when vectors are pre-normalized to unit length.

</details>

<details>
<summary><b>Q: Recall@k vs. MRR vs. NDCG for retrieval evaluation — what does each emphasize?</b></summary>

Recall@k: what fraction of relevant items appear anywhere in the top k (coverage). MRR: rewards ranking the *first* relevant result as high as possible (good when there's typically one right answer). NDCG: accounts for graded relevance levels and discounts based on rank position (best when relevance isn't just binary).

</details>

<details>
<summary><b>Q: What is hybrid search, and why add reranking on top of it?</b></summary>

Hybrid search fuses dense/semantic retrieval (paraphrase-aware) with sparse/BM25 keyword retrieval (exact terms, acronyms) via something like Reciprocal Rank Fusion (RRF). A cross-encoder reranker is then applied only to the resulting shortlist — it's accurate but too slow to run over the full corpus, so it's reserved for re-scoring a small candidate set from fast first-stage retrieval.

</details>

<details>
<summary><b>Q: Why does a single retrieval pass often fail on multi-hop questions, and what fixes it?</b></summary>

A question that requires chaining facts across multiple documents can't be answered by one similarity search against the query alone — the second "hop" fact isn't semantically close to the original question. Fixes: iterative/multi-step retrieval, query decomposition into sub-questions, or knowledge-graph-augmented retrieval.

</details>

<details>
<summary><b>Q: Name three RAG failure modes and a fix for each.</b></summary>

Irrelevant retrieval (fix: better chunking/embeddings/hybrid search); context overflow (fix: better reranking/compression before feeding the LLM); stale or version-mismatched embeddings (fix: re-index on document change, pin or migrate embedding model versions consistently between index-build and query time).

</details>

<details>
<summary><b>Q: LangChain vs. LangGraph — what problem does LangGraph solve that plain chains can't?</b></summary>

LangChain chains are linear or DAG-shaped pipelines. LangGraph models the workflow as an explicit state graph with conditional edges and native cycles — needed for agentic loops like retry, re-plan, or repeated tool-use until a stopping condition is met.

</details>

<details>
<summary><b>Q: Describe the tool-calling loop step by step.</b></summary>

The model is given tool schemas → it decides a tool call is needed → emits a structured call → the tool executes → the result is fed back to the model as an observation → the model continues reasoning or produces a final answer.

</details>

<details>
<summary><b>Q: Short-term vs. long-term agent memory — what's the mechanism and tradeoff for each?</b></summary>

Short-term: a raw conversation buffer kept in the context window — bounded by context length, but free (no extra infrastructure). Long-term: vector-store-backed semantic recall — effectively unbounded, but requires retrieval infrastructure and adds latency.

</details>

<details>
<summary><b>Q: Planner-executor vs. supervisor-worker multi-agent patterns — what's the difference?</b></summary>

Planner-executor: decomposes a task into an ordered sequence of sub-tasks that get executed in order. Supervisor-worker: routes sub-tasks to specialized peer agents and aggregates their results, without necessarily a strict linear order.

</details>

<details>
<summary><b>Q: What's the key guardrail principle for content retrieved by a RAG system or returned by a tool call?</b></summary>

Treat retrieved/tool content as untrusted *data*, never as trusted *instructions* — validate structured outputs against a schema/business rules before executing them, and rate-limit or gate high-risk tool calls.

</details>

<details>
<summary><b>Q: Name three concrete techniques to reduce or detect hallucination in an LLM system.</b></summary>

Grounding via RAG, lowering temperature, tuning the model to say "I don't know" when uncertain, and requiring citations. Detection: faithfulness/entailment checks against the retrieved context, or self-consistency across multiple samples.

</details>

<details>
<summary><b>Q: What are the known weaknesses of using an LLM as a judge for evaluation?</b></summary>

Self-preference bias (favoring outputs similar to its own style) and positional bias (favoring whichever answer appears first/second in the prompt) — despite being fast, cheap, scalable, and reasonably correlated with human judgment, it should be periodically calibrated/spot-checked against real human evaluation.

</details>

<details>
<summary><b>Q: Why are BLEU/ROUGE insufficient for evaluating RAG system correctness?</b></summary>

They measure surface-level n-gram overlap only — they can't tell if a response is factually correct or grounded in the retrieved context. Faithfulness/groundedness metrics specifically check whether the response's claims are supported by the retrieved context, which is the metric that actually matters for RAG.

</details>

<details>
<summary><b>Q: What's different about A/B testing a GenAI feature vs. a traditional feature?</b></summary>

Same core statistical principles apply, but you also need to account for model non-determinism (larger samples or repeated sampling per input), rely more on proxy metrics (thumbs up/down, completion rate) since "correctness" is harder to label at scale, and report latency/cost jointly with quality metrics.

</details>

<details>
<summary><b>Q: Name three levers for controlling LLM inference cost and latency.</b></summary>

Model cascades (route simple queries to a small/cheap model, complex ones to a large model), quantization (INT8/INT4, shrinking weight precision), and distillation (train a small student model from a large teacher's outputs).

</details>

---

## 13. ML System Design (→ `14_system_design_ml.md`)

<details>
<summary><b>Q: What's the recommended 8-step framework to structure any ML system design answer?</b></summary>

(1) Problem clarification — target, objective, success metric, constraints. (2) Data availability. (3) Feature engineering (no leakage). (4) Model choice — simplest baseline first. (5) Training/validation strategy — correct splits. (6) Deployment architecture — batch vs. real-time. (7) Monitoring/feedback loop. (8) Scaling considerations — what breaks at 10x/100x.

</details>

<details>
<summary><b>Q: For large-scale demand forecasting across many SKUs, what's the single most important architectural choice to lead with?</b></summary>

Use one **global** model across all SKUs rather than millions of per-SKU local models — this is the only practical way to handle cold-start SKUs and keep training/maintenance operationally sane at scale.

</details>

<details>
<summary><b>Q: For real-time fraud detection (<100ms), what's the key architectural pattern?</b></summary>

A **two-stage architecture**: a cheap, fast-path model handles ~100% of traffic; an expensive slow-path model or human review is reserved only for the ambiguous middle segment — driven by the asymmetric cost of false positives vs. false negatives.

</details>

<details>
<summary><b>Q: For a recommendation system, what's the standard high-level architecture, and why not rank the whole catalog directly?</b></summary>

Candidate generation (high recall, ANN search over a large catalog) → ranking (high precision, rich features on a small shortlist). Ranking the full catalog directly with a rich model is computationally infeasible at scale; cold-start items are handled via content-based fallback until collaborative signal accumulates.

</details>

<details>
<summary><b>Q: For marketing attribution, why isn't multi-touch attribution (rules/Markov/Shapley) enough on its own?</b></summary>

Multi-touch attribution is purely correlational. Only randomized geo-holdout / incrementality experiments establish true causal lift — the always-on attribution layer should be calibrated against periodic experiments, not trusted alone.

</details>

<details>
<summary><b>Q: For a RAG-based customer support chatbot, what's the key design principle beyond standard RAG?</b></summary>

Access-control-aware retrieval (permission filtering must happen at the retrieval layer, not after) plus groundedness-gated escalation — low-confidence or ungrounded answers should escalate to a human rather than risk a confident hallucination.

</details>

<details>
<summary><b>Q: For ad-budget optimization, how do linear programming and multi-armed bandits complement each other?</b></summary>

LP gives a globally optimal allocation *given known response curves*, but can't handle uncertainty about those curves. Bandits (Thompson Sampling/UCB) solve the explore-exploit problem LP can't. Strongest answer: a hybrid — LP for the primary allocation, with a reserved bandit slice continually refining the response curves feeding back into the LP.

</details>

<details>
<summary><b>Q: In a system design interview, what should you clarify before jumping to model choice?</b></summary>

The business objective and target metric, explicit scope/constraints (latency, cost, regulatory), what data actually exists (labeled or not, volume/freshness/quality), and what "success" means operationally — not just technically.

</details>

<details>
<summary><b>Q: How do you decide between batch and real-time deployment architecture in a system design answer?</b></summary>

Driven by the latency requirement and cost tolerance of the use case — real-time/streaming for scenarios needing sub-second decisions (fraud detection), batch for scenarios where daily/hourly freshness suffices and simplicity/cost matter more (demand forecasting).

</details>

<details>
<summary><b>Q: What's the most common feature-engineering mistake to explicitly guard against in a system design interview?</b></summary>

Feature leakage — using a feature that would not actually be available at inference time (e.g., a label-derived aggregate, or a "future" value relative to the prediction point).

</details>

<details>
<summary><b>Q: What should a strong "monitoring/feedback loop" answer include for any ML system design?</b></summary>

Drift detection (data and/or concept drift), defined retraining triggers, and online proxy metrics to catch problems before ground-truth labels arrive (which can be delayed by days/weeks in many real systems).

</details>

<details>
<summary><b>Q: What should you discuss under "scaling considerations" in a system design answer?</b></summary>

What specifically breaks at 10x or 100x current scale (e.g., feature store latency, model serving throughput, retraining pipeline runtime) and the concrete mitigation for each — not just a vague "it would scale using more servers."

</details>

</details>
