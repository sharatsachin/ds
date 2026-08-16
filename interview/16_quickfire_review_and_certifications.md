# Quick-Fire Review & Certification Talking Points

This is the final rapid-recall pass of the kit — read it the morning of, or the night before, the interview, not as a first exposure to the material. Every row below assumes you have already studied the full derivation in its companion file; here you get only the formula, the one-line meaning, and the trap interviewers set. Part 1 preps you to defend your certifications and degrees in conversation; Part 2 is a wall-to-wall concept sweep across the kit's topic files; Part 3 is the literal walk-in-the-door checklist; Part 4 points you to the practice tools added later in the kit (flashcards, problem sets, case studies, mock-interview drills).

## Table of Contents

1. [Part 1 — Certifications & Academic Background](#part-1--certifications--academic-background)
   - 1.1 [AWS Certified Machine Learning – Specialty: depth beyond the resume line](#11-aws-certified-machine-learning--specialty-depth-beyond-the-resume-line)
   - 1.2 [Deep Learning Specialization refresh pointer](#12-deep-learning-specialization-refresh-pointer)
   - 1.3 [MTech / BTech: defending your thesis and coursework](#13-mtech--btech-defending-your-thesis-and-coursework)
2. [Part 2 — Master Quick-Fire Concept Review](#part-2--master-quick-fire-concept-review)
   - 2.1 [Statistics & Probability (→ file 01)](#21-statistics--probability--file-01)
   - 2.2 [Hypothesis Testing & A/B Testing (→ file 02)](#22-hypothesis-testing--ab-testing--file-02)
   - 2.3 [ML Fundamentals (→ file 03)](#23-ml-fundamentals--file-03)
   - 2.4 [Trees & Boosting (→ file 04)](#24-trees--boosting--file-04)
   - 2.5 [Other ML Algorithms (→ file 05)](#25-other-ml-algorithms--file-05)
   - 2.6 [Model Evaluation & Feature Engineering (→ file 06)](#26-model-evaluation--feature-engineering--file-06)
   - 2.7 [Time Series & Forecasting (→ file 07)](#27-time-series--forecasting--file-07)
   - 2.9 [SQL / PySpark / dbt (→ file 09)](#29-sql--pyspark--dbt--file-09)
   - 2.10 [MLOps & Cloud (→ file 10)](#210-mlops--cloud--file-10)
   - 2.11 [NLP & DL Fundamentals (→ file 11)](#211-nlp--dl-fundamentals--file-11)
   - 2.12 [GenAI & Transformers (→ file 12)](#212-genai--transformers--file-12)
   - 2.13 [RAG & Agents (→ file 13)](#213-rag--agents--file-13)
   - 2.14 [System Design (→ file 14)](#214-system-design--file-14)
   - 2.15 [Case Studies & Use-Case Reasoning (→ file 19)](#215-case-studies--use-case-reasoning--file-19)
3. [Part 3 — Final Pre-Interview Checklist](#part-3--final-pre-interview-checklist)
4. [Part 4 — Practice Tools in This Kit](#part-4--practice-tools-in-this-kit)

---

## Part 1 — Certifications & Academic Background

### 1.1 AWS Certified Machine Learning – Specialty: depth beyond the resume line

Interviewers who see this cert on a resume often probe one level below "I know SageMaker" to check the certification wasn't just memorized multiple-choice trivia. Be ready to explain *why* each service exists and *when* you'd reach for it, not just its name.

| Capability | What it actually does | Why an interviewer probes it |
|---|---|---|
| **Built-in algorithm containers** | SageMaker ships pre-built, horizontally-scalable containers for common algorithms: **XGBoost** (as a standalone managed container, not just the OSS library — gets distributed training + spot instance support for free), **Linear Learner** (linear/logistic regression with built-in L1/L2 and automatic threshold tuning for imbalanced classes), **Random Cut Forest** (unsupervised anomaly detection via tree-based isolation scoring, used for streaming/time-series outliers), **Object2Vec** (general-purpose embedding learner for pairs of objects — sentences, users-items — when you need embeddings but don't have a pretrained model). | Tests whether you understand these are managed, scaled infra wrappers around known algorithms, not "AWS's own ML" — and that you know which one fits which problem shape (e.g., RCF for fraud/IoT anomaly detection vs. a supervised classifier). |
| **SageMaker Feature Store** | A managed repository with an **online store** (low-latency key-value lookup, typically backed by DynamoDB, for real-time inference) and an **offline store** (append-only, backed by S3 + Glue/Athena, for training and batch use). The core value proposition is guaranteeing **training/serving feature consistency** — the same feature computation feeds both stores so you avoid training-serving skew. | Interviewers ask "how do you stop features differing between training and production?" — this is the textbook AWS answer, and it maps to the general MLOps concept of a feature store (see file 10). |
| **SageMaker Model Monitor** | Automatically captures inference request/response payloads and compares live data statistics against a training-time baseline to flag **data quality drift**, and can be extended with Model Quality, Bias Drift, and Feature Attribution Drift monitors. Runs on a schedule (e.g., hourly) and emits CloudWatch alarms. | This is the concrete AWS-native answer to the generic "how do you detect drift in production?" interview question — pair it with the data drift vs. concept drift distinction in file 10. |
| **SageMaker Clarify** | Provides **pre-training bias metrics** (e.g., class imbalance, KL divergence between groups) and **post-training bias metrics** (e.g., disparate impact, recall difference across groups), plus **SHAP-based explainability** reports integrated into training jobs and endpoints. | Tests whether you can connect "responsible AI / fairness" buzzwords to a concrete tool, and whether you know Clarify literally computes SHAP values under the hood (ties to file 06). |
| **SageMaker Ground Truth** | Managed **data labeling** service: routes unlabeled data to human labelers (your own workforce, Mechanical Turk, or vendor workforces) with active-learning-based **automated labeling** that uses a model to auto-label "easy" examples once confidence is high, sending only hard examples to humans. | Interviewers use this to check you understand labeling cost is a real bottleneck and that "auto-labeling with human-in-the-loop" is a standard mitigation, not something you invented on the spot. |
| **Automatic Model Tuning (HPO)** | Runs **Bayesian optimization** (a Gaussian-Process-based sequential search) over a defined hyperparameter range, treating each training job as an expensive black-box evaluation — smarter than grid/random search because it uses prior results to choose the next candidate point. | Common trap: interviewer asks "grid search vs. random search vs. Bayesian — which does AWS use by default and why is it more sample-efficient?" Answer: Bayesian, because it models the objective surface and balances exploration/exploitation instead of sampling blindly. |

### 1.2 Deep Learning Specialization refresh pointer

The deeplearning.ai specialization is foundational, not applied-NLP-specific — expect an interviewer to sanity-check that the basics are still fresh even if your day job is more classical ML / GenAI-application work.

| Course topic | One-line reminder | Full depth in |
|---|---|---|
| Neural network basics (forward/backprop) | Backprop is repeated chain-rule application layer-by-layer to get $\partial L/\partial W$ for every weight. | File 03 (ML fundamentals — optimization) |
| Hyperparameter tuning, batch norm, regularization (Course 2) | Batch norm normalizes layer activations to zero mean/unit variance per mini-batch, then rescales with learnable $\gamma,\beta$ — stabilizes and speeds up training. Dropout randomly zeroes activations at training time to prevent co-adaptation. | File 11 (NLP & DL fundamentals) |
| Structuring ML projects (Course 3) | Train/dev/test set splitting strategy, human-level performance as a bias/variance proxy, error analysis on misclassified examples. | File 03, File 06 |
| CNNs (Course 4) | Convolution = learned local filters sliding over the input; pooling downsamples for translation invariance; used for grid-structured data (images, and 1D conv for sequences). | File 11 |
| RNNs / LSTMs / GRUs (Course 5) | Vanilla RNNs suffer vanishing gradients over long sequences; LSTM/GRU gates (forget/input/output) let gradients flow through a near-linear cell state, mitigating this. | File 11 |
| Attention & sequence models (Course 5, later material) | Attention lets a decoder look back at all encoder states weighted by relevance instead of compressing everything into one fixed vector — direct precursor to the Transformer. | File 12 (GenAI & Transformers) |

### 1.3 MTech / BTech: defending your thesis and coursework

You will very likely get "tell me about your thesis/capstone" or "what was the most interesting project in your MTech." A strong 60-second answer has four beats, in this order:

1. **The problem** (10-15s): state it as a real decision or business/research question, not a dataset name — "the goal was to predict X so that Y decision could be made," not "I used dataset Z."
2. **Why it was hard** (15-20s): name the actual obstacle — noisy/imbalanced labels, small sample size, non-stationarity, computational constraints, conflicting objectives (accuracy vs. interpretability) — this is what separates a real project from a tutorial replication.
3. **What you specifically contributed** (15-20s): be precise about your individual role if it was a group effort — the modeling choice, the feature engineering insight, the evaluation design — avoid "we" for the parts that were actually yours.
4. **What you'd do differently now** (10-15s): with 4 years of industry experience since, name one thing you'd change with hindsight — e.g., "I'd add proper walk-forward validation instead of a random split," or "I'd instrument the pipeline for drift monitoring from day one." This signals growth and is the single highest-leverage sentence in the answer because it proves the project isn't frozen in amber.

Keep a one-paragraph mental summary ready for both the MTech thesis and one strong BTech course project — interviewers sometimes ask for the *undergraduate* one specifically to test how far back your fundamentals go. Don't fabricate metrics you don't remember; say "directionally it was around X%" rather than inventing false precision.

---

## Part 2 — Master Quick-Fire Concept Review

### 2.1 Statistics & Probability (→ file 01)

| Concept | One-liner / formula | Gotcha |
|---|---|---|
| Mean vs. median vs. mode | Mean = arithmetic average; median = 50th percentile; mode = most frequent value. | Mean is pulled by outliers/skew; median is robust — always ask "which one does the business actually want" (e.g. median household income). |
| Variance / Std Dev | $\text{Var}(X) = E[(X-\mu)^2]$, $\sigma = \sqrt{\text{Var}(X)}$ | Sample variance divides by $n-1$ (Bessel's correction), not $n$ — forgetting this is a classic slip. |
| Skewness | Measures asymmetry; positive skew = long right tail (mean > median). | People confuse "right-skewed" with "skewed to the right visually" — right skew means the *tail*, not the bulk of data, is on the right. |
| Kurtosis | Measures tailedness; excess kurtosis > 0 = heavier tails than Normal (leptokurtic). | High kurtosis ≠ high variance — it's about outlier frequency, not spread. |
| Bernoulli | Single binary trial, $P(X=1)=p$. | Use when: one yes/no event (one coin flip, one click). |
| Binomial | Sum of $n$ i.i.d. Bernoullis, $P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$. | Use when: fixed number of independent trials, counting successes. |
| Poisson | $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$, mean = variance = $\lambda$. | Use when: counting rare events over a fixed interval (calls/hour); breaks if events aren't independent (bursty arrivals). |
| Normal / Gaussian | $f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | Use when: sums of many small independent effects (CLT-driven); don't assume real-world data is Normal without checking. |
| Exponential | $f(x)=\lambda e^{-\lambda x}$, memoryless. | Use when: time between Poisson events / time-to-failure; "memoryless" is the key trap word. |
| Uniform | Constant density over $[a,b]$. | Use when: no information favors any value in a range; often the "uninformative prior" choice. |
| Log-normal | $\ln(X)$ is Normal. | Use when: multiplicative processes (stock prices, income) — always right-skewed, never negative. |
| CLT | Sample mean of i.i.d. draws → Normal as $n\to\infty$, regardless of population shape. | CLT is about the *sampling distribution of the mean*, not about individual data becoming Normal. |
| LLN | Sample mean converges to true mean as $n\to\infty$. | LLN says nothing about the *rate* of convergence or the shape of the distribution — that's CLT's job. |
| Bayes' theorem | $P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$ | Base-rate neglect is the classic trap: ignoring $P(A)$ (prior prevalence) makes disease-test problems come out wrong. |
| Bayesian vs. Frequentist | Bayesian treats parameters as random variables with a prior/posterior; Frequentist treats parameters as fixed, unknown, and reasons about long-run frequency of data. | Interview trap: "what does a 95% CI mean" — frequentist answer is about procedure coverage, not "95% probability the parameter is in this interval" (that's the Bayesian credible interval). |
| Simpson's Paradox | A trend appears in aggregated data but reverses (or disappears) within subgroups. | Always ask "is there a lurking/confounding variable splitting the groups differently?" |
| Survivorship bias | Analyzing only the "survivors" (successful funds, planes that returned) skews conclusions. | Classic WWII bomber armor example — reinforce where the survivors *weren't* hit. |
| Selection bias | Sample isn't representative of the population due to how it was collected. | Watch for self-selection in surveys / opt-in data. |
| Confirmation bias | Seeking/interpreting evidence to confirm a pre-existing belief. | In an interview, cite it as a reason to pre-register A/B test hypotheses. |
| Regression to the mean | Extreme observations tend to be followed by less extreme ones purely from randomness, no causal mechanism needed. | Don't confuse with a real treatment effect — "Sophomore slump" and "Sports Illustrated cover jinx" are the canonical examples. |

### 2.2 Hypothesis Testing & A/B Testing (→ file 02)

| Concept | One-liner | Gotcha |
|---|---|---|
| Type I error ($\alpha$) | Rejecting a true $H_0$ (false positive). | $\alpha$ is set *before* the test, it's not "the probability $H_0$ is true." |
| Type II error ($\beta$) | Failing to reject a false $H_0$ (false negative). | Power = $1-\beta$; low power is often caused by an underpowered sample size, not a weak effect. |
| p-value (correct definition) | Probability of observing data *at least as extreme* as what you got, **assuming $H_0$ is true**. | It is NOT "the probability $H_0$ is true" and NOT "the probability the result is due to chance" — this is the single most-tested misinterpretation in DS interviews. |
| Statistical power | Probability of correctly rejecting a false $H_0$; driven by effect size, $\alpha$, sample size, variance. | Underpowered tests give unreliable "no significant difference" conclusions — absence of evidence ≠ evidence of absence. |
| t-test | Compares means when variance is unknown / sample small; uses $t$-distribution. | One-sample vs. two-sample vs. paired — picking the wrong one is a common trap (paired needed when same units measured twice). |
| z-test | Compares means/proportions when population variance is known or $n$ is large. | Rarely truly applicable in practice since population variance is rarely known — often t-test is the safer default. |
| Chi-square test | Tests independence/goodness-of-fit for categorical data via $\sum \frac{(O-E)^2}{E}$. | Needs sufficiently large expected cell counts (rule of thumb: ≥5) or the approximation breaks down. |
| ANOVA | Tests whether ≥3 group means differ, via between-group vs. within-group variance ratio ($F$-statistic). | Significant ANOVA only tells you *some* group differs — needs post-hoc tests (Tukey) to find which. |
| Confidence interval (correct interpretation) | If you repeated the sampling procedure many times, ~95% of such intervals would contain the true parameter. | NOT "95% probability the true value is in this specific interval" — that's a Bayesian-flavored misstatement of a frequentist object. |
| Sample size formula | Exists as a function of desired power, $\alpha$, effect size (MDE), and variance — memorize that it *exists* and scales as $n \propto \sigma^2/\text{MDE}^2$. | Halving the minimum detectable effect roughly quadruples the required sample size — a common estimation trap. |
| Bonferroni correction | Divide $\alpha$ by number of comparisons $m$: $\alpha/m$ — controls family-wise error rate. | Very conservative; loses power fast as $m$ grows. |
| FDR / Benjamini-Hochberg | Controls the *expected proportion* of false discoveries among rejected hypotheses, less conservative than Bonferroni. | Preferred when running many simultaneous tests (e.g., many metrics in one A/B test) and some false positives are tolerable. |
| Peeking problem | Repeatedly checking significance during an ongoing experiment inflates the true Type I error rate far above the nominal $\alpha$. | Fix with sequential testing methods (e.g., always-valid p-values) or pre-committing to a fixed sample size/duration. |
| SRM (Sample Ratio Mismatch) | Observed traffic split between variants deviates significantly from the intended split (e.g., 48/52 instead of 50/50). | A red flag that invalidates the whole experiment's results — always check this via a chi-square test *before* trusting the metric readout. |

### 2.3 ML Fundamentals (→ file 03)

| Concept | Formula / one-liner | Gotcha |
|---|---|---|
| Bias-variance decomposition | $\text{Err} = \text{Bias}^2 + \text{Variance} + \sigma^2$ | Irreducible error ($\sigma^2$) cannot be removed by any model — don't chase it. |
| L1 vs. L2 geometry | L1 (Lasso) constraint region is a diamond → hits axes → sparse solutions. L2 (Ridge) constraint region is a circle → shrinks but rarely zeroes coefficients. | "Why does L1 give sparsity" is a very common follow-up — answer with the geometric corner-intersection argument, not just "it just does." |
| OLS normal equation | $\hat\beta = (X^TX)^{-1}X^Ty$ | Fails when $X^TX$ is singular (perfect multicollinearity) — motivates Ridge's $(X^TX+\lambda I)^{-1}$. |
| VIF | $\text{VIF}_j = \frac{1}{1-R_j^2}$, measures multicollinearity of feature $j$ against other features. | VIF > 5–10 is the common rule-of-thumb concern threshold. |
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$, maps logits to $[0,1]$ probabilities. | Saturates for large $|z|$ → vanishing gradient territory in deep nets. |
| Log-loss (binary cross-entropy) | $-\frac{1}{n}\sum [y\log\hat p + (1-y)\log(1-\hat p)]$ | Heavily penalizes confident wrong predictions — a single confident miss can dominate the loss. |
| Odds ratio | $\text{odds} = \frac{p}{1-p}$; logistic regression coefficient $\beta_j$ = change in log-odds per unit increase in $x_j$, so $e^{\beta_j}$ = odds ratio. | Odds ratio ≠ probability ratio — a very common misstatement in interviews. |

### 2.4 Trees & Boosting (→ file 04)

| Concept | Formula / one-liner | Gotcha |
|---|---|---|
| Gini impurity | $Gini = 1-\sum p_i^2$ | Computationally cheaper than entropy, usually gives similar splits. |
| Entropy / Information Gain | $H=-\sum p_i\log_2 p_i$; $IG = H(\text{parent}) - \sum \frac{n_i}{n}H(\text{child}_i)$ | IG is biased toward high-cardinality features (many splits) — gain ratio corrects this. |
| Bagging vs. boosting | Bagging trains independent models in parallel on bootstrap samples to reduce **variance**; boosting trains models sequentially, each correcting the previous one's errors, to reduce **bias**. | Random Forest = bagging + feature subsampling; XGBoost/GBM = boosting. |
| Random Forest OOB error | Each tree is evaluated on the ~37% of data ($1/e$) not in its bootstrap sample — gives a "free" validation estimate without a held-out set. | OOB is not identical to k-fold CV error but is a very good unbiased proxy. |
| XGBoost 2nd-order Taylor expansion | Loss approximated as $L \approx \sum [g_i f(x_i) + \frac12 h_i f(x_i)^2] + \Omega(f)$ where $g_i,h_i$ are 1st/2nd derivatives. | Using the Hessian ($h_i$) is XGBoost's key innovation over plain gradient boosting — gives more precise leaf weight estimates. |
| XGBoost split gain | $Gain = \frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma$ | $\gamma$ is the minimum-gain pruning threshold — this is how XGBoost decides whether a split is "worth it" (regularized). |
| LightGBM leaf-wise growth | Grows the leaf with max loss reduction (not level-by-level) → deeper, more accurate trees faster, but higher overfitting risk on small data. | Must control with `max_depth`/`num_leaves` or it overfits more readily than level-wise XGBoost. |
| GOSS (Gradient-based One-Side Sampling) | Keeps all high-gradient (under-trained) samples, randomly samples low-gradient ones — speeds up training with minimal accuracy loss. | Distinguish from plain random subsampling — GOSS is gradient-informed. |
| EFB (Exclusive Feature Bundling) | Bundles mutually-exclusive sparse features into one to reduce feature dimensionality. | Key reason LightGBM is fast on high-dimensional sparse/categorical data. |
| Key hyperparameters | `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`/`reg_alpha`/`reg_lambda`. | Lower learning rate + more estimators is the standard bias-variance dial in boosting. |

### 2.5 Other ML Algorithms (→ file 05)

| Concept | One-liner | Gotcha |
|---|---|---|
| SVM margin | Maximizes the distance between the decision boundary and the nearest points (support vectors). | Only support vectors matter for the boundary — removing non-support-vector points doesn't change it. |
| SVM kernel trick | Maps data to higher dimension implicitly via kernel function $K(x_i,x_j)$ without computing the mapping explicitly (e.g. RBF: $K=e^{-\gamma\|x_i-x_j\|^2}$). | "Implicitly" is the key word — never actually materializes the high-dim vectors. |
| SVM C | Controls the penalty for misclassified/margin-violating points — small C = wider margin, more tolerance (more bias, less variance); large C = narrow margin, less tolerance (less bias, more variance). | Often confused in direction — larger C means *less* regularization, not more. |
| SVM gamma (RBF) | Controls how far the influence of a single training point reaches — high gamma = tight, wiggly boundary (overfit risk); low gamma = smoother, more global boundary. | High gamma + high C together is a classic severe-overfitting combo. |
| kNN & curse of dimensionality | As dimensions grow, all points become roughly equidistant, so "nearest" neighbor loses meaning. | Motivates dimensionality reduction (PCA) before kNN in high-dim spaces. |
| Naive Bayes independence assumption | Assumes features are conditionally independent given the class — rarely true, but works surprisingly well in practice (esp. text classification). | The "naive" part is exactly this false independence assumption — say so explicitly if asked why it's called naive. |
| k-means | Partitions into $k$ clusters by minimizing within-cluster sum of squares; requires $k$ chosen upfront. | Sensitive to initialization (use k-means++) and assumes roughly spherical, similar-sized clusters. |
| Hierarchical clustering | Builds a dendrogram via agglomerative (bottom-up merge) or divisive (top-down split) linkage; no need to pre-specify $k$. | Choice of linkage (single/complete/average/ward) changes cluster shape assumptions significantly. |
| DBSCAN | Density-based clustering; groups points with enough neighbors within radius $\epsilon$, marks sparse points as noise/outliers. | Unlike k-means, it discovers arbitrary-shaped clusters and doesn't need $k$ — but struggles with varying density clusters. |
| GMM | Soft/probabilistic clustering assuming data is a mixture of Gaussians, fit via EM algorithm. | Gives cluster membership *probabilities*, not hard assignments — the key differentiator vs. k-means. |
| PCA | Linear projection onto orthogonal directions of maximum variance (eigenvectors of covariance matrix). | Sensitive to feature scaling — always standardize first; components are linear combinations, losing direct interpretability. |
| t-SNE | Nonlinear dimensionality reduction preserving local neighborhood structure for visualization. | Distances *between* clusters and cluster sizes in a t-SNE plot are **not meaningfully interpretable** — a classic interview trap. |
| UMAP | Similar goal to t-SNE (nonlinear, local-structure-preserving) but faster and better preserves some global structure. | Still primarily a visualization tool, not a general-purpose preprocessing step for downstream modeling. |

### 2.6 Model Evaluation & Feature Engineering (→ file 06)

| Concept | Formula / one-liner | Gotcha |
|---|---|---|
| Precision | $\frac{TP}{TP+FP}$ — of predicted positives, how many are correct. | Optimize when false positives are costly (e.g., spam filter flagging real email). |
| Recall | $\frac{TP}{TP+FN}$ — of actual positives, how many were caught. | Optimize when false negatives are costly (e.g., disease screening). |
| F1 | $2\cdot\frac{P\cdot R}{P+R}$, harmonic mean of precision/recall. | Harmonic mean punishes imbalance between P and R more than arithmetic mean would. |
| ROC-AUC | Area under TPR vs. FPR curve across thresholds; probability a random positive ranks above a random negative. | Misleadingly optimistic on imbalanced data because FPR denominator (large negative class) stays small. |
| PR-AUC | Area under precision vs. recall curve. | Preferred over ROC-AUC on imbalanced data — precision directly reflects how the minority class positives are being handled. |
| RMSE | $\sqrt{\frac1n\sum(y_i-\hat y_i)^2}$ | Penalizes large errors more (squared term) — sensitive to outliers. |
| MAE | $\frac1n\sum|y_i-\hat y_i|$ | Robust to outliers but not differentiable at 0 — can complicate gradient-based optimization. |
| MAPE | $\frac{1}{n}\sum\left|\frac{y_i-\hat y_i}{y_i}\right|\times100$ | Undefined/explodes when $y_i \approx 0$ — big trap for series with near-zero values. |
| R² | $1-\frac{SS_{res}}{SS_{tot}}$ | Can be negative on test data if the model is worse than predicting the mean; adding more predictors never decreases *training* R² (use adjusted R²). |
| Calibration methods | Platt scaling (logistic fit on scores) and isotonic regression (non-parametric monotonic fit) both remap raw scores to calibrated probabilities. | A high-AUC model can still be poorly calibrated — AUC only cares about ranking, not probability magnitude. |
| Walk-forward CV | Train on past window, validate on the next chronological chunk, then roll the window forward — never shuffle time-ordered data. | Standard k-fold CV on time series leaks future information into training — a very common red-flag mistake to call out. |
| SMOTE | Synthetic Minority Oversampling — generates synthetic minority points by interpolating between existing minority neighbors. | Can create unrealistic synthetic points in noisy/overlapping regions — apply only on the training fold, never before the train/test split. |
| Class weighting | Penalize misclassification of the minority class more heavily in the loss function instead of resampling data. | Often preferred to SMOTE for tree-based/boosting models since most implementations have a native `class_weight`/`scale_pos_weight` param. |
| SHAP (Shapley values) | $\phi_i = \sum_{S\subseteq F\setminus\{i\}}\frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S\cup\{i\})-f(S)]$ — average marginal contribution of feature $i$ over all feature subset orderings. | Three axioms: **efficiency** (contributions sum to the prediction), **symmetry** (equal contribution → equal credit), **dummy/null** (zero contribution → zero credit). |
| LIME | Fits a local, interpretable (linear) surrogate model around a single prediction by perturbing inputs. | Local explanations can be unstable — small input perturbations can yield noticeably different LIME explanations. |
| Permutation importance | Shuffle one feature's values and measure the drop in model performance. | Correlated features dilute each other's importance score (shuffling one still leaves the info available via its correlated twin). |

### 2.7 Time Series & Forecasting (→ file 07)

| Concept | One-liner | Gotcha |
|---|---|---|
| Stationarity | Statistical properties (mean, variance, autocorrelation) don't change over time. | Most classical models (ARIMA) require stationarity — always check/transform first. |
| ADF test | Tests $H_0$: series **has** a unit root (non-stationary). | Reject $H_0$ (low p-value) → series **is** stationary — the null/alternative direction is a common mix-up. |
| KPSS test | Tests $H_0$: series **is** stationary — opposite null to ADF. | Best practice: run both; if they disagree, treat the series as inconclusive/borderline. |
| ACF vs. PACF identification | ACF shows correlation with all shorter lags combined; PACF shows correlation with a specific lag after removing effect of shorter lags. | Rule of thumb: PACF cuts off sharply → suggests AR(p) order; ACF cuts off sharply → suggests MA(q) order. |
| ARIMA(p,d,q) | $p$ = autoregressive lag order, $d$ = differencing order (to induce stationarity), $q$ = moving-average lag order. | $d$ is often forgotten in verbal explanations — it's specifically the differencing step, distinct from $p$/$q$. |
| Prophet decomposition | $y(t) = trend + seasonality + holidays + \epsilon$, fit via a Bayesian curve-fitting procedure rather than classical ARIMA recursion. | Handles multiple seasonalities and holiday effects easily but is not itself a "smarter" model than ARIMA — it trades some accuracy for interpretability/ease of use. |
| Holt-Winters | Exponential smoothing with three components: level, trend, seasonality — additive or multiplicative variants. | Multiplicative seasonality is needed when seasonal swings grow proportionally with the level (not a fixed absolute amount). |
| MAPE / SMAPE / WAPE / MASE tradeoffs | MAPE explodes near zero; SMAPE bounds the denominator but is asymmetric in practice; WAPE (weighted absolute % error) aggregates well across many series; MASE compares against a naive seasonal baseline. | For intermittent/near-zero-demand series, prefer WAPE or MASE over MAPE. |
| Hierarchical reconciliation | Forecasts at different aggregation levels (SKU → category → total) are adjusted (top-down, bottom-up, or optimal combination/MinT) to be mutually consistent. | Bottom-up sums are simple but can be noisy; top-down is smooth but can misallocate to sub-series; MinT-style reconciliation aims for the statistically optimal blend. |
| Croston's method | Specialized method for **intermittent demand** (many zero periods) — separately forecasts demand size and inter-demand interval. | Standard ARIMA/exponential smoothing perform poorly on intermittent demand; Croston's is the go-to alternative. |

### 2.9 SQL / PySpark / dbt (→ file 09)

| Concept | One-liner | Gotcha |
|---|---|---|
| ROW_NUMBER vs. RANK vs. DENSE_RANK | ROW_NUMBER: unique sequential number, no ties. RANK: ties share rank, next rank skips (1,1,3). DENSE_RANK: ties share rank, no skip (1,1,2). | The skip-vs-no-skip behavior on ties is exactly what gets tested — draw a tiny example table if asked to disambiguate live. |
| Broadcast vs. shuffle join | Broadcast join sends the smaller table to every executor (no shuffle) — fast for small-table joins. Shuffle join repartitions both large tables by join key across the cluster — expensive but necessary when both sides are big. | Spark auto-broadcasts below a size threshold (`spark.sql.autoBroadcastJoinThreshold`) — know that you can hint it manually too. |
| Lazy evaluation | Spark builds a logical DAG of transformations and only executes when an action (e.g. `.collect()`, `.write()`) is called. | Enables the Catalyst optimizer to reorder/fuse operations before running anything — explains why a chain of `.filter().select()` doesn't run line-by-line. |
| dbt materializations | `view` (recomputed each query), `table` (rebuilt fully each run), `incremental` (appends/merges only new rows), `ephemeral` (inlined as a CTE, not persisted). | Picking `incremental` on a large fact table but forgetting a proper unique key / merge strategy is a classic dbt production bug. |

### 2.10 MLOps & Cloud (→ file 10)

| Concept | One-liner | Gotcha |
|---|---|---|
| Data drift | Input feature distribution changes over time ($P(X)$ shifts) while the true relationship $P(Y\mid X)$ stays the same. | Detectable via statistical distance metrics (PSI, KL divergence) on feature distributions without needing new labels. |
| Concept drift | The relationship between inputs and target changes ($P(Y\mid X)$ shifts) even if $P(X)$ looks the same. | Much harder to detect without fresh ground-truth labels — this is the more dangerous, silent failure mode. |
| Flask vs. FastAPI | Flask is a mature, synchronous WSGI micro-framework; FastAPI is async (ASGI), has built-in Pydantic validation and automatic OpenAPI docs, generally faster for I/O-bound ML serving. | FastAPI's async advantage only materializes if your inference code and I/O calls are actually written to be non-blocking. |
| Rebase vs. merge | Rebase replays your commits on top of the target branch, producing a linear history; merge creates a new merge commit preserving both histories as-is. | Never rebase a shared/public branch others have already pulled — rewrites commit hashes and breaks their history. |
| Key AWS services | S3 (object storage), SageMaker (managed ML lifecycle), Lambda (serverless compute), ECS/EKS (containers), CloudWatch (monitoring/logs), Glue (managed ETL/catalog). | Know one sentence per service — "what does it do and when would you use it" — not just the acronym. |
| Key GCP services | BigQuery (serverless data warehouse), Vertex AI (managed ML platform, GCP's SageMaker analog), Dataflow (managed Apache Beam for stream/batch ETL), Cloud Composer (managed Airflow). | Interviewers may ask you to map an AWS service you know to its GCP equivalent — practice this translation both ways. |

### 2.11 NLP & DL Fundamentals (→ file 11)

| Concept | Formula / one-liner | Gotcha |
|---|---|---|
| TF-IDF | $\text{TF-IDF}(t,d) = tf(t,d) \times \log\frac{N}{df(t)}$ | Down-weights common words across the corpus (high $df$) — doesn't capture semantic meaning like embeddings do. |
| Word2Vec CBOW vs. Skip-gram | CBOW predicts the target word from surrounding context words (faster, better for frequent words); Skip-gram predicts surrounding context words from the target word (better for rare words, more training data effectively generated). | Direction of prediction is exactly reversed between the two — a very commonly mixed-up detail. |
| Vanishing gradient cause | Repeated multiplication of small derivative terms (e.g., sigmoid/tanh derivatives < 1, or long weight-matrix chains) through many layers/timesteps shrinks the gradient toward zero. | Mitigations: ReLU activations, residual/skip connections, LSTM/GRU gating, careful weight initialization, batch norm. |
| LSTM gates | Forget gate (what to drop from cell state), input gate (what new info to add), output gate (what to expose as hidden state) — all sigmoid-gated. | The cell state's near-linear path (only element-wise operations, no repeated matrix multiply saturation) is *why* gradients survive longer than vanilla RNNs. |
| Batch normalization | Normalizes layer inputs to zero mean/unit variance per mini-batch, then learns scale/shift params $\gamma,\beta$. | Behaves differently at train time (batch statistics) vs. inference time (running averages) — a classic "gotcha" bug if not handled correctly in deployment code. |
| Dropout | Randomly zeroes a fraction of activations during training to prevent co-adaptation/overfitting. | Must be disabled (or rescaled) at inference time — forgetting `model.eval()` in PyTorch is a very common practical bug. |

### 2.12 GenAI & Transformers (→ file 12)

| Concept | Formula / one-liner | Gotcha |
|---|---|---|
| Scaled dot-product attention | $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ | The $\sqrt{d_k}$ scaling prevents dot products from growing large in magnitude as dimension grows, which would otherwise push softmax into saturated, near-zero-gradient regions. |
| Encoder-only / decoder-only / encoder-decoder | Encoder-only (e.g., BERT): bidirectional context, best for understanding/classification tasks. Decoder-only (e.g., GPT): autoregressive, best for open-ended generation. Encoder-decoder (e.g., T5): best for sequence-to-sequence tasks like translation/summarization. | Match the architecture family to the task type when asked to "design a system" — this is a frequent quick-check question. |
| LoRA | Freezes pretrained weights $W$, learns a low-rank update $W + BA$ where $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r \ll d,k$. | Drastically reduces trainable parameters for fine-tuning; the low-rank assumption is that the *weight update* (not the weight itself) has low intrinsic rank. |
| RLHF pipeline stages | (1) Supervised fine-tuning (SFT) on demonstration data, (2) train a reward model on human preference comparisons, (3) optimize the policy against the reward model via RL (typically PPO). | Three distinct stages/models — don't collapse them into "just RL on human feedback" when asked to explain the pipeline. |
| DPO vs. RLHF | DPO (Direct Preference Optimization) skips training a separate reward model and RL loop — it directly optimizes the policy on preference pairs using a closed-form loss derived from the same underlying objective as RLHF. | Simpler, more stable, and cheaper to train than full RLHF, at the cost of some flexibility (no explicit reward model to reuse elsewhere). |

### 2.13 RAG & Agents (→ file 13)

| Concept | One-liner | Gotcha |
|---|---|---|
| Chunking | Splitting documents into retrievable pieces (fixed-size, sentence-aware, or semantic chunking). | Chunk size is a precision/recall tradeoff — too small loses context, too large dilutes relevance and wastes context window. |
| Embeddings | Dense vector representations of text capturing semantic similarity for retrieval via distance/cosine similarity. | Embedding model choice must match the domain (general vs. code vs. multilingual) or retrieval quality silently degrades. |
| Vector DB | Specialized store for high-dimensional vectors with approximate nearest neighbor (ANN) search (e.g., Pinecone, FAISS, Weaviate, pgvector). | Exact kNN doesn't scale — production systems always trade a bit of recall for ANN speed. |
| HNSW vs. IVF | HNSW: graph-based ANN index, high recall and fast query, higher memory/build cost. IVF (inverted file index): clusters vectors into buckets (via k-means-like partitioning) and searches only relevant buckets, lower memory, faster build, slightly lower recall. | Know the memory-vs-recall tradeoff direction — HNSW for accuracy-critical/smaller corpora, IVF (often IVF+PQ) for massive-scale/memory-constrained corpora. |
| Recall@k | Fraction of relevant items found in the top-$k$ retrieved results. | Doesn't account for *ranking order* within the top-$k$ — that's what MRR/NDCG add. |
| MRR | Mean Reciprocal Rank — averages $1/\text{rank of first relevant result}$ across queries. | Only cares about the *first* relevant hit, ignores everything after it. |
| NDCG | Discounted Cumulative Gain normalized against the ideal ranking — rewards relevant results appearing earlier, with graded (not just binary) relevance. | The metric of choice when relevance is graded (e.g., 0-3 relevance scale) rather than purely binary. |
| LangChain vs. LangGraph | LangChain: linear/sequential chains and tool-calling abstractions. LangGraph: models agent workflows as an explicit graph/state machine, supporting cycles, branching, and persistent state — needed for more complex multi-step agent loops. | LangGraph exists specifically because linear chains can't naturally express loops/conditional branching in agent reasoning. |
| Hallucination mitigation | Grounding generation in retrieved context (RAG), citation/source attribution, response verification/self-consistency checks, constrained decoding, and lowering temperature. | RAG reduces but does not eliminate hallucination — the model can still misread or ignore the retrieved context. |

### 2.14 System Design (→ file 14)

**The 8-step ML system design framework (checklist):**

1. Clarify requirements — functional and non-functional (latency, scale, constraints).
2. Frame it as an ML problem — define input/output and the metric to optimize.
3. Data — sources, labeling strategy, storage.
4. Feature engineering — what signals, how computed, online/offline consistency.
5. Model selection — start simple, justify complexity increases.
6. Evaluation — offline metrics + online (A/B test) plan.
7. Deployment — serving architecture, latency budget, scaling.
8. Monitoring & iteration — drift detection, retraining triggers, feedback loop.

| Practice design | One differentiator to remember |
|---|---|
| Feed ranking system | Candidate generation (retrieval) + ranking (scoring) two-stage funnel to manage latency at scale. |
| Search relevance system | Blend of lexical (BM25/inverted index) and semantic (embedding) retrieval, fused via a re-ranker. |
| Fraud detection system | Extreme class imbalance + need for very low-latency real-time scoring at the transaction moment. |
| Recommendation system | Cold-start problem for new users/items is the central design challenge, not just the ranking model itself. |
| Ad click-through-rate prediction | Calibration matters as much as ranking — the predicted probability feeds directly into an auction/bidding formula. |
| Content moderation system | Precision/recall tradeoff is explicitly a policy decision, not just a modeling one — false positives (over-blocking) have real user-trust cost. |

### 2.15 Case Studies & Use-Case Reasoning (→ file 19)

| Concept | One-liner | Gotcha |
|---|---|---|
| Case-study opening move | Before proposing anything, restate the business objective, ask 2-3 clarifying questions, and state your assumptions out loud. | Jumping straight to a model/algorithm name is the single most common way candidates lose points on case-study rounds — structure beats speed. |
| Structuring the answer | Objective → success metric → data → approach (baseline first, then refine) → risks/tradeoffs → how you'd validate impact. | Interviewers are grading your *reasoning process*, not whether you land on the "correct" architecture — narrate your tradeoffs explicitly. |
| Ensembling rationale (forecasting-style case studies) | Different model families capture different signal types (linear trend/seasonality vs. nonlinear feature interactions); blending reduces variance of the overall forecast error versus any single model. | Be ready to name the blending mechanism (weighted average vs. stacking) and justify the weights, not just assert "ensembling is better." |
| Attribution reasoning (Markov / Shapley-style case studies) | Removal effect = drop in conversion probability when a channel is removed from the transition graph; Shapley value = a channel's marginal contribution averaged over all orderings. | Know why Shapley is fairer (order-independence) but costlier ($2^n$ subsets) — and that it's usually approximated via sampling in practice. |
| MDP / RL-style case studies | Frame the problem as states, actions, transition dynamics, and a reward function *before* naming an algorithm. | Naming "I'd use PPO" without first defining the MDP is a red flag to an interviewer — the formulation is the hard part, the algorithm choice is secondary. |

---

## Part 3 — Final Pre-Interview Checklist

Run through this literally in the last 30 minutes before the call:

- [ ] **Recall these formulas cold, from memory, no notes**: bias-variance decomposition, precision/recall/F1, sigmoid + log-loss, scaled dot-product attention ($QK^T/\sqrt{d_k}$), p-value's correct definition, ROC-AUC vs. PR-AUC distinction.
- [ ] **Have your project narratives ready** at three lengths: a 20-second one-liner, a 2-minute walkthrough, and a "go deep for 10 minutes on the hardest part" version — for each flagship project on your resume (forecasting, attribution, RL, or whatever you personally shipped).
- [ ] **Have 2 behavioral stories per category ready** (conflict, failure/mistake, leadership/influence-without-authority, ambiguity, tight deadline) in STAR format (Situation → Task → Action → Result), each with a quantified result.
- [ ] **Rehearse the certification talking points out loud once** — Feature Store, Model Monitor, Clarify — so they don't come out as memorized definitions but as natural explanations.
- [ ] **Pre-load your thesis 60-second answer** (problem / hard part / your contribution / what you'd change now) so it doesn't get invented on the spot.
- [ ] **Prepare 3 questions to ask the interviewer** — e.g., about the team's model retraining/monitoring maturity, how the team balances model complexity vs. interpretability for stakeholders, and what a strong first 90 days looks like in the role.
- [ ] **Sanity-check logistics**: resume/portfolio open in a tab, quiet environment, water nearby, notepad and pen for system design/whiteboard questions, and a calm 5 minutes of silence before joining the call.

## Part 4 — Practice Tools in This Kit

Reading this file is the *recall* check — it tells you what you should already know. It doesn't test whether you can actually produce it under pressure. Four other files in this kit are built for that:

| File | What it drills | Use it when |
|---|---|---|
| `17_flashcards_active_recall.md` | Self-test flashcards (question hidden behind an expandable answer) across every topic file | Daily, in short bursts — this is spaced-repetition material, not a single read-through |
| `18_practice_problems_and_code.md` | SQL problems against a sample schema, probability/stats problems, "derive it from scratch" prompts, and runnable Python snippets (toy SHAP, k-means, gradient descent) | When you want to test whether you can *produce* a derivation, not just recognize it |
| `19_case_studies_and_use_cases.md` | A large bank of open-ended case-study/use-case prompts with structured approaches (not full solutions to memorize — frameworks to reuse) | For the "design/reason through this business problem" portion of a loop, distinct from the deep-dive system designs in file 14 |
| `20_mock_interview_and_progress_tracker.md` | A rehearsal script/rubric for saying a design out loud under a timer, plus a checklist to track what you've actually drilled vs. only read | In the final week, to convert passive familiarity into active readiness |
