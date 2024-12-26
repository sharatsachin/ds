# Miscellaneous

## What are the differences between supervised and unsupervised learning?

Supervised learning : models with labeled data (target output known)
    - classification, regression
    - real world example: predicting house prices based on features
Unsupervised learning : unlabeled data to discover hidden patterns or structures
    - clustering, dimensionality reduction
    - real world example: customer segmentation based on purchasing behavior

Differences:
- data labels: supervised requires labeled data; unsupervised doesn't
- output: supervised predicts specific outputs; unsupervised finds patterns
- validation: supervised can be evaluated with metrics like accuracy; unsupervised often requires human interpretation
- applications: supervised for prediction tasks; unsupervised for exploration and pattern discovery

## Differentiate between univariate, bivariate, and multivariate analysis

Univariate analysis : examines one variable at a time
    - measures like mean, median, mode
    - visualizations like histograms, box plots
    - real world example: analyzing the distribution of customer ages

Bivariate analysis : studies relationships between two variables
    - scatter plots, correlation coefficients, contingency tables
    - real world example: examining the relationship between years of experience and salary

Multivariate analysis : explores relationships among three or more variables simultaneously
    - techniques like principal component analysis, factor analysis, multiple regression
    - real world example: analyzing how multiple features like education, experience, and location influence salary predictions

## What are the feature selection methods used to select the right variables?

Feature selection methods can be categorized into three main approaches:
1. filter methods: statistical measures to score features based on correlation with target variable
    - examples: correlation coefficients, chi-square tests, information gain
    - computationally efficient but may miss feature interactions
2. wrapper methods: evaluate feature subsets using the model itself
    - examples: recursive feature elimination (RFE), forward/backward selection
    - computationally intensive but often find better feature combinations
3. embedded methods: perform feature selection during model training
    - examples: LASSO, Ridge regression
    - use regularization to reduce feature importance
    - good balance between computational cost and performance

## How would you handle dataset with missing values?

Systematic approach to handle missing data:
1. Understand missing data mechanism:
    - MCAR (missing completely at random) - no pattern in missing data, eg: random sensor failures
    - MAR (missing at random) - depends on observed data. eg: missing salary data for unemployed people
    - MNAR (missing not at random) - depends on unobserved data, eg: people with high income not reporting salary
2. For variables with >30% missing values:
    - drop the variable if not crucial
    - create a "missing" category for categorical variables
    - use advanced imputation techniques like MICE (Multiple Imputation by Chained Equations)
    - use model-based imputation methods
3. Document the approach and validate that handling missing values doesn't introduce bias

## What are dimensionality reduction and its benefits?

Dimensionality reduction is the process of reducing the number of features in a dataset while preserving important information, benefits include:
1. Reduced computational complexity
2. Minimized overfitting through feature space reduction
3. Easier visualization of high-dimensional data
4. Reduced storage requirements
5. Removal of multicollinearity

Common techniques:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- LDA (Linear Discriminant Analysis)
- Autoencoders

## How would you calculate eigenvalues and eigenvectors of a 3x3 matrix?

To find eigenvalues:
1. Write the characteristic equation: $\text{det}(A - \lambda I) = 0$
2. Solve the resulting cubic equation
3. For each eigenvalue $\lambda$, find the corresponding eigenvector by solving $(A - \lambda I)v = 0$

## What are the aspects of maintaining a deployed machine learning model?

Some aspects are:
1. Monitoring model performance
    - model metrics, data drift and concept drift, system health
2. Regular retraining
    - schedule based on domain knowledge, triggered by performance degradation
    - version control for models
3. Documentation
    - model cards, deployment configurations, monitoring thresholds
4. A/B testing for updates
    - compare new model versions with existing ones
5. Fallback procedures
    - ensure system can revert to previous versions if needed

## How do recommender systems work?

Recommender systems are algorithms that suggest relevant items to users based on their preferences and behavior. They can be categorized into:
1. Collaborative filtering: based on user-item interactions
    - user-based: recommend items liked by similar users
    - item-based: recommend items similar to those the user liked
2. Content-based filtering: based on item features
    - build user profiles based on item features
3. Hybrid systems: combine multiple approaches
    - ensemble methods, combining collaborative and content-based filtering

## How do you treat outliers in a dataset?

1. Detection methods:
   - Z-score
   - IQR (Interquartile Range)
   - Local Outlier Factor
   - Isolation Forest
2. Treatment options:
   - Removal (if clearly erroneous)
   - Capping/Winsorization
   - Transformation (log, sqrt)
   - Separate modeling for outliers

## How does Amazon's 'people who bought this also bought' feature work?

This is typically implemented using association rule mining and collaborative filtering:
1. Association Rules:
   - Apriori algorithm : find frequent itemsets, generate rules, e.g., {A, B} -> {C}
   - FP-growth algorithm : frequent pattern growth, more efficient than Apriori
2. Item-based collaborative filtering: recommend items similar to those the user has interacted with
3. Session-based recommendations: consider user's current session for real-time suggestions

## How does A/B testing work?

You split users into two groups (A and B) and show them different versions of a product or feature. You then measure the impact of the changes by comparing key metrics between the groups. Key steps include:
1. Define the hypothesis
    - for example, "adding a new feature will increase user engagement"
2. Randomly assign users to groups
3. Run the experiment
4. Analyze results using statistical tests
5. Make data-driven decisions based on the outcome

## How would you compare continuous variables in an A/B test?

1. T-test: compares means of two groups
    - $$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
    - where $\bar{X}_1, \bar{X}_2$ are sample means, $s_1, s_2$ are sample standard deviations, $n_1, n_2$ are sample sizes
    - degrees of freedom: $n_1 + n_2 - 2$
2. ANOVA: compares means of multiple groups
    - one-way ANOVA for one factor
        $$ F = \frac{MS_{\text{between}}}{MS_{\text{within}}}$$ 
        where $MS$ is mean square error for between and within groups
    - two-way ANOVA for two factors
        $$ F = \frac{MS_{\text{factor A}}}{MS_{\text{error}}} $$
    - calculates F-statistic and p-value
        $$ p = P(F > F_{\text{observed}}) $$
3. Mann-Whitney U test: non-parametric test for two groups

## What is the goal of A/B Testing?

A/B Testing aims to:
1. Compare two or more versions of a variable
2. Determine statistical significance of differences
3. Make data-driven decisions
4. Minimize business risk through experimentation

Key aspects:
- Random assignment - to avoid bias, we randomly assign users to groups A and B
- Statistical power - to detect true effects, we need sufficient sample size
- Sample size determination - to ensure results are reliable and generalizable, we calculate the required sample size
- Hypothesis testing - to evaluate the significance of differences, we use statistical tests like t-test, chi-square, etc.

## What are confounding variables and how do you control them?

Confounding variables are factors that:
1. Influence both independent and dependent variables
2. Can lead to spurious correlations
3. Need to be controlled in experimental design

An example 

Control methods:
- Randomization
- Stratification
- Statistical adjustment
- Matching

## What determines the update frequency of a model?

Factors determining update frequency:
1. Data drift rate
2. Business requirements
3. Model performance degradation
4. Computational resources
5. Cost considerations

Best practices:
- Monitor performance metrics
- Set up automated triggers
- Regular validation checks
- Document update history

## What are the definitions of KPI, lift, model fitting, robustness and DOE?

KPI (Key Performance Indicator): Quantifiable measures used to evaluate success of an organization, employee, or activity in meeting objectives.

Lift: Measure of the performance of a targeting model compared to a random selection baseline. Calculated as: (Results with model)/(Results with random selection)

Model Fitting: Process of training a model on data to find optimal parameters that minimize prediction error.

Robustness: Model's ability to maintain performance despite variations in input data or conditions.

DOE (Design of Experiments): Systematic method for determining cause-and-effect relationships. Controls variables to ensure valid, unbiased, and statistically sound conclusions.

## What is the difference between data analytics and data science?

Data Analytics focuses on:
- Analyzing historical data
- Descriptive and diagnostic analysis
- Business intelligence reporting
- Known questions and metrics
- Shorter-term insights

Data Science encompasses:
- Predictive modeling
- Machine learning algorithms
- Advanced statistical methods
- Unknown patterns discovery
- Longer-term research
- Product development
- Creating new algorithms

## What are Support Vectors in SVM?

Support Vectors are the data points that lie closest to the decision boundary (hyperplane) in SVM. They:
- Define the margin of the classifier
- Are the most difficult to classify
- Are critical elements as they support the hyperplane
- Are the only points used to determine the optimal hyperplane
- Influence the position and orientation of the hyperplane

## What do you understand by a kernel trick?

The kernel trick is a method used in SVM to handle non-linearly separable data by:
1. Mapping data into a higher-dimensional space without explicitly computing the transformation
2. Using kernel functions (like RBF, polynomial) to compute inner products in the transformed space
3. Making non-linear problems linearly separable in higher dimensions
4. Reducing computational complexity through implicit transformation

## What is the k-nearest neighbors (KNN) algorithm and how does it work?

KNN is a non-parametric algorithm that:
1. Stores all training cases
2. Classifies new points based on majority vote of k nearest neighbors
3. Uses distance metrics (usually Euclidean) to find nearest neighbors

Key components:
- Choice of k (number of neighbors)
- Distance metric selection
- Voting scheme (weighted/unweighted)
- Feature scaling requirement

## What is K-Means and how does it work?

K-Means is an unsupervised clustering algorithm that:
1. Initializes k cluster centers randomly
2. Assigns points to nearest cluster center
3. Updates cluster centers to mean of assigned points
4. Repeats steps 2-3 until convergence

Important considerations:
- Choosing optimal k
- Handling initialization sensitivity
- Dealing with outliers
- Understanding limitations (assumes spherical clusters)

## What is the chi-squared distribution and when is it used?

The chi-squared distribution:
- Is a continuous probability distribution
- Is always positive and right-skewed
- Has one parameter (degrees of freedom)
- Used for:
  1. Goodness of fit tests
  2. Tests of independence
  3. Variance analysis
  4. Feature selection in classification

## What is normalization in database design?

Normalization is a systematic process to:
1. Eliminate data redundancy
2. Ensure data integrity
3. Reduce data anomalies

Common normal forms:
- 1NF: Atomic values, no repeating groups
- 2NF: 1NF + no partial dependencies
- 3NF: 2NF + no transitive dependencies
- BCNF: 3NF + every determinant is a candidate key

## How do we choose the appropriate kernel function in SVM?

Selection criteria include:
1. Data characteristics:
   - Linear kernel for linearly separable data
   - RBF kernel for non-linear relationships
   - Polynomial kernel for curved decision boundaries

2. Domain knowledge:
   - Text data: String kernels
   - Image data: RBF kernels
   - Sequential data: Custom kernels

3. Computational resources:
   - Linear kernels are fastest
   - RBF and polynomial more computationally intensive

## How does Naïve Bayes handle categorical and continuous features?

Categorical Features:
- Uses frequency counts
- Applies Laplace smoothing
- Calculates conditional probabilities directly

Continuous Features:
- Assumes normal distribution (Gaussian Naïve Bayes)
- Estimates mean and variance for each class
- Can use kernel density estimation for non-normal distributions

## What is Laplace smoothing and why is it used in Naïve Bayes?

Laplace (add-one) smoothing:
- Adds a small constant (usually 1) to all feature counts
- Prevents zero probabilities
- Handles unseen features in test data
- Improves model generalization
- Particularly important for sparse data

## How do we deal with categorical text values in machine learning?

Methods include:
1. Label Encoding:
   - For ordinal data
   - Maintains order relationship

2. One-Hot Encoding:
   - For nominal data
   - Creates binary columns

3. Feature Hashing:
   - For high-cardinality features
   - Reduces dimensionality

4. Target Encoding:
   - Replaces categories with target statistics
   - Handles high cardinality efficiently

## What is DBSCAN and how is it used?

DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
- Finds clusters of arbitrary shape
- Identifies noise points
- Doesn't require specifying number of clusters
- Based on density connectivity

Parameters:
- eps (neighborhood radius)
- minPts (minimum points for core point)

## How does the EM (Expectation-Maximization) algorithm work in clustering?

EM iteratively:
1. E-Step: Calculates probability of each point belonging to each cluster
2. M-Step: Updates cluster parameters based on weighted point assignments
3. Repeats until convergence

Applications:
- Gaussian Mixture Models
- Hidden Markov Models
- Latent variable models

## What is silhouette score and how is it used in clustering evaluation?

Silhouette score measures:
- Cluster cohesion (intra-cluster distance)
- Cluster separation (inter-cluster distance)
- Ranges from -1 to 1 (higher is better)

Used for:
- Evaluating clustering quality
- Determining optimal number of clusters
- Comparing different clustering algorithms

## What is the Apriori algorithm and how does it work in Association Rule Mining?

Apriori algorithm:
1. Finds frequent itemsets iteratively
2. Uses minimum support threshold
3. Generates association rules
4. Employs "downward closure" property

Key concepts:
- Support (frequency of itemset)
- Confidence (strength of rule)
- Lift (correlation measure)
- Pruning strategies for efficiency

## What are bias & variance?

Bias: Difference between average model prediction and true value. High bias leads to underfitting.

Variance: Variability of model prediction for a given data point. High variance leads to overfitting.

## How can we say if model is overfitted?

1. High training accuracy, low test accuracy
2. Large gap between training and validation error
3. Complex model structure relative to data size
4. Poor generalization to new data

## How to reduce overfitting?

1. Regularization (L1, L2)
2. More training data
3. Feature selection/reduction
4. Early stopping in iterative algorithms
5. Ensemble methods
6. Cross-validation

## What is expected from LR or other models in terms of bias & variance?

Ideal models balance bias and variance. Linear regression tends to have higher bias but lower variance. Complex models like deep neural networks often have lower bias but higher variance.

## How to handle biased data sets?

1. Resampling techniques (oversampling, undersampling)
2. Synthetic data generation (SMOTE)
3. Adjusting class weights
4. Ensemble methods with balanced subsets
5. Collect more diverse data

## Different sampling techniques?

1. Simple random sampling
2. Stratified sampling
3. Cluster sampling
4. Systematic sampling
5. Oversampling/Undersampling for imbalanced data

## How to handle biased data set for medical & banking domain? (SMOTE is prohibited)

1. Collect more data from underrepresented groups
2. Use weighted loss functions
3. Ensemble methods with balanced subsets
4. Adjust decision thresholds
5. Use domain-specific knowledge to create synthetic examples.

## What is SVM?

Support Vector Machine finds the hyperplane that best separates classes with maximum margin. For non-linear separation, it uses kernel tricks to map data to higher dimensions.

## What are the discriminative & generative approaches?

Discriminative: Models p(y|x) directly (e.g., logistic regression)
Generative: Models p(x|y) and uses Bayes rule to get p(y|x) (e.g., Naive Bayes)

## How can we reduce features?

1. Feature selection (filter, wrapper, embedded methods)
2. Dimensionality reduction (PCA, t-SNE)
3. Lasso regularization
4. Domain knowledge-based selection
5. Correlation analysis

## What is PCA & what is the math behind it?

Principal Component Analysis finds orthogonal axes (principal components) that maximize variance in the data.

Math: Eigendecomposition of covariance matrix $\Sigma = XX^T$. Principal components are eigenvectors, ordered by eigenvalues.

## How can we choose important features of a model?

1. Correlation with target variable
2. Feature importance from tree-based models
3. Regularization techniques (Lasso)
4. Mutual information
5. Backward/forward selection
6. Domain expertise

## How does P-value help us to identify the right features?

P-value measures the probability of observing the data given the null hypothesis. Low p-values (<0.05) suggest features are statistically significant, potentially important for the model.

## What is K-means clustering?

Unsupervised algorithm that partitions n observations into k clusters. Iteratively assigns points to nearest centroid and updates centroids.

## How do we find "K" in K-means clustering?

1. Elbow method
2. Silhouette analysis
3. Gap statistic
4. Cross-validation
5. Domain knowledge

## What is the elbow method?

Plots within-cluster sum of squares (WCSS) vs number of clusters. "Elbow" point where WCSS decrease slows down suggests optimal K.

## What values are there in the X-axis & Y-axis in the elbow method?

X-axis: Number of clusters (K)
Y-axis: Within-cluster sum of squares (WCSS)

## What is hierarchical clustering?

Builds a hierarchy of clusters. Two main approaches: agglomerative (bottom-up) and divisive (top-down).

## What is the bottom-up & top-down approach?

Bottom-up (Agglomerative): Starts with each point as a cluster, iteratively merges closest clusters.
Top-down (Divisive): Starts with all points in one cluster, recursively splits into smaller clusters.

## What is the difference between K-means & hierarchical clustering?

1. K-means requires specifying K; hierarchical doesn't
2. K-means is iterative; hierarchical is deterministic
3. K-means is more scalable to large datasets
4. Hierarchical provides a dendrogram for visualization

## What are the different steps in feature engineering?

1. Feature creation
2. Transformations (log, sqrt, etc.)
3. Encoding categorical variables
4. Handling missing values
5. Scaling/normalization
6. Binning
7. Feature interaction/polynomial features

## What are the different approaches of binning?

1. Equal-width binning
2. Equal-frequency binning
3. Custom binning based on domain knowledge
4. Quantile binning
5. K-means binning

## How do we bin data if it is not normally distributed?

1. Quantile binning
2. Log-transform then equal-width binning
3. Custom bins based on data distribution
4. K-means clustering for binning

## What is normalization & standardization?

Normalization: Scales features to [0,1] range. $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$

Standardization: Transforms to zero mean, unit variance. $x_{std} = \frac{x - \mu}{\sigma}$

## How do we handle missing data sets? What are the different approaches?

1. Deletion (listwise, pairwise)
2. Mean/median/mode imputation
3. Regression imputation
4. Multiple imputation
5. KNN imputation
6. Using algorithms that handle missing values (e.g., tree-based methods)

## What are outliers & how do we identify them?

Outliers: Data points significantly different from other observations.

Identification methods:
1. Z-score
2. IQR method
3. DBSCAN clustering
4. Isolation Forest
5. Visual methods (box plots, scatter plots)

## What is the importance of scaling?

1. Ensures features contribute equally to model
2. Necessary for distance-based algorithms (K-means, KNN)
3. Improves convergence speed for gradient-based optimization
4. Helps interpret feature importance

## How do we identify the right algorithm for a problem?

1. Understand problem type (classification, regression, clustering)
2. Consider data size and dimensionality
3. Evaluate interpretability requirements
4. Consider computational resources
5. Analyze data characteristics (linear/non-linear, sparse)
6. Use domain knowledge
7. Experiment with multiple algorithms