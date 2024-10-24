# Miscellaneous

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
5. Use domain-specific knowledge to create synthetic examples

## What is bagging & boosting?

Bagging: Bootstrap Aggregating. Trains multiple models on random subsets of data and averages predictions.

Boosting: Sequentially trains weak learners, focusing on misclassified examples from previous iterations.

## What is decision tree?

A tree-like model of decisions. Each internal node represents a feature, each branch represents a decision rule, and each leaf node represents an outcome.

## When do we use bagging & when can we boosting?

Bagging: When base models are complex and prone to overfitting.
Boosting: When base models are simple (weak learners) and underfitting.

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

## What is cross-validation?

Technique to assess model performance on unseen data. Common method: k-fold CV, where data is split into k subsets, model trained on k-1 folds and tested on remaining fold, repeated k times.