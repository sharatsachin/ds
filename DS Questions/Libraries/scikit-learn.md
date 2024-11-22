# Scikit-learn

## What is scikit-learn and how does it differ from other ML libraries?

Scikit-learn is a machine learning library that provides:
- Consistent API across different algorithms
- Focus on classical machine learning (not deep learning)
- Extensive data preprocessing capabilities
- Built-in cross-validation and model selection
- Strong integration with NumPy and pandas
- Emphasis on ease of use and documentation

Key differences from other frameworks:
- More focused on traditional ML than deep learning
- Better for smaller datasets than deep learning frameworks
- Simpler API than TensorFlow/PyTorch
- Stronger preprocessing and feature engineering tools
- Better model evaluation and selection utilities

## How do you handle data preprocessing in scikit-learn?

Common preprocessing operations:
```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures
)

# Numerical scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-max scaling
min_max = MinMaxScaler(feature_range=(0, 1))
X_minmax = min_max.fit_transform(X)

# Robust scaling (handles outliers)
robust = RobustScaler()
X_robust = robust.fit_transform(X)

# Categorical encoding
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# One-hot encoding
onehot = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_onehot = onehot.fit_transform(X_categorical)

# Ordinal encoding
ordinal = OrdinalEncoder()
X_ordinal = ordinal.fit_transform(X_categorical)

# Feature generation
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

## How do you handle missing values and feature selection?

Missing value imputation and feature selection:
```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, RFE, SelectFromModel,
    f_classif, mutual_info_classif
)

# Simple imputation
imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', 'constant'
X_imputed = imputer.fit_transform(X)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = knn_imputer.fit_transform(X)

# Univariate feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Recursive feature elimination
from sklearn.linear_model import LogisticRegression
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Model-based feature selection
selector = SelectFromModel(LogisticRegression(penalty='l1'))
X_selected = selector.fit_transform(X, y)
```

## What are the main classification algorithms in scikit-learn?

Common classification algorithms:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
clf = LogisticRegression(penalty='l2', C=1.0)

# Decision Tree
clf = DecisionTreeClassifier(max_depth=5)

# Random Forest
clf = RandomForestClassifier(n_estimators=100)

# Gradient Boosting
clf = GradientBoostingClassifier(n_estimators=100)

# Support Vector Machine
clf = SVC(kernel='rbf', C=1.0)

# K-Nearest Neighbors
clf = KNeighborsClassifier(n_neighbors=5)

# Naive Bayes
clf = GaussianNB()

# Basic usage pattern
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

## What are the main regression algorithms in scikit-learn?

Common regression algorithms:
```python
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR

# Linear Regression
reg = LinearRegression()

# Ridge Regression
reg = Ridge(alpha=1.0)

# Lasso Regression
reg = Lasso(alpha=1.0)

# Elastic Net
reg = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Decision Tree
reg = DecisionTreeRegressor(max_depth=5)

# Random Forest
reg = RandomForestRegressor(n_estimators=100)

# Gradient Boosting
reg = GradientBoostingRegressor(n_estimators=100)

# Support Vector Regression
reg = SVR(kernel='rbf', C=1.0)

# Basic usage pattern
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

## How do you implement model evaluation and validation?

Model evaluation techniques:
```python
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    KFold, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, r2_score
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Custom cross-validation
kf = KFold(n_splits=5, shuffle=True)
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
# Classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# Regression metrics
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)
```

## How do you implement hyperparameter tuning?

Hyperparameter optimization:
```python
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV
)

# Grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Random search
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

## How do you implement pipelines?

Building ML pipelines:
```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

# Simple pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer()),
    ('classifier', RandomForestClassifier())
])

# Alternative syntax
pipeline = make_pipeline(
    StandardScaler(),
    SimpleImputer(),
    RandomForestClassifier()
)

# Complex pipeline with different column transformations
numeric_features = ['age', 'salary']
categorical_features = ['department', 'title']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit and predict
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

## How do you implement clustering algorithms?

Common clustering algorithms:
```python
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering
)

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

# Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=5)
clusters = hc.fit_predict(X)

# Spectral clustering
spectral = SpectralClustering(n_clusters=5)
clusters = spectral.fit_predict(X)

# Evaluate clustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

silhouette = silhouette_score(X, clusters)
calinski = calinski_harabasz_score(X, clusters)
davies = davies_bouldin_score(X, clusters)
```

## How do you implement dimensionality reduction?

Dimensionality reduction techniques:
```python
from sklearn.decomposition import (
    PCA, TruncatedSVD, NMF
)
from sklearn.manifold import (
    TSNE, MDS, Isomap
)

# Principal Component Analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_

# Truncated SVD (works with sparse matrices)
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

# Non-negative Matrix Factorization
nmf = NMF(n_components=2)
X_nmf = nmf.fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# Multi-dimensional Scaling
mds = MDS(n_components=2)
X_mds = mds.fit_transform(X)

# Isomap
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X)
```

## How do you save and load models?

Model persistence:
```python
import joblib
import pickle

# Save model using joblib
joblib.dump(model, 'model.joblib')

# Load model using joblib
loaded_model = joblib.load('model.joblib')

# Save model using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model using pickle
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Save pipeline
joblib.dump(pipeline, 'pipeline.joblib')

# Load pipeline
loaded_pipeline = joblib.load('pipeline.joblib')
```

## How do you handle imbalanced datasets?

Handling imbalanced data:
```python
from imblearn.over_sampling import (
    SMOTE, ADASYN, RandomOverSampler
)
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks
)
from imblearn.pipeline import Pipeline as ImbPipeline

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# ADASYN oversampling
adasyn = ADASYN(random_state=42)
X_res, y_res = adasyn.fit_resample(X, y)

# Random undersampling
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# Tomek links undersampling
tomek = TomekLinks()
X_res, y_res = tomek.fit_resample(X, y)

# Imbalanced pipeline
imb_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('sampler', SMOTE()),
    ('classifier', RandomForestClassifier())
])

imb_pipeline.fit(X_train, y_train)
```