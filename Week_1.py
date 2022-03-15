import numpy as np
from sklearn.datasets import make_regression, make_classification, load_breast_cancer
from sklearn.linear_model import Ridge, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn import model_selection
from sklearn.model_selection import cross_validate, GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib as mpl
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure

### Lecture 1 Questions ###
# Question 1 #
X, y = make_regression(n_samples=80, n_features=600, noise=10, random_state=0) #Generates a random linear combination of random features, with noise.

model = Ridge(alpha=1e-8)
model.fit(X, y)
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)

print(f"\nMean Squared Error: {mse}")
print("MSE is 0 up to machine precision:", np.allclose(mse, 0))

X, y = make_regression(n_samples=160, n_features=600, noise=10, random_state=0)
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]
estimator = Ridge(alpha=1e-8)
estimator.fit(X_train,y_train)
predictions1 = estimator.predict(X_test)
mse1 = mean_squared_error(y_test,predictions1)
print(f"\nMean Squared Error: {mse1}")
#MSE: 25412.01984
print("MSE is 0 up to machine precision:", np.allclose(mse1, 0))
#False

# Question 2 #
X, y = make_regression(noise=10) #Generates a random linear combination of random features, with noise.
model = Ridge()
scores = cross_validate(model, X, y, scoring='neg_mean_squared_error')['test_score']
print(scores)
model.fit(X,y)

# Question 3 #
X, y = make_classification() #Generate random data
parameters = {'C':[.01, .1, 1], 'penalty':('l1','l2')}
log_reg = LogisticRegression(solver='liblinear')
clf = GridSearchCV(log_reg, parameters)
print(clf)
clf.fit(X, y)
print(clf.cv_results_)
fig, ax = plt.subplots(figsize=(4,4))
sns.scatterplot(x=clf.cv_results_['param_C'], y=clf.cv_results_['split0_test_score'] ,hue=clf.cv_results_['param_penalty'],
                palette='gist_rainbow')
sns.scatterplot(x=clf.cv_results_['param_C'], y=clf.cv_results_['split1_test_score'] ,hue=clf.cv_results_['param_penalty'],
                palette='gist_rainbow', legend=False)
sns.scatterplot(x=clf.cv_results_['param_C'], y=clf.cv_results_['split2_test_score'] ,hue=clf.cv_results_['param_penalty'],
                palette='gist_rainbow', legend=False)
sns.scatterplot(x=clf.cv_results_['param_C'], y=clf.cv_results_['split3_test_score'] ,hue=clf.cv_results_['param_penalty'],
                palette='gist_rainbow', legend=False)
sns.scatterplot(x=clf.cv_results_['param_C'], y=clf.cv_results_['split4_test_score'] ,hue=clf.cv_results_['param_penalty'],
                palette='gist_rainbow', legend=False)
ax.set_xlabel("Hyperparam: C")
ax.set_ylabel('Test_Score')
ax.legend(title="Penalty", loc='best')
plt.show()

# Question 4 #
mnist = fetch_openml('mnist_784', version=1, as_frame= False) #~130MB, might take a little time to download!
mnist.keys()
X, y = mnist["data"], mnist["target"]
print(X.shape)
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
print(y[0])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train = y_train.astype(np.int8) #Casting labels from strings to integers

#Here we are binarizing our labels. All labels that are 5 are converted to True, and the rest to False.
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42) #42 is arbitrarily chosen. From documentation: "Pass an int for reproducible output across multiple function
sgd_fit = sgd_clf.fit(X_train, y_train_5)
print(sgd_fit)
sgd_predict = sgd_clf.predict([X_train[1,:]])
print(sgd_predict)
print(y_test_5[1])
scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print(scores)
val_predict = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
y_conf_matrix = confusion_matrix(y_train_5, val_predict)
print(y_conf_matrix)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(y_conf_matrix, cmap=plt.cm.viridis, alpha=0.5)
for i in range(y_conf_matrix.shape[0]):
    for j in range(y_conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=y_conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print('Precision:', precision_score(y_train_5, val_predict))
print('Recall:', recall_score(y_train_5, val_predict))
print('Accuracy:', accuracy_score(y_train_5, val_predict))
print('F1 score:', f1_score(y_train_5, val_predict))

## Lecture 2 Questions ###

# Question 1 #
X = np.asarray([[0, 1, -10], [0, -1, 0], [1, 0, 10], [1, 0, 0]])
# print(f"X:\n{X}\n")
scaler = StandardScaler()
X_fit = scaler.fit(X)
X_scaled = scaler.transform(X)
print(X_scaled)
print(f"mean: {X_scaled.mean(axis=0)}\nstd: {X_scaled.std(axis=0)}")

# Question 2 #
X, y = make_regression(noise=10, n_features=5000, random_state=0)

X_reduced = SelectKBest(f_regression).fit_transform(X, y)
scores = cross_validate(Ridge(), X_reduced, y)["test_score"]
print("feature selection in 'preprocessing':", scores)

model = make_pipeline(SelectKBest(f_regression), Ridge())
scores_pipe = cross_validate(model, X, y)["test_score"]
print("feature selection on train set:", scores_pipe)

# Plotting our results!
plt.boxplot(
    [scores_pipe, scores],
    vert=False,
    labels=[
        "feature selection on train set",
        "feature selection on whole data" ],)
plt.gca().set_xlabel("R² score")
plt.tight_layout()
plt.show()

# Question 3#
cancer = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
print(X_scaled)
print(cancer)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print('Original Shape:', X_scaled.shape)
print('Reduced Shape:', X_pca.shape)
print('Transformed PCA values:', X_pca)
print(pca.components_.shape)
print(pca.components_)
print(pca.explained_variance_[0])
print(pca.explained_variance_[1])


df = pd.DataFrame({'var': pca.explained_variance_ratio_, 'PC': ['PC1', 'PC2']})
df_pca = pd.DataFrame(X_pca, columns=['PC1','PC2'])
sns.barplot(x='PC', y='var', data=df, color='c')
plt.xlabel('Principle Components')
plt.ylabel('Variance Explained')
plt.show()

sns.scatterplot(x='PC1', y='PC2', data=df_pca, c=cancer.target)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.show()

### Data Exercise ###
def load_timeseries_and_site(n_subjects=100):
    """Load ABIDE timeseries and participants' site.
    Returns X, a list with one array of shape (n_samples, n_rois) per
    participant, and y, an array of length n_participants containing integers
    representing the site each participant belongs to.
    """
    data = datasets.fetch_abide_pcp(
        n_subjects=n_subjects, derivatives=["rois_ho"], quality_checked=False)
    X = data["rois_ho"]
    y = LabelEncoder().fit_transform(data["phenotypic"]["SITE_ID"])
    return X, y

def prepare_pipelines():
    """Prepare scikit-learn pipelines for fmri classification with connectivity.
    Returns a dictionary where each value is a scikit-learn estimator (a
    `Pipeline`) and the corresponding key is a descriptive string for that
    estimator.
    As an exercise you need to add a pipeline that performs dimensionality
    reduction with PCA.
    """
    connectivity = ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True)

    scaling = StandardScaler()
    logreg = LogisticRegressionCV(solver="liblinear", cv=3, Cs=3)
    logistic_reg = make_pipeline(
        clone(connectivity), clone(scaling), clone(logreg))
    # make_pipeline is a convenient way to create a Pipeline by passing the
    # steps as arguments. clone creates a copy of the input estimator, to avoid
    # sharing the state of an estimator across pipelines.

    pca = PCA(n_components=20)
    pca_pipe = make_pipeline(clone(connectivity), clone(scaling), clone(pca), clone(logreg))

    dummy = make_pipeline(clone(connectivity), DummyClassifier())
    # TODO: add a pipeline with a PCA dimensionality reduction step to this
    # dictionary. You will need to import `sklearn.decomposition.PCA`.
    return {
        "Logistic no PCA": logistic_reg,
        "Dummy": dummy,
        "Logistic with PCA" : pca_pipe}

def compute_cv_scores(models, X, y):
    """Compute cross-validation scores for all models
    `models` is a dictionary like the one returned by `prepare_pipelines`, ie
    of the form `{"model_name": estimator}`, where `estimator` is a
    scikit-learn estimator.
    `X` and `y` are the design matrix and the outputs to predict.
    Returns a `pd.DataFrame` with one row for each model and cross-validation
    fold. Columns include `test_score` and `fit_time`.
    """
    all_scores = []
    for model_name, model in models.items():
        print(f"Computing scores for model: '{model_name}'")
        model_scores = pd.DataFrame(cross_validate(model, X, y))
        model_scores["model"] = model_name
        all_scores.append(model_scores)
    all_scores = pd.concat(all_scores)
    return all_scores
X, y = load_timeseries_and_site()
models = prepare_pipelines()
all_scores = compute_cv_scores(models, X, y)
print(all_scores.groupby("model").mean())
sns.stripplot(data=all_scores, x="test_score", y="model")
plt.tight_layout()
plt.show()

### Could set the PCA to define the number of components by variance explained as defined in the model parameters.
### Largest number = highest rank of the design matrix.
### Logistic regression = 6105 + 1 (intercept)
### PCA = 20 * 6105
### Compressed matrix = 100 * 20
### Memory for Design matrix = 8 * 100 * 6105
### Memory for Dimensionality reduction = 8 * 100 * 20
### Memory for PCA = 8 * 20 * 6105
### Hyper-parameter = accuracy
### cross_validate = accuracy
### Yes, the default metric is appropriate for both PCA and logistic regression
### cross-validate = K-fold
### Logsitic Regression = Stratified K-fold
### Stratified K-fold is best because observations are nested within person and a person's data should not be
### divided among different folds 