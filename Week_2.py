import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import fetch_california_housing, load_wine, make_blobs, make_moons
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, accuracy_score, pairwise_distances_argmin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import clone
import sklearn.utils.estimator_checks
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import SpectralEmbedding
import scipy.io


### Problem 1 ###
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"].astype(np.float64) #3 classes of flowers.

clf = Pipeline([('scaler', StandardScaler()),('svc', LinearSVC(random_state = 0, loss = 'hinge', C = 10))])
scores_pipe = cross_validate(clf, X, y)
print('Feature selection:', scores_pipe)

print(clf.fit(X,y))
print(clf.predict(X))

### if I wanted to know about the number of classes. Cannot combine cross_validate and cross_val_predict ###
# scores_pipe = cross_val_predict(clf, X, y, method = 'decision_function')
# print('N samples', scores_pipe.shape[0])
# print('N classes', scores_pipe.shape[1])

non_linear_clf = make_pipeline(StandardScaler(), SVC(decision_function_shape='ovo', random_state = 0, C = 10, kernel = 'rbf'))
non_linear_scores_pipe = cross_validate(non_linear_clf, X, y)
print(non_linear_scores_pipe)
print('Feature selection:', non_linear_scores_pipe['test_score'])

### Using cross val predict ###
# non_linear_scores_pipe = cross_val_predict(clf, X, y, method = 'decision_function')
# print('N samples', non_linear_scores_pipe.shape[0])
# print('N classes', non_linear_scores_pipe.shape[1])
### Same number of samples for the non-linear model vs the linear model ###


### Problem 2 ###

chd = fetch_california_housing()
X, y = chd["data"], chd["target"]

print("X shape: ", X.shape)
print("y shape: ", y.shape)
print("\n Data X")
print(X)
print("\n Target y:")
print(y)
print(chd.DESCR)

dc_regression = DecisionTreeRegressor(random_state=0, max_depth=3)
dc_reg_scores = cross_validate(dc_regression, X, y)
print(dc_reg_scores)

parameters = {'random_state': [0], 'max_depth': [3, 4, 5, 6, 7, 8]}
regression = DecisionTreeRegressor()
dc_regression2 = GridSearchCV(regression, parameters)
dc_fit = dc_regression2.fit(X,y)
df = pd.DataFrame(dc_regression2.cv_results_)
#print(df)
print(dc_regression2.best_score_)
print(dc_regression2.best_params_)
print(dc_regression2.best_estimator_.feature_importances_)
dc_reg_import = dc_regression2.best_estimator_.feature_importances_
print(dc_reg_import)
dc_reg_import_df = pd.DataFrame(dc_reg_import, columns = ['Feature Importance'])
dc_reg_import_df['Feature'] = range(len(dc_reg_import))
print(dc_reg_import_df)

plot = sns.barplot(y=dc_reg_import_df['Feature Importance'], x=dc_reg_import_df['Feature'], data=dc_reg_import_df, palette='magma')
plot.set_xlabel('Feature')
plt.show()

### best score 0.51214
### best params: max depth: 7, random state: 0
### feature importances: Feature 0 highest at 0.7069 - likely median income in block group ###

parameters2 = {'n_estimators': [10, 25, 50], 'random_state': [0], 'max_depth': [3, 4, 5, 6, 7, 8]}
rf_regression = RandomForestRegressor()
rf_regression2 = GridSearchCV(rf_regression, parameters2)
rf_fit = rf_regression2.fit(X,y)
df2 = pd.DataFrame(rf_regression2.cv_results_)
# print(df2)
print(rf_regression2.best_score_)
print(rf_regression2.best_params_)
print(rf_regression2.best_estimator_.feature_importances_)
rf_reg_import = rf_regression2.best_estimator_.feature_importances_
rf_reg_import_df = pd.DataFrame(rf_reg_import, columns = ['Feature Importance'])
rf_reg_import_df['Feature'] = range(len(rf_reg_import))
print(rf_reg_import_df)

plot1 = sns.barplot(y=rf_reg_import_df['Feature Importance'], x=rf_reg_import_df['Feature'], data=rf_reg_import_df, palette='magma')
plot1.set_xlabel('Feature')
plt.show()

### best score 0.6389
### best params: max depth: 10, n_estimators: 50, random state: 0
### feature importances: Feature 0 highest at 0.5964 - likely median icome in block group ###

ereg = VotingRegressor(estimators=[('dt', dc_fit), ('rf', rf_fit)])
ereg2 = ereg.fit(X, y)
# print(ereg2.score)
# print(X.shape)
# print(y.shape)

### the first 50 samples. The shape of X is 20640, 8 - a tuple of 20640 samples and 8 features. ###
### X[:50, ] is first sample through 50 and all 8 features ###
xt = X[:50, ]

## predictions using the first 50 samples ###
pred1 = dc_regression2.predict(xt)
print(pred1)
pred2 = rf_regression2.predict(xt)
pred3 = ereg2.predict(xt)

### plot the predictions ###
plt.figure()
plt.plot(pred1, 'gd', label='DecisionTreeRegressor')
plt.plot(pred2, 'b^', label='RandomForestRegressor')
plt.plot(pred3, 'r*', ms=10, label='VotingRegressor')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel('Predicted')
plt.xlabel('Training Samples')
plt.legend(loc='best')
plt.title('Regressor predictions and their average')

plt.show()

## Problem 3: Wine Classification ###
wine=load_wine(as_frame=True)
# print(wine.DESCR)
X,y = wine['data'].values, wine['target'].values

#See features for 10 random wines
data = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names']+['target'])
data.sample(10)
# print('X shape:' + str(X.shape))
# print('y shape:' + str(y.shape))

#See features and labels for first 4 wines
# for n in range(0,4):
#     print('Wine', str(n+1), '\n Features:', list(X[n]))

### create functions ###
### prepare pipelines returns a dictionary of key, value pairs of each model ###
def prepare_pipelines():
    scaling = StandardScaler()

    logreg = LogisticRegressionCV(solver='liblinear', random_state=0, cv=5, Cs=[3, 6, 10])
    logistic = make_pipeline(clone(scaling), clone(logreg))

    SVC_reg = SVC(decision_function_shape='ovo', random_state=0, C=10, kernel='rbf')
    svc_pipe = make_pipeline(clone(scaling), clone(SVC_reg))

    DecTreeReg = DecisionTreeRegressor(random_state=0, max_depth=10)
    dt = make_pipeline(clone(scaling), clone(DecTreeReg))

    RanForReg = RandomForestRegressor(random_state=0, max_depth=10)
    rf = make_pipeline(clone(scaling), clone(RanForReg))

    return {"LogisticRegression": logistic, "SVC": svc_pipe, "DecisionTreeRegressor": dt,
            "RandomForestRegressor": rf}

### Compute cv scores creates a dataframe of all the cross validation scores ###
def compute_cv_scores(models, X, y):
    all_scores = []
    for model_name, model in models.items():
        model_scores = pd.DataFrame(cross_validate(model, X, y))
        model_scores['model'] = model_name
        all_scores.append(model_scores)
    all_scores = pd.concat(all_scores)
    return all_scores

### compute cv fit scores creates dictionaries of the test score ###
def compute_cv_fit(models, X, y):
    fit_score = {}
    for model_name, model in models.items():
        f_score = cross_val_score(model, X, y)
        mean_f_score = f_score.mean()
        fit_score[model_name] = f_score
    return fit_score

## Problem 3 Voting Classifier ###

wine = load_wine(as_frame=True)
X, y = wine['data'].values, wine['target'].values

models = prepare_pipelines()
all_scores = compute_cv_scores(models, X, y)
fit_score = compute_cv_fit(models, X, y)
# print(all_scores.groupby('model').mean())
# print(fit_score)

lrc = LogisticRegressionCV(solver='liblinear', random_state=0, cv=5, Cs=[3, 6, 10])
svc = SVC(decision_function_shape='ovo', random_state=0, C=10, kernel='rbf', probability=True)
dt = DecisionTreeClassifier(random_state=0, max_depth=10)
rf = RandomForestClassifier(random_state=0, max_depth=10)

eclf1 = VotingClassifier(estimators=[('lrc', lrc.fit(X,y)),
                                 ('svc', svc.fit(X,y)),
                                 ('dt', dt.fit(X,y)),
                                 ('rf', rf.fit(X,y))], voting='hard')

eclf1 = eclf1.fit(X,y)

for clf, label in zip([lrc, svc, dt, rf, eclf1], ['Logistic Regression', 'SVC', 'Decision Tree', 'Random Forest', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

### add class weights to account for unbalanced data ###
print(np.bincount(y))
weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(y),
                                                          y=y)

lrc2 = LogisticRegressionCV(solver='liblinear', random_state=0, cv=5, Cs=[3, 6, 10], class_weight='balanced')
svc2 = SVC(decision_function_shape='ovo', random_state=0, C=10, kernel='rbf', probability=True, class_weight='balanced')
dt2 = DecisionTreeClassifier(random_state=0, max_depth=10, class_weight='balanced')
rf2 = RandomForestClassifier(random_state=0, max_depth=10, class_weight='balanced')

eclf2 = VotingClassifier(estimators=[('lrc', lrc2.fit(X,y)),
                                 ('svc', svc2.fit(X,y)),
                                 ('dt', dt2.fit(X,y)),
                                 ('rf', rf2.fit(X,y))], voting='soft')
acc_scores = []
est_acc1 = {}
for clf2, label2 in zip([lrc2, svc2, dt2, rf2, eclf2], ['Logistic Regression', 'SVC', 'Decision Tree', 'Random Forest', 'Ensemble']):
    scores2 = cross_val_score(clf2, X, y, scoring='accuracy', cv=5)
    fit = clf2.fit(X, y)
    predictions2 = clf2.predict(X)
    if label2 == "Ensemble":
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores2.mean(), scores2.std(), label2))
    else:
        acc = accuracy_score(y, predictions2)
        acc_scores.append(acc)
        print("Model Performance: ", acc, label2)
        est_acc = {label2:acc}
    est_acc1.update(est_acc)
print(est_acc1)
### use estimator weighted average accuracy ###
eclf3 = VotingClassifier(estimators=[('lrc', lrc2.fit(X,y)),
                                 ('svc', svc2.fit(X,y)),
                                 ('dt', dt2.fit(X,y)),
                                 ('rf', rf2.fit(X,y))], voting='soft', weights=acc_scores)
eclf3.fit(X,y)
predictions3 = eclf3.predict(X)
score3 = accuracy_score(y, predictions3)
acc_scores.append(score3)
est_acc1.update({'ensemble':score3})
print('Weighted Avg Accuracy:  %.3f' % (score3*100))
##

######## Lesson 2: Unsupervised Machine Learning #########
## Problem 1: K-means Clustering ###

X, y = make_blobs(random_state=1)
X_2, y_2 = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
kmeans = KMeans(init='k-means++', random_state=1, n_clusters=3)
kmeans.fit(X_norm)
x_df = pd.DataFrame(X_norm)
clusters = np.unique(kmeans.labels_)
x_df['labels'] = pd.DataFrame(kmeans.labels_)
x_df['labels'].astype(str)
km = kmeans.predict(X_norm)
km_c_centers = kmeans.cluster_centers_

k_means_labels = pairwise_distances_argmin(X_norm, km_c_centers)
x_df['m_labels'] = k_means_labels.tolist()

for i, k in X_norm:
    scat = sns.scatterplot(x=x_df[0],
        y=x_df[1], hue=x_df['labels'], data=x_df, legend=True, palette='magma', s=60)
plt.legend(title='Labels', loc='best', labels=['0', '1', '2', '3', '4', '5', '6'])
plt.show()

X2_norm = scaler.fit_transform(X_2)
kmeans2 = KMeans(init='k-means++', random_state=0, n_clusters=2)
kmeans2.fit(X2_norm)
x2_df = pd.DataFrame(X2_norm)
clusters = np.unique(kmeans2.labels_)
x2_df['labels'] = pd.DataFrame(kmeans2.labels_)
x2_df['labels'].astype(str)
km2 = kmeans2.predict(X2_norm)
km2_c_centers = kmeans2.cluster_centers_

k_means2_labels = pairwise_distances_argmin(X2_norm, km2_c_centers)
x2_df['m_labels'] = k_means2_labels.tolist()

for i2, k2 in X2_norm:
    scat2 = sns.scatterplot(x=x2_df[0],
        y=x2_df[1], hue=x2_df['labels'], data=x2_df, legend=True, palette='magma', s=60)
plt.legend(title='Labels', loc='best', labels=['0', '1', '2', '3', '4', '5', '6'])
plt.show()


## k-means doesn't perform well on either of these dataset. The data are not well distinguished from ###
## labels that are close. Our blobs may have unequal variance or maybe unevenly sized. In looking at the ###
## documentation for the moons data it is posssible that agglomerative clustering would be a better fit ###
## to account for the shape of the data/clusters ###


## Problem 1 Clusters without predefined cluster sizes ###
X, y = make_blobs(random_state=5)
plt.scatter(X[:, 0], X[:, 1], s=60)

distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

# Plotting the distortions of K-Means
plt.figure(figsize=(16, 8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method for finding the optimal k')
plt.show()


## Problem 2 Gaussian Mixture Models, Generating Data, and Embeddings ###
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

components = 2
gmm = GaussianMixture(n_components=components, covariance_type='full')
gmm.fit(X)
gmm_predict = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=gmm_predict, s=60, cmap='viridis')
plt.show()
### This method was much better than the K-means. It was able to account for more ###
### of the shape of the components ###
#
#
### Spectral Embeddings ###
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
print(X.shape)

embedding = SpectralEmbedding(n_components=2)
trans_X = embedding.fit_transform(X)
print(trans_X.shape)
gmm2 = GaussianMixture(n_components=2, covariance_type='full')
gmm2.fit(trans_X)
gmm2_predict = gmm2.predict(trans_X)
plt.scatter(trans_X[:, 0], trans_X[:, 1], c=gmm2_predict, s=60, cmap='viridis')
plt.show()

### The transformed data allows most of the data to fall into the correct components ###
### The shape of the data seems to be rotated in space ###


Problem 2: Generate Moon, GMM, 16 components

X, y = make_moons(n_samples=500, noise=0.05, random_state=2)

components3 = 16
gmm3 = GaussianMixture(n_components=components3, covariance_type='full')
gmm3.fit(X)
gmm3_predict = gmm3.predict(X)
X3, y3 = gmm3.sample(n_samples=200)

plt.scatter(X3[:, 0], X3[:, 1], s=60, cmap='viridis')
plt.show()
## The generated points look like the original data ###


### Problem 3: Applying these concepts to Neuroscience ###
spec_components = 2

mat = scipy.io.loadmat('/home/mcalvert/Downloads/acq-64dir_space-T1w_desc-preproc_space-T1w_msmtconnectome.mat')

#Obtain the streamline count weighted by both SIFT and inverse node volumes
#Using AAL coz it is anatomically defined, as opposed to functionally defined like most other atlases in the file.
connectivity = mat["aal116_sift_invnodevol_radius2_count_connectivity"]

con = np.asarray(connectivity)
print(con.shape)
print(con)

embedding = SpectralEmbedding(n_components=2)
trans_con = embedding.fit_transform(con)
print(trans_con.shape)
gmm4 = GaussianMixture(n_components=2, covariance_type='full')
gmm4.fit(trans_con)
gmm4_predict = gmm4.predict(trans_con)
plt.scatter(trans_con[:, 0], trans_con[:, 1], c=gmm4_predict, s=60, cmap='viridis', zorder=2)
plt.show()