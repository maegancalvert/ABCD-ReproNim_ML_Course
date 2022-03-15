import numpy as np
from sklearn.datasets import make_regression, make_classification, load_breast_cancer
from sklearn.linear_model import Ridge, LogisticRegression
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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

### Lecture 1 Questions ###
# Question 1 #
# X, y = make_regression(n_samples=80, n_features=600, noise=10, random_state=0) #Generates a random linear combination of random features, with noise.
#
# model = Ridge(alpha=1e-8)
# model.fit(X, y)
# predictions = model.predict(X)
# mse = mean_squared_error(y, predictions)
#
# print(f"\nMean Squared Error: {mse}")
# print("MSE is 0 up to machine precision:", np.allclose(mse, 0))
#
# X, y = make_regression(n_samples=160, n_features=600, noise=10, random_state=0)
# X_train, X_test = X[:80], X[80:]
# y_train, y_test = y[:80], y[80:]
# estimator = Ridge(alpha=1e-8)
# estimator.fit(X_train,y_train)
# predictions1 = estimator.predict(X_test)
# mse1 = mean_squared_error(y_test,predictions1)
# print(f"\nMean Squared Error: {mse1}")
# #MSE: 25412.01984
# print("MSE is 0 up to machine precision:", np.allclose(mse1, 0))
# #False
#
# Question 2 #
# X, y = make_regression(noise=10) #Generates a random linear combination of random features, with noise.
# model = Ridge()
# scores = cross_validate(model, X, y, scoring='neg_mean_squared_error')['test_score']
# print(scores)
# model.fit(X,y)
#
# Question 3 #
# X, y = make_classification() #Generate random data
# parameters = {'C':[.01, .1, 1], 'penalty':('l1','l2')}
# log_reg = LogisticRegression(solver='liblinear')
# clf = GridSearchCV(log_reg, parameters)
# print(clf)
# clf.fit(X, y)
# print(clf.cv_results_)
# fig, ax = plt.subplots(figsize=(4,4))
# sns.scatterplot(x=clf.cv_results_['param_C'], y=clf.cv_results_['split0_test_score'] ,hue=clf.cv_results_['param_penalty'],
#                 palette='gist_rainbow')
# sns.scatterplot(x=clf.cv_results_['param_C'], y=clf.cv_results_['split1_test_score'] ,hue=clf.cv_results_['param_penalty'],
#                 palette='gist_rainbow', legend=False)
# sns.scatterplot(x=clf.cv_results_['param_C'], y=clf.cv_results_['split2_test_score'] ,hue=clf.cv_results_['param_penalty'],
#                 palette='gist_rainbow', legend=False)
# sns.scatterplot(x=clf.cv_results_['param_C'], y=clf.cv_results_['split3_test_score'] ,hue=clf.cv_results_['param_penalty'],
#                 palette='gist_rainbow', legend=False)
# sns.scatterplot(x=clf.cv_results_['param_C'], y=clf.cv_results_['split4_test_score'] ,hue=clf.cv_results_['param_penalty'],
#                 palette='gist_rainbow', legend=False)
# ax.set_xlabel("Hyperparam: C")
# ax.set_ylabel('Test_Score')
# ax.legend(title="Penalty", loc='best')
# plt.show()
#
# Question 4 #
# mnist = fetch_openml('mnist_784', version=1, as_frame= False) #~130MB, might take a little time to download!
# mnist.keys()
# X, y = mnist["data"], mnist["target"]
# print(X.shape)
# some_digit = X[0]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()
# print(y[0])

# X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# y_train = y_train.astype(np.int8) #Casting labels from strings to integers
#
# #Here we are binarizing our labels. All labels that are 5 are converted to True, and the rest to False.
# y_train_5 = (y_train == 5)
# y_test_5 = (y_test == 5)
#
# sgd_clf = SGDClassifier(random_state=42) #42 is arbitrarily chosen. From documentation: "Pass an int for reproducible output across multiple function
# sgd_fit = sgd_clf.fit(X_train, y_train_5)
# print(sgd_fit)
# sgd_predict = sgd_clf.predict([X_train[1,:]])
# print(sgd_predict)
# print(y_test_5[1])
# scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
# print(scores)
# val_predict = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# y_conf_matrix = confusion_matrix(y_train_5, val_predict)
# print(y_conf_matrix)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.matshow(y_conf_matrix, cmap=plt.cm.viridis, alpha=0.5)
# for i in range(y_conf_matrix.shape[0]):
#     for j in range(y_conf_matrix.shape[1]):
#         ax.text(x=j, y=i, s=y_conf_matrix[i, j], va='center', ha='center', size='xx-large')
#
# plt.xlabel('Predictions', fontsize=18)
# plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
# plt.show()
#
# print('Precision:', precision_score(y_train_5, val_predict))
# print('Recall:', recall_score(y_train_5, val_predict))
# print('Accuracy:', accuracy_score(y_train_5, val_predict))
# print('F1 score:', f1_score(y_train_5, val_predict))

### Lecture 2 Questions ###

# Question 1 #
# X = np.asarray([[0, 1, -10], [0, -1, 0], [1, 0, 10], [1, 0, 0]])
# # print(f"X:\n{X}\n")
# scaler = StandardScaler()
# X_fit = scaler.fit(X)
# X_scaled = scaler.transform(X)
# print(X_scaled)
# print(f"mean: {X_scaled.mean(axis=0)}\nstd: {X_scaled.std(axis=0)}")

# Question 2 #
# X, y = make_regression(noise=10, n_features=5000, random_state=0)
#
# X_reduced = SelectKBest(f_regression).fit_transform(X, y)
# scores = cross_validate(Ridge(), X_reduced, y)["test_score"]
# print("feature selection in 'preprocessing':", scores)
#
# model = make_pipeline(SelectKBest(f_regression), Ridge())
# scores_pipe = cross_validate(model, X, y)["test_score"]
# print("feature selection on train set:", scores_pipe)
#
# # Plotting our results!
# plt.boxplot(
#     [scores_pipe, scores],
#     vert=False,
#     labels=[
#         "feature selection on train set",
#         "feature selection on whole data",
#     ],
# )
# plt.gca().set_xlabel("R² score")
# plt.tight_layout()
# plt.show()

#Question 3#
cancer = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
print(X_scaled)
print(cancer)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
#
print('Original Shape:', X_scaled.shape)
print('Reduced Shape:', X_pca.shape)
print('Transformed PCA values:', X_pca)
print(pca.components_.shape)
print(pca.components_)
print(pca.explained_variance_[0])
print(pca.explained_variance_[1])
#
#
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

