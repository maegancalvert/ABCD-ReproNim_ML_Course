import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.model_selection import cross_validate, GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import matplotlib as mpl
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

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
