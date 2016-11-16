import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn import svm, neighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

iris_df = pd.read_csv('Iris.csv')

# mapping 'Species' column with number
iris_df.loc[iris_df['Species'] == 'Iris-setosa', 'Species'] = 0
iris_df.loc[iris_df['Species'] == 'Iris-versicolor', 'Species'] = 1
iris_df.loc[iris_df['Species'] == 'Iris-virginica', 'Species'] = 2
iris_df['Species'] = iris_df['Species'].astype('int')

iris_train, iris_test = train_test_split(iris_df, test_size=.3, random_state = 1)

iris_train_y = iris_train['Species']
iris_train_x = iris_train.drop('Species', axis = 1)
iris_test_y = iris_test['Species']
iris_test_x = iris_test.drop('Species', axis = 1)



# (1) Logistic Regression
logreg = LogisticRegression()
# train the model on the training set
logreg.fit(iris_train_x, iris_train_y)
# make predictions on the testing set
y_pred = logreg.predict(iris_test_x)

# Logistic Regression Model Evaluation
# compare actual response values with predicted response values
print metrics.accuracy_score(iris_test_y, y_pred)


k_range = list(range(1, 26))
scores = []
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(iris_train_x, iris_train_y)
	y_pred = knn.predict(iris_test_x)
	scores.append(metrics.accuracy_score(iris_test_y, y_pred))

# plt.plot(k_range, scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Testing Accuracy')
# plt.show()

# From the plot above, high scores
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(iris_train_x, iris_train_y)
y_pred = knn.predict(iris_test_x)
print metrics.accuracy_score(iris_test_y, y_pred)
from sklearn.metrics import precision_recall_curve
# precision, recall, thresholds = precision_recall_curve(iris_train_y, knn.predict(iris_train_x))
# print precision, recall, thresholds


# (3) PCA is a dimensionality reduction technique
# reduce dimensions from 4 to 2
iris_x = iris_df.drop('Species', axis = 1)
iris_y = iris_df['Species']
pca = PCA(n_components=2)
pca.fit(iris_x)
X2d = pca.transform(iris_x)
X2d_train, X2d_test, y_train, y_test = train_test_split(X2d, iris_y, test_size = .3, random_state = 1)


# plt.scatter(X2d_train[:, 0], X2d_train[:, 1], c = y_train)
# plt.show()

print '-- Preserved Variance --'
print pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)

kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X2d_train)
y_pred = kmeans.predict(X2d_test)

print metrics.accuracy_score(y_test, y_pred)


from sklearn.svm import SVC
svm_classifier = SVC()
svm_classifier.fit(iris_train_x, iris_train_y)
y_pred = svm_classifier.predict(iris_test_x)
print metrics.accuracy_score(y_test, y_pred)

from sklearn.tree import DecisionTreeClassifier
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(iris_train_x, iris_train_y)
y_pred = tree_classifier.predict(iris_test_x)
print metrics.accuracy_score(y_test, y_pred)

from sklearn.learning_curve import validation_curve
from sklearn import learning_curve

# train_sizes, train_scores, test_scores = learning_curve(svm_classifier, iris_train_x, iris_train_y)
# print train_sizes, train_scores, test_scores
# train_scores_mean = np.mean(train_scores, axis=1)
# print train_scores_mean
# train_scores_std = np.std(train_scores, axis=1)
# print train_scores_std
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# plt.grid()
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')

# plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
# plt.plot(train_sizes, test_scores_mean, '*-', color='g', label='Cross-validationg score')
# plt.legend(loc='best')
# plt.show()
