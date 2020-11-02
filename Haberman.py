# author: Uri Hanunov, Olga Mazo

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def AdaBoost():
    # AdaBoostClassifier with 50 estimators
    temp = AdaBoostClassifier()
    # Train Adaboost Classifer
    temp.fit(X_train, y_train)
    accForTest = temp.score(X_test, y_test) * 100 # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100 # the percentage of success on the train set
    return accForTest, accForTrain

# SVMs with different kernels
def SVMLinear ():
    temp = SVC(kernel='linear')  # SVM with linear kernel
    temp.fit(X_train, y_train)  # Train SVM Classifier
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

def SVMPoly ():
    temp = SVC(kernel='poly')  # SVM with poly kernel
    temp.fit(X_train, y_train)  # Train SVM Classifier
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

def SVMSigmoid ():
    temp = SVC(kernel='sigmoid')  # SVM with sigmoid kernel
    temp.fit(X_train, y_train)  # Train SVM Classifier
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

def SVMRbf ():
    temp = SVC(kernel='rbf')  # SVM with rbf kernel
    temp.fit(X_train, y_train)  # Train SVM Classifier
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

# DecisionTreeClassifier
def DecisionTree():
    temp = DecisionTreeClassifier()
    temp.fit(X_train, y_train)  # Train DecisionTreeClassifier Classifier
    accForTest = temp.score(X_test, y_test) * 100 # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100 # the percentage of success on the train set
    return accForTest, accForTrain

# KNeighborsClassifier with k neighbors when k = 1,3,5,7,9 and p is l2 = Euclidean Distance
def KNN1EuclideanDistance():
    temp = KNeighborsClassifier(n_neighbors=1, p=2)
    temp.fit(X_train, y_train)  # Train KNeighbors Classifier (k=1)
    accForTest = temp.score(X_test, y_test) * 100 # the percentage of success on the test set
    accForTrain = temp.score(X_train,y_train) * 100 # the percentage of success on the train set
    return accForTest, accForTrain

def KNN3EuclideanDistance():
    temp = KNeighborsClassifier(n_neighbors=3, p=2)
    temp.fit(X_train, y_train)  # Train KNeighbors Classifier (k=3)
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

def KNN5EuclideanDistance():
    temp = KNeighborsClassifier(n_neighbors=5, p=2)
    temp.fit(X_train, y_train)  # Train KNeighbors Classifier (k=5)
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

def KNN7EuclideanDistance():
    temp = KNeighborsClassifier(n_neighbors=7, p=2)
    temp.fit(X_train, y_train)  # Train KNeighbors Classifier (k=7)
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

def KNN9EuclideanDistance():
    temp = KNeighborsClassifier(n_neighbors=9, p=2)
    temp.fit(X_train, y_train)  # Train KNeighbors Classifier (k=9)
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

# KNeighborsClassifier with k neighbors when k = 1,3,5,7,9 and p is l1 = Manhattan Distance
def KNN1ManhattanDistance():
    temp = KNeighborsClassifier(n_neighbors=1, p=1)
    temp.fit(X_train, y_train)  # Train KNeighbors Classifier (k=1)
    accForTest = temp.score(X_test, y_test) * 100 # the percentage of success on the test set
    accForTrain = temp.score(X_train,y_train) * 100 # the percentage of success on the train set
    return accForTest, accForTrain

def KNN3ManhattanDistance():
    temp = KNeighborsClassifier(n_neighbors=3, p=1)
    temp.fit(X_train, y_train)  # Train KNeighbors Classifier (k=3)
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

def KNN5ManhattanDistance():
    temp = KNeighborsClassifier(n_neighbors=5, p=1)
    temp.fit(X_train, y_train)  # Train KNeighbors Classifier (k=5)
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

def KNN7ManhattanDistance():
    temp = KNeighborsClassifier(n_neighbors=7, p=1)
    temp.fit(X_train, y_train)  # Train KNeighbors Classifier (k=7)
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

def KNN9ManhattanDistance():
    temp = KNeighborsClassifier(n_neighbors=9, p=1)
    temp.fit(X_train, y_train)  # Train KNeighbors Classifier (k=9)
    accForTest = temp.score(X_test, y_test) * 100  # the percentage of success on the test set
    accForTrain = temp.score(X_train, y_train) * 100  # the percentage of success on the train set
    return accForTest, accForTrain

# main
# load the data set
haberman = pd.read_csv("haberman.txt", header=None, names=['age', 'year_of_treatment', 'positive_lymph_nodes', 'survival_status_after_5_years'])
numpy_array = np.genfromtxt("haberman.txt", delimiter=",", skip_header=0)
X = numpy_array[0:306, 0:-1]
y = numpy_array[0:306, -1]
sumForTrain = 0
sumForTest = 0
# All the machines learning
machineLearning = [AdaBoost, SVMLinear, SVMPoly, SVMRbf, SVMSigmoid, DecisionTree,
                   KNN1ManhattanDistance, KNN3ManhattanDistance, KNN5ManhattanDistance, KNN7ManhattanDistance, KNN9ManhattanDistance,
                   KNN1EuclideanDistance, KNN3EuclideanDistance, KNN5EuclideanDistance, KNN7EuclideanDistance, KNN9EuclideanDistance]
for i in machineLearning:
    for j in range(100):
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test
        tempTest, tempTrain = i()
        sumForTrain += tempTrain
        sumForTest += tempTest
    print(i.__name__ , "for train:", sumForTrain/100)  # the Success rates of the machine on the training set
    print(i.__name__, "for test:", sumForTest/100)  # the Success rates of the machine on the testing set
    sumForTrain = 0
    sumForTest = 0

# print some data of the data set(e.g min value, numbers of lines.. )
pd.options.display.width = 0
print(haberman.describe())

# show us Distribution plots - Univariate Analysis
# Here the height of the bar denotes the percentage of data points under the corresponding group
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    fg = sns.FacetGrid(haberman, hue='survival_status_after_5_years', height=5)
    fg.map(sns.distplot, feature).add_legend()
    plt.show()