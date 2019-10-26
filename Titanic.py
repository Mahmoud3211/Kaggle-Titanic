# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:58:27 2019

@author: Mahmoud Nada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')
test =  pd.read_csv('test.csv')

y = train['Survived']
droped_colums = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
X = train.drop(droped_colums, axis = 1)

print(X.isnull().sum())
print(y.isnull().sum())

X['Age'].fillna(X['Age'].mean(), inplace=True)
X['Embarked'].fillna(method='bfill', inplace=True)

dummies = ['Embarked', 'Sex']
X1 = pd.get_dummies(X[dummies])
X = X.drop(dummies, axis=1)
X = X.join(X1)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

####################################Imports#########################################
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
################################Classifiers#################################################

# SVC Classifiear
svc = SVC(random_state=0)
svc.fit(X_train, y_train)
y_predSVC = svc.predict(X_valid)
cmSVC = confusion_matrix(y_valid, y_predSVC)
aScoreSVC = accuracy_score(y_valid, y_predSVC)
fScoreSVC = f1_score(y_valid, y_predSVC)
print("SVC Classifier Accuracy : ", aScoreSVC)
print("SVC Classifier f1 score : ", fScoreSVC)
print("confusion matrix of SVC Classifier : ")
print(cmSVC)
########################################################################################
#AdaBoost Classifiear using DecisionTreeClassifier
boost = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())
boost.fit(X_train, y_train)
y_pred = boost.predict(X_valid)
cm = confusion_matrix(y_valid, y_pred)
aScore = accuracy_score(y_valid, y_pred)
fScore = f1_score(y_valid, y_pred)
print("AdaBoost Classifier Accuracy : ", aScore)
print("AdaBoost Classifier f1 score : ", fScore)
print("confusion matrix of AdaBoost Classifier : ")
########################################################################################
# Gaussian Naive Bayes Classifier
clf_gnb = GaussianNB()
clf_gnb.fit(X_train, y_train)
GaussianNB(priors=None, var_smoothing=1e-09)
y_pred_gnb = clf_gnb.predict(X_valid)
cm_gnb = confusion_matrix(y_valid, y_pred_gnb)

aScore_gnb = accuracy_score(y_valid, y_pred_gnb)
fScore_gnb = f1_score(y_valid, y_pred_gnb)

print("Gaussian Naive Bayes Classifier Accuracy : ", aScore_gnb)
print("Gaussian Naive Bayes Classifier f1 score : ", fScore_gnb)
print("confusion matrix of Gaussian Naive Bayes Classifier : ")
print(cm_gnb)
########################################################################################
# K Nearest Neighbors Classifier
clf_k = KNeighborsClassifier()
clf_k.fit(X_train, y_train)
y_pred_k = clf_k.predict(X_valid)
cm_k = confusion_matrix(y_valid, y_pred_k)

aScore_k = accuracy_score(y_valid, y_pred_k)
fScore_k = f1_score(y_valid, y_pred_k)

print("K Nearest Neighbors Classifier Accuracy : ", aScore_k)
print("K Nearest Neighbors Classifier f1 score : ", fScore_k)
print("confusion matrix of K Nearest Neighbors Classifier : ")
print(cm_k)
########################################################################################
# K Nearest Neighbors Hyper-Parameter Tuning using grid search
clf_kgc = KNeighborsClassifier()
parameter_kgc = {
        'n_neighbors':[3,5,10,15,20],
        'weights':['uniform', 'distance'],
        'metric':['euclidean', 'manhattan', 'minkowski']
        }
scorer_kgc = make_scorer(accuracy_score)
grid_obj_kgc = GridSearchCV(clf_kgc,
                            parameter_kgc,
                            scoring = scorer_kgc,
                            verbose=1
                            , cv=3,
                            n_jobs=-1)
grid_fit_kgc = grid_obj_kgc.fit(X_train, y_train)
best_clf_kgc = grid_fit_kgc.best_estimator_
best_clf_kgc.fit(X_train, y_train)
best_train_prediction_kgc = best_clf_kgc.predict(X_train)
best_valid_prediction_kgc = best_clf_kgc.predict(X_valid)
print('the training accuracy of Optimized K Nearest Neighbors Classifier is :', accuracy_score(best_train_prediction_kgc, y_train))
print('the validation accuracy of Optimized K Nearest Neighbors Classifier is :', accuracy_score(best_valid_prediction_kgc, y_valid))
print('the validation F1 score of Optimized K Nearest Neighbors Classifier is :', f1_score(best_valid_prediction_kgc, y_valid))
########################################################################################
# Tuning the hyper-parameters of DecisionTreeClassifier using Gridsearch
clf_gcd = DecisionTreeClassifier(random_state=42)
parameters_gcd = {'max_depth':[50, 100, 200],
                  'min_samples_split':[2, 4, 6, 8], 'min_samples_leaf':[2,4,6,8]}
scorer_gcd = make_scorer(accuracy_score)
grid_obj_gcd = GridSearchCV(clf_gcd, parameters_gcd, scoring = scorer_gcd)
grid_fit_gcd = grid_obj_gcd.fit(X_train, y_train)
best_clf_gcd = grid_fit_gcd.best_estimator_
best_clf_gcd.fit(X_train, y_train)
best_train_prediction_gcd = best_clf_gcd.predict(X_train)
best_valid_prediction_gcd = best_clf_gcd.predict(X_valid)
print('the training accuracy of Optimized DecisionTree Classifier is :', accuracy_score(best_train_prediction_gcd, y_train))
print('the validation accuracy of Optimized DecisionTree Classifier is :', accuracy_score(best_valid_prediction_gcd, y_valid))
print('the validation F1 score of Optimized DecisionTree Classifier is :', f1_score(best_valid_prediction_gcd, y_valid))
########################################################################################
rf = RandomForestClassifier(n_estimators=200, max_depth=2,random_state=0)
rf.fit(X, y)  
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_valid)
cm_rf = confusion_matrix(y_valid, y_pred_k)

aScore_rf = accuracy_score(y_valid, y_pred_k)
fScore_rf = f1_score(y_valid, y_pred_k)

print("rf Classifier Accuracy : ", aScore_k)
print("rf f1 score : ", fScore_k)
print("confusion matrix of rf: ")
print(cm_k)
########################################################################################

rf2 = RandomForestRegressor(random_state = 42)

from pprint import pprint
print('Parameters currently in use:\n')
pprint(rf2.get_params())

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf2 = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf2_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf2_random.fit(X_train, y_train)
rf2_random.best_params_

best_random = rf2_random.best_estimator_
best_random.fit(X_train, y_train)

best_train_prediction_rf = best_random.predict(X_train)
best_valid_prediction_rf = best_random.predict(X_valid)
print('the training accuracy of Optimized DecisionTree Classifier is :',
      mean_squared_error(best_train_prediction_rf, y_train))
print('the validation accuracy of Optimized DecisionTree Classifier is :',
      mean_squared_error(best_valid_prediction_rf, y_valid))
print('the validation F1 score of Optimized DecisionTree Classifier is :',f1_score(best_valid_prediction_rf, y_valid))

########################################################################################
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf3 = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf3, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)
grid_search.best_params_

best_grid = grid_search.best_estimator_
best_grid.fit(X_train, y_train)
"""
best_train_prediction_rf3 = best_grid.predict(X_train)
best_valid_prediction_rf3 = best_grid.predict(X_valid)
print('the training accuracy of Optimized DecisionTree Classifier is :',
      accuracy_score(best_train_prediction_rf3, y_train))
print('the validation accuracy of Optimized DecisionTree Classifier is :',
      accuracy_score(best_valid_prediction_rf3, y_valid))
print('the validation F1 score of Optimized DecisionTree Classifier is :',
      f1_score(best_valid_prediction_rf3, y_valid))

cm_rf3 = confusion_matrix(y_valid, best_valid_prediction_rf3)
best_grid.score(y_valid, best_valid_prediction_rf3)
"""
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

grid_accuracy = evaluate(best_grid, X_valid, y_valid)



########################################################################################
"""
########################################################################################
# Neural network Classifier
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

clf_nn = Sequential()
clf_nn.add(Dense(128, activation='tanh', input_shape=(10,)))

clf_nn.add(Dense(64, activation='tanh'))
clf_nn.add(Dropout(.3))

clf_nn.add(Dense(32, activation='tanh'))
clf_nn.add(Dropout(.3))

clf_nn.add(Dense(128, activation='relu'))
clf_nn.add(Dropout(.3))
clf_nn.add(Dense(64, activation='relu'))
clf_nn.add(Dropout(.2))
clf_nn.add(Dense(1, activation='sigmoid'))

clf_nn.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
clf_nn.summary()

clf_nn.fit(X_train, y_train, epochs=100,batch_size=50)
score_nn=clf_nn.evaluate(X_valid, y_valid)
print("Neural network Classifier Accuracy : ", score_nn[1])


import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective="binary:logistic", max_depth=2, eta=1, random_state=0)

xgb_model.fit(X_train, y_train)
from sklearn.metrics import  mean_squared_error

y_pred_x = xgb_model.predict(X_valid)
aScore_x = mean_squared_error(y_valid, y_pred_x)
print(aScore_x)
"""





