
import pandas as pd
#imports
import numpy as np
from sklearn import model_selection
from numpy import percentile
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn import ensemble
from sklearn import tree
from sklearn import naive_bayes
from sklearn.feature_selection import RFECV
import random
from sklearn import neighbors
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

employeeDataset = pd.read_csv("EmployeeAttrition.csv")
employeeDataset["Attrition"].replace({"Yes":1,"No":0} , inplace = True)

print(employeeDataset.head())

def changeDiscrete(seriesName):
  valuesInSeries = list(set(employeeDataset[seriesName].values))
  dictTemp = {} 
  for i in range(len(valuesInSeries)):
    dictTemp[valuesInSeries[i]] = i
  
  employeeDataset[seriesName].replace(dictTemp , inplace = True)

discreteSeriesNames = [
    "Attrition",
    "BusinessTravel",
    "Department",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "Over18",
    "OverTime",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance"
]

for i in discreteSeriesNames[1:]:
  changeDiscrete(i)


print(employeeDataset.head())

X = employeeDataset.drop("Attrition" , axis = 1)
y = employeeDataset["Attrition"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

logreg = logreg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=45, solver='liblinear', max_iter=100, multi_class='warn', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
from sklearn import metrics, cross_validation
predicted = cross_validation.cross_val_predict(logreg, X, y, cv=10)
metrics.accuracy_score(y, predicted) 


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
