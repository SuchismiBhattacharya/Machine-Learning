import pandas as pd
url = 'C:\diabetis.csv'
col_names=['Pregnancies','Glucose','Diastolic','Triceps','Insulin','BMI','DPF','Age','Diabetes']
dia= pd.read_csv(url, header=None,names=col_names)
X = dia.drop(['Diabetes'], axis=1)
y = dia['Diabetes'].values
from sklearn.model_selection import train_test_split
## Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(40),max_iter=150, alpha=1e-4,
                  solver='adam', verbose=10, tol=1e-4 ,learning_rate_init=.1)
mlp.fit(X_train,y_train)
print("Training set score: %f" % mlp.score(X_train,y_train))
print("Test set score: %f " % mlp.score(X_train,y_train))
