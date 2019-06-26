import pandas as pd
wine=[]
col_names=[]
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data' 
col_names = ['Alcohol','Malic_Acid','Ash','Alvcalinity_of_Ash','Magnesium','Total_Phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280/OD315 of diluted wines', 'Proline','Wine_num']
wine= pd.read_csv(url, header=None,names=col_names)
wine_class = {1:1,2:2,3:3}
wine['Wine_num'] = [wine_class[i] for i in wine.Alcohol]
X = wine.drop(['Alcohol', 'Wine_num'], axis=1)
y = wine.Wine_num
from sklearn.model_selection import train_test_split
## Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(40),max_iter=150, alpha=1e-4,
                  solver='adam', verbose=10, tol=1e-4 ,learning_rate_init=.1)
mlp.fit(X_train,y_train)
print("Training set score: %f" % mlp.score(X_train,y_train))
print("Test set score: %f " % mlp.score(X_train,y_train))
