import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data' 
col_names = ['Alcohol','Malic_Acid','Ash','Alvcalinity_of_Ash','Magnesium','Total_Phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280/OD315 of diluted wines', 'Proline','Wine_num']
wine= pd.read_csv(url, header=None, names=col_names)
wine_class = {1:1,2:2,3:3}
wine['Wine_num'] = [wine_class[i] for i in wine.Alcohol]
## Create an 'X' matrix by dropping the irrelevant columns.
X = wine.drop(['Alcohol', 'Wine_num'], axis=1)
y = wine.Wine_num
from sklearn.model_selection import train_test_split
## Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)
