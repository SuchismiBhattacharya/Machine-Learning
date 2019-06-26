import pandas as pd
url = 'C:\diabetis.csv'
col_names=['Pregnancies','Glucose','Diastolic','Triceps','Insulin','BMI','DPF','Age','Diabetes']
dia= pd.read_csv(url, header=None,names=col_names)
X = dia.drop(['Diabetes'], axis=1)
y = dia['Diabetes'].values
from sklearn.model_selection import train_test_split
## Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
from sklearn.neighbors import KNeighborsClassifier
## Instantiate the model with 5 neighbors. 
knn = KNeighborsClassifier(n_neighbors=19,leaf_size=45)
## Fit the model on the training data.
print(knn.fit(X_train, y_train))
## See how the model performs on the test data.
print(knn.score(X_test, y_test))
