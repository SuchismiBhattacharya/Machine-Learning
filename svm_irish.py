#import the pandas and seaborn liabrary 
import pandas as pd
import seaborn as sb
#read in the data using seaborn
df=sb.load_dataset('iris')
#create a dataframe with all training data except the target column.
#Here the Target column is Species 
x=df.drop(columns=['species'])
#Load the test data set 
y=df['species'].values
#split dataset into train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,
                    test_size=0.2,random_state=6,stratify=y)
from sklearn import svm

clf = svm.SVC()
clf.fit(x_train, y_train)
confidence = clf.score(x_test, y_test)
print(confidence)
