import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('data/train.csv')

# benchmark of using logistic regression.
X_train_disc = train[['Pclass','Sex','Cabin','Embarked','Ticket']]
X_train_cont = train[['Age','SibSp','Parch','Fare']]

imputer_disc = SimpleImputer(strategy='constant')
imputer_cont = SimpleImputer(strategy='median')

enc = OneHotEncoder(handle_unknown='ignore')

X_train_disc = enc.fit_transform(imputer_disc.fit_transform(X_train_disc)).todense()
X_train_cont = imputer_cont.fit_transform(X_train_cont)
X_train = np.concatenate((X_train_disc,X_train_cont),axis=1)

y_train = train['Survived']

lr = LogisticRegression(solver='liblinear').fit(X_train,y_train) # C=0.01 degrade a lot?
print(lr.score(X_train,y_train))

# on test set
test = pd.read_csv('data/test.csv')

# benchmark of using logistic regression.
X_test_disc = test[['Pclass','Sex','Cabin','Embarked','Ticket']]
X_test_cont = test[['Age','SibSp','Parch','Fare']]

X_test_disc = enc.transform(imputer_disc.transform(X_test_disc)).todense()
X_test_cont = imputer_cont.transform(X_test_cont)
X_test = np.concatenate((X_test_disc,X_test_cont),axis=1)

y_test = lr.predict(X_test)

out = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':y_test})
out.to_csv('logistic.csv',index=False)