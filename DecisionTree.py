import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


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

tree = DecisionTreeClassifier()
para=[{'max_depth':[3,4,5,6,7,8,9],'criterion':['gini','entropy']}]
search=GridSearchCV(estimator=tree, param_grid=para, scoring='accuracy',cv=5)
# scores_rf=cross_val_score(search,X_train,y_train,scoring='accuracy',cv=5)
search.fit(X_train,y_train)
print(search.best_score_)
print(search.best_params_)
#
# print(scores_rf.mean())
# #
# # on test set
test = pd.read_csv('data/test.csv')

# benchmark of using logistic regression.
X_test_disc = test[['Pclass','Sex','Cabin','Embarked','Ticket']]
X_test_cont = test[['Age','SibSp','Parch','Fare']]

X_test_disc = enc.transform(imputer_disc.transform(X_test_disc)).todense()
X_test_cont = imputer_cont.transform(X_test_cont)
X_test = np.concatenate((X_test_disc,X_test_cont),axis=1)

y_test = search.predict(X_test)
#
out = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':y_test})
out.to_csv('tree.csv',index=False)