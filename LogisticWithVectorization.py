import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import textdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('data/train.csv')


# vectorize name
X_name = train['Name']
vector_name = CountVectorizer(binary = False, min_df=2, ngram_range=(1,3), analyzer='word')
X_name = vector_name.fit_transform(X_name.tolist()).todense()
# sorted_vol = sorted(vector_name.vocabulary_.keys(),key=lambda a:vector_name.vocabulary_[a])
# freq = {sorted_vol[i]: name.sum(axis=0)[0,i] for i in range(len(sorted_vol))}
# print({k: v for k, v in sorted(freq.items(), key=lambda a: a[1])})
# print(len(vector_name.vocabulary_))

# vectorize ticket
X_ticket = train['Ticket']
vector_ticket = CountVectorizer(binary = False, min_df=2, ngram_range=(1,3), analyzer='word')
X_ticket = vector_ticket.fit_transform(X_ticket.tolist()).todense()
# sorted_vol = sorted(vector_ticket.vocabulary_.keys(),key=lambda a:vector_ticket.vocabulary_[a])
# freq = {sorted_vol[i]: ticket.sum(axis=0)[0,i] for i in range(len(sorted_vol))}
# print({k: v for k, v in sorted(freq.items(), key=lambda a: a[1])})
# print(len(vector_ticket.vocabulary_))

# vectorize cabin
X_cabin = train['Cabin']
vector_cabin = CountVectorizer(binary = False, min_df=2, ngram_range=(1,3), max_df=0.5, analyzer='word')
X_cabin = vector_cabin.fit_transform(X_cabin.values.astype(str)).todense()
# sorted_vol = sorted(vector_cabin.vocabulary_.keys(),key=lambda a:vector_cabin.vocabulary_[a])
# freq = {sorted_vol[i]: cabin.sum(axis=0)[0,i] for i in range(len(sorted_vol))}
# print({k: v for k, v in sorted(freq.items(), key=lambda a: a[1])})
# print(len(vector_cabin.vocabulary_))

# benchmark of using logistic regression.
X_train_disc = train[['Pclass','Sex','Embarked']]
X_train_cont = train[['Age','SibSp','Parch','Fare']]
# print(X_train_disc)

imputer_disc = SimpleImputer(strategy='constant')
imputer_cont = SimpleImputer(strategy='median')
enc = OneHotEncoder(handle_unknown='ignore')



X_train_disc = enc.fit_transform(imputer_disc.fit_transform(X_train_disc)).todense()
X_train_cont = imputer_cont.fit_transform(X_train_cont)
X_train = np.concatenate((X_train_disc,X_train_cont,X_name,X_cabin,X_ticket),axis=1)

y_train = train['Survived']

lr = LogisticRegression(solver='liblinear').fit(X_train,y_train) # C=0.01 degrade a lot?
print(lr.score(X_train,y_train))

para=[{'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
search=GridSearchCV(estimator=lr, param_grid=para, scoring='accuracy',cv=100)
search.fit(X_train,y_train)
print(search.best_score_)
print(search.best_params_)

# on test set
test = pd.read_csv('data/test.csv')

# benchmark of using logistic regression.
X_test_name = vector_name.transform(test['Name'].tolist()).todense()
X_test_ticket = vector_ticket.transform(test['Ticket'].tolist()).todense()
X_test_cabin = vector_cabin.transform(test['Cabin'].values.astype(str)).todense()

X_test_disc = test[['Pclass','Sex','Embarked']]
X_test_cont = test[['Age','SibSp','Parch','Fare']]

X_test_disc = enc.transform(imputer_disc.transform(X_test_disc)).todense()
X_test_cont = imputer_cont.transform(X_test_cont)
X_test = np.concatenate((X_test_disc,X_test_cont,X_test_name,X_test_cabin,X_test_ticket),axis=1)

y_test = search.predict(X_test)

out = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':y_test})
out.to_csv('logisticVector.csv',index=False)