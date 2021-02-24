import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


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

# on test set
test = pd.read_csv('data/test.csv')

# benchmark of using logistic regression.
X_test_disc = test[['Pclass','Sex','Cabin','Embarked','Ticket']]
X_test_cont = test[['Age','SibSp','Parch','Fare']]

X_test_disc = enc.transform(imputer_disc.transform(X_test_disc)).todense()
X_test_cont = imputer_cont.transform(X_test_cont)
X_test = np.concatenate((X_test_disc,X_test_cont),axis=1)


# dataset
class Data(Dataset):
    def __init__(self, train = True):
        if train:
            self.len = len(X_train)
            self.x = torch.tensor(X_train)
            self.y = torch.tensor(y_train).view(-1,1)
        else:
            self.len = len(X_test)
            self.x = torch.tensor(X_test)
            self.y = torch.zeros(self.len,1).view(-1,1)
    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        return sample

    # Get Length
    def __len__(self):
        return self.len

train_data = Data()
test_data = Data(train=False)

model = nn.Sequential(
    nn.Linear(842, 100),
    nn.ReLU(),
    nn.Linear(100, 1),
    nn.Sigmoid(),
)

learning_rate=0.01
criterion=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loader=DataLoader(dataset=train_data,batch_size=100)

def train(model,criterion, train_loader, optimizer, epochs):
    LOSS = []
    accuracy = []
    for _ in range(epochs):
        epoch_loss = 0
        for X,y in train_loader:
            yhat = model(X.float())
            loss = criterion(yhat,y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        LOSS.append(epoch_loss)
        X,y = train_data[:]
        yhat = torch.round(model(X.float()))
        acc = (yhat == y.float()).numpy().mean()
        accuracy.append(acc)

    return LOSS, accuracy

loss, accuracy =train(model,criterion, train_loader, optimizer, epochs=100)

print(loss)
print(accuracy)
plt.plot(loss)
plt.plot(accuracy)
# plt.show()


yhat = torch.round(model(test_data.x.float())).detach().view(-1).numpy().astype(int)
# print(yhat)

out = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':yhat})
out.to_csv('nn.csv',index=False)