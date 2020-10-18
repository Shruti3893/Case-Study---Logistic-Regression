# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:01:33 2020

@author: Lenovo
"""

# =============================================================================
# Logistic regression for Bank data set
# =============================================================================

import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt

#Importing Data
Bank = pd.read_csv("bank-full.csv", sep = ';')
Bank.head()
Bank.drop(['contact','poutcome'],inplace =True,axis =1)

#Convert State Column from Character to Binary
from sklearn import preprocessing
df = preprocessing.LabelEncoder()
Bank['job'] = df.fit_transform(Bank['job'])
Bank['marital'] = df.fit_transform(Bank['marital'])
Bank['education'] = df.fit_transform(Bank['education'])
Bank['default'] = df.fit_transform(Bank['default'])
Bank['housing'] = df.fit_transform(Bank['housing'])
Bank['loan'] = df.fit_transform(Bank['loan'])
Bank['month'] = df.fit_transform(Bank['month'])
Bank['y'] = df.fit_transform(Bank['y'])

Bank.columns
Bank.isnull().sum()

import seaborn as sns
sns.boxplot(x="age",y="job",data=Bank)
plt.boxplot(Bank.education)
Bank.describe()

Bank.mean()
Bank.shape

Bank.y.value_counts()
Bank.y.value_counts().index[0] # gets you the most occuring value
Bank.y.value_counts().plot(kind="bar")

# Splitting data into train and test
from sklearn.model_selection import train_test_split

train,test = train_test_split(Bank,test_size = 0.3,random_state=42)

trainX = train.drop(["y"],axis=1)
trainY = train["y"]
testX = test.drop(["y"],axis=1)
testY = test["y"]
 
testY.value_counts() 
# Checking na values 
train.isnull().sum()
test.isnull().sum()

#Model building 
from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression()
logit_model.fit(trainX, trainY)
logit_model.coef_
logit_model.predict_proba(trainX)
y_pred = pd.Series(logit_model.predict(trainX))

y_prob = pd.DataFrame(logit_model.predict_proba(trainX.iloc[:,:]))
Bank['y_pred'] = y_pred
Bank_new  = pd.concat([Bank,y_prob],axis=1)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(trainY,y_pred)
pd.crosstab(trainY,y_pred)
accuracy_test = (20147+52)/(22155) # 91.17%
accuracy_test

# filling all the cells with zeroes
train["train_pred"] = np.zeros(31647)
Bank['pred_prob'] = y_pred
# Creating new column for storing predicted class of y

# filling all the cells with zeroes
Bank["y_val"] = 0

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
Bank.loc[y_pred>=0.5,"y_val"] = 1
Bank.y_val

from sklearn.metrics import classification_report
classification_report(Bank.y_val,Bank.y)

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(Bank.y_val,Bank.y)
# the above function is applicable for binary classification class 
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 

# Prediction on Test data set
# Creating new column for storing predicted class of y
test_pred = logit_model.predict(testX)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
testX.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion = pd.crosstab(testY,y_pred)
pd.crosstab(testY,y_pred)
accuracy_test = (8608+24)/(9492) # 90.93%
accuracy_test