#!/usr/bin/env python
# coding: utf-8

# # #BHARAT INTERNSHIP
# 
# ## NAME-SHAIK TABASSUM SHABANA
#  
#  ## TASK2-TITANIC CLASSIFICATION
#   - In this task we predicts if a passenger will survive on the titanic or not
#   
#   # Import Libraries
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report


# # Loading the Dataset 
titanic=pd.read_csv(r"C:\Users\dell\Downloads\tested.csv")

titanic


# # Cleaning the data
titanic.shape
titanic.info()   #statistical info
titanic.describe()     
titanic.head(10)
titanic.columns


# #  Exploratory Data Analysis
titanic["Sex"].value_counts
titanic['Survived'].value_counts()       #count of no.of survivors


# # Countplot of survived vs not survived
sns.countplot(x='Survived',data=titanic)


# # Male vs Female Survival
sns.countplot(x='Survived',data=titanic,hue='Sex')
sns.countplot(x='Survived',hue='Pclass',data=titanic)


# # Missing Data
titanic.isna()     #check for null
titanic.isnull().sum()


# # Visualize null values
sns.heatmap(titanic.isna())

(titanic['Age'].isna().sum()/len(titanic['Age']))*100

(titanic['Cabin'].isna().sum()/len(titanic['Cabin']))*100

sns.distplot(titanic['Age'].dropna(),kde=False,color='blue',bins=40)

titanic['Age'].hist(bins=40,color='blue',alpha=0.4)

sns.countplot(x='SibSp',data=titanic)

titanic['Fare'].hist(color='red',bins=40,figsize=(8,4))

plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass',y='Age',data=titanic,palette='winter')

titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)

titanic['Age'].isna().sum()

sns.heatmap(titanic.isna())


# - We can see cabin  column has a no.of of null values ,as such we can not use it for prediction so we will drop it 

titanic.drop('Cabin',axis=1,inplace=True)

titanic.head()


# # Preparing Data for Model

titanic.info()


# # convert sex column to numerical values
gender=pd.get_dummies(titanic['Sex'],drop_first=True)

titanic['Gender']=gender
titanic.head()


# # Drop the columns which are not required 

titanic.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)

titanic.head()


# # Separate Dependent and Independent variables

x=titanic[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=titanic['Survived']

y


# # Data Modeling

# >Building model using logestic regression

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.22,random_state=41)


yr=LogisticRegression()

yr.fit(x_train,y_train)

predict=yr.predict(x_train)           #predict


# # Testing

# >See how our model works

pd.DataFrame(confusion_matrix(y_train,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])

print(classification_report(y_train,predict))
