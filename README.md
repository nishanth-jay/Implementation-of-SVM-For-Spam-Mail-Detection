# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
/*
Program to implement the SVM For Spam Mail Detection..

Developed by: J Nishanth

RegisterNumber:  212225040284

*/
import chardet

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn import metrics


file = 'spam.csv'

with open(file, 'rb') as rawdata:
  
    result = chardet.detect(rawdata.read(100000))


data = pd.read_csv('spam.csv', encoding='Windows-1252')

x = data["v2"].values

y = data["v1"].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


model = Pipeline([

    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())

])


model.fit(x_train, y_train)

y_pred = model.predict(x_test)


accuracy = metrics.accuracy_score(y_test, y_pred)

print(accuracy)

print(metrics.classification_report(y_test, y_pred))

print(metrics.confusion_matrix(y_test, y_pred))

## Output:
<img width="708" height="150" alt="398595784-ba42ca2d-85a5-4795-b40d-63df37236a81" src="https://github.com/user-attachments/assets/46f2e286-804a-4daf-a1bc-046035ab581e" />
<img width="726" height="237" alt="398595804-22edcf48-dd25-4513-867d-6c7430da985d" src="https://github.com/user-attachments/assets/2e47c8c5-0f36-4bcb-84a6-16c93fc82c6e" />
<img width="414" height="274" alt="398595822-618324db-35cd-4c45-8469-466a22c5cf86" src="https://github.com/user-attachments/assets/adac6b01-6679-445c-9c2d-029482c9d6f0" />
<img width="220" height="169" alt="398595845-81bc5619-eb01-41f1-9218-e95a64875d42" src="https://github.com/user-attachments/assets/67ab163d-6f70-47cd-8e05-ace60ad2a8cf" />
<img width="422" height="131" alt="398595912-3da73419-4d11-45bf-b361-b1fd344e732e" src="https://github.com/user-attachments/assets/35058df0-f2c1-4b5d-8865-602f10687649" />
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
