# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:GREFFINA SANCHEZ P 
RegisterNumber: 212222040048 
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

1. Data head

![image](https://github.com/greffinaprem/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475603/376c2cef-01b0-4b10-8cc5-027fea359d06)

2. Data set info

![image](https://github.com/greffinaprem/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475603/535bc5af-7455-4b6f-a64a-092676232dd9)

3. Null dataset

![image](https://github.com/greffinaprem/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475603/51ddb129-2a35-49cc-877b-8652a24bfc2a)

4. Values count in left column 

![image](https://github.com/greffinaprem/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475603/5aea28b6-deba-4aaa-93b6-acc735fdf34c)

5. Dataset transformed head

![image](https://github.com/greffinaprem/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475603/60004176-f43e-44f7-b7cf-ac59a9eac460)

6. x.head

![image](https://github.com/greffinaprem/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475603/e728900f-f717-4d77-8d2c-2535851bc8b0)

7. Accuracy

![image](https://github.com/greffinaprem/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475603/b6ede513-9d87-408e-a228-8acb5bcbdaf9)

8. Data prediction

![image](https://github.com/greffinaprem/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119475603/a9f5ab17-8ab8-4fef-aaf8-3d321229eb10)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
