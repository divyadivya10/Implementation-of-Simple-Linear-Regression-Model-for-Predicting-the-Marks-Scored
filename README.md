# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the necessary libraries such as pandas, numpy, matplotlib, and sklearn modules for data handling, visualization, and model building.

2.Load the dataset using pandas and display the first and last few rows to understand the data structure.
3.Separate the dataset into independent variable (x - Hours studied) and dependent variable (y - Marks scored).
4.Split the dataset into training and testing sets using train_test_split, with one-third data reserved for testing.
5.Create and train the Linear Regression model using the training data by fitting the LinearRegression object.
6.Predict the output (marks scored) for the test data using the trained model.
7.Evaluate the model performance by calculating Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
8.Visualize the results for the training set by plotting a scatter graph of the actual data points and drawing the best-fit line.
9.Visualize the results for the test set similarly, plotting actual versus predicted points to assess the model accuracy.
10.Display the student's name and register number to personalize the project output.
```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Divya R
RegisterNumber: 212222040040
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse = np.sqrt(mse)
print("RMSE = ",rmse)
plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name: Divya R")
print("Reg no : 212222040040")
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="black")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name: Divya R")
print("Reg no : 212222040040")
```

## Output:

## head()
![image](https://github.com/user-attachments/assets/169ba70b-a69f-46dd-9c94-b45711dff4d2)
## tail()
![image](https://github.com/user-attachments/assets/407d0490-d456-41ad-a251-b581719afd0b)
## Array
![image](https://github.com/user-attachments/assets/e81a3b1e-9ecf-480d-b533-f28c5ca9e724)
![image](https://github.com/user-attachments/assets/c13a7273-2633-4f32-8c50-ef590514d305)
## Errors
![image](https://github.com/user-attachments/assets/17a29cc1-841d-4883-bb17-3c14cc5b03a3)
## Hours vs Scores training set
![image](https://github.com/user-attachments/assets/27230549-5ee0-48c0-a2c0-1e61b764832a)
## Hours vs Scores test set
![image](https://github.com/user-attachments/assets/bb81cbea-81ca-46dd-b87e-c92869a7d827)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
