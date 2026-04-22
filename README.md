# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### 1. Import necessary libraries and load the dataset; extract input (R&D Spend) and output (Profit).
### 2. Normalize the input data and initialize parameters (slope m, intercept b), learning rate, and epochs.
### 3. Apply gradient descent to update m and b iteratively by minimizing the error.
### 4. Predict output using the trained model and plot the regression line with the data points.

## Program:

## Program to implement the linear regression using gradient descent.
### Developed by: Kabilan S
### RegisterNumber: 212225230119
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("startup.csv")
X=data['R&D Spend'].values
Y=data['Profit'].values
X=(X-X.mean())/X.std()
m=0
b=0
lr=0.01
epochs=1000
n=len(X)
for i in range(epochs):
    y_pred=m*X + b
    dm=(-2/n)*np.sum(X*(Y-y_pred))
    db=(-2/n)*np.sum(Y-y_pred)
    m=m-lr*dm
    b=b-lr*db
print("Slope :",m)
print("Intercept :",b)
y_pred=m*X+b
plt.scatter(X,Y)
plt.plot(X,y_pred)
plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")
plt.show()

```

## Output:
<img width="1037" height="493" alt="image" src="https://github.com/user-attachments/assets/02df9319-d60b-4142-944b-5b8ed60c931f" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
