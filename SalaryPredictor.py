# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 19:59:34 2018

@author: yatheen!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython # the plot is shown as a seperate screen
get_ipython().run_line_magic('matplotlib', 'qt')

print("\n\nStep 1 : All packages imported Successfully")

dataset = pd.read_csv(r'E:\Mini Projects\Machine Learning\What will be your salary\Salary_Data.csv')

print("Step 2 : Dataset Imported")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,1].values

print("Step 3 : Feature Matrix and Output values are determined.")

from sklearn import cross_validation
from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 1/3 , random_state = 4)

print("Step 4 : Training data and test Data is seperated ")

print("Step 5 : Initial Visualisation")

np.random.seed(19680801)
N = 50
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

import matplotlib.patches as mpatches
"""red_patch = mpatches.Patch(color='red', label='Raw Data')
plt.figure(1)
plt.title("Raw Data")
plt.xlabel('Experience')
plt.legend(handles=[red_patch])
plt.ylabel('Salary')
plt.scatter(X,Y, s=area, c='red', alpha=0.5)
plt.show()

plt.figure(2)
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
blue_patch = mpatches.Patch(color='blue', label='Training data')
green_patch = mpatches.Patch(color='green', label='Test data')
plt.legend(handles=[blue_patch,green_patch])
plt.scatter(X_train,Y_train, s=area, c='blue', alpha=0.5)
plt.scatter(X_test,Y_test, s=area, c='green', alpha=0.5)
plt.show()"""

print("Step 6 : Linear Regression ")

from sklearn.linear_model import LinearRegression
regression_object = LinearRegression()
regression_object.fit(X_train,Y_train)

print("Step 7 : Prediction")
y_pred = regression_object.predict(X_test)

print("Step 8 : Model Visualisation")
"""plt.figure(3)
plt.title('Salary vs Experience ')
plt.xlabel('Experience')
plt.ylabel('Salary')
black_line = mpatches.Patch(color='black', label='Predicted Line')
plt.legend(handles=[black_line])
plt.scatter(X_test,Y_test, s=area, c='green', alpha=0.5)
plt.plot(X_train,regression_object.predict(X_train),color='black')
plt.show()"""



print("Model Trained! \nStep 9 : Interactive Model")
experience = int(input("Enter the Experience in years to predict the salary:\t"))

print("The salary for the specified experience will be:%10d" % regression_object.predict(np.array([[experience]])))
