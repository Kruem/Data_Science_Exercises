# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:42:23 2023

@author: JULIAN
"""
# Regresion lineal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar data set
dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

# dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3, random_state=0)


#crear model de regresion lineal con conjunto de entrenamiento

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,Y_train)

#Predecir el conjunto de test
y_pred = regression.predict(X_test)

#visualizar los datos de entrenamiento


plt.scatter(X_train,Y_train, color="red")
plt.plot(X_train,regression.predict(X_train), color="blue")
plt.title("Sueldo vs Años de Experiencia(Conjunto de Entrenamiento)")
plt.xlabel("Años de Experciencia")
plt.ylabel("Sueldo {em $}")
plt.show()

#visualizar los datos de test

plt.scatter(X_test,Y_test, color="red")
plt.plot(X_train,regression.predict(X_train), color="blue")
plt.title("Sueldo vs Años de Experiencia(Conjunto de Test)")
plt.xlabel("Años de Experciencia")
plt.ylabel("Sueldo {em $}")
plt.show()
