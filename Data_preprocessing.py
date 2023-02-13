# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:03:19 2023

@author: JULIAN
"""

# como importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar data set
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values




# dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,
                                                  random_state=0)

#escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""




 