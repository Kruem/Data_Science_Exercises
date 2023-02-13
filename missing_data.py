# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 23:00:27 2023

@author: JULIAN
"""
#Datos faltantes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar data set
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#tratamiento de datos NA

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None, 
                      verbose=0, copy=True, add_indicator=False)
imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])