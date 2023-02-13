# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 22:59:58 2023

@author: JULIAN
"""
#Datos categoricos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar data set
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values


# codificar datos categoricos
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])
le_Y = preprocessing.LabelEncoder()
Y= le_Y.fit_transform(Y)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)