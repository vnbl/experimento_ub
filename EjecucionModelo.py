#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:37:17 2020

@author: fernanda
"""
import pandas as pd #Para manipular datasets
import numpy as np #Funciones matematicas de manera vectorial
import random as rand
import sklearn.metrics as metrics # Para utilizar algoritmos de machine learning. Metrics: utiliza m'etricas de desempeno, funciones de score, etc. de forma a cuantificar la calidad de predicciones
import seaborn as sns # Visualizacion de datos

# matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.model_selection import train_test_split #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import LeaveOneOut # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html?highlight=leaveoneout




def train_and_predict(model, X_train, Y_train, X_test, Y_test):
    columns = Y_test.columns # pandas.core.indexes.base.Index
#    print ("aca empieza")
#    print(type(Y_test))
#    print ("aca termina")
    Y_pred = pd.DataFrame(index=Y_test.index) #Y_pred esta inicialmente vacio, tiene dimension n' = cantidad de alumnos del set de pruebas.


# Usar for suele ser lento para entrenar, mejor usar algo tipo producto vectorial.
    # numpy.dot en vez de for.
    for subject in columns:
        #print(type(subject))
        subject = [subject]
        #print(type(subject))
        model.fit(X_train, Y_train[subject])
        partial_pred = pd.DataFrame(model.predict(X_test), index=Y_test[subject].index, columns=subject)
        Y_pred[subject] = partial_pred


    return Y_pred


def fit_and_predict_model(model, XY, X_labels, Y_labels, train_size=0.2): #train_size= proporcion del data set
    ''' Split in train and test, train model and run predictions for all output vectors '''
    train, test = train_test_split(XY, random_state=0, train_size=train_size)
    X_train = train[X_labels]
    Y_train = train[Y_labels]
    X_test = test[X_labels]
    Y_test = test[Y_labels]

    Y_pred = train_and_predict(model, X_train, Y_train, X_test, Y_test)

    # # Quantize prediction to nearest .5
    # Y_pred = (Y_pred * 2).round(0) / 2

    return X_test, Y_test, Y_pred 

def fit_and_predict_model_loo(model, XY, X_labels, Y_labels):
    ''' Fit model on N-1 samples and Predict last sample '''

    loo = LeaveOneOut()

    X = XY[X_labels]
    Y = XY[Y_labels]

    #nsplits = loo.get_n_splits(X)
#    print(X.shape)
#    print("empieza nsplits")
#    print(nsplits)
    #Y = XY[Y_labels]

    Y_pred = pd.DataFrame(columns=Y.columns)
    split = 1

#    matriz = [loo.split(X)]
#    print(matriz)
#   no anda
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        #print(str(X_train) + str(" ") + str(X_test))
        Y_pred_sample = train_and_predict(model, X_train, Y_train, X_test, Y_test)

        Y_pred = pd.concat([Y_pred, Y_pred_sample])

        split += 1
        #print(split)
    return Y_pred