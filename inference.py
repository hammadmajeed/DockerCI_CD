#!/usr/bin/python3
# inference.py

import platform;
import sys;
import numpy;
import scipy;
from sklearn.metrics import confusion_matrix
import statistics as stat

import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
from joblib import load
from sklearn import preprocessing



def inference():

    MODEL_DIR = os.environ["MODEL_DIR"]
    MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
    MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
    MODEL_FILE_RFC = os.environ["MODEL_FILE_RFC"]
    MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
    MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)
    MODEL_PATH_RFC = os.path.join(MODEL_DIR, MODEL_FILE_RFC)

    # Load, read and normalize training data
    testing = "test.csv"
    data_test = pd.read_csv(testing)
        
    y_test = data_test['# Letter'].values
    X_test = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
   
   # print("Shape of the test data")
   # print(X_test.shape)
   # print(y_test.shape)
    
    # Data normalization (0,1)
    X_test = preprocessing.normalize(X_test, norm='l2')
    
    # Models training
    
    # Run model
    #print(MODEL_PATH_LDA)
    clf_lda = load(MODEL_PATH_LDA)
    #print("LDA score and classification:")
    clf_lda.score(X_test, y_test)
    #prediction of the LDA
    pred_lda = clf_lda.predict(X_test)
    #print(pred_lda)
 #   print(confusion_matrix(y_test, predicted))

    # random forest
    from sklearn.ensemble import RandomForestClassifier
    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(X_test, y_test)
    #print("RFC score and classification:")
    clf_rfc.score(X_test, y_test)
    pred_rfc = clf_rfc.predict(X_test)
    #print(pred_rfc)

   # print(confusion_matrix(y_test, predicted))
    # Run model
    clf_nn = load(MODEL_PATH_NN)
    #print("NN score and classification:")
    clf_nn.score(X_test, y_test)
    pred_nn = clf_nn.predict(X_test)
   # print(pred_nn)
    #Use majority voting for prediction
    predictions=pd.DataFrame([pred_nn,pred_rfc,pred_lda])
    final = predictions.apply(stat.mode)
    #Recommendation for the boss
    print("Based on my training, I recommend the following")
    ## Python program to understand, how to print tables using pandas data frame
    data = {'Estimator 1':pred_lda,'Estimator 2':pred_rfc,'Estimator 3':pred_nn, 'Decision':final}
    print(pd.DataFrame(data))
if __name__ == '__main__':
    inference()
