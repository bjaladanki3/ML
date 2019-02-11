# -*- coding: utf-8 -*-
"""
@author : Bhavani Jaladanki (bjaladanki3@gatech.edu)

"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

adult = pd.read_hdf('adult.hdf','adult')
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values



adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)

pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])


d = adultX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]
params_adult = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_adult}
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'ANN','adult')

adult_final_params =adult_clf.best_params_
adult_OF_params =adult_final_params.copy()
adult_OF_params['MLP__alpha'] = 0

pipeA.set_params(**adult_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})
makeTimingCurve(adultX,adultY,pipeA,'ANN','adult')

pipeA.set_params(**adult_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','adult')

pipeA.set_params(**adult_OF_params)
pipeA.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','adult')



# SPAM
spam = pd.read_hdf('spam.hdf','spam')
spamX = spam.drop('clas',1).copy().values
spamY = spam['clas'].copy().values

spam_trgX, spam_tstX, spam_trgY, spam_tstY = ms.train_test_split(spamX, spamY, test_size=0.3, random_state=0,stratify=spamY)

pipeB = Pipeline([('Scale',StandardScaler()),
                  ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

d = spamX.shape[1]
hiddens_spam = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]

params_spam = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_spam}




spam_clf = basicResults(pipeB,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params_spam,'ANN','spam')
spam_final_params =spam_clf.best_params_
spam_OF_params =spam_final_params.copy()
spam_OF_params['MLP__alpha'] = 0

pipeB.set_params(**spam_final_params)
pipeB.set_params(**{'MLP__early_stopping':False})
makeTimingCurve(spamX,spamY,pipeB,'ANN','spam')

pipeB.set_params(**spam_final_params)
pipeB.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeB,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','spam')

pipeB.set_params(**spam_OF_params)
pipeB.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeB,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','spam')
#
#
# #

#loans_clf = basicResults(pipeW,loans_trgX,loans_trgY,loans_tstX,loans_tstY,params_loans,'ANN','loans')        
