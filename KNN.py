# -*- coding: utf-8 -*-
"""
@author : Bhavani Jaladanki (bjaladanki3@gatech.edu)

"""


import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel



adult = pd.read_hdf('adult.hdf','adult')
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values

#madelon = pd.read_hdf('datasets.hdf','madelon')
#madelonX = madelon.drop('Class',1).copy().values
#madelonY = madelon['Class'].copy().values
#


adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)
#madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)


d = adultX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(1,9.01,1/2)]
#d = madelonX.shape[1]
#hiddens_madelon = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]

#
#pipeM = Pipeline([('Scale',StandardScaler()),
#                 ('Cull1',SelectFromModel(RandomForestClassifier(),threshold='median')),
#                 ('Cull2',SelectFromModel(RandomForestClassifier(),threshold='median')),
#                 ('Cull3',SelectFromModel(RandomForestClassifier(),threshold='median')),
#                 ('Cull4',SelectFromModel(RandomForestClassifier(),threshold='median')),
#                 ('KNN',knnC())])

pipeA = Pipeline([('Scale',StandardScaler()),
                 ('KNN',knnC())])



#params_madelon= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
params_adult= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}

#madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,params_madelon,'KNN','madelon')
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'KNN','adult')


#madelon_final_params={'KNN__n_neighbors': 43, 'KNN__weights': 'uniform', 'KNN__p': 1}
#adult_final_params={'KNN__n_neighbors': 142, 'KNN__p': 1, 'KNN__weights': 'uniform'}
#madelon_final_params=madelon_clf.best_params_
adult_final_params=adult_clf.best_params_



#pipeM.set_params(**madelon_final_params)
#makeTimingCurve(madelonX,madelonY,pipeM,'KNN','madelon')
pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'KNN','adult')




######################################################################################
#
#
#banknote= pd.read_hdf('datasets.hdf','banknote') 
#banknote.describe()
#banknoteX = banknote.drop('clas',1).copy().values
#banknoteY = banknote['clas'].copy().values
#banknote_trgX, banknote_tstX, banknote_trgY, banknote_tstY = ms.train_test_split(banknoteX, banknoteY, test_size=0.7, random_state=0,stratify=banknoteY)     
#
#
##banknote_trgX, banknote_tstX, banknote_trgY, banknote_tstY = ms.train_test_split(banknoteX, banknoteY, test_size=0.3, random_state=0,stratify=banknoteY)     
#d = banknoteX.shape[1]
#hiddens_banknote = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
#alphas = [10**-x for x in np.arange(1,7.01,1/8)]
#pipeB = Pipeline([('Scale',StandardScaler()),                
#                 ('KNN',knnC())])  
#
#params_banknote= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,20,2),'KNN__weights':['uniform','distance']}
#banknote_clf = basicResults(pipeB,banknote_trgX,banknote_trgY,banknote_tstX,banknote_tstY,params_banknote,'KNN','banknote')        
#
#banknote_final_params=banknote_clf.best_params_
#pipeB.set_params(**banknote_final_params)
#makeTimingCurve(banknoteX,banknoteY,pipeB,'KNN','banknote')



##########################################################################################

#wine = pd.read_hdf('datasets.hdf','wine_1') 
#
#wine.describe()
#wineX = wine.drop('quality',1).copy().values
#wineY = wine['quality'].copy().values
#wine_trgX, wine_tstX, wine_trgY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.42, random_state=0,stratify=wineY)     
#
#d = wineX.shape[1]
#hiddens_wine = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
#alphas = [10**-x for x in np.arange(1,9.01,1/8)]
#pipeW = Pipeline([('Scale',StandardScaler()),                
#                 ('KNN',knnC())])  
#
#params_wine= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
#wine_clf = basicResults(pipeW,wine_trgX,wine_trgY,wine_tstX,wine_tstY,params_wine,'KNN','wine')        
#
#wine_final_params=wine_clf.best_params_
#pipeW.set_params(**wine_final_params)
#makeTimingCurve(wineX,wineY,pipeW,'KNN','wine')

#########################################################################################





# SPAM
# spam = pd.read_hdf('spam.hdf','spam')
# spamX = spam.drop('clas',1).copy().values
# spamY = spam['clas'].copy().values
#
#
#
# spam_trgX, spam_tstX, spam_trgY, spam_tstY = ms.train_test_split(spamX, spamY, test_size=0.3, random_state=0,stratify=spamY)
#
# d = spamX.shape[1]
# hiddens_spam = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
# alphas = [10**-x for x in np.arange(1,9.01,1/2)]
#
# pipeB = Pipeline([('Scale',StandardScaler()),
#                  ('KNN',knnC())])
#
# params_spam= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
# spam_clf = basicResults(pipeB,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params_spam,'KNN','spam')
#
# spam_final_params=spam_clf.best_params_
# pipeB.set_params(**spam_final_params)
# makeTimingCurve(spamX,spamY,pipeB,'KNN','spam')
