
# -*- coding: utf-8 -*-
"""
@author : Bhavani Jaladanki (bjaladanki3@gatech.edu)

"""

import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

class primalSVM_RBF(BaseEstimator, ClassifierMixin):
    '''http://scikit-learn.org/stable/developers/contributing.html'''
    
    def __init__(self, alpha=1e-9,gamma_frac=0.1,n_iter=2000):
         self.alpha = alpha
         self.gamma_frac = gamma_frac
         self.n_iter = n_iter
         
    def fit(self, X, y):
         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         
         # Get the kernel matrix
         dist = euclidean_distances(X,squared=True)
         median = np.median(dist) 
         del dist
         gamma = median
         gamma *= self.gamma_frac
         self.gamma = 1/gamma
         kernels = rbf_kernel(X,None,self.gamma )
         
         self.X_ = X
         self.classes_ = unique_labels(y)
         self.kernels_ = kernels
         self.y_ = y
         self.clf = SGDClassifier(loss='hinge',penalty='l2',alpha=self.alpha,
                                  l1_ratio=0,fit_intercept=True,verbose=False,
                                  average=False,learning_rate='optimal',
                                  class_weight='balanced',n_iter=self.n_iter,
                                  random_state=55)         
         self.clf.fit(self.kernels_,self.y_)
         
         # Return the classifier
         return self

    def predict(self, X):
         # Check is fit had been called
         check_is_fitted(self, ['X_', 'y_','clf','kernels_'])
         # Input validation
         X = check_array(X)
         new_kernels = rbf_kernel(X,self.X_,self.gamma )
         pred = self.clf.predict(new_kernels)
         return pred
    





adult = pd.read_hdf('adult.hdf','adult')
adultX = adult.drop('income',1).copy().values
adultY = adult['income'].copy().values

adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)

N_adult = adult_trgX.shape[0]

alphas = [10**-x for x in np.arange(1,9.01,1/2)]


#Linear SVM
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])

params_adult = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_adult)/.8)+1]}

# print("target",np.unique(adult_trgY))
# print("train",np.unique(adult_tstY))
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'SVM_Lin','adult')

adult_final_params =adult_clf.best_params_
#adult_OF_params ={'SVM__n_iter': 55, 'SVM__alpha': 1e-16}
#
#
adult_OF_params = adult_final_params.copy()
adult_OF_params['SVM__alpha'] = 1e-16

pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'SVM_Lin','adult')

pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_Lin','adult')
#
pipeA.set_params(**adult_OF_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,200,5)},'SVM_LinOF','adult')
#pipeM.set_params(**madelon_OF_params)
#iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_LinOF','madelon')






#RBF SVM
gamma_fracsA = np.arange(0.2,2.1,0.2)

pipeA = Pipeline([('Scale',StandardScaler()),
                 ('SVM',primalSVM_RBF())])


params_adult = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_adult)/.8)+1],'SVM__gamma_frac':gamma_fracsA}
#params_madelon = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_madelon)/.8)+1],'SVM__gamma_frac':gamma_fracsM}
#
#madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,params_madelon,'SVM_RBF','madelon')
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'SVM_RBF','adult')


adult_final_params =adult_clf.best_params_
adult_OF_params = adult_final_params.copy()
adult_OF_params['SVM__alpha'] = 1e-16

pipeA.set_params(**adult_final_params)
makeTimingCurve(adultX,adultY,pipeA,'SVM_RBF','adult')

pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','adult')

pipeA.set_params(**adult_OF_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','adult')


#####################################################################################

# SPAM
spam = pd.read_hdf('spam.hdf','spam')
spamX = spam.drop('clas',1).copy().values
spamY = spam['clas'].copy().values

spam_trgX, spam_tstX, spam_trgY, spam_tstY = ms.train_test_split(spamX, spamY, test_size=0.3, random_state=0,stratify=spamY)

N_spam = spam_trgX.shape[0]

alphas = [10**-x for x in np.arange(1,4.01,1/2)]

#Linear SVM
pipeB = Pipeline([('Scale',StandardScaler()),
                 ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])

params_spam = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_spam)/.8)+1]}

spam_clf = basicResults(pipeB,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params_spam,'SVM_Lin','spam')

spam_final_params =spam_clf.best_params_
#biodeg_OF_params ={'SVM__n_iter': 55, 'SVM__alpha': 1e-16}
#
#
spam_OF_params = spam_final_params.copy()
spam_OF_params['SVM__alpha'] = 1e-16

pipeB.set_params(**spam_final_params)
makeTimingCurve(spamX,spamY,pipeB,'SVM_Lin','spam')

pipeB.set_params(**spam_final_params)
iterationLC(pipeB,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_Lin','spam')
#
pipeB.set_params(**spam_OF_params)
iterationLC(pipeB,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'SVM__n_iter':np.arange(1,200,5)},'SVM_LinOF','spam')


#RBF SVM
gamma_fracsA = np.arange(0.2,2.1,0.2)

pipeB = Pipeline([('Scale',StandardScaler()),
                 ('SVM',primalSVM_RBF())])


params_spam = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_spam)/.8)+1],'SVM__gamma_frac':gamma_fracsA}

spam_clf = basicResults(pipeB,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params_spam,'SVM_RBF','spam')

spam_final_params =spam_clf.best_params_
spam_OF_params = spam_final_params.copy()
spam_OF_params['SVM__alpha'] = 1e-16

pipeB.set_params(**spam_final_params)
makeTimingCurve(spamX,spamY,pipeB,'SVM_RBF','spam')

pipeB.set_params(**spam_final_params)
iterationLC(pipeB,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','spam')

pipeB.set_params(**spam_OF_params)
iterationLC(pipeB,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','spam')
