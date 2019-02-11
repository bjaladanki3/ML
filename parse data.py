# -*- coding: utf-8 -*-
"""
@author : Bhavani Jaladanki (bjaladanki3@gatech.edu)

"""

import pandas as pd
import numpy as np
import xlsxwriter
from sklearn.model_selection import train_test_split


# Preprocess with adult dataset

#df = pd.read_csv('./adult.csv', header=None)
#adult = pd.read_csv('./adult2.csv',header=None)
adultf = pd.read_csv('./adult.data', skiprows= 1)
Y = [1]*adultf.shape[0]


adult, wast1, wast2, wast3 = train_test_split(adultf, Y, test_size=0.45, random_state=0)


adult.columns = ['age','employer','fnlwt','edu','edu_num','marital','occupation','relationship','race','sex','cap_gain','cap_loss','hrs','country','income']
#adult['cap_gain'][5]

#adult.cap_gain.dtype = pd.np.float32
#adult['cap_gain'] = adult['cap_gain'].convert_objects(convert_numeric=True)
#adult['cap_loss'] = adult['cap_loss'].convert_objects(convert_numeric=True)

adult.describe()

#pd.to_numeric(adult, errors='coerce')
# Note that cap_gain > 0 => cap_loss = 0 and vice versa. Combine variables.
#print(adult.ix[adult.cap_gain>0].cap_loss.abs().max())
#print(adult.ix[adult.cap_loss>0].cap_gain.abs().max())
adult['cap_gain_loss'] = adult['cap_gain']-adult['cap_loss']
adult = adult.drop(['fnlwt','edu','cap_gain','cap_loss'],1)
adult['income'] = pd.get_dummies(adult.income)
print(adult.groupby('occupation')['occupation'].count())
print(adult.groupby('country').country.count())
#http://scg.sdsu.edu/dataset-adult_r/
replacements = { 'Cambodia':' SE-Asia',
                'Canada':' British-Commonwealth',
                'China':' China',
                'Columbia':' South-America',
                'Cuba':' Other',
                'Dominican-Republic':' Latin-America',
                'Ecuador':' South-America',
                'El-Salvador':' South-America ',
                'England':' British-Commonwealth',
                'France':' Euro_1',
                'Germany':' Euro_1',
                'Greece':' Euro_2',
                'Guatemala':' Latin-America',
                'Haiti':' Latin-America',
                'Holand-Netherlands':' Euro_1',
                'Honduras':' Latin-America',
                'Hong':' China',
                'Hungary':' Euro_2',
                'India':' British-Commonwealth',
                'Iran':' Other',
                'Ireland':' British-Commonwealth',
                'Italy':' Euro_1',
                'Jamaica':' Latin-America',
                'Japan':' Other',
                'Laos':' SE-Asia',
                'Mexico':' Latin-America',
                'Nicaragua':' Latin-America',
                'Outlying-US(Guam-USVI-etc)':' Latin-America',
                'Peru':' South-America',
                'Philippines':' SE-Asia',
                'Poland':' Euro_2',
                'Portugal':' Euro_2',
                'Puerto-Rico':' Latin-America',
                'Scotland':' British-Commonwealth',
                'South':' Euro_2',
                'Taiwan':' China',
                'Thailand':' SE-Asia',
                'Trinadad&Tobago':' Latin-America',
                'United-States':' United-States',
                'Vietnam':' SE-Asia',
                'Yugoslavia':' Euro_2'}
adult['country'] = adult['country'].str.strip()
adult = adult.replace(to_replace={'country':replacements,
                                  'employer':{' Without-pay': ' Never-worked'},
                                  'relationship':{' Husband': 'Spouse',' Wife':'Spouse'}})
adult['country'] = adult['country'].str.strip()
print(adult.groupby('country').country.count())
for col in ['employer','marital','occupation','relationship','race','sex','country']:
    adult[col] = adult[col].str.strip()

# adult = adult.dropna(axis=1,how='all')
adult = pd.get_dummies(adult)
adult = adult.rename(columns=lambda x: x.replace('-','_'))

adult.to_hdf('adult.hdf','adult',complib='blosc',complevel=9)
writer = pd.ExcelWriter('adult.xlsx', engine='xlsxwriter')
adult.to_excel(writer, sheet_name='Sheet1')
writer.save()






## Spam
spam = pd.read_csv('./spambase.data',header=None)
spam.columns = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order",
                "word_freq_mail", "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free", "word_freq_business",
                "word_freq_email", "word_freq_you", "word_freq_credit", "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl",
                "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
                "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original",
                "word_freq_project", "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
                "char_freq_$", "char_freq_#", "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total", "clas"]

spam["clas"] = pd.get_dummies(spam.clas)
# spam = spam.dropna(axis=1,how='all')
spam = pd.get_dummies(spam)
spam.describe()
spam.to_hdf('spam.hdf','spam',complib='blosc',complevel=9)
writer = pd.ExcelWriter('spam.xlsx', engine='xlsxwriter')
spam.to_excel(writer, sheet_name='Sheet1')
writer.save()

# biodeg = pd.read_csv('./biodeg.csv')
# biodeg.columns= ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19","A20","A21","A22","A23","A24","A25","A26","A27","A28","A29","A30","A31","A32","A33","A34","A35","A36","A37","A38","A39","A40","A41","clas"]
# biodeg["clas"] = pd.get_dummies(biodeg.clas)
# biodeg = pd.get_dummies(biodeg)
# biodeg.describe()
# biodeg.to_hdf('datasets.hdf','biodeg',complib='blosc',complevel=9)
#


