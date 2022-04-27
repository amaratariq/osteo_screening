import numpy as np
import pandas as pd
import sys
import pickle as pkl

import pandas as pd
import pickle as pkl
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import os
from matplotlib import pyplot as plt

from scipy.special import expit as sigmoid
from PIL import Image

header_data = '../data/'

def train():
    df = pd.read_csv(os.path.join(header_data, 'data_w_ehr_info.csv'))
    print(len(df))
    def labeling(x):
        if x<=-1:
            return 1
        else:
            return 0
        
    df['Label'] = df['TOTAL_SPINE_TSCORE'].apply(labeling)
    print(df['Label'].value_counts())

    df['PatientAge_mapped'] = df.PatientAge.apply(age_mapping)
    print(df['PatientAge_mapped'].value_counts())


    dct = pkl.load(open('../bucket/Amara/Osteo/mrn_split_basic1235678.pkl', 'rb'))
    mrn_train = dct['mrn_train']
    mrn_test = dct['mrn_test']
    mrn_val = dct['mrn_val']

    df_train = df.loc[df.mrn.isin(mrn_train)]
    df_test = df.loc[df.mrn.isin(mrn_test)]
    df_val = df.loc[df.mrn.isin(mrn_val)]

    dct_axial = pkl.load(open('Model/Axial/train/train-01/train_predictions.pkl', 'rb'))
    dct_coronal = pkl.load(open('Model/Coronal/train/train-01/train_predictions.pkl', 'rb'))
    dct_baseline = pkl.load(open('Models/EHR/train_predictions.pkl', 'rb'))
    dct_axial['probs'] = np.array(dct_axial['probs'])
    dct_coronal['probs'] = np.array(dct_coronal['probs'])

    mat = np.array([
                    (dct_coronal['probs']), 1-(dct_coronal['probs']),(dct_axial['probs']), 1-(dct_axial['probs']), 
                sigmoid(dct_baseline['probs'][:,1]), 1-sigmoid(dct_baseline['probs'][:,1])]).transpose()
    labels_train = dct_axial['labels']


    print(mat.shape, len(labels_train), mat_test.shape, len(labels_test))

    
    op = MinMaxScaler()#StandardScaler()#
    op = op.fit(mat)
    mat_norm = op.transform(mat)
    mat_test_norm = op.transform(mat_test)


    clf = RandomForestClassifier(random_state=0, max_depth  = 5)

    clf.fit(mat_norm, labels_train)
    pkl.dump(clf, open(os.path.join('Models/Fusion/model.sav'), 'wb'))

    
    
def test():
    df = pd.read_csv(os.path.join(header_data, 'data_w_ehr_info.csv'))
    print(len(df))
    def labeling(x):
        if x<=-1:
            return 1
        else:
            return 0
        
    df['Label'] = df['TOTAL_SPINE_TSCORE'].apply(labeling)
    print(df['Label'].value_counts())

    df['PatientAge_mapped'] = df.PatientAge.apply(age_mapping)
    print(df['PatientAge_mapped'].value_counts())


    dct = pkl.load(open('../bucket/Amara/Osteo/mrn_split_basic1235678.pkl', 'rb'))
    mrn_train = dct['mrn_train']
    mrn_test = dct['mrn_test']
    mrn_val = dct['mrn_val']

    df_train = df.loc[df.mrn.isin(mrn_train)]
    df_test = df.loc[df.mrn.isin(mrn_test)]
    df_val = df.loc[df.mrn.isin(mrn_val)]

    dct_axial = pkl.load(open('Model/Axial/train/train-01/train_predictions.pkl', 'rb'))
    dct_coronal = pkl.load(open('Model/Coronal/train/train-01/train_predictions.pkl', 'rb'))
    dct_baseline = pkl.load(open('Models/EHR/train_predictions.pkl', 'rb'))
    dct_axial['probs'] = np.array(dct_axial['probs'])
    dct_coronal['probs'] = np.array(dct_coronal['probs'])

    mat = np.array([
                    (dct_coronal['probs']), 1-(dct_coronal['probs']),(dct_axial['probs']), 1-(dct_axial['probs']), 
                sigmoid(dct_baseline['probs'][:,1]), 1-sigmoid(dct_baseline['probs'][:,1])]).transpose()
    labels_train = dct_axial['labels']

    #Test
    dct_axial = pkl.load(open('Model/Axial/train/train-01/test_predictions.pkl', 'rb'))
    dct_coronal = pkl.load(open('Model/Coronal/train/train-01/test_predictions.pkl', 'rb'))
    dct_baseline = pkl.load(open('Models/EHR/test_predictions.pkl', 'rb'))dct_axial['probs'] = np.array(dct_axial['probs'])
    dct_coronal['probs'] = np.array(dct_coronal['probs'])
    mat_test = np.array([ 
                    (dct_coronal['probs']), 1-(dct_coronal['probs']), (dct_axial['probs']), 1-(dct_axial['probs']),
                sigmoid(dct_baseline['probs'][:,1]), 1-sigmoid(dct_baseline['probs'][:,1])]).transpose()
    labels_test = dct_axial['labels']

    print(mat.shape, len(labels_train), mat_test.shape, len(labels_test))

    
    op = MinMaxScaler()#StandardScaler()#
    op = op.fit(mat)
    mat_norm = op.transform(mat)
    mat_test_norm = op.transform(mat_test)


    clf = pkl.load(open(os.path.join('Models/Fusion/model.sav'), 'rb'))
    
    preds = clf.predict(mat_test_norm)
    probs = clf.predict_proba(mat_test_norm)

    preds2 = np.array([1 if probs[i,1]>=0.5 else 0 for i in range(len(preds))])

    print(classification_report(labels_test, preds2))
    print('AUC-ROC:\t', roc_auc_score(labels_test, probs[:,1]))
    cm = confusion_matrix(labels_test, preds2, labels = [0, 1])
    print('Confusion Matrix:\n', cm)
    dct = {}
    dct['labels'] = labels_test
    dct['preds'] = preds
    dct['probs'] = probs

    labels_test = np.array(dct['labels'])
    preds = np.array(dct['preds'])
    probs = np.array(dct['probs'])

    pkl.dump(dct, open('Models/Fusion/test_predictions.pkl', 'wb'))