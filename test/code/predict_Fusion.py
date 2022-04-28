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


    
def test(check_performance):
    df = pd.read_csv(os.path.join(header_data, 'data_w_ehr_info.csv'))
    print(len(df))
    
    if check_performance:
        def labeling(x):
            if x<=-1:
                return 1
            else:
                return 0
        df['Label'] = df['TOTAL_SPINE_TSCORE'].apply(labeling)
        print(df['Label'].value_counts())

    df['PatientAge_mapped'] = df.PatientAge.apply(age_mapping)
    print(df['PatientAge_mapped'].value_counts())


    df_test = df.copy()

    #Test
    dct_axial = pkl.load(open('Model/Axial/test/test-01/test_predictions.pkl', 'rb'))
    dct_coronal = pkl.load(open('Model/Coronal/test/test-01/test_predictions.pkl', 'rb'))
    dct_baseline = pkl.load(open('Models/EHR/test_predictions.pkl', 'rb'))dct_axial['probs'] = np.array(dct_axial['probs'])
    dct_coronal['probs'] = np.array(dct_coronal['probs'])
    mat_test = np.array([ 
                    (dct_coronal['probs']), 1-(dct_coronal['probs']), (dct_axial['probs']), 1-(dct_axial['probs']),
                sigmoid(dct_baseline['probs'][:,1]), 1-sigmoid(dct_baseline['probs'][:,1])]).transpose()
    if check_performance:
        labels_test = dct_axial['labels']

    
    op = MinMaxScaler()
    op = op.fit(mat_test)
    mat_test_norm = op.transform(mat_test)


    clf = pkl.load(open(os.path.join('Models/Fusion/model.sav'), 'rb'))
    
    preds = clf.predict(mat_test_norm)
    probs = clf.predict_proba(mat_test_norm)

    preds2 = np.array([1 if probs[i,1]>=0.5 else 0 for i in range(len(preds))])
    dct = {}
    dct['preds'] = preds
    dct['probs'] = probs

    if check_performance:
        print(classification_report(labels_test, preds2))
        print('AUC-ROC:\t', roc_auc_score(labels_test, probs[:,1]))
        cm = confusion_matrix(labels_test, preds2, labels = [0, 1])
        print('Confusion Matrix:\n', cm)
        dct['labels'] = labels_test


    pkl.dump(dct, open('Models/Fusion/test_predictions.pkl', 'wb'))