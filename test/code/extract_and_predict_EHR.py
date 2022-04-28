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
from sklearn.neural_network import MLPClassifier

import numpy as np
import os
from matplotlib import pyplot as plt

from scipy.special import expit as sigmoid
from PIL import Image

import skimage
from scipy import ndimage
from matplotlib import pyplot as plt

header_data = '../data/'


def extract_EHR_from_dicom_metadata():
    df = pd.read_csv(os.path.join(header_data, 'data.csv'))
    #data file has CT_acc (CT accessiom number), 'TOTAL_SPINE_TSCORE', 'Original DICOM file location-Axial', 'Original DICOM file location-Coronal'
    meta = pd.concat(os.path.join(header_data, 'niftis/metadata.csv'))
    mapp = pd.concat(os.path.join(header_data, 'niftis/mapping.csv'))
    
    meta_mapped = meta.loc[meta.file.isin(mapp['Original DICOM file location'].values)]


    cols = [ 'Manufacturer', 'ManufacturerModelName', 'OtherPatientIDs', 'PatientAge', 
        'PatientBirthDate', 'PatientID', 'PatientSex', 'PatientSize', 'PatientWeight', 
        'ImageOrientationPatient', 'ImagePositionPatient', 'PatientPosition', 
        'PatientOrientation',  'PixelSpacing_Axial',  'PixelSpacing_Coronal']
    axial  = 0
    coronal = 0
    df[cols] = None
    for i,j in df.iterrows():
        temp = meta_mapped.loc[(meta_mapped.AccessionNumber==df.at[i, 'CT_acc']) & (meta_mapped.file==df.at[i, 'Original DICOM file location-Transverse'])]
        if len(temp)>0:
            df.at[i, 'PixelSpacing_Axial'] = temp.at[temp.index[0], 'PixelSpacing']
            axial +=1
        else:
            temp = meta_mapped.loc[(meta_mapped.AccessionNumber==df.at[i, 'CT_acc']) & (meta_mapped.file==df.at[i, 'Original DICOM file location-Coronal'])]
            if len(temp)>0:
                df.at[i, 'PixelSpacing_Coronal'] = temp.at[temp.index[0], 'PixelSpacing']
                coronal+=1
        if len(temp)>0:
            for c in cols:
                df.at[i, c] = temp.at[temp.index[0], c]

    print('axial found:\t', axial, 'coronal found:\t', coronal)
    df.to_csv(os.path.join(header_data, 'data_w_ehr_info.csv'))

    ##CROSS SECTION COMPUTATION
    
    def vol_window(vol, level, window):
        maxval = level + window/2
        minval = level - window/2
        vol[vol<minval] = minval
        vol[vol>maxval] = maxval
        return vol
   
    for ii in range(0, len(df)):
        
        i = df.index[ii]
        CT_acc = df.at[i, 'CT_acc']
        slice_number = int(df.at[i, 'L3_slice'])

        header = os.path.join(header_data, 'slices/axial/')
        image_name = header+str(CT_acc)+'_'+str(slice_number)+'.npy'
        a = np.load(image_name)
        a = a.astype(float)

        a = vol_window(a, 500, 1500)
        a = (a-np.min(a))/(np.max(a)-np.min(a))
        a = 255.0*a
        mask = a.copy()
        mask[mask>0] = 255
        image = Image.fromarray(np.uint8(a)).convert('RGB')
        mask_image = Image.fromarray(np.uint8(mask)).convert('RGB')

        
        labels = skimage.measure.label(mask, return_num=False)

        maxCC_withbcg = labels == np.argmax(np.bincount(labels.flat))
        maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=mask.flat))
        largest = maxCC_nobcg.copy()
        largest[largest==False] = 0
        largest[largest==True] = 1
        
        largest_filled = ndimage.binary_fill_holes(largest).astype(int)
        max_width = max([len([i for i in range(512) if largest_filled[i, y]>0]) for y in range(512)])
        max_depth = max([len([i for i in range(512) if largest_filled[y,i]>0]) for y in range(512)])
        
        #print(max_width, max_depth)
        df.at[i, 'Width'] = max_width
        df.at[i, 'Depth'] = max_depth

        if ii % 100==0:
            print('cross section computation done for ', ii)

    df['CrossSection'] = np.sqrt(df['Width']*df['Depth'])

    def pixelspacing_0(x):
        return float(x.split(',')[0][2:-1])
    def pixelspacing_1(x):
        return float(x.split(',')[1][2:-2])
    df['PixelSpacing_Transverse_0'] = df.PixelSpacing_Transverse.apply(pixelspacing_0)
    df['PixelSpacing_Transverse_1'] = df.PixelSpacing_Transverse.apply(pixelspacing_1)
    a = df['PixelSpacing_Transverse_1'] == df['PixelSpacing_Transverse_0']

    df['WidthSpacing'] = df['Width']*df['PixelSpacing_Transverse_1']
    df['DepthSpacing'] = df['Depth']*df['PixelSpacing_Transverse_0']
    a = df['WidthSpacing']*df['DepthSpacing']
    df['CrossSectionSpacing'] = np.sqrt(a)

    df.to_csv(os.path.join(header_data, 'data_w_ehr_info.csv'))


def EHR_model_test_embed(check_performance=False):
    
    def labeling(x):
        if x<=-1:
            return 1
        else:
            return 0
    def age_mapping(x):
        if type(x) is str and len(x)==4:
            return x[1]
        elif type(x) is str and len(x)==3:
            return x[0]
        else:
            return 'UNKNOWN'

    #136.1302176492294, 462.9853871884079
    df = pd.read_csv(os.path.join(header_data, 'data_w_ehr_info.csv'))
    print(len(df))
    df = df.dropna(subset=['Original DICOM file location-Axial', 'Original DICOM file location-Coronal', 'TOTAL_SPINE_TSCORE', 'L3_slice'], how='any')
    print(len(df))

    if check_performance:
        df['Label'] = df['TOTAL_SPINE_TSCORE'].apply(labeling)
        print(df['Label'].value_counts())

    df['PatientAge_mapped'] = df.PatientAge.apply(age_mapping)
    print(df['PatientAge_mapped'].value_counts())


    df['CrossSectionSpacing_binned'] = pd.cut(df['CrossSectionSpacing'].values, 10)

    a = df['CrossSectionSpacing'].values
    df['CrossSectionSpacing_norm'] = (a-np.min(a))/(np.max(a)-np.min(a))



    df_test = df.copy()

    if check_performance:
        labels_test = df_test.Label.values



    discrete_cols = ['PatientAge_mapped', 'PatientSex', 'CrossSectionSpacing_binned']

    npd_discrete = df_test[discrete_cols].copy().to_numpy()
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(npd_discrete)


    mat_test = enc.transform(npd_discrete).todense()

    mat_test = np.concatenate((mat_test[:, :5], mat_test[:, 6:]), axis=1)


    clf = pkl.load(open(os.path.join('Models/EHR/model.sav'), 'rb'))

    #######################
    preds = clf.predict(mat_test)
    probs = clf.predict_proba(mat_test)

    if check_performance:
        print(classification_report(labels_test, preds))
        print('AUC-ROC:\t', roc_auc_score(labels_test, probs[:,1]))
        cm = confusion_matrix(labels_test, preds, labels = [0, 1])
        print('Confusion Matrix:\n', cm)
    
    dct = {}
    if check_performance:
        dct['labels'] = labels_test
    dct['preds'] = preds
    dct['probs'] = probs
    pkl.dump(dct, open(os.path.join('Models/EHR/test_predictions.pkl'), 'wb'))



