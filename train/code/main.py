from niftiextraction import ImageExtractorNifti
from nifitextracion import nifit_score_matching
import extract_and_predict_EHR
import coronal_slice
import predict_Axial
import predict_Fusion

import os
header_data = '../data/'

def main():
    ## MRN based split
    df = pd.read_csv(os.path.join(header_data, 'data.csv'))
    mrn = list(df.MRN.values)
    mrn_train = np.random.choice(MRN, round(0.8*len(mrn)), replace=False)
    mrn_test = [i for i in mrn if i not in mrn_train]
    mrn_val = np.random.choice(mrn_train, round(0.1*len(mrn_train)), replace=False)
    MRN_train = [i for i in mrn_train if i not in mrn_val]
    dct = {'mrn_train':mrn_train, 'mrn_test': mrn_test, 'mrn_val': mrn_val}
    pkl.dump(dct, open(os.path.join(header_data, 'mrn_split.pkl'), 'wb'))

    ## Extraction of volumes from CT dicoms
    print('extracting nifti volumes from DCM')
    # check https://github.com/Emory-HITI/Niffler/tree/master/modules/nifti-extraction for exlanation of argument values
    os.system("python3 ImageExtractorNifti --DICOMHome osteo_screening/train/data/dicoms  --OutputDirectory osteo_screening/train/data/nifit --SplitIntoChunks 50 --Depth 3 --PrintImages false --CommonHeadersOnly true --UseProcesses 12 --FlattenedToLevel study --is16Bit true --SendEmail false YourEmail test@test.edu ")

    print('matching coronal and axial views as nifti volumes with T-score values using CT_acc of data.csv file and updating data.csv')
    os.system("python3 nifti_score_matching.py")

    print('detetcting coronal slice with largets hip bone area visible and updating data.csv')
    os.system("python3 coronal_slice.py")

    print('detecting axial slices with L3 vertebra and updating data.csv')
    # assumes sarcopenia_ai has been installed https://github.com/fk128/sarcopenia-ai
    os.system("python3 axial_slice.py")

    print('extracting demographic information (Age, Gender, Cross section) from DICOM meta data')
    extract_and_predict_EHR.extract_EHR_from_dicom_metadata()

    print('running EHR based prediction model model and generating embeddings')
    extract_and_predict_EHR.EHR_model_train_test_embed()    

    print('training axial slice based prediction')
    predict_Axial.trian()

    print('testing and generating axial slice based prediction')
    predict_Axial.test_and_embed()

    print('training coronal slice based prediction')
    predict_Coronal.trian()

    print('testing and generating coronal slice based prediction')
    predict_Coronal.test_and_embed()

    print('training fusion model')
    predict_Fusion.train()

    print('testing fusion model')
    predict_Fusion.test()

if __name__ == "__main__":
    main()