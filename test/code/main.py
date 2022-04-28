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

    ## Extraction of volumes from CT dicoms
    print('extracting nifti volumes from DCM')
    # check https://github.com/Emory-HITI/Niffler/tree/master/modules/nifti-extraction for exlanation of argument values
    os.system("python3 ImageExtractorNifti --DICOMHome osteo_screening/test/data/dicoms  --OutputDirectory osteo_screening/test/data/nifit --SplitIntoChunks 50 --Depth 3 --PrintImages false --CommonHeadersOnly true --UseProcesses 12 --FlattenedToLevel study --is16Bit true --SendEmail false YourEmail test@test.edu ")

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
    # set check_performance to True if data.csv contains labels in 'TOTAL_SPINE_TSCORE' column
    extract_and_predict_EHR.EHR_model_test_embed(check_performance=False)    

    print('testing and generating axial slice based prediction')
    # set check_performance to True if data.csv contains labels in 'TOTAL_SPINE_TSCORE' column
    predict_Axial.test_and_embed(check_performance=False)

    
    print('testing and generating coronal slice based prediction')
    # set check_performance to True if data.csv contains labels in 'TOTAL_SPINE_TSCORE' column
    predict_Coronal.test_and_embed(check_performance=False)

    print('testing fusion model')
    # set check_performance to True if data.csv contains labels in 'TOTAL_SPINE_TSCORE' column
    predict_Fusion.test(check_performance=False)

if __name__ == "__main__":
    main()