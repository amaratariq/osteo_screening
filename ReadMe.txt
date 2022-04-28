Osteoporosis screening from CT scan

Dependencies:
install https://github.com/fk128/sarcopenia-ai
L3 slice detection model is used for slice detection from CT axial view 

Train/Data
dicoms - DCM files
nifti - nifti files extracted from DCM
slices - selected axial and coronal slices
data.csv - CT accession numbers corresponding to dcm file location and T-score label 

Train/Code
Models - trained models
niftiextraction/Imageextractor.py - code for extracting nifti files extracted from dcm files
niftiextraction/nifit_score_matching.py - code for matching coronal and axial views Train/Data/data.csv

axial_slice.py - code for extracting L3 slice from axial view
coronal_slice.py - code for extracting slice with largets visible hip bones from coronal view view
extract_and_predict_EHR.py - code to extract patient demographic from dcm metadata and compute effective diameter from L3 slice, EMR-based model training 
predict_axial.py - code for L3 slice based model training
predict_coronal.py - code for selected coronal slice based model training
predict_Fusion.py - code for late fusion model training using saved probability estimates from EMR, coronal, and axial based prediction models


Test/Data
dicoms - DCM files
nifti - nifti files extracted from DCM
slices - selected axial and coronal slices
data.csv - CT accession numbers corresponding to dcm file location and T-score label 

Test/Code
Models - trained models
niftiextraction/Imageextractor.py - code for extracting nifti files extracted from dcm files
niftiextraction/nifit_score_matching.py - code for matching coronal and axial views Train/Data/data.csv

axial_slice.py - code for extracting L3 slice from axial view
coronal_slice.py - code for extracting slice with largets visible hip bones from coronal view view
extract_and_predict_EHR.py - code to extract patient demographic from dcm metadata and compute effective diameter from L3 slice, EMR-based model prediction
predict_axial.py - code for L3 slice based prediction
predict_coronal.py - code for selected coronal slice based prediction
predict_Fusion.py - code for late fusion model prediction using saved probability estimates from EMR, coronal, and axial based prediction models


