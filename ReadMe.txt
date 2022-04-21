Osteoporosis screening from CT scan

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
ehr_into_extraction.ipynb - code to extract patient demographic from dcm metadata and compute effective diameter from L3 slice, EMR-based model training 
predict_axial.py - code for L3 slice based model training
predict_coronal.py - code for selected coronal slice based model training
fusion_modeling.ipynb - code for late fusion model training using saved probability estimates from EMR, coronal, and axial based prediction models


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
ehr_into_extraction_ehr_model_evaluation.ipynb - code to extract patient demographic from dcm metadata and compute effective diameter from L3 slice, EMR-based model prediction
predict_axial.py - code for L3 slice based prediction
predict_coronal.py - code for selected coronal slice based prediction
fusion_modeling_evluation.ipynb - code for late fusion model prediction using saved probability estimates from EMR, coronal, and axial based prediction models


