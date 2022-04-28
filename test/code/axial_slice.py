import SimpleITK as sitk
import cv2
import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
from scipy import ndimage
from PIL import Image
import skimage
import keras
from tensorflow.keras.utils import Sequence

from sarcopenia_ai.core.input_parser import InputParser
from sarcopenia_ai.core.model_wrapper import BaseModelWrapper, allocate_tf_gpu_devices
from sarcopenia_ai.io import load_image
from sarcopenia_ai.preprocessing.preprocessing import to256
from sarcopenia_ai.apps.slice_detection.utils import place_line_on_img, decode_slice_detection_prediction, preprocess_sitk_image_for_slice_detection, adjust_detected_position_spacing


print('all loaded')
sys.stdout.flush()

def main():
    header_data = '../data/'

    df = pd.read_csv(os.path.join(header_data, 'data.csv'))
    print('data frame read', len(df))
    sys.stdout.flush()

    def load_model(model_path):
        # Setup model
        model_wrapper = BaseModelWrapper(model_path,
                                        name=None,
                                        config=None,
                                        data_loader=None,
                                        is_multi_gpu=False
                                        )
        model_wrapper.setup_model()

        print(model_wrapper.model.summary())

        return model_wrapper.model

    # Setup model
    model_wrapper = BaseModelWrapper('./sarcopenia_ai/models/slice_detection/')
    model_wrapper.setup_model()
    print('setup')
    sys.stdout.flush()


    for i,j in df.iterrows():
        image_path = df.at[i, 'NIFTI location-Axial']

        try:
            sitk_image, image_name = load_image(image_path)
            image2d = preprocess_sitk_image_for_slice_detection(sitk_image)

            spacing = sitk_image.GetSpacing()
            preds = model_wrapper.model.predict(image2d[0])
            pred_z, prob = decode_slice_detection_prediction(preds)
            slice_z = adjust_detected_position_spacing(pred_z, spacing)



            slice_name = header_data+'/slices/axial/'+str(df.at[i, 'CT_acc'])+'_'+str(slice_z)+'.npy'
            img = nib.load(image_path)#./data/volume.nii.gz')
            a = np.array(img.dataobj)
            np.save(open(slice_name, 'wb'), a[:,:,slice_z])
            df.at[i, 'L3_slice'] =slice_z
            df.at[i, 'L3_slice_prob'] = prob
            print('{}: Slice detected at position {} with confidence {} saved '.format(i, slice_z, prob))
        except:
            print(i, 'Exception')
            img = nib.load(image_path)
            a = np.array(img.dataobj)
            slice_z = round(0.5*a.shape[2])
            slice_name = header_data+'/slices/axial/'+str(df.at[i, 'CT_acc'])+'_'+str(slice_z)+'.npy'
            prob = -1
            np.save(open(slice_name, 'wb'), a[:,:,slice_z])
            df.at[i, 'L3_slice'] =slice_z
            df.at[i, 'L3_slice_prob'] = prob
            print('{}: Slice detected at position {} with confidence {} saved '.format(i, slice_z, prob))
        if i%100 == 0:
            df.to_csv(os.path.join(header_data, 'data.csv'))
            sys.stdout.flush()

            
    df.to_csv(os.path.join(header_data, 'data.csv'))
    print('done')

if __name__ == "__main__":
    main()