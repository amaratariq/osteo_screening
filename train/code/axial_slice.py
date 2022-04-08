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

print('all loaded')
sys.stdout.flush()

header_data = '../data/'

df = pd.read_csv(os.path.join(header_data, 'data.csv'))


print('data frame read', len(df))


def check_largest_segment_slice(img):
    idx = [i for i in range(int(img.shape[1]*0.3), int(img.shape[1]*0.7))]
    sizes = []
    for i in idx:
        a = img[:,i, :].copy()
        mask = a.copy()
        mask[mask>=250] = 255
        mask[mask!=255] = 0
        labels = skimage.measure.label(mask, return_num=False)
        cc = np.argsort(np.bincount(labels.flat, weights=mask.flat))

        cc = [ccc for ccc in cc if ccc!=0]
        #print(cc)
        if len(cc)>=2:
            maxCC_nobcg = labels.copy()
            maxCC_nobcg[maxCC_nobcg==cc[-1]] = 1
            maxCC_nobcg[maxCC_nobcg==cc[-2]] = 1
            maxCC_nobcg[maxCC_nobcg!= True] = 0

            largest = maxCC_nobcg.copy()
            largest_filled = ndimage.binary_fill_holes(largest).astype(int)
            sizes.append(np.bincount(largest_filled.flat)[1])
        else:
            sizes.append(0)
    return idx[np.argmax(sizes)]


for i,j in df.iterrows():
  
        image_path = df.at[i, 'NIFTI location-Coronal']

        img = nib.load(image_path)
        #print('read')
        img = np.array(img.dataobj)
        #print('numpy')
        slice_z = check_largest_segment_slice(img) #round(0.4*a.shape[1])
        print('{}: Slice detected at position {} '.format(i, slice_z))
        slice_name = header_data+'/slices/coronal/'+str(df.at[i, 'CT_acc'])+'_'+str(slice_z)+'.npy'
  
        np.save(open(slice_name, 'wb'), img[:,slice_z,:])
        #print('saved')
        df.at[i, 'coronal_slice'] =slice_z
    except:
        print(i, 'EXCEPTION')
    if i%100 == 0:
        df.to_csv(os.path.join(header_data, 'data.csv'))
        sys.stdout.flush()

        
df.to_csv(os.path.join(header_data, 'data.csv'))
print('done')
