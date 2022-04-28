import pandas as pd
import sys
import os
import numpy as np
import nibabel as nib


## assumes NIFTI volumes have been extracted from DICOM files and mapping.csv and metadata.csv has been stores in the niftis folder alongwith all volumes
def main():
    header_data = '../../data/'

    mapp = pd.read_csv(os.path.join(header_data, 'niftis/mapping.csv'))
    print(len(mapp))
    sys.stdout.flush()
    meta = pd.read_csv(os.path.join(header_data, 'niftis/metadata.csv'))
    print(len(meta))
    sys.stdout.flush()
    df = pd.read_excel(os.path.join(header_data, 'data.csv'))
    print(len(df))
    sys.stdout.flush()

    def to_string(x):
        return str(x)
    meta['AccessionNumber_str'] = meta.AccessionNumber.apply(to_string)

    meta_mapped = meta.loc[meta.file.isin(mapp['Original DICOM file location'].values)]
    print(len(meta_mapped))
    sys.stdout.flush()
    df = df.loc[df.CT_acc.isin(meta_mapped.AccessionNumber_str.values)]
    print(len(df))
    sys.stdout.flush()


    for iii in range(0,len(df)):
        i = df.index[iii]

        ct_acc = df.at[i, 'CT_acc']
        meta_temp = meta_mapped.loc[meta_mapped.AccessionNumber_str==ct_acc]
        mapp_temp = mapp.loc[mapp['Original DICOM file location'].isin(meta_temp['file'].values)]
        print(len(meta_temp), len(mapp_temp))
        sys.stdout.flush()
        tlen = 0
        clen = 0
        slen = 0
        cnloc = ""
        coloc = ""
        snloc = ""
        soloc = ""
        tnloc = ""
        toloc = ""
        for ii,jj in mapp_temp.iterrows():
            loc = mapp_temp.at[ii, ' NIFTI location ']
            img = nib.load(loc)
            a = np.array(img.dataobj)
            desc = mapp_temp.at[ii, 'Original DICOM file location'].split('/')[-1].lower()
            print(desc, a.shape)
            sys.stdout.flush()
            if a.shape[0]==512 and a.shape[1]==512 and a.shape[2]>tlen:
                tlen = a.shape[2]
                tnloc = mapp_temp.at[ii, ' NIFTI location ']
                toloc = mapp_temp.at[ii, 'Original DICOM file location']
            elif 'cor' in desc:
                clen = a.shape[1]
                cnloc = mapp_temp.at[ii, ' NIFTI location ']
                coloc = mapp_temp.at[ii, 'Original DICOM file location']
            
        df.at[i, 'Original DICOM file location-Coronal'] = coloc
        df.at[i, 'NIFTI location-Coronal'] = cnloc
        df.at[i, 'Sequence Length-Coronal'] = clen
        

        
        df.at[i, 'Original DICOM file location-Axial'] = toloc
        df.at[i, 'NIFTI location-Axial'] = tnloc
        df.at[i, 'Sequence Length-Axial'] = tlen
        print(tlen, clen, slen)
        if iii%100==0:
            df.to_csv(os.path.join(header_data, 'data.csv'))
        sys.stdout.flush()
    df.to_csv(os.path.join(header_data, 'data.csv'))



if __name__ == "__main__":
    main()