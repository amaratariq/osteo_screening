import pandas as pd
import pickle as pkl
import numpy as np
from tqdm import tqdm
import os
import sys
from os import path
import pickle as pkl
import matplotlib.pyplot as plt
import datetime
import utils
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, pairwise_distances, classification_report
from sklearn.decomposition import PCA

import nibabel as nib
# PyTorch libraries and modules
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import make_grid
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py

import sys
import time
print(torch.__version__)
print(torchvision.__version__)
import ssl


 

def test_and_embed(check_perforamnce=False):   

    do_train=False
    n_iters = 25
    print_every = 1
    plot_every = 1
    batch_size = 32
    model_name = 'densenet121-finetuned'
    save_dir = 'Models/Axial/'
    best_model_path = save_dir+"train/train-01/best.pth.tar"
    metric_name = 'auroc'
    predictor = 'l3_slice'
    lr = 1e-4
    maximize_metric=True
    patience = 10
    early_stop=False
    prev_val_loss = 1e10
    iter = 0

    header_data = '../data/'


    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1024, 1))#, nn.Sigmoid())

    print('modal loaded')
    sys.stdout.flush()

    def vol_window(vol, level, window):
        maxval = level + window/2
        minval = level - window/2
        vol[vol<minval] = minval
        vol[vol>maxval] = maxval
        return vol

    class CTDataset(Dataset):
        def __init__(self, df, check_performance):
            self.df = df
            self.check_performance = check_performance
            
        def __len__(self):
            return len(self.df)
        
        def image_loader(self, image_name):
            trans = transforms.Compose([
                                                                        transforms.Scale(256),
                                                                        transforms.CenterCrop(256),
                                                                        transforms.ToTensor(),
                                                                        #transforms.Normalize(mean=[0.056, 0.056, 0.056],std=[0.053, 0.053, 0.053]), 
                                                                        transforms.Normalize(mean=[14.3, 14.3, 14.3],std=[13.7, 13.7, 13.7])
                                                                        ])


            a = np.load(image_name)
            a = a.astype(float)
            a = vol_window(a, 500, 1500)
            a = (a-np.min(a))/(np.max(a)-np.min(a))
            a = 255.0*a
            image = Image.fromarray(np.uint8(a)).convert('RGB')
            
            image = trans(image).float()
            image = Variable(image, requires_grad=True)
            #image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
            return image  
        
        def __getitem__(self, index):
            header_slices = '../data/slices/'
            if self.check_performance:
                y = self.df.at[self.df.index[index], 'TOTAL_SPINE_TSCORE']
                if y<=-1:
                    y = 1
                else:
                    y = 0
            else:
                y = -1 #dummy value
            CT_acc = self.df.at[self.df.index[index], 'CT_acc']
            slice_number = int(self.df.at[self.df.index[index], 'L3_slice'])
            prob = self.df.at[self.df.index[index], 'L3_slice_prob']
            
            image_name = header_slices+str(CT_acc)+'_'+str(slice_number)+'.npy'
            
            x = self.image_loader(image_name)
            y = torch.tensor([y], dtype=torch.float)
            return x, y 
        
   
    df = pd.read_csv(os.path.join(header_data, 'data.csv'))

    print('data frame read', len(df))
    sys.stdout.flush()


    df_test = df.copy()


    datagen_test = CTDataset(df = df_test.copy()) 

    test_loader = DataLoader(dataset=datagen_test,  shuffle=False, batch_size=batch_size)


    y_pred = []
    y_true = []
    y_prob= []
    with torch.no_grad():
        st = time.time()
        for j, data in enumerate(test_loader):
            data[0] = data[0].to(device)
            print(data[0].shape)
            output = model(data[0].view(-1,3, 256, 256)).reshape(-1)

            probs = torch.sigmoid(output).cpu().detach().numpy()  # (batch_size, )
            preds = (probs >=0.5).astype(int)
            y_pred += list(preds)
            y_prob +=list(probs)
            y_true += list(data[1].numpy().reshape(-1))
            ed = time.time()
            print(ed-st)
            st = ed
            sys.stdout.flush()
    if check_performance:
        pkl.dump({'labels': y_true, 'preds': y_pred, 'probs': y_prob}, open(save_dir+'/test_predictions.pkl', 'wb'))
    
        test_results = utils.eval_dict(y=y_true, 
                                            y_pred=y_pred, 
                                            y_prob=y_prob, 
                                            average='macro',
                                            best_thresh = 0.5)

        test_results_str = ", ".join(
            "{}: {:.4f}".format(k, v) for k, v in test_results.items()
        )
        logger.info("TEST - {}".format(test_results_str))
        print(classification_report(y_true, y_pred))
        y_pred = (y_prob>val_results['best_thresh']).astype(int)
        print(classification_report(y_true, y_pred))

    else:
        pkl.dump({'preds': y_pred, 'probs': y_prob}, open(save_dir+'/test_predictions.pkl', 'wb'))