import pandas as pd
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
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def image_loader(self, image_name):
        trans = transforms.Compose([
            transforms.Scale((256,256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0016, 0.0016, 0.0016],std=[0.0033, 0.0033, 0.0033])
            #transforms.Normalize(mean=[28.5, 28.5, 28.5],std=[13.99, 13.99, 13.99])
        ])


        a = np.load(image_name)
        a = a.astype(float)
        a = vol_window(a, 500, 1500)#1000, 1600)#
        a = (a-np.min(a))/(np.max(a)-np.min(a))
        a = 255.0*a#[:,:512]
        image = Image.fromarray(np.uint8(a)).convert('RGB')
        
        image = trans(image).float()
        image = Variable(image, requires_grad=True)

        return image  
    
    def __getitem__(self, index):
        header_slice = '../data/slices/coronal'
        y = self.df.at[self.df.index[index], 'TOTAL_SPINE_TSCORE']
        if y<=-1:
            y = 1
        else:
            y = 0
        CT_acc = self.df.at[self.df.index[index], 'CT_acc']
        slice_number = int(self.df.at[self.df.index[index], 'coronal_slice'])
        
        image_name = header+str(CT_acc)+'_'+str(slice_number)+'.npy'
        x = self.image_loader(image_name)
        y = torch.tensor([y], dtype=torch.float)
        return x, y
    
    

do_train=False
batch_size = 32
model_name = 'densenet121-finetuned'
save_dir = 'Models/Coronal/'
model_path = 'Models/Coronal/train/train-01/'
metric_name = 'auroc'
maximize_metric=True
header_data = '../data/'

df = pd.read_csv(os.path.join(header_data, 'data_w_ehr_info.csv'))
print('data frame read', len(df))
print(len(df))


sys.stdout.flush()


df_test = df.copy()
print('Test', np.unique([1 if t<=-1 else 0 for t in df_test['TOTAL_SPINE_TSCORE'].values], return_counts=True))


datagen_test = CTDataset(df = df_test.copy()) 

test_loader = DataLoader(dataset=datagen_test,  shuffle=False, batch_size=batch_size)


print('loader created')
sys.stdout.flush()


save_dir = utils.get_save_dir(
    save_dir, training=True if do_train else False
)

logger = utils.get_logger(save_dir, "densenet121-finetuning")
logger.propagate = False

logger.info('do_train: {}, batch_size: {}, predictor: {}, saving to {}, detected coronal slice'.format(do_train, batch_size, predictor, save_dir))

## checkpoint saver
saver = utils.CheckpointSaver(
    save_dir=save_dir,
    metric_name=metric_name,
    maximize_metric=maximize_metric,
    log=logger,
)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(cuda, device)


if cuda:
    model = model.to(device)

best_path = os.path.join(os.path.join(model_path, "best.pth.tar"))
model = utils.load_model_checkpoint(best_path, model) 
model.eval()
model.to(device)
print('model loaded')
sys.stdout.flush()



y_pred = []
y_true = []
y_prob= []
with torch.no_grad():
    st = time.time()
    for j, data in enumerate(test_loader):
        data[0] = data[0].to(device)
        #Sprint(data[0].shape)
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
print(set(y_true), set(y_pred), len(y_true), len(y_pred), len(y_prob))
pkl.dump({'labels': y_true, 'preds': y_pred, 'probs': y_prob}, open(save_dir+'/test_predictions.pkl', 'wb'))
test_results = utils.eval_dict(y=y_true, 
                                    y_pred=y_pred, 
                                    y_prob=y_prob, 
                                    average='macro',
                                    thresh_search=True)

test_results_str = ", ".join(
    "{}: {:.4f}".format(k, v) for k, v in test_results.items()
)
logger.info("TEST - {}".format(test_results_str))
print(classification_report(y_true, y_pred))
y_pred = (y_prob>test_results['best_thresh']).astype(int)
print(classification_report(y_true, y_pred))


