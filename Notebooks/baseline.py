
# coding: utf-8

# In[7]:


import warnings
warnings.filterwarnings('ignore')
import torch
from torchvision import transforms, models, datasets
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F


# In[8]:


from collections import OrderedDict


# In[9]:


import matplotlib.pyplot as plt
from PIL import Image


# In[10]:


labels_csv = {'train': "/scratch/scratch6/akansh12/DeepEXrays/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_train.csv",
             'test': "/scratch/scratch6/akansh12/DeepEXrays/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv"
             }

data_dir = {'train': "/scratch/scratch6/akansh12/DeepEXrays/data/data_256/train/",
           'test': "/scratch/scratch6/akansh12/DeepEXrays/data/data_256/test/"}


# In[11]:


#Normalization values:
global_labels = ['Pleural effusion', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']


# In[12]:


#dataset
class Vin_big_dataset(Dataset):
    def __init__(self, image_loc, label_loc, transforms, data_type = 'train'):
        global_labels = ['Pleural effusion', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
        filenames = os.listdir(image_loc)
        self.full_filenames = [os.path.join(image_loc, i) for i in filenames]
        
        label_df = pd.read_csv(label_loc)
        label_df.set_index("image_id", inplace = True)
        self.labels = [label_df[global_labels].loc[filename[:-4]].values for filename in filenames]
            
        self.transforms = transforms
        self.data_type = data_type
    def __len__(self):
        return len(self.full_filenames)
    
    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transforms(image)
        
        if self.data_type == 'train':
            return image, self.labels[idx][np.random.choice([0,1,2], size = 1)[0]]
        else:
            return image, self.labels[idx]
    
    
            


# ### Get mean and STD

# In[8]:


# train_data = Vin_big_dataset(image_loc = data_dir['train'],
#                           label_loc = labels_csv['train'],
#                           transforms = transforms.ToTensor(),
#                           data_type = 'train')

# def get_mean_std(loader):
#     channels_sum, channels_squared_sum, num_batches = 0,0,0
    
#     for data,_ in tqdm(loader):
#         channels_sum += torch.mean(data, dim = [0,2,3])
#         channels_squared_sum += torch.mean(data**2, dim = [0,2,3])
#         num_batches += 1
        
#     mean = channels_sum/num_batches
#     std = (channels_squared_sum/num_batches - mean**2)**0.5
    
#     return mean, std

# mean, std = get_mean_std(DataLoader(train_data,batch_size = 16,shuffle = True))
# print(mean, std)


# In[13]:


data_transforms = { 
    "train": transforms.Compose([
#         transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.RandomHorizontalFlip(p = 0.5), 
        transforms.RandomRotation((-20,20)),
        transforms.ToTensor(),
        transforms.Normalize([0.5490, 0.5490, 0.5490], [0.2679, 0.2679, 0.2679])
    ]),
    
    "test": transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5490, 0.5490, 0.5490], [0.2679, 0.2679, 0.2679])        
    ])
    
}


# In[14]:


train_data = Vin_big_dataset(image_loc = data_dir['train'],
                          label_loc = labels_csv['train'],
                          transforms = data_transforms['train'],
                          data_type = 'train')

test_data = Vin_big_dataset(image_loc = data_dir['test'],
                          label_loc = labels_csv['test'],
                          transforms = data_transforms['test'],
                          data_type = 'test')


# In[20]:


trainloader = DataLoader(train_data,batch_size = 16,shuffle = True)
testloader = DataLoader(test_data,batch_size = 16,shuffle = False)


# In[16]:


model = models.densenet121(pretrained=False)


# In[17]:


model.load_state_dict(torch.load("/storage/home/akansh12/Vin-ChestXR-Abnormality-detection/model/imageNet.pth"))


# In[18]:


model.classifier = nn.Sequential(OrderedDict([
    ('fcl1', nn.Linear(1024,256)),
    ('dp1', nn.Dropout(0.3)),
    ('r1', nn.ReLU()),
    ('fcl2', nn.Linear(256,32)),
    ('dp2', nn.Dropout(0.3)),
    ('r2', nn.ReLU()),
    ('fcl3', nn.Linear(32,6)),
    ('out', nn.Sigmoid()),
]))


# In[21]:


#metric
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score,recall_score, f1_score
# roc_auc_score(y, clf.decision_function(X), average=None)
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro')*100,
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro')*100,
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro')*100,
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro')*100,
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro')*100,
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro')*100,
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples')*100,
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples')*100,
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples')*100,
            }


# In[22]:


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor = 0.1,patience = 5, verbose= True)
epochs = 40
test_loss_min = np.Inf


# In[23]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[24]:


if torch.cuda.is_available():
    model = model.cuda()


# In[30]:


train_loss = []
test_loss = []

for epoch in range(0,epochs):
    train_loss = 0.0
    test_loss = 0.0
    model_train_result = []
    train_target = []
    model_test_result = []
    test_target = []
    
    
    model.train()
    for images,labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        ps = model(images)
        
        #for metric computing
        model_train_result.extend(ps.detach().cpu().numpy())
        train_target.extend(labels.cpu().numpy())
        
        
        loss = criterion(ps,labels.type(torch.float))
        
        optimizer.zero_grad()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    avg_train_loss = train_loss / len(trainloader)
    train_loss.append(avg_train_loss)
    
    train_result = calculate_metrics(model_train_result, train_target)
    
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images = images.to(device)
            ps = model(images)
            
            model_test_result.extend(ps.cpu().numpy())
            test_target.extend(labels.cpu().numpy())
            
            
            loss = criterion(ps,labels.type(torch.float))
            test_loss += loss.item()
            
        avg_test_loss = test_loss / len(testloader)
        test_loss.append(avg_test_loss)
        
        schedular.step(avg_test_loss)

        test_result = calculate_metrics(model_test_result, test_target)
        
        if avg_test_loss <= test_loss_min:
                    print('testation loss decreased ({:.6f} --> {:.6f}).   Saving model ...'.format(test_loss_min,avg_test_loss))
                    torch.save({
                        'epoch' : i,
                        'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'test_loss_min' : avg_test_loss
                    },'DenseNet_size224.pt')
                    
    
    
    print("epoch:{:2d} iter:{:3d} test: "
                  "micro f1: {:.3f} "
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}".format(epoch,0,
                                              train_result['micro/f1'],
                                              train_result['macro/f1'],
                                              train_result['samples/f1']))
    
    
    print("epoch:{:2d} iter:{:3d} test: "
                  "micro f1: {:.3f} "
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f}".format(epoch,0,
                                              test_result['micro/f1'],
                                              test_result['macro/f1'],
                                              test_result['samples/f1']))    
    

