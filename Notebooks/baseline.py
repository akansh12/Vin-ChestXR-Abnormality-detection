
# coding: utf-8

# In[1]:


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


# In[2]:


from collections import OrderedDict


# In[3]:


import matplotlib.pyplot as plt
from PIL import Image


# In[4]:


labels_csv = {'train': "/scratch/scratch6/akansh12/DeepEXrays/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_train.csv",
             'test': "/scratch/scratch6/akansh12/DeepEXrays/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv"
             }

data_dir = {'train': "/scratch/scratch6/akansh12/DeepEXrays/data/data_1024/train/",
           'test': "/scratch/scratch6/akansh12/DeepEXrays/data/data_1024/test/"}


# In[5]:


#Normalization values:
global_labels = ['Pleural effusion', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']




# In[6]:


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
    
    
            


# In[7]:


data_transforms = { 
    "train": transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.RandomHorizontalFlip(p = 0.5), 
        transforms.RandomRotation((-20,20)),
        transforms.ToTensor(),
        transforms.Normalize([123.675,116.28,103.53], [58.395,57.12,57.375])
    ]),
    
    "test": transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([123.675,116.28,103.53], [58.395,57.12,57.375])        
    ])
    
}


# In[8]:


train_data = Vin_big_dataset(image_loc = data_dir['train'],
                          label_loc = labels_csv['train'],
                          transforms = data_transforms['train'],
                          data_type = 'train')

print("Train Data loaded")
test_data = Vin_big_dataset(image_loc = data_dir['test'],
                          label_loc = labels_csv['test'],
                          transforms = data_transforms['test'],
                          data_type = 'test')

print("Test Data loaded")

# In[64]:


trainloader = DataLoader(train_data,batch_size = 16,shuffle = True)
testloader = DataLoader(test_data,batch_size = 16,shuffle = True)


# In[10]:


model = models.densenet121(pretrained=True)


# In[11]:


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


# In[ ]:


#metric
from sklearn.metrics import roc_auc_score
# roc_auc_score(y, clf.decision_function(X), average=None)
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }


# In[97]:


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor = 0.1,patience = 5)
epochs = 40
test_loss_min = np.Inf


# In[98]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# In[106]:


if torch.cuda.is_available():
    model = model.cuda()


# In[109]:


train_loss_hist = []
test_loss_hist = []

train_auc_hist = []
test_auc_hist = []

for i in range(epochs):
    
    train_loss = 0.0
    test_loss = 0.0
    train_auc = 0.0
    test_auc = 0.0 
    
    dummy_ps_train = []
    dummy_ps_test = []
    
    dummy_labels_train = []
    dummy_labels_test = []
    
    model.train()
    
    for images,labels in tqdm(trainloader):
        
        images = images.to(device)
        labels = labels.to(device)
        dummy_labels_train.extend(labels.numpy())
        
        ps = model(images)
        dummy_ps_train.extend(ps.detach().numpy())
        loss = criterion(ps,labels.type(torch.float))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(trainloader)
    train_loss_hist.append(avg_train_loss)
    
    train_auc = roc_auc_score(dummy_labels_train,dummy_ps_train,average=None)
    train_auc_hist.append(train_auc)
   



    model.eval()
    with torch.no_grad():
        
        for images,labels in tqdm(testloader):
            
            images = images.to(device)
            labels = labels.to(device)
            dummy_labels_test.extend(labels.numpy())
            
            ps = model(images)
            loss = criterion(ps,labels.type(torch.float))
            dummy_ps_test.extend(ps.detach().numpy())

            test_loss += loss.item()
              

        avg_test_loss = test_loss / len(testloader)
        test_loss_hist.append(avg_test_loss)

        test_auc = roc_auc_score(dummy_labels_test,dummy_ps_test,average=None)
        test_auc_hist.append(test_auc)

        
        schedular.step(avg_test_loss)
        
        if avg_test_loss <= test_loss_min:
            print('testation loss decreased ({:.6f} --> {:.6f}).   Saving model ...'.format(test_loss_min,avg_test_loss))
            torch.save({
                'epoch' : i,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'test_loss_min' : avg_test_loss
            },'DenseNet_size224.pt')
            
            test_loss_min = avg_test_loss
            

            
            
    print("Epoch : {} Train Loss : {:.6f} Average Train AUC : {:.6f}".format(i+1,avg_train_loss,np.average(train_auc)))
    print("Epoch : {} Test Loss : {:.6f} Average Test AUC : {:.6f}".format(i+1,avg_test_loss,np.average(test_auc)))

