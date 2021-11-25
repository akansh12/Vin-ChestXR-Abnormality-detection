# from torchvision import datasets
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from PIL import Image


class Vin_big_dataset(Dataset):
    def __init__(self, image_loc, label_loc, transforms, data_type, selec_radio, radio_id = None, label_type = None):
        if label_type == 'global':
            global_labels = ['Pleural effusion', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
        if label_type == 'local':
            global_labels = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Clavicle fracture', 
                             'Consolidation', 'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration',
                             'Lung Opacity', 'Lung cavity', 'Lung cyst', 'Mediastinal shift',
                             'Nodule/Mass', 'Pleural effusion', 'Pneumothorax',
                             'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'No finding']
        
        if data_type == 'train':
            label_df = pd.read_csv(label_loc)
            if selec_radio == 'rand_one':
                label_df['labels'] = label_df['image_id']
                label_df.set_index("labels", inplace = True)
                filenames = np.unique(label_df.index.values).tolist()
                self.full_filenames = [os.path.join(image_loc, i +'.png') for i in filenames]
                self.labels = []
                for i in (filenames):
                    self.labels.append(label_df[global_labels].loc[i].values.tolist()[np.random.choice([0,1,2])])
                self.labels = torch.tensor(self.labels)
            if selec_radio == 'agree_two':
                label_df['labels'] = label_df['image_id']
                label_df.set_index("labels", inplace = True)
                filenames_temp = np.unique(label_df.index.values).tolist()
                self.labels = []
                filenames = []
                for i in filenames_temp:
                    a,b = np.unique(label_df.loc[i][global_labels].values, axis = 0, return_counts=True)
                    if b[0] >= 2:
                        filenames.append(i)
                        self.labels.append(a[0])
                self.labels = torch.tensor(self.labels)
                self.full_filenames = [os.path.join(image_loc, i +'.png') for i in filenames]
            if selec_radio == 'agree_three':
                label_df['labels'] = label_df['image_id']
                label_df.set_index("labels", inplace = True)
                filenames_temp = np.unique(label_df.index.values).tolist()
                self.labels = []
                filenames = []
                for i in filenames_temp:
                    a,b = np.unique(label_df.loc[i][global_labels].values, axis = 0, return_counts=True)
                    if b[0] == 3:
                        filenames.append(i)
                        self.labels.append(a[0])
                self.labels = torch.tensor(self.labels)
                self.full_filenames = [os.path.join(image_loc, i +'.png') for i in filenames]
            if selec_radio == 'radio_per_epoch':
                label_df['labels'] = label_df['image_id']
                label_df.set_index("labels", inplace = True)
                filenames = np.unique(label_df.index.values).tolist()
                self.labels = []
                for i in filenames:
                    self.labels.append(label_df.loc[i][global_labels].values[radio_id].tolist())
                self.labels = torch.tensor(self.labels)
                self.full_filenames = [os.path.join(image_loc, i +'.png') for i in filenames]
            if selec_radio == 'all': 
                label_df['labels'] = label_df['image_id'] +'_'+ label_df['rad_id']
                label_df.set_index("labels", inplace = True)
                filenames = label_df.index.values.tolist()
            
                self.full_filenames = [os.path.join(image_loc, i.split('_')[0]+'.png') for i in filenames]
                self.labels = []
                for i in tqdm(filenames):
                    self.labels.append(label_df[global_labels].loc[i].values.tolist())         
                self.labels = torch.tensor(self.labels)
                
        if data_type == 'test':                     
            filenames = os.listdir(image_loc)
            self.full_filenames = [os.path.join(image_loc, i) for i in filenames]
            label_df = pd.read_csv(label_loc)
            label_df.set_index("image_id", inplace = True)
            self.labels = [label_df[global_labels].loc[filename[:-4]].values for filename in filenames]
            
        self.transforms = transforms
#         self.data_type = data_type
    def __len__(self):
        return len(self.full_filenames)
    
    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transforms(image)
        
        return image, self.labels[idx]