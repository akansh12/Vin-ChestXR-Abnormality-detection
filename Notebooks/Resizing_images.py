

import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
import sys
sys.path.append("../src")
from utils import read_xray
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')



import warnings
warnings.filterwarnings('ignore')



def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    return im


# ## Train



input_dir = "/scratch/scratch6/akansh12/DeepEXrays/physionet.org/files/vindr-cxr/1.0.0/train/"
output_dir = "/scratch/scratch6/akansh12/DeepEXrays/data/data_1024/"





filenames = [i for i in os.listdir(input_dir) if i.endswith('.dicom')]



os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)

for i in tqdm(filenames):
    original = read_xray(os.path.join(input_dir, i))
    im = resize(original, size = 1024)
    im = im.convert('RGB')
    im.save(os.path.join(output_dir, 'train/')+i[:-6]+'.png', 'PNG', quality= 95)

