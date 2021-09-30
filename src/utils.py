import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    #VOI stands for Value Of Interest and LUT stands for Lock Up Table: TO know more: https://help.accusoft.com/ImageGear/v17.2/Windows/DLL/topic468.html
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # Fixing Monochrome, Monochrome 2 is preferable           
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    #Converting uint16 to uint8.     
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data


    