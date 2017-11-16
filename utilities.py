# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:29:23 2017

@author: pmacias
"""
import numpy as np
from collections import defaultdict
from skimage.segmentation import slic
import SimpleITK 
import tempfile 
import time
import os

from Filters import SRM



def itkImageToSLIC(itk_image, n_seg = 1000, compactness = 0.0001, return_itk_image = True):
    """
    Save space
    """
    def type_conversion(image):
        d = defaultdict(lambda: np.uint64, {0:np.uint8, 1:np.uint16, 2:np.uint32, 3:np.uint32})
        maxLabel = len(np.unique(image))
        return image.astype( d[(maxLabel.bit_length() - 1)/8])
        
    if not isinstance(itk_image, SimpleITK.Image):
        print("SLIC Segmentation from a itk_image needs a itk_image!!")
        return None
    slic_mask = type_conversion(slic(SimpleITK.GetArrayFromImage(itk_image), n_segments=n_seg, compactness=compactness,
                     spacing=itk_image.GetSpacing(),multichannel=False, enforce_connectivity=True))
    if not return_itk_image:
        return slic_mask
    itk_slic_mask = SimpleITK.GetImageFromArray(slic_mask)
    itk_slic_mask.CopyInformation(itk_image)
    print(type(itk_slic_mask))
    return itk_slic_mask

def read_dicom(path):
    reader = SimpleITK.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(path))
    return reader.Execute()

def resamplig(fix_image, to_resample_image, interpolator = SimpleITK.sitkBSpline):
    resample = SimpleITK.ResampleImageFilter()
    resample.SetReferenceImage(fix_image)
    resample.SetInterpolator(interpolator)
    return resample.Execute(to_resample_image)

def srm(itk_image, q = 25, three_dim = True, averages = False ):
    temp_image_path_input = os.path.join(tempfile.gettempdir(),str(time.time())+'.mhd')
    SimpleITK.WriteImage(itk_image, temp_image_path_input)
    temp_image_path_output = os.path.join(tempfile.gettempdir(),str(time.time())+'.mhd')
    srm = SRM(save_img=(temp_image_path_output, False),q=25)
    srm.execute(temp_image_path_input)
    srm_itk_image = SimpleITK.ReadImage(temp_image_path_output)
    srm_itk_image.CopyInformation(itk_image)
    return srm_itk_image
    
        

if __name__ == "__main__":
    t1_fat_sup_dcm = '/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131121/102152_812000/t1_vibe_tra_bh_fatsat_exsp_0037'
    t1_fat_sup = read_dicom(t1_fat_sup_dcm)
#    t1_fat_sup_slic = itkImageToSLIC(t1_fat_sup)
#    SimpleITK.WriteImage(t1_fat_sup_slic, '/tmp/slic_res.mhd')
    t1_seg = srm(t1_fat_sup)
    pt_dcm = '/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131121/102152_812000/_Tho_MRAC_PET_15_min_list_AC_Images_0020/'
    pt = read_dicom(pt_dcm)
    SimpleITK.WriteImage(resamplig(pt,t1_seg), 'labels_to_pet.mhd')
    
    
