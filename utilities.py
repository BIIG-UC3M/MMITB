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

import glob

from Filters import SRM, Keep_N_Objects
from UsefullFunctions import AttrDict





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
    srm = SRM(save_img=(temp_image_path_output, False),q=q)
    srm.execute(temp_image_path_input)
    srm_itk_image = SimpleITK.ReadImage(temp_image_path_output)
    srm_itk_image.CopyInformation(itk_image)
    return srm_itk_image

def get_background(itk_image, q = 15):
    temp_image_path_output = os.path.join(tempfile.gettempdir(),str(time.time())+'.mhd')
    prob_bck = srm(itk_image, q=q) 
    connected_bck = SimpleITK.ConnectedComponent(prob_bck == 0)
    keep_bck = Keep_N_Objects(n_objects=1,save_img=(temp_image_path_output, False))
    keep_bck.execute(connected_bck)
    return keep_bck.output_path_and_image.image

def get_bck_images(itk_images_list, fixed_itk_image_indx = 0):
    fixed_itk_img = itk_images_list[fixed_itk_image_indx]
    fixed_itk_img = fixed_itk_img if isinstance(fixed_itk_img, SimpleITK.Image) else read_dicom(fixed_itk_img)
    img_out = SimpleITK.Image(fixed_itk_img.GetSize(), SimpleITK.sitkUInt8)+1
    img_out.CopyInformation(fixed_itk_img)
    for i,img in enumerate(itk_images_list):
        img_obj = img if isinstance(img, SimpleITK.Image) else read_dicom(img)
        img_bck = get_background(img_obj)
        img_bck = resamplig(fixed_itk_img, img_bck, interpolator=SimpleITK.sitkNearestNeighbor) if i != fixed_itk_image_indx else img_bck
        print '/tmp/img_bck_'+img.split('/')[-1]+'.mhd'
        img_out += img_bck
        SimpleITK.WriteImage(img_bck , '/tmp/img_bck_'+img.split('/')[-1]+'.mhd')
    SimpleITK.WriteImage(img_out, '/tmp/img_bck_out_pre.mhd')
    SimpleITK.WriteImage(img_out > (len(itk_images_list)+1) / 2, '/tmp/img_bck_out.mhd')
    
    

def test_label_bck():
    folders = glob.glob('/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB7*/*/')
    for i, folder in enumerate(folders):
        images = glob.glob(folder+'/*/*UMAP*')
        print i, images
        for image in images:
            t1s = glob.glob(os.path.split(image)[0]+'/*t1*')
            for t1 in t1s:
                print t1
                t1_im = read_dicom(t1)
                study = image.split('/')[-2]
                image_im = read_dicom(image)
                mask = image_im==0
                SimpleITK.WriteImage(mask, os.path.join('/tmp/'+study+'.mhd'))
                SimpleITK.WriteImage(t1_im, os.path.join('/tmp/t1'+study+'.mhd'))
                SimpleITK.WriteImage(resamplig(t1_im, mask, interpolator=SimpleITK.sitkNearestNeighbor),  os.path.join('/tmp/resam'+study+'.mhd')) 


class Region():
    SIZE = 'Physical Size'
    MEAN = 'Mean'
    def __init__(self, mask_itk, label, stats_filter = None):
        self.mask_itk = mask_itk
        self.label = label
        self.cluster = np.nan #Not assigned or regressed at the begining
        self.stats_filter = stats_filter if stats_filter != None else SimpleITK.LabelIntensityStatisticsImageFilter()
        self.FEATS = AttrDict({REGION.SIZE:stats_filter.GetPhysicalSize,
                      REGION.MEAN:stats_filter.GetMean})
        
        
    def set_feat(self, feature = SIZE):
        if self.stats_filter.GetProgress() == 0.0:
            sel
        

if __name__ == "__main__":
#    t1_fat_sup_dcm = '/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131121/102152_812000/t1_vibe_tra_bh_fatsat_exsp_0037'
#    t1_fat_sup = read_dicom(t1_fat_sup_dcm)
#    t1_fat_sup_slic = itkImageToSLIC(t1_fat_sup)
#    SimpleITK.WriteImage(t1_fat_sup_slic, '/tmp/slic_res.mhd')
#    t1_seg = srm(t1_fat_sup, q=2)
#    pt_dcm = '/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131121/102152_812000/_Tho_MRAC_PET_15_min_list_AC_Images_0020/'
#    pt = read_dicom(pt_dcm)
#    SimpleITK.WriteImage(resamplig(pt,t1_seg, interpolator=SimpleITK.sitkNearestNeighbor), '/tmp/labels_to_pet_NN.mhd')
#    pt = SimpleITK.Cast(pt, SimpleITK.sitkFloat32)
#    SimpleITK.WriteImage(pt,'/tmp/pt.mhd')
    #test_label_bck()
    images = glob.glob('/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB7*/*/'+'/*/*t1*')
    #for im in images:
        #get_background(read_dicom(im))
    
    
