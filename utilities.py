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
import pandas as pd

import glob

from Filters import SRM, Keep_N_Objects
from UsefullFunctions import AttrDict




def read_image(image):
    """
    image could be a itk_image object, a path to a image or a path to a DICOM folder
    """
    if isinstance(image, SimpleITK.Image):
        return image
    return read_dicom(image) if os.path.isdir(image) else SimpleITK.ReadImage(image)
    

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
    #print(type(itk_slic_mask))
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

def srm(itk_image, q = 25, three_dim = True, averages = False, fully_connected = True, smooth = None ):
    temp_image_path_input = os.path.join(tempfile.gettempdir(),str(time.time())+'.mhd')
    if smooth is not None:
        itk_image = SimpleITK.Median(itk_image, smooth)
        
    SimpleITK.WriteImage(itk_image, temp_image_path_input)
    temp_image_path_output = os.path.join(tempfile.gettempdir(),str(time.time())+'.mhd')
    srm = SRM(save_img=(temp_image_path_output, False),q=q)
    srm.execute(temp_image_path_input)
    srm_itk_image = SimpleITK.ReadImage(temp_image_path_output)
    srm_itk_image.CopyInformation(itk_image)
    
    if fully_connected:
        return srm_itk_image
    
    img_out = SimpleITK.Image(srm_itk_image.GetSize(), SimpleITK.sitkUInt32)
    img_out.CopyInformation(srm_itk_image)
    SimpleITK.WriteImage(img_out,"/tmp/imou.mhd")
    fake_labelling = SimpleITK.LabelShapeStatisticsImageFilter()
    fake_labelling.Execute(srm_itk_image)
    max_offset = 0
    for l in fake_labelling.GetLabels():
        partial_img = SimpleITK.Cast(SimpleITK.ConnectedComponent(srm_itk_image == l, fullyConnected=True), SimpleITK.sitkUInt32) 
        stupid_filter = SimpleITK.MinimumMaximumImageFilter()
        stupid_filter.Execute(partial_img)
        maxi = int(stupid_filter.GetMaximum())
        img_out += (partial_img  + max_offset)*partial_img
        if l % 500 == 0:
            SimpleITK.WriteImage(partial_img, "/tmp/partial_"+str(l)+".mhd")
            SimpleITK.WriteImage(img_out, "/tmp/partial_out_"+str(l)+".mhd")
        max_offset += maxi
        print maxi,max_offset, fake_labelling.GetNumberOfLabels()
    
    return img_out
        
    
    


def get_background(itk_image, q = 15):
    temp_image_path_output = os.path.join(tempfile.gettempdir(),str(time.time())+'.mhd')
    prob_bck = srm(itk_image, q=q) 
    connected_bck = SimpleITK.ConnectedComponent(prob_bck == 0, fullyConnected=True)
    SimpleITK.WriteImage(connected_bck,'/tmp/conn.mhd')
    keep_bck = Keep_N_Objects(n_objects=1,save_img=(temp_image_path_output, False))
    keep_bck.execute(connected_bck)
    SimpleITK.WriteImage(keep_bck.output_path_and_image.image,'/tmp/connObje.mhd')
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
    

def multi_label_mask(itk_image, labels):
    img_out = SimpleITK.Image(itk_image.GetSize(), SimpleITK.sitkUInt8)
    img_out.CopyInformation(itk_image)
    for l in labels:
        img_out+= itk_image == l
    return img_out
    

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
                SimpleITK.WriteImage(mask, os.path.join('/tmp/mask'+study+'.mhd'))
                SimpleITK.WriteImage(t1_im, os.path.join('/tmp/t1'+study+'.mhd'))
                SimpleITK.WriteImage(resamplig(t1_im, mask, interpolator=SimpleITK.sitkNearestNeighbor),  os.path.join('/tmp/resam'+study+'.mhd')) 


def simple_covergence(path_images_list):
    d = []
    for p in path_images_list:
        info_list = p.split('/')[-1].split('_')
        n_imgs = int(info_list[5])
        k_prior = int(info_list[7])
        con_prior = float(info_list[-1][:-4])
        k = len(np.unique(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(p) ) ) ) - 1 #Background i not considered a cluster
        d.append({'n_imgs':n_imgs, 'k_prior':k_prior, 'con_prior':con_prior, 'k':k})
    return pd.DataFrame(d)
        
        


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
#    test_label_bck()
#    images = glob.glob('/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB7*/*/'+'/*/*t1*')
    #for im in images:
        #get_background(read_dicom(im))
        
    t1 = read_dicom('/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t1_vibe_tra_bh_fatsat_exsp_0034')
    umap = read_dicom('/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/Tho_MRAC_PET_15_min_list_in_UMAP_0007')
    pet = read_dicom('/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/_Tho_MRAC_PET_15_min_list_AC_Images_0018')
    iverted_mask = multi_label_mask(umap, [224,1000])
    iverted_mask2 = SimpleITK.InvertIntensity(get_background(t1, q=25), maximum=1 )
    SimpleITK.WriteImage(iverted_mask, '/tmp/inverted_mask_umap.mhd')
    SimpleITK.WriteImage(iverted_mask2, '/tmp/inverted_mask_t1.mhd')
    
    iverted_mask = resamplig(t1, iverted_mask, interpolator=SimpleITK.sitkNearestNeighbor )
    no_bck_image = SimpleITK.Mask(t1, iverted_mask*iverted_mask2)
    SimpleITK.WriteImage(no_bck_image, '/tmp/no_bck_img.mhd')
    #t1_eg_discc = srm(no_bck_image, q=200 ,fully_connected=False, smooth=True)
    #SimpleITK.WriteImage(t1_eg_discc, '/tmp/t1_seg_dis.mhd')
    #itk_slic_img = SimpleITK.Median(itkImageToSLIC(no_bck_image, n_seg=50000, compactness = 0.00001), [1,1,1])
    #SimpleITK.WriteImage(itk_slic_img, '/tmp/itkslicMedian.mhd')
    #SimpleITK.WriteImage(SimpleITK.Mask(itk_slic_img, iverted_mask), '/tmp/itkslicMedian_no_bck.mhd')
    
    mask_pet = resamplig(pet, iverted_mask*iverted_mask2, interpolator = SimpleITK.sitkNearestNeighbor)
    for i in range(172):
        for j in range(172):
            mask_pet.SetPixel(i,j,0,0)
            mask_pet.SetPixel(i,j,126,0)
    masked_pet = SimpleITK.Mask(pet,mask_pet )
    
    srm_pet = srm(masked_pet, q = 200)
    SimpleITK.WriteImage(SimpleITK.Mask(srm_pet, mask_pet), '/tmp/srm_pet200.mhd')
    slic_pet = itkImageToSLIC(masked_pet, n_seg = 10000, compactness=0.0000001 )
    SimpleITK.WriteImage(SimpleITK.Mask(slic_pet, mask_pet), '/tmp/slic_pet.mhd')
    srm_smothed_image = srm(no_bck_image, q=50, smooth=[1,1,1])
    SimpleITK.WriteImage(SimpleITK.Mask(itkImageToSLIC( SimpleITK.Median(no_bck_image,[1,1,1]), n_seg=2000 ),iverted_mask*iverted_mask2 ),'/tmp/img_slic.mhd' )
    SimpleITK.WriteImage(srm_smothed_image, '/tmp/srm_smth.mhd')
    srm_pet_resampled = resamplig(t1, srm_pet, interpolator=SimpleITK.sitkNearestNeighbor)
    SimpleITK.WriteImage(srm_pet_resampled, '/tmp/srm_pet_resampled.mhd')
    SimpleITK.WriteImage(srm_smothed_image*srm_pet_resampled, '/tmp/srm_t1_pet.mhd')
    SimpleITK.WriteImage( SimpleITK.Mask( itkImageToSLIC( srm_smothed_image, n_seg=500, compactness=0.001 ),iverted_mask*iverted_mask2 ), '/tmp/slic_srm_umap.mhd')
    #SimpleITK.WriteImage( SimpleITK.Mask( srm(no_bck_image, q=1600, smooth= [1,1,1]),iverted_mask ), '/tmp/slic_srm3.mhd')
    
    
