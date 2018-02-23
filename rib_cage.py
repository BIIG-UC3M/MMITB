#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:46:33 2018

@author: pmacias
"""
import SimpleITK
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi

import glob

def multi_label_mask(itk_image, labels):
    img_out = SimpleITK.Image(itk_image.GetSize(), SimpleITK.sitkUInt8)
    img_out.CopyInformation(itk_image)
    for l in labels:
        img_out+= itk_image == l
    return img_out

def distance_objects(itk_image, image_center = None):
    image_center = itk_image.TransformIndexToPhysicalPoint( np.array(itk_image.GetSize())/2) if image_center is None else image_center
    shape_stats = SimpleITK.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(itk_image)
    return np.array(shape_stats.GetLabels()),[euclidean(shape_stats.GetCentroid(label), image_center) for label in shape_stats.GetLabels() ]

def get_scapula_labels(bone_image, ref_slice = None, num_s = 2):
    if ref_slice is not None:
        ref_slice = ref_slice
        bone_image = bone_image[:,:,ref_slice]
    labels,distances = distance_objects(bone_image)

    return labels[np.argsort(distances)[-num_s:]].reshape(num_s,1)

def get_CT_bones_rude(itk_image, n_th = 3, bone_limit = [500,1200], size_limit = 0.5, ref_slice = -1):
    otsu = SimpleITK.OtsuMultipleThresholds(itk_image, numberOfThresholds = n_th)
    intensity_stats = SimpleITK.LabelIntensityStatisticsImageFilter()
    intensity_stats.Execute(otsu, itk_image)

    bone_label = np.argmax([intensity_stats.GetMedian(l) for l in intensity_stats.GetLabels() ]) + 1

    bone_mask = otsu == bone_label
    bone_mask = SimpleITK.Median(bone_mask, [3,3,3])
    connected_bones = SimpleITK.ConnectedComponent(bone_mask)
    
    intensity_stats.Execute(connected_bones, itk_image)
    labels = np.array(intensity_stats.GetLabels())
    intensities = np.array([intensity_stats.GetMedian(l) for l in labels ])
    labels = labels [(intensities > bone_limit[0]) * (intensities < bone_limit[1])]
    sizes = np.array([intensity_stats.GetPhysicalSize(l) for l in labels])
    
    labels = labels[ sizes > size_limit  ]
    
    connected_bones = SimpleITK.Mask(connected_bones, multi_label_mask(connected_bones, labels))
    scp_labels = get_scapula_labels(connected_bones, ref_slice = ref_slice)
    print scp_labels, type(scp_labels)
    labels = labels[ np.prod(labels != scp_labels, axis = 0, dtype = np.bool) ]
    no_scp_bones = multi_label_mask(connected_bones, labels)
    return otsu, bone_mask, SimpleITK.Mask(itk_image, bone_mask), connected_bones, SimpleITK.Mask(connected_bones, no_scp_bones), SimpleITK.Mask(itk_image, no_scp_bones), no_scp_bones


class MASK_DOWNSAMPLING():
    CONTOUR = SimpleITK.BinaryContourImageFilter()
    THINNING = SimpleITK.BinaryThinningImageFilter()

def get_rib_cage_convex_hull(rib_cage_mask, downsamplig = MASK_DOWNSAMPLING.THINNING):
    rib_cage_mask = downsamplig.Execute(rib_cage_mask) if downsamplig is not None else rib_cage_mask
    mask_array = SimpleITK.GetArrayFromImage(rib_cage_mask)
    points = np.stack([indx for indx in np.where(mask_array)], axis = 1)
    return points,Voronoi(points) #ConvexHull(points)
    

if __name__ == "__main__":
    image_path = ('/media/pmacias/DATA2/amunoz/LOTE_2-study_5007/Study-5007/049JID/2.3WEEKS_26JUL14/5007_049JID_3WEEKSW_Segmentation/5007_049JID_3WEEKSW_Segmentation_oneFileVolume.mhd')
    
    images = glob.glob('/media/pmacias/DATA2/amunoz/LOTE_2-study_5007/Study-5007/*/*/*/*_oneFileVolume.mhd')
    
    for image in images:
        fields = image.split('/')[-2].split('_')
        s = fields[1]
        w = fields[2].split('W')[0]
        image = SimpleITK.ReadImage(image_path)
        name = s+'_'+w+'.mhd'
        print (name)
        a = get_CT_bones_rude(image, n_th=3)
        #SimpleITK.WriteImage(a[0],'/tmp/otsu.mhd')
        #SimpleITK.WriteImage(a[1],'/tmp/bone_mask.mhd')
        #SimpleITK.WriteImage(a[2],'/tmp/bone.mhd')
        #SimpleITK.WriteImage(a[3],'/tmp/connected_bone.mhd')
        #SimpleITK.WriteImage(a[4],'/tmp/connected_bone_no_scp.mhd')
        #SimpleITK.WriteImage(a[5],'/tmp/bones_no_scp.mhd')
        SimpleITK.WriteImage(a[6],'/tmp/'+name)
