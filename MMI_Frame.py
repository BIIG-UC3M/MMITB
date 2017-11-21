#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:06:52 2017

@author: pmacias
"""

from utilities import read_image, resamplig
import SimpleITK
import pandas as pd

class Region():
    SIZE = 'Physical Size'
    FERET_DIAMETER = 'Feret Diameter'
    PERIMETER = 'Perimeter'
    ELONGATION = 'Elongation'
    SPHERICAL_DIAMETER = 'Spherical Diameter' 
    SPHERICAL_RADIUS = 'Spherical Radius'
    FLATNESS = 'Flatness'
    PIXELS = 'Number of Pixels'
    PIXELS_ON_BORDER = 'Number of pixels on border'
    PERIMETER_ON_BORDER = 'Perimeter on border'
    PERIMETER_ON_BORDER_RATIO = 'Perimeter on border ratio'
    ROUNDNESS = 'Roundness'
    MEAN = 'Mean'
    def __init__(self, mask_itk, label):
        self.mask_itk = mask_itk
        self.label = label
        self.cluster = np.nan #Not assigned or regressed at the begining
        self.Region = Region if Region != None else SimpleITK.LabelIntensityStatisticsImageFilter()
        self.FEATS = AttrDict({Region.SIZE:Region.GetPhysicalSize,
                      Region.MEAN:Region.GetMean})
        
        
    def set_feat(self, feature = SIZE):
        if self.Region.GetProgress() == 0.0:
            pass
        

class ModalityImage():
    T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT = '''Volume Interpolated Breathhold Examination - T1-weighted 3D spoiled turbo gradient echo sequence with fat saturation
    Acquired in a single breath-hold on the exspiration
    High sensitivity for small nodular lesions
    TRA: Transverse
    BH: breath-hold
    EXPS: exspiration
    PRECONT: No Contrast'''
    
    T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT = '''Volume Interpolated Breathhold Examination - T1-weighted 3D spoiled turbo gradient echo sequence with fat saturation
    Acquired in a single breath-hold on the exspiration
    High sensitivity for small nodular lesions
    TRA: Transverse
    BH: breath-hold
    EXPS: exspiration
    POSTCONT: With Contrast\n'''
    
    
    THO_MRAC_PET_15_MIN_LIST_IN_UMAP = "MU-MAP"
    
    THO_MRAC_PET_15_MIN_LIST_AC_IMAGES = "PET"
    
    T2_HASTE_COR_BH_SPAIR_EXSP = '''HASTE: Half-Fourier Acquired Single-shot Turbo spin Echo - T2-weighted echo-planar fast spin echo
    Acquired in a single breath-hold on the exspiration
    High sensitivity for infiltrates
    SPAIR: Spectrally Adiabatic Inversion Recovery - fat suppression method
    COR: Coronal
    BH: breath-hold
    EXPS: exspiration
    Acquired in a single breath-hold on the exspiration\n High sensitivity for infiltrates\n'''
                                 
    T2_HASTEIRM_TRA_MBH_EXSP = '''HASTE: Half-Fourier Acquired Single-shot Turbo spin Echo - T2-weighted echo-planar fast spin echo
    Acquired in a single breath-hold on the exspiration
    High sensitivity for infiltrates
    IRM:Inversion Recovery Magnitude - fat suppression method
    TRA: Transverse
    MBH: multi-breath-hold - concatenates
    EXPS: exspiration
    Acquired in a single breath-hold on the exspiration
    High sensitivity for infiltrates'''
    
    THO_T2_SPC_COR_PACE = '''SPC: Sampling Perfection with application optimised Contrasts using different flip angle evolution - T2-weighted 3D fast spin echo
    Motion-compensated by tracking of respiratory cycle
    Motion correction allows improved depiction of masses with chest wall invasion and mediastinal pathology such as masses, lymph nodes or cysts
    PACE: Prospective Acquisition CorrEction - motion correction from tracked respiratory cycle
    COR:Coronal'''
    
    THO_T2_TRUFI_COR_P2_FREE_BREATHING_IPAT = '''TRUFI: TRUe Fast Imaging with steady-state free precession - T2-weighted
    Free breathing
    Provides functional information on pulmonary motion during respiratory cycle and heart function. High sensitivity for pulmonary embolism and gross cardiac or respiratory dysfunction
    COR: Coronal
    IPAT: Integrated PArallel imaging Techniques (p2 - grading for ipat)\n'''
    
    modalities_list = [T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT, T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT, 
                                       THO_MRAC_PET_15_MIN_LIST_IN_UMAP, THO_MRAC_PET_15_MIN_LIST_AC_IMAGES, 
                                       T2_HASTE_COR_BH_SPAIR_EXSP, T2_HASTEIRM_TRA_MBH_EXSP,
                                       THO_T2_SPC_COR_PACE, THO_T2_TRUFI_COR_P2_FREE_BREATHING_IPAT]
    
    short_id = {T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:'T1_VIBE', T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:'T1_VIVE_C',
                THO_MRAC_PET_15_MIN_LIST_IN_UMAP:'MU_MAP',THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:'PET',
                T2_HASTE_COR_BH_SPAIR_EXSP: 'T2_HASTE_SPAIR', T2_HASTEIRM_TRA_MBH_EXSP: 'T2_HASTE_IRM',
                THO_T2_SPC_COR_PACE:'T2_SPC_PACE',  THO_T2_TRUFI_COR_P2_FREE_BREATHING_IPAT:'T2_TRUFI_IPAT'}
    @staticmethod
    def check_modality(image_modality):
        return image_modality in ModalityImage.modalities_list

    def __init__(self, image_modality, image_path_or_object ):
        """
        image_modality: one of the statics fields defined within the class
        image_path_or_object: An ITK image object or the dicom/image path.
        """
        if self.check_modality(image_modality):
            self.image_modality = image_modality
        else:
            print('Incorrect image Modality '+image_modality+". Specify one from "+self.modalities_list)
            return None
        self.itk_image = read_image(image_path_or_object)
        
    def get_features(self, labels_mask, features = [Region.MEAN]):
        resample_labels = resamplig(self.itk_image, labels_mask, interpolator=SimpleITK.sitkNearestNeighbor)
        SimpleITK.WriteImage(resample_labels, '/tmp/resampled_labels.mhd')
        stats_filter = SimpleITK.LabelIntensityStatisticsImageFilter()
        FEATS = {Region.SIZE:stats_filter.GetPhysicalSize,
                 Region.ELONGATION:stats_filter.GetElongation,
                 Region.SPHERICAL_DIAMETER:stats_filter.GetEquivalentSphericalPerimeter,
                 Region.SPHERICAL_RADIUS:stats_filter.GetEquivalentSphericalRadius,
                 Region.FERET_DIAMETER:stats_filter.GetFeretDiameter,
                 Region.FLATNESS:stats_filter.GetFlatness, 
                 Region.PIXELS:stats_filter.GetNumberOfPixels, 
                 Region.PIXELS_ON_BORDER:stats_filter.GetNumberOfPixelsOnBorder,
                 Region.PERIMETER:stats_filter.GetPerimeter, 
                 Region.PERIMETER_ON_BORDER:stats_filter.GetPerimeterOnBorder,
                 Region.PERIMETER_ON_BORDER_RATIO:stats_filter.GetPerimeterOnBorderRatio,
                 Region.ROUNDNESS:stats_filter.GetRoundness,
                 Region.MEAN:stats_filter.GetMean}
        stats_filter.Execute( resample_labels,self.itk_image)
        print stats_filter.GetNumberOfLabels()
        d = [{'Label':label, self.short_id[self.image_modality]+'_'+feat: FEATS[feat](label)} for feat in features  for label in stats_filter.GetLabels()]
        return pd.DataFrame(d)
    

class MultiModalityImage():
    def __init__(self, modalities_imgs, labels_mask_image = None):
        """
        modalities_imgs: List of ModalityImage
        labels_mask_image: itk image used to set Regions
        """
        self.modalities_imgs = modalities_imgs
        self.label_mask_image = read_image(labels_mask_image)
        #self.regions = 

