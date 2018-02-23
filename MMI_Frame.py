#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:06:52 2017

@author: pmacias
"""
from utilities import read_image, resamplig,  multi_label_mask,get_background, srm, proper_uint_type,save_clf
import SimpleITK
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler,StandardScaler
import glob
from abc import ABCMeta, abstractmethod
import inspect
from sklearn import mixture
from sklearn.externals import joblib
import os

import seaborn as sns
import matplotlib.pyplot as plt
import itertools




class Region():
    SIZE = 'Physical_Size'
    FERET_DIAMETER = 'Feret_Diameter'
    PERIMETER = 'Perimeter'
    ELONGATION = 'Elongation'
    SPHERICAL_DIAMETER = 'Spherical_Diameter' 
    SPHERICAL_RADIUS = 'Spherical_Radius'
    FLATNESS = 'Flatness'
    PIXELS = 'Number_of_Pixels'
    PIXELS_ON_BORDER = 'Number_of_pixels_on_border'
    PERIMETER_ON_BORDER = 'Perimeter_on_border'
    PERIMETER_ON_BORDER_RATIO = 'Perimeter_on_border_ratio'
    ROUNDNESS = 'Roundness'
    MEAN = 'Mean'
    MIN = 'Minimum'
    MAX = 'Maximum'
    MEDIAN = 'Median'
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
        self.label_mask = None
        self.label_mask_array = None
        self.resample_image = None
        self.resample_image_array = None
        
    def resample_image_to_modality(self, image_to_resample, interpolator = SimpleITK.sitkNearestNeighbor):
        return resamplig(self.itk_image, image_to_resample, interpolator= interpolator)
        
        
    def get_features(self, labels_mask, features = [Region.MEAN]):
        self.label_mask = self.resample_image_to_modality(labels_mask)

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
                 Region.MEAN:stats_filter.GetMean, 
                 Region.MIN:stats_filter.GetMinimum,
                 Region.MAX:stats_filter.GetMaximum,
                 Region.MEDIAN:stats_filter.GetMedian}
        stats_filter.Execute( self.label_mask,self.itk_image)
        print stats_filter.GetNumberOfLabels()
        d = {}
        for feat in features:
            d[self.short_id[self.image_modality]+'_'+feat]=pd.Series([FEATS[feat](label) for label in stats_filter.GetLabels()], index = stats_filter.GetLabels())
        return pd.DataFrame(d)
    
    def get_voxels_label(self, label_mask, label):
        if self.label_mask is not label_mask:
            #print (self.short_id[self.image_modality],"label_mask is", shape_stats  = SimpleITK.LabelShapeStatisticsImageFilterself.label_mask, 'change by ',label_mask ) 
            self.label_mask = label_mask
            self.label_mask_array = SimpleITK.GetArrayFromImage(self.label_mask)
            self.resample_image = resamplig(self.label_mask, self.itk_image, interpolator = SimpleITK.sitkBSpline)
            self.resample_image_array = SimpleITK.GetArrayFromImage(self.resample_image) 
        #print('Intensities for: ',self.short_id[self.image_modality], 'with mask', self.label_mask, 'and resampled image', self.resample_image)
        intensities = self.resample_image_array[self.label_mask_array == label]
        return pd.DataFrame({self.short_id[self.image_modality]:pd.Series( intensities.tolist() ,index = range(len(intensities)) ) })
        
    
    def __str__(self):
        return self.short_id[self.image_modality]

class ClusterModel():
    __metaclass__ = ABCMeta
   
    @abstractmethod
    def predict(self, features): pass
    
    @abstractmethod
    def fit(self, features): pass

    @abstractmethod
    def get_components(self): pass

class SklearnModel(ClusterModel):
    def __init__(self, skmodel):
        mixtures = [e[1] for e in  inspect.getmembers(mixture,inspect.isclass)]
        
        def some_instance(skmodel):
            for m in mixtures:
                if isinstance(skmodel, m): return True
            return False
    
        if not some_instance(skmodel):
            print ('No Valid skmmodel mixture')
            return None
        self.skmodel = skmodel
    
    def predict(self, features):
        return self.skmodel.predict(features)
    
    def fit(self, features):
        self.skmodel.fit(features)
        
    def get_components(self):
        return self.skmodel.n_components
    
    def save_clf(self, name ):
        joblib.dump(self.skmodel, name, compress=9)
        
    
    def __str__(self):
        return self.skmodel.__str__()

class MultiModalityImage():
    def __init__(self, modalities_imgs, labels_mask_image = None):
        """
        modalities_imgs: List of ModalityImage
        labels_mask_image: itk image used to set Regions
        """
        self.modalities_imgs = []
        self.label_mask_image = None
        self.multimodality_feats = None
        self.cluster_map = {}
        
        
        self.add_modality_images(modalities_imgs)
        self.set_label_mask(labels_mask_image)
        
        
    def set_regions_feats(self, features = [Region.MEAN]):
        pos = self.__get_regions_postions__()
        self.multimodality_feats = pd.concat([ modality_img.get_features(self.label_mask_image,features) for modality_img in self.modalities_imgs] , axis=1)
        self.multimodality_feats['x'],self.multimodality_feats['y'],self.multimodality_feats['z'] = np.array(pos).T
        
    def __get_regions_postions__(self):
        ##Careful!!! the label_mask_image change for computational reasons.
        ##Labels could change
        shape_stats  = SimpleITK.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(self.label_mask_image)
        return [list(shape_stats.GetCentroid(label)) for label in shape_stats.GetLabels()]
        
    def density_label(self, labels, plot = True, max_samples = 2000):
        df = pd.DataFrame()
        df_plot = pd.DataFrame()
        for label in labels:
            print('label: ',label)
            df_voxels = pd.concat([ modality_img.get_voxels_label(self.label_mask_image,label) for modality_img in self.modalities_imgs] , axis=1)
            df_voxels['Label'] = label
            df_voxels = pd.DataFrame() if len(df_voxels) < df_voxels.shape[1] else df_voxels
            df = pd.concat([df, df_voxels])
            step = len(df_voxels)/max_samples + 1
            df_plot = pd.concat([df_plot, df_voxels[0::step]])
        if plot:
            g = sns.PairGrid(df_plot, hue='Label', vars=df_plot.columns[[0,2,3,4,5,6]])
            g.map_upper(plt.scatter, alpha=.5)
            g.map_lower(sns.kdeplot)
            g.map_diag(sns.kdeplot, lw=2);
            g.add_legend(fontsize=10, bbox_to_anchor=(0.9, 0.5, 0, 0))
        return df,df_plot
        
        
    def add_modality_image(self, modality_img):
        if  isinstance(modality_img,ModalityImage):
            self.modalities_imgs += [modality_img]
        elif isinstance(modality_img,dict):
            for k in modality_img.keys():
                self.add_modality_image(ModalityImage(k, modality_img[k]))
        else:
            print ('Incorrect modality image argument. Must be a ModalityImage object or dict({modality:image})')
            
    def add_modality_images(self, modality_images):
        if isinstance(modality_images, list):
            for modality in modality_images:
                self.add_modality_image(modality)
        elif isinstance(modality_images, dict):
            for k in modality_images.keys():
                self.add_modality_image({k:modality_images[k]})
        else:
            print ('Iconrrect list or dictionary of modality images')
            
    def set_label_mask(self, labels_mask_image):
        self.label_mask_image = read_image(labels_mask_image) if labels_mask_image is not None else self.get_labels_mask()
        
    def get_modality_image_by_id(self, modality_id):
        try:
            return self.modalities_imgs[  [modality.image_modality for modality in self.modalities_imgs].index(modality_id) ]
        except ValueError:
            print ('Incorrect modality_id '+modality_id)
            return  None
    
    def __str__(self):
        return 'modalities_'+str(len(self.modalities_imgs))
    
    def get_umap_mask(self, labels_out = [224,1000]):
        umap = self.get_modality_image_by_id(ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP)
        if umap is None:
            raise Exception('umap image is not defined')
        return multi_label_mask(umap.itk_image, labels_out)
    
    
    def get_t1post_mask(self, q = 25):
        t1 = self.get_modality_image_by_id(ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT)
        if t1 is None:
            raise Exception('t1 image is not defined')
        return SimpleITK.InvertIntensity(get_background(t1.itk_image, q=q), maximum=1 )
    
    def get_pet_mask(self, umap_mask = None, t1_mask = None, avoid_limits = True):
        pet = self.get_modality_image_by_id(ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES)
        if pet is None:
            raise Exception('pet image is not defined')
        umap_mask = self.get_umap_mask() if umap_mask is None else umap_mask
        t1_mask = self.get_t1post_mask() if t1_mask is None else t1_mask
        
        t1 = self.get_modality_image_by_id(ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT)
        umap_mask = resamplig(t1.itk_image, umap_mask, interpolator=SimpleITK.sitkNearestNeighbor )
        
        mask_pet = resamplig(pet.itk_image, umap_mask*t1_mask, interpolator = SimpleITK.sitkNearestNeighbor)
        
        if avoid_limits:
            size = mask_pet.GetSize()
            for i in range(size[0]):
                for j in range(size[1]):
                    mask_pet.SetPixel(i,j,0,0)
                    mask_pet.SetPixel(i,j,size[2]-1,0)
        return mask_pet
    
    def get_labels_mask(self,q_pet_srm = 200, q_t1_srm = 50, **kargs):
        umap_mask =  kargs.pop('umap_mask',None)
        umap_mask = self.get_umap_mask() if umap_mask is None else umap_mask
        
        t1_mask = kargs.pop('t1_mask' , None)
        t1_mask = self.get_t1post_mask() if t1_mask is None else t1_mask
        
        t1 = self.get_modality_image_by_id(ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT).itk_image
        pet = self.get_modality_image_by_id(ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES).itk_image
        
        mask_pet = self.get_pet_mask(umap_mask = umap_mask, t1_mask = t1_mask)
        
        t1_masked = SimpleITK.Mask(t1, resamplig(t1, umap_mask, interpolator=SimpleITK.sitkNearestNeighbor )*t1_mask)
        masked_pet = SimpleITK.Mask(pet,mask_pet)
        
        srm_pet_resampled = resamplig(t1, srm(masked_pet, q = q_pet_srm ), interpolator=SimpleITK.sitkNearestNeighbor)

        t1_srm = srm(t1_masked, q=q_t1_srm, smooth=[1,1,1])
        pet_srm = srm_pet_resampled
        t1_srm = SimpleITK.GetArrayFromImage(t1_srm)
        pet_srm = SimpleITK.GetArrayFromImage(pet_srm)
        

        t1_srm = t1_srm.astype(np.uint64 )
        pet_srm = pet_srm.astype(np.uint64)
        B = np.max(pet_srm)       
        res = t1_srm*B+pet_srm #position is unique
        res[t1_srm == 0] = 0
        res[pet_srm == 0] = 0

        clusters = np.unique(res)[1:]
        s_res = np.zeros(res.shape, dtype = proper_uint_type(len(clusters)))
        for i,c in enumerate(clusters):
            ix = np.isin(res,c)
            s_res[ix] = i+1
            
        s_res = SimpleITK.GetImageFromArray(s_res)
        s_res.CopyInformation(t1)
        
        return s_res
        
    
    def get_cluster_map(self, modality_image_id, region_cluster_dict):
        clusters = region_cluster_dict.values()
        data_type = np.uint8 if np.max(clusters) < 256 else np.uint16 #TODO Ajuste dtype generico
        img_label = SimpleITK.GetArrayFromImage(self.label_mask_image)
        img_out = np.zeros(img_label.shape, dtype = data_type)
        
        for c in np.unique(clusters):
            labels = [l for l in region_cluster_dict.keys() if region_cluster_dict[l]==c]
            ix = np.isin(img_label, labels)
            img_out[ix] = c
        img_out = SimpleITK.GetImageFromArray(img_out)
        img_out.CopyInformation(self.label_mask_image)
        return self.get_modality_image_by_id(modality_image_id).resample_image_to_modality(img_out)
        
    
    def mixture_map(self, model = SklearnModel(mixture.BayesianGaussianMixture(n_components = 4, max_iter = 3000)),
                    save_map = (ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT,None), use_feats = None, return_map = False  ):
        if not isinstance(model, ClusterModel):
            print('Incorrect Model. Must be a ClusterMode Instance')
            return None
        clean_feats = self.multimodality_feats.dropna() if use_feats is None else self.multimodality_feats[use_feats].dropna()
        model.fit(clean_feats)
        y = model.predict(clean_feats)
        f = {clean_feats.index.values[i]:y[i]+1 for i in range(len(y))}
        mix_map = self.get_cluster_map(save_map[0], f)
        if save_map[1] is not None: #Otherwise the dir path
            SimpleITK.WriteImage(mix_map,save_map[1])
        
        return mix_map,model if return_map else model
            



IMGS_PATH = '/media/pmacias/DATA2/amunoz/NUS_DATA_2016/'


MMIS = [
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB706/20131121/102152_812000/t1_vibe_tra_bh_fatsat_exsp_0037' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB706/20131121/102152_812000/Tho_MRAC_PET_15_min_list_in_UMAP_0007',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB706/20131121/102152_812000/_Tho_MRAC_PET_15_min_list_AC_Images_0020',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB706/20131121/102152_812000/t1_vibe_tra_bh_fatsat_exsp_0017',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB706/20131121/102152_812000/t2_hasteirm_tra_mbh_exsp_0018',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB706/20131121/102152_812000/t2_haste_cor_bh_spair_exsp_0002'},
        
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB706/20131203/110408_515000/t1_vibe_tra_bh_fatsat_exsp_0034' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB706/20131203/110408_515000/Tho_MRAC_PET_15_min_list_in_UMAP_0007',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB706/20131203/110408_515000/_Tho_MRAC_PET_15_min_list_AC_Images_0018',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB706/20131203/110408_515000/t1_vibe_tra_bh_fatsat_exsp_0019',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB706/20131203/110408_515000/Tho_t2_spc_cor_pace_0014',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB706/20131203/110408_515000/t2_hasteirm_tra_mbh_exsp_0017',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB706/20131203/110408_515000/t2_haste_cor_bh_spair_exsp_0002'},
         
         
         
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB708/20140925/145452_125000/t1_vibe_tra_bh_fatsat_exsp_0036' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB708/20140925/145452_125000/Tho_MRAC_PET_15_min_list_in_UMAP_0009',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB708/20140925/145452_125000/_Tho_MRAC_PET_15_min_list_AC_Images_0016',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB708/20140925/145452_125000/t1_vibe_tra_bh_fatsat_exsp_0021',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB708/20140925/145452_125000/t2_spc_cor_pace_0018',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB708/20140925/145452_125000/t2_hasteirm_tra_mbh_exsp_0020',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB708/20140925/145452_125000/t2_haste_cor_bh_spair_exsp_0004'},
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB708/20141002/150830_593000/t1_vibe_tra_bh_fatsat_exsp_0036' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB708/20141002/150830_593000/Tho_MRAC_PET_15_min_list_in_UMAP_0009',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB708/20141002/150830_593000/_Tho_MRAC_PET_15_min_list_AC_Images_0020',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB708/20141002/150830_593000/t1_vibe_tra_bh_fatsat_exsp_0021',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB708/20141002/150830_593000/Tho_t2_spc_cor_pace_0016',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB708/20141002/150830_593000/t2_hasteirm_tra_mbh_exsp_0018',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB708/20141002/150830_593000/t2_haste_cor_bh_spair_exsp_0004'},
         
         
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB709/20140926/145911_859000/Tho_t1_vibe_tra_bh_fatsat_exsp_0016' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB709/20140926/145911_859000/Tho_MRAC_PET_15_min_list_in_UMAP_0013',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB709/20140926/145911_859000/_Tho_MRAC_PET_15_min_list_AC_Images_0025',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB709/20140926/145911_859000/t1_vibe_tra_bh_fatsat_exsp_0036',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB709/20140926/145911_859000/Tho_t2_spc_cor_pace_0015',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB709/20140926/145911_859000/t2_hasteirm_tra_mbh_exsp_0023',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB709/20140926/145911_859000/t2_haste_cor_bh_spair_exsp_0004'},  
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB709/20141003/112659_187000/t1_vibe_tra_bh_fatsat_exsp_0034' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB709/20141003/112659_187000/Tho_MRAC_PET_15_min_list_in_UMAP_0007',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB709/20141003/112659_187000/_Tho_MRAC_PET_15_min_list_AC_Images_0017',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB709/20141003/112659_187000/t1_vibe_tra_bh_fatsat_exsp_0019',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB709/20141003/112659_187000/Tho_t2_spc_cor_pace_0014',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB709/20141003/112659_187000/t2_hasteirm_tra_mbh_exsp_0018',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB709/20141003/112659_187000/t2_haste_cor_bh_spair_exsp_0002'},   
         
         
         
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB710/20150116/105242_031000/t1_vibe_tra_bh_fatsat_exsp_0137' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB710/20150116/105242_031000/Tho_MRAC_PET_15_min_list_in_UMAP_0110',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB710/20150116/105242_031000/_Tho_MRAC_PET_15_min_list_AC_Images_0119',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB710/20150116/105242_031000/t1_vibe_tra_bh_fatsat_exsp_0122',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB710/20150116/105242_031000/Tho_t2_spc_cor_pace_0118',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB710/20150116/105242_031000/t2_hasteirm_tra_mbh_exsp_0120',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB710/20150116/105242_031000/t2_haste_cor_bh_spair_exsp_0105'},
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB710/20150325/104549_656000/t1_vibe_tra_bh_fatsat_exsp_0034' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB710/20150325/104549_656000/Tho_MRAC_PET_15_min_list_in_UMAP_0007',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB710/20150325/104549_656000/_Tho_MRAC_PET_15_min_list_AC_Images_0014',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB710/20150325/104549_656000/t1_vibe_tra_bh_fatsat_exsp_0019',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB710/20150325/104549_656000/Tho_t2_spc_cor_pace_0016',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB710/20150325/104549_656000/t2_hasteirm_tra_mbh_exsp_0018',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB710/20150325/104549_656000/t2_haste_cor_bh_spair_exsp_0002'},  
         
         
         
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB712/20150304/105103_015000/t1_vibe_tra_bh_fatsat_exsp_0035' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB712/20150304/105103_015000/Tho_MRAC_PET_15_min_list_in_UMAP_0008',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB712/20150304/105103_015000/_Tho_MRAC_PET_15_min_list_AC_Images_0018',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB712/20150304/105103_015000/t1_vibe_tra_bh_fatsat_exsp_0020',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB712/20150304/105103_015000/Tho_t2_spc_cor_pace_0015',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB712/20150304/105103_015000/t2_hasteirm_tra_mbh_exsp_0019',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB712/20150304/105103_015000/t2_haste_cor_bh_spair_exsp_0003'}, 
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB712/20150715/125021_359000/t1_vibe_tra_bh_fatsat_exsp_0040' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB712/20150715/125021_359000/Tho_MRAC_PET_15_min_list_in_UMAP_0009',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB712/20150715/125021_359000/_Tho_MRAC_PET_15_min_list_AC_Images_0017',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB712/20150715/125021_359000/t1_vibe_tra_bh_fatsat_exsp_0022',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB712/20150715/125021_359000/Tho_t2_spc_cor_pace_0018',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB712/20150715/125021_359000/t2_hasteirm_tra_mbh_exsp_0021',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB712/20150715/125021_359000/t2_haste_cor_bh_spair_exsp_0004'},
         
         
         
         
        
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB717/20150511/121743_406000/Tho_t1_vibe_tra_bh_fatsat_exsp_0019' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB717/20150511/121743_406000/Tho_MRAC_PET_30_min_list_in_UMAP_0007',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB717/20150511/121743_406000/_Tho_MRAC_PET_30_min_list_AC_Images_0022',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB717/20150511/121743_406000/Tho_t1_vibe_tra_bh_fatsat_exsp_0017',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB717/20150511/121743_406000/Tho_t2_spc_cor_pace_0009',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB717/20150511/121743_406000/Tho_t2_hasteirm_tra_mbh_exsp_0010',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB717/20150511/121743_406000/t2_haste_cor_bh_spair_exsp_0002'},  
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB717/20150703/105242_671000/t1_vibe_tra_bh_fatsat_exsp_0034' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB717/20150703/105242_671000/Tho_MRAC_PET_15_min_list_in_UMAP_0007',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB717/20150703/105242_671000/_Tho_MRAC_PET_15_min_list_AC_Images_0014',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB717/20150703/105242_671000/t1_vibe_tra_bh_fatsat_exsp_0019',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB717/20150703/105242_671000/Tho_t2_spc_cor_pace_0016',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB717/20150703/105242_671000/t2_hasteirm_tra_mbh_exsp_0018',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB717/20150703/105242_671000/t2_haste_cor_bh_spair_exsp_0002'},   
         
         
         
         
         
        {ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB718/20150618/110422_984000/Tho_MRAC_PET_15_min_list_in_UMAP_0007',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB718/20150618/110422_984000/_Tho_MRAC_PET_15_min_list_AC_Images_0017',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB718/20150618/110422_984000/Tho_t2_spc_cor_pace_0015',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB718/20150618/110422_984000/t2_hasteirm_tra_mbh_exsp_0018',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB718/20150618/110422_984000/t2_haste_cor_bh_spair_exsp_0020'},  
         
        {ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB718/20150831/105257_421000/Tho_MRAC_PET_15_min_list_in_UMAP_0008',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB718/20150831/105257_421000/_Tho_MRAC_PET_15_min_list_AC_Images_0017',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB718/20150831/105257_421000/Tho_t2_spc_cor_pace_0020',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB718/20150831/105257_421000/t2_hasteirm_tra_mbh_exsp_0018',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB718/20150831/105257_421000/t2_haste_cor_bh_spair_exsp_0003'}, 
         
         
         
         
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB721/20150825/125625_531000/t1_vibe_tra_bh_fatsat_exsp_0035' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB721/20150825/125625_531000/Tho_MRAC_PET_15_min_list_in_UMAP_0008',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB721/20150825/125625_531000/_Tho_MRAC_PET_15_min_list_AC_Images_0015',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB721/20150825/125625_531000/t1_vibe_tra_bh_fatsat_exsp_0020',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB721/20150825/125625_531000/Tho_t2_spc_cor_pace_0017',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB721/20150825/125625_531000/t2_hasteirm_tra_mbh_exsp_0019',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB721/20150825/125625_531000/t2_haste_cor_bh_spair_exsp_0003'},      
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB721/20150928/120817_843000/t1_vibe_tra_bh_fatsat_exsp_0015' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB721/20150928/120817_843000/Tho_MRAC_PET_30_min_list_in_UMAP_0007',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB721/20150928/120817_843000/_Tho_MRAC_PET_30_min_list_AC_Images_0019',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB721/20150928/120817_843000/t1_vibe_tra_bh_fatsat_exsp_0015',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB721/20150928/120817_843000/Tho_t2_spc_cor_pace_0012',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB721/20150928/120817_843000/t2_hasteirm_tra_mbh_exsp_0014',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB721/20150928/120817_843000/t2_haste_cor_bh_spair_exsp_0002'},     
         
         
         
         
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB724/20160621/114146_843000/t1_vibe_tra_bh_fatsat_exsp_0040' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB724/20160621/114146_843000/Tho_MRAC_PET_15_min_list_in_UMAP_0007',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB724/20160621/114146_843000/_Tho_MRAC_PET_15_min_list_AC_Images_0018',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB724/20160621/114146_843000/t1_vibe_tra_bh_fatsat_exsp_0022',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB724/20160621/114146_843000/Tho_t2_spc_cor_pace_0016',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB724/20160621/114146_843000/t2_hasteirm_tra_mbh_exsp_0020',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB724/20160621/114146_843000/t2_haste_cor_bh_spair_exsp_0002'},    
         
        {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:IMGS_PATH+'/PLTB724/20161207/113300_421000/t1_vibe_tra_bh_fatsat_exsp_0037' ,
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:IMGS_PATH+'/PLTB724/20161207/113300_421000/Tho_MRAC_PET_15_min_list_in_UMAP_0007',
        ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:IMGS_PATH+'/PLTB724/20161207/113300_421000/_Tho_MRAC_PET_15_min_list_AC_Images_0015',
        ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:IMGS_PATH+'/PLTB724/20161207/113300_421000/t1_vibe_tra_bh_fatsat_exsp_0022',
        ModalityImage.THO_T2_SPC_COR_PACE:IMGS_PATH+'/PLTB724/20161207/113300_421000/Tho_t2_spc_cor_pace_0016',
        ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:IMGS_PATH+'/PLTB724/20161207/113300_421000/t2_hasteirm_tra_mbh_exsp_0021',
        ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:IMGS_PATH+'/PLTB724/20161207/113300_421000/t2_haste_cor_bh_spair_exsp_0002'},
          ]



COMBOS = [['PET_Mean', 'T2_SPC_PACE_Mean', 'T2_HASTE_SPAIR_Mean', 'x'],
          ['PET_Mean','T2_SPC_PACE_Mean','T2_HASTE_SPAIR_Mean','T1_VIBE_Mean','x','z'],
          ['PET_Mean','T2_SPC_PACE_Mean','T2_HASTE_SPAIR_Mean','T1_VIVE_C_Mean','x','y','z'] ]

########GET COMBOS###############
#        res_path = '/media/pmacias/DATA2/amunoz/NUS_R4/'
#        fs = list(mm.multimodality_feats.columns)
#        fs.remove('MU_MAP_Mean',idi)
#        a =  [ list(itertools.combinations(fs,i)) for i in range(1,len(fs)) ]
#        feats_combos =  []
#        for t in a:
#            for tup in t:
#                feats_combos.append(list(tup))
#        
#        combs = len(feats_combos)
        
#######################################3
        
        
if __name__ == "__main__":
    RES_PATH = '/media/pmacias/DATA2/amunoz/NUS_R7/'
    df = []
    #im_d = {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t1_vibe_tra_bh_fatsat_exsp_0034' }
#    im_d = {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t1_vibe_tra_bh_fatsat_exsp_0034' ,
#            ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/Tho_MRAC_PET_15_min_list_in_UMAP_0007',
#            ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/_Tho_MRAC_PET_15_min_list_AC_Images_0018',
#            ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t1_vibe_tra_bh_fatsat_exsp_0019',
#            ModalityImage.THO_T2_SPC_COR_PACE:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/Tho_t2_spc_cor_pace_0014',
#            ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t2_hasteirm_tra_mbh_exsp_0017',
#            ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t2_haste_cor_bh_spair_exsp_0002'
#            }
    for mul_im in MMIS:
        study_fields = mul_im[ModalityImage.THO_MRAC_PET_15_MIN_LIST_IN_UMAP].split('/')
        subject = study_fields[7]
        date = study_fields[8]
        idi = subject+'_'+date
        PATH_IDI = os.path.join(RES_PATH,idi)
        if not os.path.exists(PATH_IDI):
            os.makedirs(PATH_IDI)
        
        try:
            mm = MultiModalityImage(mul_im)
        except:
            print('Cannot creatre MMI for', idi)
            continue
        SimpleITK.WriteImage( mm.label_mask_image,os.path.join(PATH_IDI,idi+'_labels.mhd'))
        print('MM UP',idi)
        mm.set_regions_feats()
        print('Regions UP',idi)

        for c,combo in enumerate(COMBOS):
            print(len(COMBOS) - c, combo, idi)
            name = os.path.join(PATH_IDI,'combo_'+str(c))
            try:               
                model = mm.mixture_map(SklearnModel(mixture.BayesianGaussianMixture(n_components = 7, max_iter=10000, n_init = 4,random_state=69)),
                                   save_map = (ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT,name+'.mhd'), use_feats=combo)
            except:
                print('Improper model for',idi,'at combo', combo)
                continue
            model = model[1]
            print('Model UP')
            model.save_clf(name+'.pkl')
            ps = model.skmodel.weight_concentration_
            df.append({'subject':subject,'date':date, 'feats':c, 'iters':model.skmodel.n_iter_, 'lw':model.skmodel.lower_bound_, 'ps': 1 - np.sum(ps < 0.01) })
        df2 = pd.DataFrame(df)
        df2.to_csv(RES_PATH +'df_e.csv')
    df3 = pd.DataFrame(df)
    df3.to_csv(RES_PATH +'df_f_e.csv')
  
        
#    #mm.density_label([8256,2,12,34,69])
#    #a = mm.density_label([2932,444,274,267,265,3763,2762,272,263,17337], max_samples=100)
#    #habitants = mm.density_label([20915,20917,20918,697,760,768,741,767,20526], max_samples=100)
#    lungs_labels = list(set([3764,2747, 2932,2741,3894,1469,3763,2762,1477]))row_offsets =  [n_row + j for j in range(-n,n+1) ]
#    tissue_labels = list(set([267,274,414,186,993,264,263,754,1361]))
#    bone_labels = list(set([272,1572,1364,946,240,940]))
#    cavities_labels = list(set([4454,6902,13468,3163,8596]))
#    lesions_labels = list(set([4719,1220,8553,8556,578,574,592,741,759,20464,760]))
#    nodules_labels = list(set([]))
#    pulmonaty_veseels = list(set([4285,424,4451,10689,7957,272,293]))
#    l = [lungs_labels,tissue_labels, bone_labels, cavities_labels, lesions_labels, nodules_labels, pulmonaty_veseels]
#    l = [lungs_labels,tissue_labels, bone_labels, cavities_labels, lesions_labels, pulmonaty_veseels]
#    ls = ['lung','tissue', 'bone', 'cavities', 'lesions', 'nodules', 'veseels']
#    ls = ['lung','tissue', 'bone', 'cavities', 'lesions', 'veseels']
#    df = pd.DataFrame()
#    for i,s in enumerate(l):
#        print(ls[i])
#        _,exact_df = mm.density_label(s, max_samples=200, plot=True)
#        exact_df['class'] = ls[i]
#        df = pd.concat([df, exact_df])
##    
      
#    g = sns.PairGrid(df,vars=df.columns[:-2],hue='class')
#    g.map_upper(plt.scatter)
#    g.map_lower(sns.kdeplot, cmap="Blues_d")
#    g.map_diag(sns.kdeplot, lw=2, legend=False);
#    
#    
#    feats_l = [Region.MEAN]
#    mm.set_regions_feats(feats_l)
#    clean_feats = mm.multimodality_feats.dropna()
#    
#    
#    g = sns.PairGrid(clean_feats,vars=clean_feats.columns[:-1],hue='label')
#    g.map_upper(plt.scatter)
#    g.map_lower(sns.kdeplot, cmap="Blues_d")
#    g.map_diag(sns.kdeplot, lw=3, legend=False);
#    
    
    
    
#    imgs = glob.glob('/tmp/map_imgs_pt_and_t1_*')
    
#    df = []
#    for j,feats_l in enumerate([[Region.MEAN],[Region.MEAN, Region.MEDIAN], [Region.MEAN, Region.MEDIAN, Region.MIN],[Region.MEAN, Region.MEDIAN, Region.MIN, Region.MAX] ]):
#        for wei in [0.001,0.5,1.0]:
#            mm = MultiModalityImage(im_d, labels_mask_image='/tmp/mask_test5.mhd')
#            mm.set_regions_feats(feats_l)
#            
#            print('Using Contrast t1')
#            for i in K:
#                n_imgs = len(mm.modalities_imgs)
#                name = '/media/pmacias/DATA2/amunoz/NUS_R2/mixVI_K_'+str(i)+'_wei_'+str(wei)+'_feats_'+str(j)+'_imgs_'+str(n_imgs) 
#                print name
#                model = mm.mixture_map(SklearnModel(mixture.BayesianGaussianMixture(n_components = i, weight_concentration_prior = wei, max_iter=10000,random_state=69)),
#                               save_map = (ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT,name+'.mhd'))
#                model = model[1]
#                model.save_clf(name+'.pkl')
#                df.append({'K':i, 'wei':wei, 'feats':j, 'iers':model.skmodel.n_iter_, 'lw':model.skmodel.lower_bound_, 'imgs':n_imgs})
#                
#            
#            mm.add_modality_images({ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/_Tho_MRAC_PET_15_min_list_AC_Images_0018'})
#            mm.set_regions_feats(feats_l)
#            print('Using PET')
#            for i in K:
#                n_imgs = len(mm.modalities_imgs)
#                name = '/media/pmacias/DATA2/amunoz/NUS_R2/mixVI_K_'+str(i)+'_wei_'+str(wei)+'_feats_'+str(j)+'_imgs_'+str(n_imgs) 
#                print name
#                model = mm.mixture_map(SklearnModel(mixture.BayesianGaussianMixture(n_components = i, weight_concentration_prior = wei, max_iter=10000,random_state=69)),
#                               save_map = (ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT,name+'.mhd'))
#                model = model[1]
#                model.save_clf(name+'.pkl')
#                df.append({'K':i, 'wei':wei, 'feats':j, 'iers':model.skmodel.n_iter_, 'lw':model.skmodel.lower_bound_, 'imgs':n_imgs})
#                
#                
#            mm.add_modality_images({ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t1_vibe_tra_bh_fatsat_exsp_0019'})   
#            mm.set_regions_feats(feats_l)
#            for i in K:
#                n_imgs = len(mm.modalities_imgs)
#                name = '/media/pmacias/DATA2/amunoz/NUS_R2/mixVI_K_'+str(i)+'_wei_'+str(wei)+'_feats_'+str(j)+'_imgs_'+str(n_imgs) 
#                print name
#                model = mm.mixture_map(SklearnModel(mixture.BayesianGaussianMixture(n_components = i, weight_concentration_prior = wei, max_iter=10000,random_state=69)),
#                               save_map = (ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT,name+'.mhd'))
#                model = model[1]
#                model.save_clf(name+'.pkl')
#                df.append({'K':i, 'wei':wei, 'feats':j, 'iers':model.skmodel.n_iter_, 'lw':model.skmodel.lower_bound_, 'imgs':n_imgs})
#                
#                
#                
#                
#            mm.add_modality_images({ModalityImage.THO_T2_SPC_COR_PACE:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/Tho_t2_spc_cor_pace_0014'})   
#            mm.set_regions_feats(feats_l)
#            print('Using T2_SPC')
#            for i in K:
#                n_imgs = len(mm.modalities_imgs)
#                name = '/media/pmacias/DATA2/amunoz/NUS_R2/mixVI_K_'+str(i)+'_wei_'+str(wei)+'_feats_'+str(j)+'_imgs_'+str(n_imgs) 
#                print name
#                model = mm.mixture_map(SklearnModel(mixture.BayesianGaussianMixture(n_components = i, weight_concentration_prior = wei, max_iter=10000,random_state=69)),
#                               save_map = (ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT,name+'.mhd'))
#                model = model[1]
#                model.save_clf(name+'.pkl')
#                df.append({'K':i, 'wei':wei, 'feats':j, 'iers':model.skmodel.n_iter_, 'lw':model.skmodel.lower_bound_, 'imgs':n_imgs})
#                
#                
#                
#                
#            mm.add_modality_images({ModalityImage.T2_HASTEIRM_TRA_MBH_EXSP:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t2_hasteirm_tra_mbh_exsp_0017'})   
#            mm.set_regions_feats(feats_l)
#            print('Using T2_HASTREIM')
#            for i in K:
#                n_imgs = len(mm.modalities_imgs)
#                name = '/media/pmacias/DATA2/amunoz/NUS_R2/mixVI_K_'+str(i)+'_wei_'+str(wei)+'_feats_'+str(j)+'_imgs_'+str(n_imgs) 
#                print name
#                model = mm.mixture_map(SklearnModel(mixture.BayesianGaussianMixture(n_components = i, weight_concentration_prior = wei, max_iter=10000,random_state=69)),
#                               save_map = (ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT,name+'.mhd'))
#                model = model[1]
#                model.save_clf(name+'.pkl')
#                df.append({'K':i, 'wei':wei, 'feats':j, 'iers':model.skmodel.n_iter_, 'lw':model.skmodel.lower_bound_, 'imgs':n_imgs})
#                
#                
#                
#                
#            mm.add_modality_images({ModalityImage.T2_HASTE_COR_BH_SPAIR_EXSP:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t2_haste_cor_bh_spair_exsp_0002'})   
#            mm.set_regions_feats(feats_l)
#            print('Using T2_HASTE')
#            for i in K:
#                n_imgs = len(mm.modalities_imgs)
#                name = '/media/pmacias/DATA2/amunoz/NUS_R2/mixVI_K_'+str(i)+'_wei_'+str(wei)+'_feats_'+str(j)+'_imgs_'+str(n_imgs) 
#                print name
#                model = mm.mixture_map(SklearnModel(mixture.BayesianGaussianMixture(n_components = i, weight_concentration_prior = wei, max_iter=10000,random_state=69)),
#                               save_map = (ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT,name+'.mhd'))
#                model = model[1]
#                model.save_clf(name+'.pkl')
#                df.append({'K':i, 'wei':wei, 'feats':j, 'iers':model.skmodel.n_iter_, 'lw':model.skmodel.lower_bound_, 'imgs':n_imgs})
#    
#    df.to_csv('/media/pmacias/DATA2/amunoz/NUS_R2/df.csv')
                

        
    
            
        

