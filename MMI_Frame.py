#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:06:52 2017

@author: pmacias
"""
from utilities import read_image, resamplig
import SimpleITK
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler,StandardScaler


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
        
    def resample_image_to_modality(self, image_to_resample, interpolator = SimpleITK.sitkNearestNeighbor):
        return resamplig(self.itk_image, image_to_resample, interpolator= interpolator)
        
        
    def get_features(self, labels_mask, features = [Region.MEAN]):
        resample_labels = self.resample_image_to_modality(labels_mask)
        SimpleITK.WriteImage(resample_labels, '/tmp/resampled_labels'+ModalityImage.short_id[self.image_modality]+'.mhd')
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
        stats_filter.Execute( resample_labels,self.itk_image)
        print stats_filter.GetNumberOfLabels()
        d = {}
        for feat in features:
            d[self.short_id[self.image_modality]+'_'+feat]=pd.Series([FEATS[feat](label) for label in stats_filter.GetLabels()], index = stats_filter.GetLabels())
        return pd.DataFrame(d)
    

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
        self.multimodality_feats = pd.concat([ modality_img.get_features(self.label_mask_image,features) for modality_img in self.modalities_imgs] , axis=1)
        
        
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
        self.label_mask_image = read_image(labels_mask_image) if labels_mask_image is not None else None
        
    def get_modality_image_by_id(self, modality_id):
        try:
            return self.modalities_imgs[  [modality.image_modality for modality in self.modalities_imgs].index(modality_id) ]
        except ValueError:
            print ('Incorrect modality_id '+modality_id)
            return  None
        
    def get_cluster_map(self, modality_image_id, region_cluster_dict):
        """
        region_cluster_dict. key must be the region and value the cluster
        TODO: correspondance betwwe labes in label_mask_image and region_cluster_dict
        """
        clusters = region_cluster_dict.values()
        data_type = SimpleITK.sitkUInt8 if np.max(clusters) < 256 else SimpleITK.sitkUInt16
        img_out = SimpleITK.Image(self.label_mask_image.GetSize(), data_type)
        img_out.CopyInformation(self.label_mask_image)
        for k in region_cluster_dict.keys():
            img_out += SimpleITK.Cast((self.label_mask_image == k) * region_cluster_dict[k],data_type)
        return self.get_modality_image_by_id(modality_image_id).resample_image_to_modality(img_out)

from sklearn import mixture
def mixture_map(multimodality_image, mix = 0, clusters = 2, wei_prior = None):
    if not isinstance(multimodality_image, MultiModalityImage):
        print ("Iconrrecto multimodality image onject")
        return None
    GMM = 'GMM'
    BMM = 'BMM'
    mixture_fun = mixture.GaussianMixture if mix == 0 else mixture.BayesianGaussianMixture
    mixture_string = GMM if mix == 0 else BMM
    
    if mixture_string == GMM:
        mixture_fun = mixture_fun(n_components = clusters, covariance_type = 'full')
    else:
        mixture_fun = mixture_fun(n_components = clusters, covariance_type = 'full', weight_concentration_prior = wei_prior, max_iter = 1000)
    #multimodality_image.set_regions_feats()
    clean_feats = multimodality_image.multimodality_feats.dropna()
    #X = StandardScaler().fit_transform(clean_feats)
    mixture_fun.fit(clean_feats)
    y = mixture_fun.predict(clean_feats)
    f = {clean_feats.index.values[i]:y[i]+1 for i in range(len(y))}
    SimpleITK.WriteImage(multimodality_image.get_cluster_map(ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT, f),
                         '/tmp/map_'+'imgs_pt_t1_'+str(len(multimodality_image.modalities_imgs))+'_'+mixture_string+'_'+str(clusters)+'_prior_'+str(wei_prior)+'.mhd')
    return mixture_fun.bic(clean_feats) if mixture_string == GMM else 'covergencia '+str(mixture_fun.converged_)+ ' iter '+ str(mixture_fun.n_iter_)+ ' lower_bound '+str( mixture_fun.lower_bound_)
    
    
min_clus = 2
max_clus = 30       
if __name__ == "__main__":
    im_d = {ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_POSTCONT:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t1_vibe_tra_bh_fatsat_exsp_0034' }
    
    for j,feats_l in enumerate([[Region.MEAN],[Region.MEAN, Region.MEDIAN], [Region.MEAN, Region.MEDIAN, Region.MIN],[Region.MEAN, Region.MEDIAN, Region.MIN, Region.MAX] ]):
        for wei in [0.0001, 0.001, 0.01, 0.1, 0.2,0.4,0.5,0.7,0.9,1.0]:
            mm = MultiModalityImage(im_d, labels_mask_image='/tmp/srm_pet200.mhd')
            mm.set_regions_feats(feats_l)
            print('Using Contrast t1')
            for i in range(min_clus,max_clus):
                print i
                #print i,mixture_map(mm, clusters=i)
                print i,j,mixture_map(mm, clusters=i, mix = 1, wei_prior=wei),'Bayes'
            
            mm.add_modality_images({ModalityImage.THO_MRAC_PET_15_MIN_LIST_AC_IMAGES:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/_Tho_MRAC_PET_15_min_list_AC_Images_0018'})
            mm.set_regions_feats(feats_l)
            print('Using PET')
            for i in range(min_clus,max_clus):
                #print i,mixture_map(mm, clusters=i)
                print i,j,mixture_map(mm, clusters=i, mix = 1, wei_prior=wei),'Bayes'
                
            mm.add_modality_images({ModalityImage.T1_VIVE_TRA_BH_FATSAT_EXPS_PRECONT:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/t1_vibe_tra_bh_fatsat_exsp_0019'})   
            mm.set_regions_feats(feats_l)
            for i in range(min_clus,max_clus):
                print('Using PreContrast T1')
                #print mixture_map(mm, clusters=i) 
                print i,j,mixture_map(mm, clusters=i, mix = 1, wei_prior=wei),'Bayes'
                
            mm.add_modality_images({ModalityImage.THO_T2_SPC_COR_PACE:'/media/pmacias/DATA2/amunoz/NUS_DATA_2016/PLTB706/20131203/110408_515000/Tho_t2_spc_cor_pace_0014'})   
            mm.set_regions_feats(feats_l)
            print('Using T2')
            for i in range(min_clus,max_clus):
                #print i,mixture_map(mm, clusters=i) 
                print i,j,mixture_map(mm, clusters=i, mix = 1, wei_prior=wei),'Bayes'
         

        
    
            
        

