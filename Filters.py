#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:06:50 2017

@author: pmacias
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 09:24:51 2017

@author: biig
"""
import numpy as np
import SimpleITK 

from Segmentation import A_Filter, FILTER_TYPE, UTILITIES_DIR, External_Filter, Simple_ITK_Filter,  Fiji_Filter


class Apply_Median(External_Filter):
    '''
    Median filter
    Inputs:
        Name input volume
    Output:
        Name of the output volume
    '''
    def __init__(self, location_path = UTILITIES_DIR,save_img =(None,True) ,radius=1):
        A_Filter.__init__(self, "apply_median", FILTER_TYPE.PRE_FILTER, location_path, built_in=False, save_img= save_img)
        self.radius = radius
    
    def set_params(self):
        self.params = [self.radius]
     

class Otsu_Threshold(External_Filter):
    '''
    Otsu Threshold    
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
    '''  
    def __init__(self, location_path = UTILITIES_DIR,save_img =(None,True)):
        A_Filter.__init__(self, 'hu_threshold', FILTER_TYPE.PRE_FILTER, location_path, built_in=False, save_img = save_img)
    
    def set_params(self):
        self.params = []
        

class Labelize(External_Filter):
    '''
    Labelize filter
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
    '''
    def __init__(self,location_path = UTILITIES_DIR,save_img =(None,True)):
        A_Filter.__init__(self, 'labelize', FILTER_TYPE.PRE_FILTER, location_path, built_in=False, save_img = save_img)
        
    def set_params(self):
        self.params = []
        


class Choose_Lungs(External_Filter):
    '''
    Choose lungs
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
    '''
    
    def __init__(self,location_path = UTILITIES_DIR,save_img =(None,True)):
        A_Filter.__init__(self, 'choose_lungsV2', FILTER_TYPE.PRE_FILTER, location_path, built_in=False, save_img = save_img)
    
    def set_params(self):
        self.params = []
    



class Extract_Trachea_FM(External_Filter):
    """
    Initial Position a tuple with the 3D position or a method to get it
    """
    MACAQUE_EXPECTED_PERIMETER = (11.5,33.5)
    MICE_EXPECTED_PERIMETER = (1.2,4)

    def __init__(self,location_path = UTILITIES_DIR, save_img = (None,True),intial_postion = None,
                 trachea_expected_perimeter = MACAQUE_EXPECTED_PERIMETER,time_Step=0.8, variable_thr =(-625, -300),fixed_thr=1.4):
        A_Filter.__init__(self, 'air_ext_fm', FILTER_TYPE.TRAQUEA_EXTRACTION, location_path, built_in=False, save_img = save_img)
        self.time_step = time_Step
        self.variable_thr = variable_thr
        self.fixed_threshold=fixed_thr
        self.trachea_expected_perimeter = trachea_expected_perimeter
        self.initial_postion = intial_postion
    
    
    def set_initial_position(self):
        pass
    
    def set_params(self):
        self.params  = [self.time_step, self.output_path_and_image.path, self.variable_thr[0], self.variable_thr[1], #TODO var_thr[1] is fixed_the and fixed_thr is prop_sigma
                        self.fixed_threshold, self.initial_postion[0], self.initial_postion[1], self.initial_postion[2]]
        
class SRM(Fiji_Filter):
    def __init__(self, location_path='/home/pmacias/Projects/MRI-PET_Tuberculosis/Fiji.app/',
                 save_img = (None, True), q = 25, three_dim = True, averages = False):
        A_Filter.__init__(self, 'ImageJ-linux64', FILTER_TYPE.OTHERS, location_path, built_in=False, save_img = save_img)
        self.q = q
        self.three_dim = three_dim
        self.averages = averages
    
    def str_three_dim(self, three_dim = None):
        if three_dim is not None:
            self.three_dim = three_dim
        if isinstance(self.three_dim, bool):
            return '3d' if self.three_dim is True else ""
        return self.three_dim
    
    def str_averages(self, averages = None):
        if averages is not None:
            self.averages = averages
        if isinstance(self.averages, bool):
            return 'showaverages' if self.averages is True else ""
        return self.averages
    
    def set_params(self):
        p1 = 'image_path='
        p2 = 'q='
        p3 = 'averages='
        p4 = 'threeD='
        p5 = 'image_out_path='
        self.params = ['/home/pmacias/Projects/MRI-PET_Tuberculosis/MMITB/SRM.ijm',
                       "'"+p1+'"'+self.input_path_and_image.path+'"'+"," +p2+str(self.q)+","+p3+
                       '"'+self.str_averages()+'"'+","+p4+'"'+self.str_three_dim()+'"'+","+p5+'"'+self.output_path_and_image.path+'"'+"'" ]
            
    

class Dilation(Simple_ITK_Filter):
    '''
    Dilation
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
    '''
    def __init__(self, radius = 1,kernel = SimpleITK.sitkBall, background = 0.0,
                 foreground = 1.0, boundaryToForeground = False,save_img =(None,True) ):
        A_Filter.__init__(self,'Dilation', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.radius = radius
        self.kernel = kernel
        self.background = background 
        self.foreground = foreground
        self.boundaryForegorund = boundaryToForeground
    
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.BinaryDilateImageFilter()
        self.simple_itk_filter.SetKernelRadius(self.radius)
        self.simple_itk_filter.SetKernelType(self.kernel)
    def set_params(self):
        self.params = [self.background, self.foreground, self.boundaryForegorund]

def find_trachea_init_ITK(image_input, trachea_start_slice = 1):
    """
    This functions needs a trachea at the "trachea_start_slice" slice of image_input
    """
    class Find_Position(External_Filter):
        
        def __init__(self,location_path = UTILITIES_DIR,save_img =(None,True), start_slice = trachea_start_slice):
            A_Filter.__init__(self, 'find_trachea', FILTER_TYPE.OTHERS, location_path, built_in=False, save_img = save_img)
            self.start_slice = start_slice
        
        def set_params(self):
            self.params = [self.start_slice]
            
    find_postion = Find_Position(start_slice=trachea_start_slice)
    find_postion.execute(image_input)
    with open(find_postion.output_path_and_image.path) as cords_file:
        return [int(coordinate) for coordinate in cords_file.readline().rstrip('\r\n').split(',')]
 
       
class Binarize(Simple_ITK_Filter):
    def __init__(self, lower_th = 1,upper_th= 255, inside_val = 255, outside_val = 0, save_img =(None,True)):
        A_Filter.__init__(self,'Binarize', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.lower_th = lower_th
        self.upper_th= upper_th
        self.inside_val = inside_val
        self.outside_val = outside_val
        
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.BinaryThresholdImageFilter()
        
    def set_params(self):
        self.params = [self.lower_th, self.upper_th, self.inside_val, self.outside_val]


class Mask(Simple_ITK_Filter):
    def __init__(self, mask_image, outside_val = 0, save_img =(None,True)):
        A_Filter.__init__(self,'Mask', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.mask_image = mask_image
        self.outside_val = outside_val
    
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.MaskImageFilter()
        
    def set_params(self):
        image = self.mask_image if isinstance(self.mask_image, SimpleITK.Image) else SimpleITK.ReadImage(self.mask_image)
        image.CopyInformation(self.input_path_and_image.image)
        self.params = [image, self.outside_val]

        
class Mask_Neg(Simple_ITK_Filter):
    def __init__(self, mask_image, save_img =(None,True)):
        A_Filter.__init__(self,'Mask', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.mask_image = mask_image 
        
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.MaskNegatedImageFilter()
        
    def set_params(self):
        image = self.mask_image if isinstance(self.mask_image, SimpleITK.Image) else SimpleITK.ReadImage(self.mask_image)
        image.CopyInformation(self.input_path_and_image.image)
        self.params = [image]


class File_Holes(Simple_ITK_Filter):
    def __init__(self, radius = 1, iterations = 20, foregorund = 255, background = 0, threads  = 4, majority_thr = 1, save_img =(None,True)):
        A_Filter.__init__(self, 'Filling', FILTER_TYPE.POST_FILTER, built_in=True, save_img=save_img )
        self.radius = radius
        self.background = background
        self.foregorund = foregorund
        self.iterations = iterations
        self.threads = threads
        self.majority_thr = majority_thr
        
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.VotingBinaryIterativeHoleFillingImageFilter()
        self.simple_itk_filter.SetNumberOfThreads(self.threads)
        
    def set_params(self):
        self.params = [self.radius, self.iterations, self.majority_thr, self.foregorund, self.background]

class Erode(Simple_ITK_Filter):
    def __init__(self, radius = 1, kernel = SimpleITK.sitkBall, backgorund = 0.0, foregorund = 255.0, boundary_foreground = False, save_img =(None,True)):
        A_Filter.__init__(self, 'Erode', FILTER_TYPE.OTHERS, built_in=True, save_img=save_img)
        self.radius = radius
        self.kernel = kernel
        self.backgorund = backgorund
        self.foregorund = foregorund
        self.boundary_foreground = boundary_foreground 
        
    def set_Filter_ITK(self):
        self.simple_itk_filter = SimpleITK.BinaryErodeImageFilter()
        self.simple_itk_filter.SetKernelRadius(self.radius)
        self.simple_itk_filter.SetKernelType(self.kernel)
    def set_params(self):
        self.params = [self.backgorund, self.foregorund, self.boundary_foreground]
       

class Hu_Threshold(A_Filter):
    '''
    Hu Threshold
    Inputs:
        Name input volume
    Outputs:
        Name of the output volume
''' 
    def __init__(self, iniTh=0.0,thrPlusOne = -600, huDiff=0.01, ct_HUnits_error=290, outside_val = 0.0, inside_val =255.0, save_img =(None,True)):
        A_Filter.__init__(self, 'Hu_trheshold', FILTER_TYPE.PRE_FILTER, built_in=True, save_img=save_img)
        self.iniTh = iniTh
        self.huDiff = huDiff
        self.errFac = ct_HUnits_error
        self.inside_val = inside_val
        self.outside_val = outside_val
        self.thrPlusOne = thrPlusOne

    
    def execute(self,inputimage, output = None):
        self.to_interface(inputimage)
        img = SimpleITK.GetArrayFromImage(self.input_path_and_image.image)
        def getHuMeans(image, thr):
            #Returns the means under and upper a thr of an array
            return ( np.mean(image[image < thr]), np.mean(image[image >= thr]) )

        while(np.abs(self.iniTh - self.thrPlusOne) > self.huDiff):
            self.iniTh = self.thrPlusOne
            uNuBMeans = getHuMeans(img, self.thrPlusOne)
            self.thrPlusOne = (uNuBMeans[0] + uNuBMeans[1]) / 2.0
            print self.iniTh,self.thrPlusOne,uNuBMeans[0],uNuBMeans[1]
        
        self.thrPlusOne += self.errFac
        print self.iniTh,self.thrPlusOne,uNuBMeans[0],uNuBMeans[1]
    
        if self.thrPlusOne < 0:    
            img[img >= self.thrPlusOne] = self.outside_val    
            img[img < self.thrPlusOne] = self.inside_val
        else:
            img[img <= self.thrPlusOne] = self.inside_val
            img[img > self.thrPlusOne] = self.outside_val    
        
        self.output_path_and_image.image = SimpleITK.GetImageFromArray(img)
        self.output_path_and_image.path = self.output_path_and_image.path if output is None else output
        SimpleITK.WriteImage(self.output_path_and_image.image,self.output_path_and_image.path)
        return self
      
        
if __name__ == "__main__":
    print 'Filpter.py'   
    srm = SRM(save_img=('/tmp/srm_filter.mhd', False),q=25)
    srm.execute("/home/pmacias/Projects/MRI-PET_Tuberculosis/PLTB724/20160621/114146_843000/t1_vibe_tra_bh_fatsat_exsp_0022/")

#    import glob
#    images = glob.glob("/media/pmacias/DATA2/M2_y_App_Pedro_Macias_Gordaliza/Macaques_M2/*.mhd")
#    print images
#    for i,image in enumerate(images):
#        srm = SRM(save_img=('/tmp/srm_filter'+str(i)+'.mhd', True),q=25)
#        srm.execute(image)
