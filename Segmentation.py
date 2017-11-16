#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:08:07 2017

@author: pmacias
"""

import tempfile
import os
import SimpleITK
from abc import ABCMeta, abstractmethod
from enum import Enum
from UsefullFunctions import calling_external, find_executable
import time

'''
SEGMENTATION FUNCTION
'''
separator=os.sep
imageFormats=['.mhd','.hdr']
FinalFolder='Segmentation'
SegmentedFolder='Segmented'

SIMPLE_ITK_ID = 'SimpleITK.SimpleITK'
#file_location  = os.path.dirname(os.path.abspath(__file__))

#TODO Loas this file and configuration options form a config file or two levels up by deafult
FILE_LOCATION  = os.path.dirname(os.path.abspath(__file__)) #Workaround chapucero
APP_DIR = os.path.dirname(FILE_LOCATION )
TLS_DIR = os.path.dirname(APP_DIR)
UTILITIES_DIR = os.path.join(os.path.dirname(TLS_DIR),'TLS_BUILD','Segmentation_Modules' ) 
RESULTS_FOLDER = "Seg_Prev_Res"

'''
SEGMENTATION CLASSES
'''
class FILTER_TYPE(Enum):
    PRE_FILTER = 1
    TRAQUEA_EXTRACTION = 2
    EXTRACTING = 3
    POST_FILTER = 4
    STOMACH_EXTRACTER = 5
    PIPE = 6
    OTHERS = 7

class PATH_IMAGE():
    def __init__(self,path,image):
        self.path = path
        self.image = image
        
    def get_path_image(self):
        return self.path,self.image
   

class A_Filter():

    """
    location_path just for external filters
    """
    __metaclass__ = ABCMeta
    def __init__(self, filter_name,kind, location_path = UTILITIES_DIR, built_in = True, save_img = (None,True) ):
        self.input_path_and_image = PATH_IMAGE(None,None)
        self.output_path_and_image = PATH_IMAGE(None,None)
        self.filter_name = filter_name
        #self.params = []

        if isinstance(save_img,tuple) and len(save_img) == 2:
            self.save_img = save_img
        else:
            raise Exception('Incorrect save format in '+self.filter_name+" must be a len 2 tuple")
        
        if isinstance(kind,FILTER_TYPE):
            self.filter_type = kind 
        else:
            raise Exception("Incorret A_Filter type: "+self.filter_type)
            
        if not built_in:
            self.path = find_executable(self.filter_name, location_path)
        
        self.set_output_name()
    
    def set_output_name(self):
        out_path,add_name = self.save_img
        if out_path is not None:
            path,file_name = os.path.split(out_path)
        if out_path is None:
            self.output_path_and_image.path = os.path.join(tempfile.gettempdir(),str(int(time.time()))+"_TLS_"+self.filter_name+".mhd")
        elif os.path.isdir(path):
            if add_name:
                file_name,ff = os.path.splitext(file_name)
                self.output_path_and_image.path = os.path.join(path,file_name+'_'+self.filter_name+ff)
            else:
                self.output_path_and_image.path = out_path
        else:
            raise Exception("Invalid output_name configutation "+self.save_img)
            
    def to_interface(self, a_filter):
        if isinstance(a_filter,Simple_ITK_Filter):
            self.input_path_and_image = a_filter.output_path_and_image
        elif isinstance(a_filter, External_Filter):
            self.input_path_and_image.path = a_filter.output_path_and_image.path
            self.input_path_and_image.image = SimpleITK.ReadImage(a_filter.output_path_and_image.path)
        else:
            self.input_path_and_image.path = a_filter #Should be a path to the image
            self.input_path_and_image.image = SimpleITK.ReadImage(a_filter) if isinstance(a_filter, str) else a_filter
        
    def flush(self):
        self.input_path_and_image.image = None
        self.output_path_and_image.image = None
    
            
    def __str__(self):
        return self.filter_name
        

    @abstractmethod
    def execute(self, A_Filter): pass


class External_Filter(A_Filter):
    __metaclass__ = ABCMeta
    def execute(self, a_filter, output = None): #Input and Output can be just paths
        """
        a_filter and image or another filter
        """
        self.input_path_and_image.path = a_filter.output_path_and_image.path if isinstance(a_filter,A_Filter) else a_filter
        self.output_path_and_image.path = self.output_path_and_image.path if output is None else output
        self.set_params()
        self.check_params()
        calling_external(self.path, [self.input_path_and_image.path]+self.params
        +[self.output_path_and_image.path])
        return self
        
    def check_params(self):
        if not isinstance(self.params,list):
            raise Exception('The parameter fot the filter '+self.filter_name+" are incorrect")
            
            
    @abstractmethod
    def set_params(self):pass

class Fiji_Filter(A_Filter):
    __metaclass__ = ABCMeta
    def execute(self, a_filter, output = None): #Input and Output can be just paths
        """
        a_filter and image or another filter
        """
        self.input_path_and_image.path = a_filter.output_path_and_image.path if isinstance(a_filter,A_Filter) else a_filter
        self.output_path_and_image.path = self.output_path_and_image.path if output is None else output
        self.set_params()
        self.check_params()
        calling_external('pkill',[ '-f', 'ImageJ*'])
        calling_external(self.path, ['--ij2', '--headless', '--run', ]+self.params)
        return self
        
    def check_params(self):
        if not isinstance(self.params,list):
            raise Exception('The parameter fot the filter '+self.filter_name+" are incorrect")
            
            
    @abstractmethod
    def set_params(self):pass
        

class Simple_ITK_Filter(A_Filter):
    __metaclass__ = ABCMeta    
    def execute(self,a_filter, output = None):
        self.to_interface(a_filter)

        self.set_params()
        self.check_params()
        self.set_Filter_ITK()
        exec_params = [self.input_path_and_image.image] + self.params
        self.output_path_and_image.image = self.simple_itk_filter.Execute(*exec_params)
        self.output_path_and_image.path = self.output_path_and_image.path if output is None else output
        SimpleITK.WriteImage(self.output_path_and_image.image,self.output_path_and_image.path)
        return self
                                          
        
                                          
    def __isSimpleITK__(self):
        if SIMPLE_ITK_ID in str(type(self.simple_itk_filter)):
            return SIMPLE_ITK_ID in str(type(self.simple_itk_filter))
        else:
            raise Exception("Incorrect SimpleITK Filter "+str(self.filter_simple))
    
    def flush_and_save(self):
        pass
    
            
    def check_params(self):
        if not isinstance(self.params,list):
            raise Exception('The parameter fot the filter '+self.filter_name+" are incorrect")
            
    @abstractmethod
    def set_params(self):pass

    @abstractmethod
    def set_Filter_ITK(self):pass
    
    
class Pipe(A_Filter):
    def __init__(self,filter_name ,filters_list):
        A_Filter.__init__(self, filter_name= filter_name ,kind = FILTER_TYPE.PIPE, location_path = UTILITIES_DIR, built_in = True, save_img = (None,True))
        self.filters_list = filters_list
        self.check_filters()
        
    def execute(self,a_filter, ouput = None):
        if isinstance(a_filter, A_Filter):
            self.input_path_and_image = a_filter.input_path_and_image
        else:
            self.input_path_and_image.path = a_filter
        
        prev_filter = a_filter
        for filt in self.filters_list:
            prev_filter = filt.execute(prev_filter)
        
        self.output_path_and_image = prev_filter.output_path_and_image
        return prev_filter
        
    def check_filters(self):
        checking = [isinstance(fil,A_Filter) for fil in self.filters_list ]
        if (sum(checking) != len(checking)):
            raise Exception("Incorrec A_Filters at "+self.filters_list)
        
    

    
    
    
    
    
