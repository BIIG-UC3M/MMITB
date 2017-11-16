#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:36:47 2017

@author: pmacias
"""
import os 
import subprocess
import numpy as np
import SimpleITK
import tempfile

def is_exe(fpath):
    """
    check if the program is executable
    """
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

def check_executable(program):
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None

def find_executable(name, path, check_on_system = False):#cutre workaround
    print "Finding for "+name+" in "+path+"\n"
    #path = os.path.expanduser('~')
    for root, dirs, files in os.walk(path):
        if name in files:
            program = check_executable(os.path.join(root, name))
            if program is not None:
                return  program
    if check_on_system:
        raise Exception("There is not execuable "+name)
    else:
        new_seek_in = os.path.expanduser('~')#TODO is multiplatform?
        print ("It was impossible to find the function "+name+" on "
        +path+". It will be seeked on "+new_seek_in+"\n")
        return find_executable(name, os.path.abspath(os.path.expanduser('~')+os.sep), check_on_system=True) # Avoid infinite loop

def find_images(initialpath,imageFormats=['.mhd','.hdr']):
    images=[]
    for root, dirs, files in os.walk(initialpath):
        for name in files:
            for formats in imageFormats:
                if formats in name:
                    images.append(os.path.join(root, name))
    return images

def calling_external(exter_fun_path,inputs):
    for i in inputs:
        exter_fun_path=exter_fun_path+' '+str(i)
    process = subprocess.Popen(exter_fun_path, shell=True)
    process.wait()  


def name_output(name_func,inputimage,finalformat=None,pattern = None, save= True):
    """
    Save:
    True for the input same place
    False for the temp
    otherwise must be a route
    """
    
    file_name,ff = os.path.splitext(inputimage)
    file_name_real = file_name.split(os.sep)[-1] if pattern is None else pattern
    if isinstance(save, bool):
        file_name = file_name if save else os.path.join(tempfile.gettempdir(),file_name_real)
    elif os.path.isdir(save):
        print 'SAVE',file_name_real
        file_name = os.path.join(save,file_name_real)
    else:
        raise Exception(save+' is not a path for the refult of filter '+name_func)
    if finalformat is not None:
        ff = finalformat
    
    return  file_name+'_'+name_func+ff
    


def argus2SimpleITK(path):
  """
  Transforms Argus/IDL image types to SimpleITK usual
  """
  typemap = {"unsigned integer":'u2', "signed integer":'i2', "unsigned long":'u4',
 "signed long":'i4', "short float":'f4', "long float":'f8', "byte":np.uint8}
   
  hdrMap = {};
  #Get fields to open raw as image
  with open(path) as f:
    for line in f:
      splitted = line.lower().split(':=') #All to lower case in order to avoid problems
      if(len(splitted) > 1):
        hdrMap[splitted[0].strip()] = splitted[1].strip()


#  print hdrMap['number format']
  rawFile = os.path.join(os.path.dirname(path), hdrMap['name of data file'])
  if os.path.exists(rawFile)==False:
      rawFile=os.path.join(os.path.dirname(path),  os.path.splitext(hdrMap['name of data file'])[0].upper()+os.path.splitext(hdrMap['name of data file'])[1])
  #TODO sometimes matrixsize[3] doesnt exists. Check for NumberTotalImages within hdrMap
  if 'matrix size [3]' in hdrMap:
      dims = (int(hdrMap['matrix size [2]']), int(hdrMap['matrix size [1]']),int( hdrMap['matrix size [3]']))
  elif 'NumberTotalImages' in hdrMap:
      dims = (int(hdrMap['matrix size [2]']), int(hdrMap['matrix size [1]']),int( hdrMap['NumberTotalImages']))
   
  spacing = ( float(hdrMap['scaling factor (mm/pixel) [2]']), float( hdrMap['scaling factor (mm/pixel) [1]']), float( hdrMap['scaling factor (mm/pixel) [3]']) )
  offset = (float(hdrMap['offset [2]']), float(hdrMap['offset [1]']), float(hdrMap['offset [3]']))
  endian = hdrMap['imagedata byte order']
  endiannes = '>' if  endian == 'bigendian' else '<'
  
#  print "Dims: ",dims, "Spacing: " ,spacing, "Offset: ",offset
  dataType = np.int16;
  if(int(hdrMap['number of bytes per pixel']) == 1):
    dataType = np.dtype(typemap["byte"])
  else:
    dataType = endiannes + typemap[hdrMap['number format']]
  
  
    
  arrayImage = np.fromfile(rawFile, dtype = dataType)
  dims = list(reversed(dims))
  arrayImage = arrayImage.reshape(dims, order = 'C') 

  itkImage = SimpleITK.GetImageFromArray(arrayImage)
  itkImage.SetOrigin(offset)
  itkImage.SetSpacing(spacing)
  

  return itkImage


def hdr_to_mhd(pathimages):
    images=find_images(pathimages,imageFormats=['.hdr'])
    for i in images:
        image=argus2SimpleITK(i)
        pathfinal=os.path.join(os.path.abspath(os.path.join(i, os.pardir)),os.path.basename(os.path.normpath(i)).split('.')[0]+'.mhd')
        SimpleITK.WriteImage(image,pathfinal)
        

if __name__ == "__main__":
    print 'Functions'

        
