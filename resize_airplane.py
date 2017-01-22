# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 08:14:32 2017

@author: abhin
"""

from PIL import Image
from resizeimage import resizeimage
import os
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def resize(): 
    directory = 'airplanes'
    for filename in os.listdir(directory):
        path  = directory + "/" + filename
        img = Image.open(path).convert('L')
        img = resizeimage.resize_cover(img, [28, 28], validate=False)
#        img = rgb2gray(np.array(img))
        img.save('resize_airplanes/'+filename, img.format)
    
def load():
    directory = 'resize_airplanes'
    img = Image.open('resize_airplanes/image_0001.jpg')
    img.load()
    st = np.array(img,dtype="int32")
    
    st = st[np.newaxis,:,:]
#    st = st[np.newaxis,:,:]
    print st.shape
    for filename in os.listdir(directory):
        if(filename == "image_0001.jpg"):
            continue
        path  = directory + "/" + filename
        img = Image.open(path)
        img.load()
        data2 = np.array(img,dtype="int32")
        if(len(data2.shape) > 1):
##         print "ST",st.shape
##         print "DT1",data2.shape
##         print "DT",data2[np.newaxis,:,:,:].shape
#         data2 = data2[np.newaxis,:,:]
         st = np.vstack((st, data2[np.newaxis,:,:]))
          
##         print "STT",st.shape
    print st.shape
#    st = np.rollaxis(st,3,1)
#        
    
    
   
#    print st.shape
    return st
    
#resize()
#load()

