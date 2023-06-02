"""
Master’s thesis - UNIR - Luis Enrique Ramirez Pelaez - 2023
Deep Learning as an alternative to deconvolution of galaxy images captured with the space telescope Hubble
Code developed
------------------------------------------------------------------
    https://www.linkedin.com/in/soy-luisen/ 
    https://github.com/soy-luisen/GalaxyHubble
------------------------------------------------------------------
"""

"""
Module for definition of architectures
"""
import cv2, os, glob, time, math
import pandas as pd
import numpy as np
import pprint as pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration, io, exposure
from skimage.metrics import structural_similarity as ssim

from astropy.visualization import astropy_mpl_style, ImageNormalize, SqrtStretch, MinMaxInterval,imshow_norm
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.table import Table

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, utils
from tensorflow.keras.layers import Input , Conv2D , PReLU, MaxPooling2D , Dropout , concatenate , UpSampling2D, Normalization, Rescaling, Add
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers


def crea_modelo(nombre_modelo,input_shape,dub=0): 
    """
    Function that creates an architecture according to the parameter
    
    Parameters:
    -----------
    nombre_modelo: Type of model to build
    input_shape: Image input, rows x columns x depth
    dub: Only for DIDN's architectures. Number of processing units
    
    Returns:
    -----------
    modelo: Built model    
    """    
    if (nombre_modelo=='unetclassic4'):
        modelo=UNet_classic_4(input_shape=input_shape)

    elif (nombre_modelo=='unetclassic8'):
        modelo=UNet_classic_8(input_shape=input_shape)

    elif (nombre_modelo=='unetclassic16'):
        modelo=UNet_classic_16(input_shape=input_shape)

    elif (nombre_modelo=='unetclassic32'):
        modelo=UNet_classic_32(input_shape=input_shape)

    elif (nombre_modelo=='unetclassic64'):
        modelo=UNet_classic_64(input_shape=input_shape)

    elif (nombre_modelo=='unetclassic128'):
        modelo=UNet_classic_128(input_shape=input_shape)
    
    elif (nombre_modelo=='DIDN16'):
        modelo=DIDN16(input_shape=input_shape,dub=dub)   
        
    elif (nombre_modelo=='AEPP2'):
        modelo=AEPP2(input_shape=input_shape)   
        
    return modelo


def UNet_classic_16(input_shape):       
        '''
        Code from:
        https://pythonawesome.com/complete-u-net-implementation-with-keras-in-python/
        https://github.com/sagnik1511/U-Net-Reduced-with-TF-keras
        
        Number filters: 16,32,64,128,256,128,64,32,16

        Parameters
        ----------
        input_shape: Image input, rows x columns x depth              

        Returns
        -------
        model: Built model        
        '''
        inputs = Input(input_shape)       
        
        conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
          
        conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
          
        conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
          
        conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
          
        up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
          
        up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
          
        up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
          
        up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
          
        outputs = layers.Conv2D(1, 1, activation = 'relu')(conv9)        
          
        model = keras.Model(inputs = inputs , outputs = outputs,name = 'UNet_generator16')
        model.summary()
          
        return model          
    
def UNet_classic_4(input_shape):       
        '''
        Code from:
        https://pythonawesome.com/complete-u-net-implementation-with-keras-in-python/
        https://github.com/sagnik1511/U-Net-Reduced-with-TF-keras
        
        Modified by: Luis Enrique Ramírez Peláez
        Number filters: 4,8,16,32,64,32,16,8,4

        Parameters
        ----------
        input_shape: Image input, rows x columns x depth              

        Returns
        -------
        model: Built model        
        '''
        inputs = Input(input_shape)       
        
        conv1 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
          
        conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
          
        conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
          
        conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
          
        up6 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
          
        up7 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
          
        up8 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
          
        up9 = Conv2D(4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
          
        outputs = layers.Conv2D(1, 1, activation = 'relu')(conv9)        
          
        model = keras.Model(inputs = inputs , outputs = outputs,name = 'UNet_generator4')
        model.summary()
          
        return model  
    
def UNet_classic_8(input_shape):       
        '''
        Code from:
        https://pythonawesome.com/complete-u-net-implementation-with-keras-in-python/
        https://github.com/sagnik1511/U-Net-Reduced-with-TF-keras
        
        Modified by: Luis Enrique Ramírez Peláez
        Number filters: 8,16,32,64,128,64,32,16,8,4

        Parameters
        ----------
        input_shape: Image input, rows x columns x depth              

        Returns
        -------
        model: Built model        
        '''
        inputs = Input(input_shape)       
        
        conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
          
        conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
          
        conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
          
        conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
          
        up6 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
          
        up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
          
        up8 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
          
        up9 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
          
        outputs = layers.Conv2D(1, 1, activation = 'relu')(conv9)        
          
        model = keras.Model(inputs = inputs , outputs = outputs,name = 'UNet_generator8')
        model.summary()
          
        return model  
    
def UNet_classic_32(input_shape):       
        '''
        Code from:
        https://pythonawesome.com/complete-u-net-implementation-with-keras-in-python/
        https://github.com/sagnik1511/U-Net-Reduced-with-TF-keras
        
        Modified by: Luis Enrique Ramírez Peláez
        Number filters: 32,64,128,256,512,256,128,64,32

        Parameters
        ----------
        input_shape: Image input, rows x columns x depth              

        Returns
        -------
        model: Built model        
        '''
        inputs = Input(input_shape)       
        
        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
          
        conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
          
        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
          
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
          
        up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
          
        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
          
        up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
          
        up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
          
        outputs = layers.Conv2D(1, 1, activation = 'relu')(conv9)        
          
        model = keras.Model(inputs = inputs , outputs = outputs,name = 'UNet_generator32')
        model.summary()
          
        return model  
    
def UNet_classic_64(input_shape):       
        '''
        Code from:
        https://pythonawesome.com/complete-u-net-implementation-with-keras-in-python/
        https://github.com/sagnik1511/U-Net-Reduced-with-TF-keras
        
        Modified by: Luis Enrique Ramírez Peláez
        Number filters: 128,256,512,1024,512,256,128

        Parameters
        ----------
        input_shape: Image input, rows x columns x depth              

        Returns
        -------
        model: Built model        
        '''
        inputs = Input(input_shape)       
        
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
          
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
          
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
          
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
          
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
          
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
          
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
          
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
          
        outputs = layers.Conv2D(1, 1, activation = 'relu')(conv9)        
          
        model = keras.Model(inputs = inputs , outputs = outputs,name = 'UNet_generator64')
        model.summary()
          
        return model  

def UNet_classic_128(input_shape):       
        '''
        Code from:
        https://pythonawesome.com/complete-u-net-implementation-with-keras-in-python/
        https://github.com/sagnik1511/U-Net-Reduced-with-TF-keras
        
        Modified by: Luis Enrique Ramírez Peláez
        Number filters: 128,256,512,1024,2048,1024,512,256,128

        Parameters
        ----------
        input_shape: Image input, rows x columns x depth              

        Returns
        -------
        model: Built model        
        '''
        inputs = Input(input_shape)       
        
        conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
          
        conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
          
        conv3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
          
        conv5 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
          
        up6 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
          
        up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
          
        up8 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
          
        up9 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
          
        outputs = layers.Conv2D(1, 1, activation = 'relu')(conv9)        
          
        model = keras.Model(inputs = inputs , outputs = outputs,name = 'UNet_generator128')
        model.summary()
          
        return model  
    
def DIDN16(input_shape,dub):
    """
    @inproceedings{yu2019deep,
    title={Deep iterative down-up CNN for image denoising},
    author={Yu, Songhyun and Park, Bumjun and Jeong, Jechang},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
    year={2019}
    
    Keras implementation by Luis Enrique Ramírez Peláez  
    
    Parameters
    ----------
    input_shape: Image input, rows x columns x depth   
    dub: Number of processing units

    Returns
    -------
    model: Built model      
    """
    
    inputs = Input(input_shape)
    
    conv_input = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(inputs)    
    relu1=PReLU()(conv_input)
    
    conv_down=Conv2D(16,3,activation = 'relu',strides=2,padding='same',use_bias=False)(relu1)    
    relu2=PReLU()(conv_down)

    if (dub==1):
        recursives = DIDN16_Residual_Block(relu2)
    
    elif (dub==2):
        recursive_A = DIDN16_Residual_Block(relu2)
        recursives = DIDN16_Residual_Block(recursive_A)
        
    elif (dub==3):
        recursive_A = DIDN16_Residual_Block(relu2)
        recursive_B = DIDN16_Residual_Block(recursive_A)
        recursives = DIDN16_Residual_Block(recursive_B)
    
    elif (dub==4):
        recursive_A = DIDN16_Residual_Block(relu2)
        recursive_B = DIDN16_Residual_Block(recursive_A)
        recursive_C = DIDN16_Residual_Block(recursive_B)
        recursives = DIDN16_Residual_Block(recursive_C)
        
    elif (dub==5):
        recursive_A = DIDN16_Residual_Block(relu2)
        recursive_B = DIDN16_Residual_Block(recursive_A)
        recursive_C = DIDN16_Residual_Block(recursive_B)
        recursive_D = DIDN16_Residual_Block(recursive_C)
        recursives = DIDN16_Residual_Block(recursive_D)
        
    elif (dub==6):    
        recursive_A = DIDN16_Residual_Block(relu2)            
        recursive_B = DIDN16_Residual_Block(recursive_A)
        recursive_C = DIDN16_Residual_Block(recursive_B)
        recursive_D = DIDN16_Residual_Block(recursive_C)
        recursive_E = DIDN16_Residual_Block(recursive_D)
        recursives = DIDN16_Residual_Block(recursive_E)
    
    elif (dub==7): 
        recursive_A = DIDN16_Residual_Block(relu2)
        recursive_B = DIDN16_Residual_Block(recursive_A)
        recursive_C = DIDN16_Residual_Block(recursive_B)
        recursive_D = DIDN16_Residual_Block(recursive_C)
        recursive_E = DIDN16_Residual_Block(recursive_D)
        recursive_F = DIDN16_Residual_Block(recursive_E)
        recursives = DIDN16_Residual_Block(recursive_F)
    
    elif (dub==8): 
        recursive_A = DIDN16_Residual_Block(relu2)
        recursive_B = DIDN16_Residual_Block(recursive_A)
        recursive_C = DIDN16_Residual_Block(recursive_B)
        recursive_D = DIDN16_Residual_Block(recursive_C)
        recursive_E = DIDN16_Residual_Block(recursive_D)
        recursive_F = DIDN16_Residual_Block(recursive_E)
        recursive_G = DIDN16_Residual_Block(recursive_F)
        recursives = DIDN16_Residual_Block(recursive_G)
        
    elif (dub==9): 
        recursive_A = DIDN16_Residual_Block(relu2)
        recursive_B = DIDN16_Residual_Block(recursive_A)
        recursive_C = DIDN16_Residual_Block(recursive_B)
        recursive_D = DIDN16_Residual_Block(recursive_C)
        recursive_E = DIDN16_Residual_Block(recursive_D)
        recursive_F = DIDN16_Residual_Block(recursive_E)
        recursive_G = DIDN16_Residual_Block(recursive_F)
        recursive_H = DIDN16_Residual_Block(recursive_G)
        recursives = DIDN16_Residual_Block(recursive_H)

    elif (dub==10): 
        recursive_A = DIDN16_Residual_Block(relu2)
        recursive_B = DIDN16_Residual_Block(recursive_A)
        recursive_C = DIDN16_Residual_Block(recursive_B)
        recursive_D = DIDN16_Residual_Block(recursive_C)
        recursive_E = DIDN16_Residual_Block(recursive_D)
        recursive_F = DIDN16_Residual_Block(recursive_E)
        recursive_G = DIDN16_Residual_Block(recursive_F)
        recursive_H = DIDN16_Residual_Block(recursive_G)
        recursive_I = DIDN16_Residual_Block(recursive_H)
        recursives = DIDN16_Residual_Block(recursive_I)

    recon = DIDN16_Recon_Block(recursives)

    conv_mid = Conv2D(16,1,activation = 'relu',strides=1,padding='same',use_bias=False)(recon)    
    relu3=PReLU()(conv_mid)
    conv_mid2 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu3)    
    relu4=PReLU()(conv_mid2)

    subpixel = UpSampling2D(size = (2,2))(relu4)
    outputs = Conv2D(1,3,activation = 'relu',strides=1,padding='same',use_bias=False)(subpixel)    

    model = keras.Model(inputs = inputs , outputs = outputs,name = 'DIDN16_DUB_'+str(dub)+'_architecture')
    model.summary()
          
    return model  


def DIDN16_Residual_Block(input):
    conv1 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(input)        
    relu2=PReLU()(conv1)    
    
    conv3 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu2)        
    relu4=PReLU()(conv3)      
   
    conv5 = Conv2D(32,3,activation = 'relu',strides=2,padding='same',use_bias=False)(relu4)        
    relu6=PReLU()(conv5)

    conv7 = Conv2D(32,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu6)        
    relu8=PReLU()(conv7)
   
    conv9 = Conv2D(64,3,activation = 'relu',strides=2,padding='same',use_bias=False)(relu8)        
    relu10=PReLU()(conv9)

    conv11 = Conv2D(64,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu10)        
    relu12=PReLU()(conv11)

    conv13 = Conv2D(128,1,activation = 'relu',strides=1,padding='same',use_bias=False)(relu12)        
    up14=UpSampling2D(size = (2,2))(conv13)

    conv15 = Conv2D(32,1,activation = 'relu',strides=1,padding='same',use_bias=False)(up14)        
    conv16 = Conv2D(32,3,activation = 'relu',strides=1,padding='same',use_bias=False)(conv15)        
    relu17=PReLU()(conv16)
    
    conv18 = Conv2D(64,1,activation = 'relu',strides=1,padding='same',use_bias=False)(relu17)        
    up19=UpSampling2D(size = (2,2))(conv18)

    conv20 = Conv2D(16,1,activation = 'relu',strides=1,padding='same',use_bias=False)(up19)   
    conv21 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(conv20)        
    relu22=PReLU()(conv21)
    conv23 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu22)        
    relu24=PReLU()(conv23)
    
    conv25 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu24)
    
    return conv25   

def DIDN16_Recon_Block(input):
    conv1 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(input)        
    relu2=PReLU()(conv1)        
    conv3 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu2)        
    relu4=PReLU()(conv3) 

    conv5 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu4)        
    relu6=PReLU()(conv5) 
    conv7 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu6)        
    relu8=PReLU()(conv7) 

    conv9 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu8)        
    relu10=PReLU()(conv9) 
    conv11 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu10)        
    relu12=PReLU()(conv11) 

    conv13 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu12)        
    relu14=PReLU()(conv13) 
    conv15 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu14)        
    relu16=PReLU()(conv15) 

    conv17 = Conv2D(16,3,activation = 'relu',strides=1,padding='same',use_bias=False)(relu16)        
    
    return conv17

    
def AEPP2(input_shape):
    """
    Hernández Afonso, J. (2022). Redes Neuronales Convolucionales para la Reconstrucción de 
    Imágenes de Galaxias [TFM]. Universidad Internacional de La Rioja.
    """
    input_img = Input(input_shape)
    
    # -----------------------------------------Upper branch (x)----------------------------------------------------
    # Compressor
    x = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)    
    x = layers.ReLU()(x)
    x1 = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (4, 4), padding="same")(x1) 
    x = layers.ReLU()(x) 
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    
    # Decompressor
    x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(x)
    x = layers.ReLU()(x)
    x = Add()([x1, x])
    output_up = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(x)
    
    # -----------------------------------------Lower branch (y)----------------------------------------------------    
    # Compressor
    y = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)
    y = layers.ReLU()(y)
    y1 = layers.MaxPooling2D((2, 2), padding="same")(y)
    y = layers.Conv2D(64, (4, 4), padding="same")(y1)
    y = layers.ReLU()(y)
    y2 = layers.MaxPooling2D((2, 2), padding="same")(y)
    y = layers.Conv2D(128, (4, 4), padding="same")(y2)
    y = layers.ReLU()(y)
    y3 = layers.MaxPooling2D((2, 2), padding="same")(y)
    y = layers.Conv2D(256, (4, 4), padding="same")(y3)
    y = layers.ReLU()(y)
    y = layers.MaxPooling2D((2, 2), padding="same")(y)
    
    # Decompressor
    y = layers.Conv2DTranspose(128, (4, 4), strides=2, padding="same")(y)
    y = layers.ReLU()(y)
    y = Add()([y3, y])
    y = layers.Conv2DTranspose(64, (4, 4), strides=2, padding="same")(y)
    y = layers.ReLU()(y)
    y = Add()([y2, y])
    y = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(y)
    y = layers.ReLU()(y)
    y = Add()([y1, y])
    output_low = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(y)
    
    # ---------------------------------------- End of Branches----------------------------------------------------
    
    pre_output = Add()([output_up, output_low])
    output = layers.ReLU()(pre_output)    
    
    # Symetric autoencoder
    model = keras.Model(inputs=input_img, outputs=output, name= 'AEPP2')
    model.summary()
    return model    