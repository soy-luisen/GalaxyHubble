"""
Master’s thesis - UNIR - Luis Enrique Ramirez Pelaez - 2023
Deep Learning as an alternative to deconvolution of galaxy images captured with the space telescope Hubble
Code developeds
------------------------------------------------------------------
    https://www.linkedin.com/in/soy-luisen/ 
    https://github.com/soy-luisen/GalaxyHubble
------------------------------------------------------------------
"""


"""
Module with functions to loading and manipulating the dataset.
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


def normaliza(valor,min,max,nuevo_min=0,nuevo_max=1):
    """
    Normalizes values (image) from the min-max range to the nuevo_min-nuevo_max range
    
    Parameters:
    -----------
    valor: data image
    min: old min value
    max: old max value
    nuevo_min: new min value
    nuevo_max: new max value    
    
    Return:
    -------
    nuevo_valor: new data image    
    """
    rango_anterior=max-min
    rango_nuevo=nuevo_max-nuevo_min
    escala=(valor-min)/rango_anterior
    nuevo_valor=(rango_nuevo*escala)+nuevo_min
    return nuevo_valor


def carga_dataset(rango_files,carpeta_dataset,dataset_files,dataset_deconv_files):   
    """
    The dataset files are read and the images are extracted.
    
    Parameters:
    -----------
    rango_files: Items of list to be processed
    carpeta_dataset: Dataset folder
    dataset_files: List of dataset files names (.fits file type) with galaxy images
    dataset_deconv_files: List of dataset files deconv names (.fits file type) with deconvolutioned galaxy images
    
    Return:
    -------
    X_data: array of dataset images
    y_data: array of dataset target images (deconvolutioned image)   
    
    """    
    
    X_data = []
    y_data = []       
    
    # All the .fits files that will be added to the dataset are traversed
    for i in np.array(rango_files): 
        nombre_fichero=carpeta_dataset+dataset_files[i] #File name of i .fits images
        print(nombre_fichero)
        fichero = get_pkg_data_filename(nombre_fichero) #Open .fits file
        hdul = fits.open(fichero)  #The entire file is read to a list of type HDUList
        num_imagenes= len(hdul)    #The number of images in the file is obtained (HDUList list size)    
        
        nombre_fichero_deconv=carpeta_dataset+dataset_deconv_files[i]  #Build the file name of i .fits target images
        fichero_deconf=get_pkg_data_filename(nombre_fichero_deconv)    #Open .fits file   
        hdul_deconv=fits.open(fichero_deconf)  #The entire file is read to a list of type HDUList

        for x in range(0,num_imagenes):
            image_data=hdul[x].data                #Extract data from image .fits file
            image_data_deconv=hdul_deconv[x].data  #Extract data from target image .fits file
            
            #Images are normalized to 0-1
            image_data=normaliza(image_data,image_data.min(),image_data.max())
            image_data_deconv=normaliza(image_data_deconv,image_data_deconv.min(),image_data_deconv.max())
            
            X_data.append(image_data)         #Image data is append to X_data list
            y_data.append(image_data_deconv)  #Target image data is append to y_data list          

        hdul.close(closed=True)          #The HDULists are closed
        hdul_deconv.close(closed=True)  
        
    return np.array(X_data), np.array(y_data)