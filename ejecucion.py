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
Module with functions for training, prediction, visualization and other operations with the networks.
"""
import cv2, os, glob, time, math
import pandas as pd
import numpy as np
import pprint as pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve2d as conv2
from random import randint

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
from tensorflow.keras.layers import Input , Conv2D , MaxPooling2D , Dropout , concatenate , UpSampling2D, Normalization, Rescaling
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers

import arquitecturas

def ssim_loss(y_true, y_pred):
    """
    SSIM Loss function
    
    Params:
    -------
    y_true: true value
    y_pred: predict value
    
    Return:
    1-SSIM. The target for LOSS function is zero, but de target for SSIM is one, so we have to subtract SSIM from 1
    """
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0,filter_size=7))


class earlyStop(tf.keras.callbacks.Callback):  
    """
    A class is defined to stop training when loss < 0.02 (ssim > 0.98)
    """
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss')< 0.02):   
            print("\nReached 3% loss rate, training is canceled!!")
            self.model.stop_training = True

# Compila y entrena el modelo.
def entrena_modelo(model, x_train, y_train, x_val, y_val, input_shape = (128, 128, 1),
                   epochs=20, batch_size=128, optimizer="adam", lr=0.001, loss="mse", metrics="mean_absolute_error"):
    """
    Build and train the model.
    
    Params:
    -------
    model: architecture model
    x_train: x train data
    y_train: y train data
    x_val:   x validation data
    y_val:   y validation data
    input_shape =  default (128, 128, 1)
    epochs=        Default 20
    batch_size=    Default 128
    optimizer=     Defautl "adam"
    lr=            Default 0.001
    loss=          Defautl "mse"
    metrics=       Defautl "mean_absolute_error"    
    
    Return:
    -------
    history: Model trained
    """
    
    callbacks = earlyStop()    
    
    if (optimizer=="adam"):
        opt = optimizers.Adam(learning_rate = lr)
    else: #Put here another optimizers
        opt = optimizers.Adam(learning_rate = lr)

    if (loss=="ssim"):
        model.compile(optimizer = opt, loss = ssim_loss, metrics = metrics)
    else:
        model.compile(optimizer = opt, loss =loss, metrics = metrics)
        
    history = model.fit(x_train,y_train , 
                        batch_size = batch_size,
                        epochs = epochs ,
                        validation_data = (x_val,y_val) , verbose = 1, 
                        callbacks=[callbacks])  
    return history

def carga_modelo(ubicacion,modelo_h5):
    """
    Pretrained model load.
    
    Parameters:
    -----------
    ubicacion: Source folder
    modelo_h5: Pretrained model file
    
    Return:
    -------
    modelo: Pretrained model load.
    
    """
    
    modelo=models.load_model(ubicacion + modelo_h5, compile=False)
    return modelo    
    

def guarda_imagen(ax, filename):
    """
    Save image file.
    
    Params:
    ------
    ax: Data image
    filename: Filename (with folder)        
    """
    ax.axis("On")
    ax.figure.canvas.draw()      
    plt.savefig(filename, dpi=200)    

def muestra_estadisticas_entrenamiento(history,model,x_test,y_test,nombre_imagen):
    """
    Show training statitics.
    
    Parameters:
    -----------
    history: Trained model
    model: Model
    x_test: x data test
    y_test: y data test
    nombre_imagen: File name image to be saved
    
    Return:
    salida: Data statistics on text    
    
    """
    #Graphs are shown with metrics
    history_frame = pd.DataFrame(history.history)
    metrica_usada=history_frame.columns[1]
    metrica_usada_val=history_frame.columns[3]    
    
    f, axes = plt.subplots(1, 2,figsize=(14,4))
    axes[0].plot(history_frame[metrica_usada])
    axes[0].plot(history_frame[metrica_usada_val])
    axes[0].set_title('Model '+metrica_usada)
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel(metrica_usada)
    axes[0].legend(['train', 'validation'], loc='upper left')
    axes[0].grid()
    axes[1].plot(history_frame['loss'])
    axes[1].plot(history_frame['val_loss'])
    axes[1].set_title('Model loss')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].legend(['train', 'validation'], loc='upper left')
    axes[1].grid()
   
    #Image file is saved
    plt.savefig(nombre_imagen+"_"+metrica_usada+"_loss"+".png", dpi=100, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
    #Text statistics are built
    epochs = len(history_frame['loss'])
    salida="Epoch "+str(epochs)+"\n"
    salida += "Model evaluation: \n"      
    score = model.evaluate(x_test, y_test,32)
    salida += "Test loss "+str(score[0])+"\n"
    salida += "Test "+metrica_usada+" "+str(score[1])+"\n"    
    
    #Show text statistics
    print(salida)
    
    return salida

def muestra_predicciones(modelo,x_test,y_test,nombre_imagen,num,imagenes_a_mostrar,az,escala=True):
    """
    Show predictions.
    
    Parameters:
    -----------
    modelo: Model to make predictions
    x_test: x test data
    y_test: y test data
    nombre_imagen: File name image to be saved
    num: Number of images to display
    imagenes_a_mostrar: List of images to display
    az: If True, displays 'num' random predictions. 
        If False and num>0, display the first `num` images
        If False and num=0, display the list of images in 'imagenes_a_mostrar'
    escala: Default True. If False, don´t show color scale
    
    Return:
    -------
    predictions: Model predictions   
    
    """
    predictions = modelo.predict(x_test) #Built predictions
    num_predicciones=len(predictions)       
    mapa_color='gist_stern'   #gist_stern mapa color
    
    if num>0:
        bucle=num   #Display 'num' images
    elif num==0:
        bucle=len(imagenes_a_mostrar)  #Display the list of images in 'imagenes_a_mostrar'
     
    for i in range(bucle):
        plt.gray()       
        f, axes = plt.subplots(1, 8,figsize=(30,30))  
        if (az):
            img_mostrar=randint(0,num_predicciones) #A random image is chosen
        elif (num>0):
            img_mostrar=i #An image from 0 to 'num' is chosen
        elif (num==0):
            img_mostrar=imagenes_a_mostrar[i]  #An imagen i from imagenes_a_mostrar[i] is chosen
        
        im=axes[0].imshow(x_test[img_mostrar])
        axes[0].set_title(str(img_mostrar)+'-Original')
        axes[0].grid(False)         
        if escala==True:
            plt.colorbar(im,ax=axes[0],location='bottom',pad=0.01)  
        
        im=axes[1].imshow(y_test[img_mostrar])
        axes[1].set_title(str(img_mostrar)+'-Target')
        axes[1].grid(False)   
        if escala==True:
             plt.colorbar(im,ax=axes[1],location='bottom',pad=0.01)
        
        im=axes[2].imshow(predictions[img_mostrar,:,:,0])
        axes[2].set_title(str(img_mostrar)+'-Predicc')
        axes[2].grid(False) 
        if escala==True:
             plt.colorbar(im,ax=axes[2],location='bottom',pad=0.01)         
        
        im=axes[3].imshow(np.log(x_test[img_mostrar]))
        axes[3].set_title(str(img_mostrar)+'-Original log ')
        axes[3].grid(False) 
        if escala==True:
             plt.colorbar(im,ax=axes[3],location='bottom',pad=0.01)
       
        
        im=axes[4].imshow(np.log(y_test[img_mostrar]))
        axes[4].set_title(str(img_mostrar)+'-Target log')
        axes[4].grid(False) 
        if escala==True:
             plt.colorbar(im,ax=axes[4],location='bottom',pad=0.01)
       
        
        im=axes[5].imshow(np.log(predictions[img_mostrar,:,:,0]))
        axes[5].set_title(str(img_mostrar)+'-Predic log')
        axes[5].grid(False)    
        if escala==True:
             plt.colorbar(im,ax=axes[5],location='bottom',pad=0.01)               
        
        diferencia=abs(predictions[img_mostrar,:,:,0] - y_test[img_mostrar,:,:])
        im=axes[6].imshow(diferencia)      
        axes[6].set_title(str(img_mostrar)+' Predic - Target')
        axes[6].grid(False)
        if escala==True:
             plt.colorbar(im,ax=axes[6],location='bottom',pad=0.01)
               
        diferencia=abs(predictions[img_mostrar,:,:,0] - y_test[img_mostrar,:,:])
        im=axes[7].imshow(diferencia, cmap = mapa_color,vmin=0, vmax=1)        
        axes[7].set_title(str(img_mostrar)+' Predic - Target')
        axes[7].grid(False)     
        if escala==True:
             plt.colorbar(im,ax=axes[7],location='bottom',pad=0.01)
       
        #Save image file
        plt.savefig(nombre_imagen+"_"+"ejemplos_"+str(img_mostrar)+".png", dpi=100, bbox_inches='tight' )
        plt.tight_layout()
        plt.show()         
    
    return predictions
        

def guarda_predicciones(images,fich_destino,carpeta_dataset):
    """
    Given an array with images, it saves them in a .fits
    The image (from predictions) has 3 dimensions [128,128,1], you must remove the last one: [:,:,0]    
    
    Parameters:
    -----------
    images: Images array
    fich_destino: Destination file 
    carpeta_dataset: Dataset folder
    
    Return: none    
    """    
    
    procesados=0
    num_imagenes= len(images)    #Number of images
   
    print("Saving "+str(num_imagenes)+" images on "+fich_destino)
    for x in range(0,num_imagenes):  #Loop with all images                   
                             
        if (procesados==0): #If it´s the first image, we create the file by overwriting
            hdu = fits.PrimaryHDU(images)
            hdul = fits.HDUList([hdu])            
            fits.writeto(carpeta_dataset+fich_destino,images[x][:,:,0],overwrite=True)   # Add image to dataset
        else:
            fits.append(carpeta_dataset+fich_destino,images[x][:,:,0])                   # Add image to dataset
        procesados +=1   
        
def is_cuda():
    """
    Check for CUDA. Display result.    
    """
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='True'  
    gpus = tf.config.list_physical_devices('GPU')    
    print("Num GPUs Available: ", len(gpus))
    if tf.test.gpu_device_name():        
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))               
    else:
        print("Please install GPU version of TF")  
    
        
def reset_keras():
    """
    The Keras session is cleaned to avoid cache.
    """
    
    # We resets state and model to avoid problems in successive workouts.
    if "modelo" in globals():
        del modelo
    if "history" in globals():
        del history
    
    keras.backend.clear_session()  
    keras.backend.reset_uids()
    tf.random.set_seed(randint(0,100))  
    np.random.seed(randint(0,100))       
    
def obtain_PSNR(MSE, bits):
    '''
    This function returns de average PSNR between the images of a reference images set and a processed images set.
    
    Parameters:
    ----------
    MSE: MSE valur
    bits: Bits number
    
    Return:
    ------
    PSNR: PSNR value calculated
    
    '''
    r = 2**bits-1
    PSNR = 20 * np.log10(r / math.sqrt(MSE))
    
    return PSNR

def obtain_SSIM(X, y):
    '''
    This function returns de average SSIM between the images of a reference images set and a processed images set.
    
    Parameters:
    ----------
    X: X data to compare
    y: y data to compare
    
    Return:
    ------
    SSIM_mean: SSIM value calculated
    
    '''
    
    SSIM_List = []
    for i in range(len(X)):
        img1 = y[i,:,:]
        img2 = X[i,:,:]
        SSIM_List.append(ssim(img1, img2, data_range=img1.max() - img1.min()))
    
    SSIM_mean = np.mean(SSIM_List)
    return SSIM_mean

def obtain_MSE(X, y, bits = 8):
    '''
    This function returns de average MSE between the images of a reference images set and a processed images set.
    
    Parameters:
    ----------
    X: X data to compare
    y: y data to compare
    bits: Default 8
    
    Return:
    ------
    MSE_mean: MSE value calculated
    
    '''
    
    MSE_List = []
    cod_type = 'uint' + str(bits)
    r = 2**bits-1
    
    for i in range(len(X)):        
        image1 = (X[i,:,:]*r).astype(cod_type)        
        image2 = (y[i,:,:]*r).astype(cod_type)        
        MSE_List.append(np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2))             
        
    MSE_mean = np.mean(MSE_List)    
    return MSE_mean