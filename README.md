### TITLE
Deep Learning as an alternative to deconvolution of galaxy images captured with the space telescope Hubble.

### ABSTRACT
The anomalies and artifacts of the real galaxy image captures cause the presence of noise that hinders the work 
of observation and investigation of astronomers. Poisson noise and the Point Spread Function (PSF) represent 
two typical cases that are usually treated for attenuation or suppression. Advances in Artificial Intelligence have 
enabled the construction of models that can be trained to reconstruct galaxy images, mitigating the aberrations 
inherent in the acquisition process. In this project, real images from the Hubble Space Telescope will be used 
to train three neural network architectures, AEPP2 (based on Autoencoders), U-Net and DIDN, and perform a 
reconstruction of them, eliminating the effect of the Point Spread Function, as would be done with a 
deconvolution algorithm. The tests carried out provide data with which it can be stated that the U-Net network 
model is the one that offers the best results in this context.

###CONTENTS
This is the code developed for the master's thesis of artificial intelligence.

**main**
|--arquitecturas.py Code of definition of neural network architectures.
|--dataset.py Code for dataset management
|--ejecucion.py Code for training, statistics and visualization tasks
|--paper-deep-learning-as-an-alternative-to-the-deconvolution-of-images-of-galaxies-captured-with-the-hubble-space-telescope.pdf 
Scientific article describing the work done

**dataset** 
Folder with dataset files. They can be donwloaded from https://1drv.ms/f/s!Aj2cPpzQoR9fkOgfgkVYXQq2Qw828w?e=2XRlPF

**pretrained_models** 
Folder with pretraining models files. Others can be donwloaded from https://1drv.ms/f/s!Aj2cPpzQoR9fkOgfgkVYXQq2Qw828w?e=2XRlPF

**results** 
Some examples of results.
