## Introduction
Reduced order models capitalise on dimensionality reduction which can lead to a computational acceleration of many orders of magnitude resulting in a model that can be efficient in multi-query problems (such as optimisation) and real-time calculations (such as predictive control).

This directory includes Opal-AE, an Python-based, convolutional, and domain-decomposition-capable, auto-encoder part of Opal. This part is implemented to learn an optimal low-dimensional representation. The  low  dimensional  representation  is  in  terms of  coordinates  on  the  expressive  nonlinear  data supporting manifold within an unstructured mesh finite element fluid model.  Auto-encoders implemented in this project include a global convolutional auto-encoder and a domain-decomposition convolutional auto-encoder. We also implemented two different ways to do the structured transformation.  We tested them on flow past a cylinder.

In this repository you can find the tools for reduced order modelling.



## System requirement
Linux and MacOS.

## Installation instructions

To download the repository to your local machine.
```bash
  git clone https://github.com/msc-acse/acse-9-independent-research-project-JiaYe-Mao.git
```
Installing all packages below.
```bash
   pip install numpy
   pip install sklearn 
   pip install keras
   pip install tensorflow
   pip install matplotlib
   pip install vtk
```
Adding environment path for Opal and IC-Ferst
```bash
   export PYTHONPATH='PATH_TO_DIRECTORY/acse-9-independent-research-project-JiaYe-Mao/multifluids_icferst-master/python:$PYTHONPATH'
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:PATH_TO_DIRECTORY/acse-9-independent-research-project-JiaYe-Mao/Opal-modified/spud
   export PATH="PATH_TO_DIRECTORY/acse-9-independent-research-project-JiaYe-Mao/Opal-modified/spud:$PATH"

```



## How to run Opal

Changing the settings for Opal

```
	cd PATH_TO_DIRECTORY/acse-9-independent-research-project-JiaYe-Mao/Opal-modified/tests/fpc_finer
	diamond -s ../../schemes/opal.rng fpc.opal
```

You can leave some settings default, and only modify some important settings. Settings in Opal/opal_operation/nirom/compression/plain_autoencoder is the core setting for this project. For Flow Past A Cylinder problem, field name should be Velocity.  You can follow the instruction in diamond to complete the setting part.

Save the file and run the following code to get the interpolated matrix in the directory.

```
python ../../opal.py fpc.opal
```

Because some problem exist in Linux version Keras (Loss in training will explode after a certain steps), the training have to be done in other OS. We have only tested it on MacOS. Copy the interpolated_matrix files into same directory with ae_training.py on a MacOS computer and run the scipt

```
python ae_training.py  (on MacOS)
```

This step will give you two models, 'encoder_dd.h5' and 'decoder_dd.h5'.

Then copy the models back to Linux system directory (the directory you set as generate_directory in Opal settings).

```
python ../../opal.py fpc.opal
```

After running all the script above, the vtu files will be generated. Use paraview to open and visualize the result.

```
paraview
```



## Dependencies

 The external libraries:

 - numpy >= 1.16.4
 - scipy >= 1.2.2
 - matplotlib >= 3.1.0
 - Keras >= 2.2.4
 - scikit-learn >= 0.20.3
 - tensorflow >= 1.14.0   
 - vtk >= 8.1.2
## Repository Information
* __Opal_master__		- contains Opal (which is the major software I developed) 
* __multifluids_icferst-master__     - IC-Ferst (support to run the Opal)
* __report__		- contains the final report, detailing this project's motivations, software design, analysis, and conclusions 
* __results__		- some pre-runed results for Opal, including the results for domain=1,2,4, and sampling methods 1 and 2. 

## Author and Course Information
__Author:__ Jiaye Mao
__Github:__ Jiaye Mao
__CID:__ 01536315

This project is completed for Imperial College's MSc in Applied Computational Science and Engineering program,
as part of the final course module ACSE9. This project was completed under the supervision of Professor Christopher Pain. 

## License  
Licensed under the MIT [license](https://github.com/msc-acse/acse-9-independent-research-project-Wade003/blob/master/LICENSE)