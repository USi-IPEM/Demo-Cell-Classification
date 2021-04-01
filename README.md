##### Machine Learning for industrial process control.

In this repository we explore predictive quality control using
machine learning.

- The measurements which where used to train and test
the model can be found in the ```01_Data``` folder,
this folder also outlines the meaning of important
keys, floats and constants.

- Our Preprocessing code is available in ```02_Preprocessing```.

- Finally the folder ```classifier``` contains code to train
and test our models. To reproduce results from the paper run
```python classifier/train_test_classifiers.py```.

###### Dependencies:
Code in this paper has been tested using python 3.7.6, 
numpy 1.19.2, pandas 1.1.3 and sklearn 0.23.2.
Install these using 
```
$ pip install -U scikit-learn
```
For the tensorboard logs to work run:
```
$ pip install -U torch
$ pip install -U tensorboard
```


###### Demo Video:
The industrial demo setting we consider is
shown in the video below:

![Alt Text](demo.gif)

###### Funding:
This project was partly supported by the German and European IKT.NRW
project "ManuBrain".
