# README #

This README would normally document whatever steps are necessary to get your application up and running.

### Overview ###

In this project we hope to use the recently released Udacity self-driving car/behavioral emulation assignment in order to reimplement ALVINN using the keras python package. Once we implement ALVINN, the goal is then to build a self-driving car evaluation suite. This was one of the lacking features of the ALVINN paper and we hope to fill in the gaps. 

Once ALVINN has been thoroughly evaluated, the next stage is to do better! We will research deep learning techniques that will address the shortcomings our evaluation identifies. During this stage, we will design several candidate deep learning networks and/or network features and implement those that perform best. 

The final goal of the project is to learn TensorFlow and implement at least one of the networks we’ve designed in this framework. Once implemented, we will apply our evaluation suite to each of the networks and to ALVINN.

###Useful Resources###
* [Udacity Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
* [Keras Docs](https://keras.io/)
* [Keras Tutorial](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)
* [How to use the simulator](https://medium.com/towards-data-science/introduction-to-udacity-self-driving-car-simulator-4d78198d301d)
* [Download Unity](https://store.unity.com/download?ref=personal)
* [Install conda](https://conda.io/miniconda)

### Startup ###

```
#!bash
virtualenv ai_final
source ai_final/bin/activate
pip install keras
pip install tensorflow
pip install flask-socketio
pip install eventlet
pip install pillow
pip install h5py
```

### Cleanup ###

```
#!bash
deactivate
```