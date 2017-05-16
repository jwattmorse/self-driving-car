"""
Jacob Watt-Morse and Ben Solis-Cohen
5/16/2017
Convolusion Network to Drive a car in 
the udacity self driving car simulator
"""
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D

#from keras.preprocessing.image import ImageDataGenerator
from CNNread_data import read_all_images as rd
from sklearn.model_selection import train_test_split

import numpy as np

import sys

max_steer = 1.0
min_steer = -1.0

def main ():
    compNN()

def compNN():
    
    # returns pictures tupe of nparrays
    # first one is of images as arrays, second one is of steering angles     
    (x_train,y_train) =  rd(sys.argv[1])
    
    # normalizing input
    x_train = x_train.astype('float32')
    x_train /= 255


    """
    Model based on that detailed in the paper
    "End to End Learning for Self-Driving Car"
    Bojarski et al. April 2016 NVIDIA Corporation
    
    Model uses 3 convolution layers of 5x5 size kernals 
    with 2,2 size strides and 2 layers with 3x3 kernals
    The output is then flatten and taken through several
    dense layers.

    Note: Normalizaation layer from paper is handeld by preprosessing
    steps take above
    """
    model = Sequential ()
    model.add(Convolution2D(24, (5,5), strides=(2,2), activation = 'relu', input_shape = (160,320,3)))
    
    model.add(Convolution2D(36, (5,5), strides= (2,2), activation = 'relu'))
    model.add(Convolution2D(48, (5,5), strides = (2,2), activation = 'relu'))
    model.add(Convolution2D(64, (3,3), activation = 'relu'))
    model.add(Convolution2D(64, (3,3), activation = 'relu'))
    
    model.add(Flatten())
    model.add(Dense(1164, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    # output layer
    model.add(Dense(30, activation = 'relu')) 

    # SGD = stochastic gradient decent
    model.compile(loss = 'mean_squared_error', optimizer = 'SGD')

    # Input size for the buffer
    # number of epochs per example determined by testing                          
    buf_size = 210
    model.fit_generator(alvinn_generator(x_train,y_train,buf_size), steps_per_epoch = int(x_train.shape[0]), epochs = 3, verbose = 2)

    # save model for later
    model.save('modelCNN.h5')
    
    return model

# e.g shape = (160,320,3) or (30,32,1)                                      
# Code structure for this method taken from                                            
# https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98                    
import numpy as np
import itertools
def alvinn_generator(x_train, y_train, buf_size):
    # Specify result feature and label shapes                                                      
    # result_feature_shape = [(160,320,3)]
    result_label_shape = []

    #feature_buf_shape = tuple([buf_size] + result_feature_shape)
    label_buf_shape = tuple([buf_size] + result_label_shape)

    image_iterator = iter(transformation_generator(x_train,y_train))
    # Initialize training buffer                                                                   
    xbuff = np.ndarray(shape= (buf_size, 160,320,3), dtype = float)
    ybuff = np.ndarray(label_buf_shape)
    tot = 0.0
    mean = 0.0
    for i in range(buf_size):
        (xbuff[i],ybuff[i]) = next(image_iterator)
        tot += ybuff[i]
    mean = tot / buf_size
    
    # yield the first buffer
    yield (xbuff, generate_steering(ybuff))

    i = 0
    for (image_array,steering_angle) in image_iterator:
        if i == 6:
            # Shuffle and yield the buffers                                               
            p = np.random.permutation(ybuff.shape[0])
            xbuff = xbuff[p]
            ybuff = ybuff[p]
            yield (xbuff,generate_steering(ybuff))
            i = 0
        i += 1
        tot = update_buffer(xbuff,ybuff,tot,image_array,steering_angle)

# Simple update buffer code                                                                  
# Evicts image/steering_angle pair that most reduces the absolute value                    
# of the mean. Returns total to keep track of mean within the buffer          
def update_buffer(xbuff,ybuff,tot,image_array,steering_angle):
    # before propagation based on buffer                                                  
    indx = -1
    newTot = tot
    buf_size = xbuff.shape[0]
    mean = tot/buf_size
    
    # loop that finds value to evict based on one that reduces the mean by the most  
    for j in range(buf_size):
       newmean = (tot + (steering_angle - ybuff[j]))/buf_size
       if abs(newmean) < abs(mean):
           mean = newmean
           indx = j
           newTot = tot + (steering_angle -ybuff[j])

    # if cant improve mean, randomly select int to evict 
    if indx == -1:
        indx = np.random.randint(0,199)
        newTot += (steering_angle-ybuff[indx])
        mean = newTot/buf_size

    # insert new image into the buffer
    xbuff[indx] = image_array
    ybuff[indx] = steering_angle
    return newTot


# Yields images that are reflected and shifted by taking the
# right and left images            
from augment import shift_image, get_new_steering_angle 
def transformation_generator(features,labels):
    # loop generates the original and 5 modified images, yeild each as it is created 
    for (image_arrays,steering_angle) in itertools.cycle(zip(features,labels)):
        center_image = image_arrays[0]
        left_image = image_arrays[1]
        right_image = image_arrays[2]

        # Yield left, right and center with modified steering angles 
        right_steer = get_new_steering_angle(steering_angle,1/3,0)
        left_steer = get_new_steering_angle(steering_angle,-1/3,0)

        yield (center_image, steering_angle)
        yield (left_image, left_steer)
        yield (right_image, right_steer)

        # Yield reflected images with modified steering angles
        # reflects left right and center images 
        center_reflected = np.fliplr(center_image)
        left_reflected = np.fliplr(left_image)
        right_reflected = np.fliplr(right_image)

        yield(center_reflected,-1*steering_angle)
        yield(left_reflected, -1*left_steer)
        yield(right_reflected, -1*right_steer)

        #blur_slope = 1/2
        #shift_range = 1/3
        #s = np.random.uniform(-1*shift_range, shift_range)
        #(out_image,new_steering_angle) = shift_image(center_image,steering_angle,s,blur_slope)
        #out_image = crop_top(out_image,blur_slope,shift_range)

# WHY DO WE HAVE THIS HAHA??
def process_image(image_array, steering_angle):
    return (image_array, steering_angle)

# Given a value between 0 and 29 it calcudates
# the steerign angle in radians that corresponds to 
# the transformation used to create our outout array
def get_steering_angle(idx):
    total_angle = max_steer-min_steer
    frac = idx/30
    return float(frac*total_angle + min_steer)

# given an np.array of steering angles it creates
# an array of length 30 that uses the "hill"     
# technique implemented by ALVINN in which a   
# distribution of weights are placed around 
# the steering angle. Angles correspond to an 
# index whihc represents a discrete distrubtion
# of the steering angle space.         
def generate_steering(angles):
    res = np.empty((len(angles),30))
    for i in range(len(angles)):
        angle = angles[i]
        new_y = [0]*30
        hill = [.1,.32,.61,.89,1,.89,.61,.32,.1]
        y_bin = bin(angle)
        j = y_bin-4

        # cuts off hill if steering angle is near end of indexing range
        for val in hill:
            if j >= 0 and j <30:
                new_y[j] = val
            j += 1
        res[i] = new_y
    return res

# computes the index in the steerign array that is
# representative of the given steering angle      
def bin(angle):
    total_angle = max_steer-min_steer
    angle_frac = (angle - min_steer)/total_angle
    return int(round(angle_frac*29))

if __name__ == "__main__": main()
