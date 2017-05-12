"""
Model for training data

"""
from keras.models import Sequential
from keras.layers import Dense # SHOULD CONFIRM
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from read_data import read_data as rd
from read_data import read_all_images
from sklearn.model_selection import train_test_split
import numpy as np

import sys

max_steer = 1.0
min_steer = -1.0

def main ():
    compNN()

def compNN():

    (x_train,y_train) =  read_all_images(sys.argv[1])
    y = y_train
    x = x_train.astype('float32')
    x /= 255
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    # For future testing code...not currently relavent
    #x_test = x_test.reshape(x_test.shape[0], 1, 30, 32)
    # sizob = x_train.shape[0]

    model = Sequential ()
    model.add(Dense(5, activation = 'relu', input_dim = 960))
    model.add(Dense(30, activation = 'relu'))

    # SGD = stochastic gradient decent
    model.compile(loss = 'mean_squared_error', optimizer = 'SGD')

    # batch size = SIZE OF TRIAING SET but for now 32
    # nb_epochs = 40 from ALVINN '88
    # verbose = 2 gives updatge after each epoch
    # shuffle set to True so that train on new samples each time

    buf_size = 210
    model.fit_generator(alvinn_generator(x_train,y_train,buf_size), steps_per_epoch = int(x_train.shape[0]), epochs = 10,verbose = 2)
#    model.fit(x_train, y_train, batch_size = 32, epochs = 40, verbose = 2, shuffle=True)


    # if we wanted to check how we did
    #score = model.evaluate(x_test, ytest, batchsize)
    #print score
    model.save('model.h5')

    # Evaluate how well the model learns the training data
#    print(model.evaluate(x_test, y_test, verbose=2))    
    return model


# e.g shape = (160,320,3) or (30,32,1)
# Code structure for this method taken from
# https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
import numpy as np
import itertools
def alvinn_generator(x_train, y_train, buf_size):
    # Specify result feature and label shapes
    result_feature_shape = [960]
    result_label_shape = []
    
    feature_buf_shape = tuple([buf_size] + result_feature_shape)
    label_buf_shape = tuple([buf_size] + result_label_shape)
    
    image_iterator = iter(transformation_generator(x_train,y_train))

    # Initialize training buffer
    xbuff = np.ndarray(feature_buf_shape)
    ybuff = np.ndarray(label_buf_shape)
    tot = 0.0
    mean = 0.0
    for i in range(buf_size):
        (xbuff[i],ybuff[i]) = next(image_iterator)
        tot += ybuff[i]
    mean = tot / buf_size

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
# of the mean
def update_buffer(xbuff,ybuff,tot,image_array,steering_angle):
    # before propagation based on buffer
    #eviIdx
    indx = -1
    newTot = tot    
    buf_size = xbuff.shape[0]
    mean = tot/buf_size
    for j in range(buf_size):
       newmean = (tot + (steering_angle - ybuff[j]))/buf_size
       if abs(newmean) < abs(mean):
           mean = newmean
           indx = j
           newTot = tot + (steering_angle -ybuff[j])
    # if cant improve mean, randomly select int to evict
#    indx = -1
    if indx == -1:
        indx = np.random.randint(0,199)
        newTot += (steering_angle-ybuff[indx])
        mean = newTot/buf_size
    xbuff[indx] = image_array
    ybuff[indx] = steering_angle
    return newTot

from augment import get_new_steering_angle
def transformation_generator(features,labels):
    for (image_arrays,steering_angle) in itertools.cycle(zip(features,labels)):
        center_image = image_arrays[0]
        left_image = image_arrays[1]
        right_image = image_arrays[2]
        
        right_steer = get_new_steering_angle(steering_angle,1/4,0)
        left_steer = get_new_steering_angle(steering_angle,-1/4,0)        

        
        # Yield left, right and center
        yield (center_image.flatten(), steering_angle)
        yield (left_image.flatten(), left_steer)
        yield (right_image.flatten(), right_steer)

        center_reflected = np.fliplr(center_image)
        left_reflected = np.fliplr(left_image)
        right_reflected = np.fliplr(right_image)

        yield(center_reflected.flatten(),-1*steering_angle)
        yield(left_reflected.flatten(), -1*left_steer)
        yield(right_reflected.flatten(), -1*right_steer)
        
def process_image(image_array, steering_angle):
    return (image_array, steering_angle)

def get_steering_angle(idx):
    total_angle = max_steer-min_steer
    frac = idx/30
    return float(frac*total_angle + min_steer)

def generate_steering(angles):
    res = np.empty((len(angles),30))
    for i in range(len(angles)):
        angle = angles[i]
        new_y = [0]*30
        hill = [.1,.32,.61,.89,1,.89,.61,.32,.1]
        y_bin = bin(angle)
        j = y_bin-4
        for val in hill:
            if j >= 0 and j <30:
                new_y[j] = val
            j += 1
        res[i] = new_y
    return res
        
def bin(angle):
    total_angle = max_steer-min_steer
    angle_frac = (angle - min_steer)/total_angle
    return int(round(angle_frac*29))

if __name__ == "__main__": main()
