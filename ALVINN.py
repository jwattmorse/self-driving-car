"""
Model for training data

"""
from keras.models import Sequential
from keras.layers import Dense # SHOULD CONFIRM
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from read_data import read_data as rd
from sklearn.model_selection import train_test_split
import numpy as np
from random import randint
from copy import deepcopy

import sys

max_steer = 1.0
min_steer = -1.0

def main ():
    compNN()

def compNN():
    xbuff = np.ndarray(shape=(200,960), dtype=float)
    ybuff = np.ndarray(shape= (200,), dtype=float)
    tot = 0.0
    mean = 0.0

    (x_train,y) =  rd(sys.argv[1])
    #print (y)
    x = x_train.astype('float32')
    x /= 255
    #print (x.shape)
    #print (xbuff.shape)
    
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)


    #print (x_train.shape)
    # For future testing code...not currently relavent
    #x_test = x_test.reshape(x_test.shape[0], 1, 30, 32)
    # sizob = x_train.shape[0]

    # fill buffer with 200 images
    for i in range(200):
        xbuff[i] = x[i]
        ybuff[i] = y[i]
        tot += y[i]
    
    mean = tot / 200
    model = Sequential ()
    model.add(Dense(5, activation = 'relu', input_dim = 960))
    model.add(Dense(30, activation = 'relu'))
    
    # SGD = stochastic gradient decent
    model.compile(loss = 'mean_squared_error', optimizer = 'SGD')
    model.fit(xbuff, generate_steering(ybuff), batch_size = 32, epochs = 40, verbose = 2, shuffle=True)

    # before propagation based on buffer
    for i in range(200, x[0].size):
        #eviIdx
        indx = -1
        val = y[i]
        newTot = tot
        for j in range(200):
            newmean = (tot + ( val - ybuff[j]))/200
            if ( abs(newmean) < abs(mean)):
                print ("changed mean")
                mean = newmean
                indx = j
                newTot = tot + (val-ybuff[j])
        # if cant improve mean, randomly select int to evict
        if indx == -1:
            indx = randint(0,199)
            newTot += (val-ybuff[indx])
            mean = newTot/200
        xbuff[indx] = x[i]
        ybuff[indx] = y[i] 

        model.fit(xbuff, generate_steering(ybuff), batch_size = 32, epochs = 40, verbose = 2, shuffle=True)
    

 


    # if we wanted to check how we did
    #score = model.evaluate(x_test, ytest, batchsize)
    #print score
    model.save('model.h5')

    # Evaluate how well the model learns the training data
    # print(model.evaluate(x_test, y_test, verbose=2))
    
    return model

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


# Right now just removes first element
# need to think about how to implement this
"""
def bufferEvic(buf, t, mean, x, y):
    t += (y-buf[0][1])
    buf[0][0] = x
    buf[0][1] = y
    mean = tot/200

    return buf, t, mean
""" 

if __name__ == "__main__": main()
