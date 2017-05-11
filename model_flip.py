"""
Model for training data

"""
from keras.models import Sequential
from keras.layers import Dense # SHOULD CONFIRM
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from read_data_flip import read_data as rd
from sklearn.model_selection import train_test_split
import numpy as np

import sys

max_steer = 1.0
min_steer = -1.0

def main ():
    compNN()

def compNN():

    (x_train,y_train) =  rd(sys.argv[1])
    y = generate_steering(y_train)
    x = x_train.astype('float32')
    x /= 255
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    print (x_train.shape)
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
    model.fit(x_train, y_train, batch_size = 32, epochs = 40, verbose = 2, shuffle=True)


    # if we wanted to check how we did
    #score = model.evaluate(x_test, ytest, batchsize)
    #print score
    model.save('modelflip.h5')

    # Evaluate how well the model learns the training data
    print(model.evaluate(x_test, y_test, verbose=2))
    
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

if __name__ == "__main__": main()
