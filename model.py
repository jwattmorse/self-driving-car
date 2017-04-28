"""
Model for training data

"""
from keras.models import Sequential
from keras.layers import Dense # SHOULD CONFIRM
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from read_data import read_data as rd
import numpy as np
import sys

max_steer = .4363
min_steer = -.4363

def main ():
    compNN()

def compNN():

    (x_train,y_train) =  rd(sys.argv[1])
    y_train = generate_steering(y_train)
    y_train = np_utils.to_categorical(y_train,30)
    
    x_train = x_train.astype('float32')
    x_train /= 255

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
    model.fit(x_train, y_train, batch_size = 32, epochs = 15, verbose = 2)

    # if we wanted to check how we did
    #score = model.evaluate(x_test, ytest, batchsize)
    #print score
    print(model)
    model.save('model.h5')
    return model
 
def generate_steering(angles):
    res = []
    for angle in angles:
        """
        new_y = [0]*30
        new_y[bin(angle)] = 1
        res.append(new_y)
        """
        res.append(bin(angle))
    return np.array(res)

def bin(angle):
    total_angle = max_steer-min_steer
    angle_frac = (angle + min_steer)/total_angle
    return int(angle_frac*30//1)

if __name__ == "__main__": main()
