"""
should take a precomipled and drive data and updates the model 

"""

from keras.models import Sequential
from keras.models import load_model as lm
from keras.layers import Dense # SHOULD CONFIRM                                                      
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from read_data import read_data as rd
from model import generate_steering
from model import bin
import numpy as np
import sys


max_steer = 1.0
min_steer = -1.0

def main (argv):
    curmodel = lm(argv[0]) 
    datafile = argv[1]
    return update(curmodel, datafile)

def update(mod, data):
    
    (x_train,y_train) =  rd(data)
    y_train = generate_steering(y_train)

    x_train = x_train.astype('float32')
    x_train /= 255

    mod.fit(x_train, y_train, batch_size = 32, epochs = 60, verbose = 2)
    
    mod.save('model.h5')
    
if __name__ == "__main__":
    main(sys.argv[1:])
