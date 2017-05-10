from keras.models import Sequential
from keras.models import load_model as lm
from keras.layers import Dense                      
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from read_data import read_data as rd
from read_data import transform_image

from model import generate_steering
from model import bin
from drive import calidx

import numpy as np
import sys


max_steer = 1.0
min_steer = -1.0

def main (argv):
    curmodel = lm(argv[0])
    datafile = argv[1]
    tone = errors (curmodel, datafile)
    print ("Correct Prediciton: ", tone[0])
    print ("Missed to Right: ", tone[1])
    print ("Missed to Left: ", tone[2])
    print ("Standard Error: ", tone[3])
    print ("Squared Error: ", tone[4])


def errors(mod, data):
    mean = 0.0
    runCal = 0.0
    err = 0.0
    serr = 0.0
    numcor = 0
    numr = 0
    numl = 0

    (x_train,y_train) =  rd(data)
    y_train = generate_steering(y_train)
    
    x_train = x_train.astype('float32')
    x_train /= 255
    
    
    l = x_train.shape[0]
    for z in range(0, l):

        q = y_train[z]   
        p = np.array([x_train[z].tolist()])
        
        max_idx = np.argmax(q)
        
        pred_array = mod.predict(p, batch_size=1)
        res = calidx(pred_array[0])
        if ( res == max_idx):
            numcor += 1
        else:
            if (res < max_idx):
                numl += 1
            else:
                numr += 1
            err += abs(res - max_idx)
            serr += (res - max_idx) * (res - max_idx) 

    return ((numcor/l), (numr/l), (numl/l), (err/l), (serr / l))

        

if __name__ == "__main__":
    main(sys.argv[1:])

