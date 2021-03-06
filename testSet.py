"""                                    
Jacob Watt-Morse and Ben Solis-Cohen   
5/16/2017
Function evalutes the accuracy of a self-driving model
it does so by bringing in a data set provided by the user
that has not been used to train the model and computing
how oftens it predictions are the same, left of and right 
off a human steering angle. Also computers the standard
and mean squared error for those predictions
"""

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
    # builds model

    curmodel = lm(argv[0])
    datafile = argv[1]
    tone = errors (curmodel, datafile)
    print ("Correct Prediciton: ", tone[0])
    print ("Missed to Right: ", tone[1])
    print ("Missed to Left: ", tone[2])
    print ("Standard Error: ", tone[3])
    print ("Squared Error: ", tone[4])


# function computes function
def errors(mod, data):
    err = 0.0 # total error
    serr = 0.0 #total squared error
    numcor = 0 # numbmer of times guesses corretly
    numr = 0 # number of times guesses to the right
    numl = 0 # number of times gueeses to the left
    

    (x_train,y_train) =  rd(data)
    y_train = generate_steering(y_train)
    
    x_train = x_train.astype('float32')
    x_train /= 255
    
    
    len = x_train.shape[0] # number of image inputs

    # loop goes over every image, takes a prediction for that
    # image and does a serties of tests to check the above values
    for z in range(0, len):
        sterang = y_train[z]   
        pic = np.array([x_train[z].tolist()])
        
        max_idx = np.argmax(sterang)
        
        # prediction for one image 
        pred_array = mod.predict(pic, batch_size=1)
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

