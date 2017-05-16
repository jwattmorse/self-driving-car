"""       
Jacob Watt-Morse and Ben Solis-Cohen                                      
5/16/2017
Bulk of code provided by Udacity for assignment.Found at: 
https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/README.md
Our code is indicated
Code drives the veichle in the udacity self driving car simulator.
Found here:
https://github.com/udacity/self-driving-car-sim
It also tracks the output of the neural net
"""
import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from scipy.ndimage.measurements import center_of_mass
from scipy import misc

from read_data import transform_image
from model import get_steering_angle

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None



class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        ######################################################################
        #### OUR CODE STARTS HERE
        ######################################################################
        image_array = np.asarray(image)
        # maniupulates image into input for neural network
        image_array_t = transform_image(image_array)
        
        # prediction array created by our model
        pred_array = model.predict(image_array_t, batch_size=1)
        
        # calculates the index to drive
        str_idx = calidx(pred_array[0])
        # gets the steering angle for that index
        steering_angle = get_steering_angle(str_idx)
        
        ######################################################################      
        #### OUR CODE ENDS  HERE                                                     
        ######################################################################
        
        throttle = controller.update(float(speed))        
        send_control(steering_angle, throttle)
        
        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
            temp1 = misc.imresize(image_array,(30,32))
            misc.imsave('{}.jpg'.format(image_filename + '_reduced'), temp1)
            temp2 = misc.imresize(image_array[:,:,-1],(30,32))
            misc.imsave('{}.jpg'.format(image_filename + '_bw'), temp2)
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


######################################################################                       
#### OUR CODE STARTS HERE                                  
######################################################################
# computes the steering index using method detailed in Parmaleu '88
# it finds a smaller array arround the max value in the output array
# (9 indexes long because input index was 9 long, and then computes
# the center of mass around that index
def calidx(arg):
    max_idx = np.argmax(arg)
    start_i = max_idx - 4
    end_i = max_idx + 5
    # edge case if start and final index are out of bounds
    if start_i < 0:
        start_i = 0
    if end_i >= 31:
        end_i = 30
    
    small_array = arg[start_i:end_i]
    # calculates center of mass
    ret_idx = center_of_mass(small_array)

    return ret_idx[0] + float(start_i)
 

######################################################################                       
 #### OUR CODE ENDS  HERE                                            
 ######################################################################  

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

