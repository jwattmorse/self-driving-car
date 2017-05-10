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
n = 0
mean = 0.0
runCal = 0.0
sd = 0.0


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
        image_array = np.asarray(image)
        image_array_t = transform_image(image_array)
        print(image_array_t)
        print(image_array_t.shape)
        pred_array = model.predict(image_array_t, batch_size=1)
        print (pred_array)
        str_idx = calidx(pred_array[0])
        steering_angle = get_steering_angle(str_idx)
        #trackingData(steering_angle)
        #print (steering_angle)
        #print (mean)
        #print (sd)
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


# using Welford Algorithm to ch k
def trackingData(angle):
    global n
    global mean
    global runCal
    global sd

    n += 1
    delta = angle - mean
    mean += delta/n
    delta2 = angle - mean
    runCal += delta*delta2
    if n < 2:
        sd = 0.0
    else:
        sd = M2 / (n - 1)

#first finds max and then computes center around the max
def calidx(arg):
    max_idx = np.argmax(arg)
    """
    mx = float("-inf")
    max_idx = -1
    for x in range(0,arg.size):
        cur = arg[x]
        if  mx < cur:
            mx = cur
            max_idx = x
    """
    #print (mx)
    #print (max_idx)
    # determine start and end index
    start_i = max_idx - 4
    end_i = max_idx + 5
    if start_i < 0:
        start_i = 0
    if end_i >= 31:
        end_i = 30
    
    small_array = arg[start_i:end_i]
    #print (small_array)
    ret_idx = center_of_mass(small_array)
    #print (ret_idx[0] + float(start_i))
    #print (ret_idx[0])
    return ret_idx[0] + float(start_i)

#function that computes center of mass
def COMJAC(arg):
    total = 0.0
    com = 0.0
    for x in range(0, arg.size):
        total += arg[x]
        com += arg[x]*(x+1)
    
    print (com/total - 1)

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
