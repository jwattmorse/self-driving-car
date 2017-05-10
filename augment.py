from math import cos,tan,sqrt,sin,asin
import sys

#==========================================================
# Calculate steering angle based on shifted/translate image
#==========================================================

# From ALVINN: time used to calculate lookahead distance l
l_time = 2.5

#From udacity buick (in feet)
wheelbase = 9.33

# r_p = previous turning radius
# d_p = A to T distance
# d = B to T distance
# s = shift in feet
# speed = speed in FEET per SECOND
# l = lookahead distance

def calc_r_p(steering_angle):
    return(wheelbase/sin(steering_angle))

def calc_r(steering_angle,speed,s,t):
    r_p = calc_r_p(steering_angle)
    l = speed*l_time
    d_p = r_p - sqrt(r_p**2-l**2)
    d = cos(t)*(d_p + s + l*tan(t))
    return (l**2 + d**2)/(2*d)

def new_steering_angle(steering_angle,speed,s,t):
    speed *= 1.4666666
    r = calc_r(steering_angle,speed,s,t)
    return asin(wheelbase/r)


#==========================
# Shift and translate image
#==========================
from scipy.ndimage.interpolation import shift,rotate

def shift_image(image_array,s):
    return shift(image_array,(s,0))
def rotate_image(image_array,t):
    pass
def interpolate_image(image_array,s,t):
    pass
def random_augment(image_array,s_range,t_range):
    pass


#=======================================
# Reflect image and negate steering angle
#======================================

from read_data import img_from_file
if __name__ == "__main__":
    file_name = sys.args[1]
    out_file_name = 'augment_test'
    image_array = misc.imread(file_name,mode = 'RGB')
    # Do operation
    image.save('{}.jpg'.format(out_file_name))

    
