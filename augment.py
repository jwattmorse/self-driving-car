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
    # Note: We need to figure out how to convert pixels to feet!
    # Question: should we crop the sides as well?
    # Definitely crop top and bottom every time so that's fine
    # I think we crop top and bottom and interpolate sides (yup!)
    # -.6 to .6 meters (roughly 1/3 of the image)
    # We could use the size of car for conversion from pixels to feet
    # figure out slope
    slope = 1/4
    s = int(s*image_array.shape[1])    
    new_image = shift(image_array,(0,s,0))
    if s > 0:
        for x in range(s):
            for y in range(int((s-x)*slope),image_array.shape[0]):
                new_image[y][x] = new_image[y-int((s-x)*slope),s]
#    else:
#        xstart = image_array.shape[1] - s        
#        for x in range(xlim,image_array.shape[1]):
#            for y in range(int((s-
                
    return new_image

def rotate_image(image_array,t):
    # -6 to 6 degrees
    return rotate(image_array,t)
def interpolate_image(image_array,s,t):
    pass
def random_augment(image_array,s_range,t_range):
    pass
def crop_top_and_bottom(image_array,t,b):
    pass

#=======================================
# Reflect image and negate steering angle
#======================================

from read_data import img_from_file
from scipy import misc
from PIL import Image
if __name__ == "__main__":
    file_name = sys.argv[1]
    out_file_name = 'augment_test.jpg'
    image_array = misc.imread(file_name,mode = 'RGB')
    out_image = rotate_image(image_array,0)
    out_image = shift_image(out_image,1/8)
    misc.imsave(out_file_name, out_image)
    img = Image.open(out_file_name)
    img.show()

    
