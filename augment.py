from math import cos,tan,sqrt,sin,asin,radians
import sys

#==========================================================
# Calculate steering angle based on shifted/translate image
#==========================================================

# From ALVINN: time used to calculate lookahead distance l (in seconds)
#l_time = 2
#Changed lookahead to 20 ft because this is the lowest r_p can be and we don't want
# negative squareroot!
#From udacity buick (in feet)
wheelbase = 9.33
lens_width = 6 #ft (estimated based on car width)
# r_p = previous turning radius
# d_p = A to T distance
# d = B to T distance
# s = shift in feet
# speed = speed in FEET per SECOND
# l = lookahead distance

def calc_r_p(steering_angle):
    if steering_angle == 0:
        #if going straight then turning radius is infinite so return big number
        return 10000000
    return(wheelbase/sin(steering_angle))

def calc_r(steering_angle,speed,s,t):
    # find if we are shifting with or against d_p
    # This determines the sign of s
    # Similar idea for theta
    
    r_p = calc_r_p(steering_angle)
    l = 20
    sign = 1
    print('alpha',steering_angle)
    print('speed',speed)
    print('wheelbase',wheelbase)    
    print('s',s)
    print('theta',t)
    print('l',l)
    print('r_p',r_p)
    d_p = r_p - sqrt(r_p**2-l**2)
    d = cos(t)*(d_p + sign*abs(s) + l*tan(t))
    print('d_p',d_p)
    print('d',d)
    return sqrt((l**2 + d**2)/(4*cos(steering_angle)**2))
#    return (l**2 + d**2)/(2*d)

def get_new_steering_angle(steering_angle,speed,s,t):    
    # Steering angle passed in range -1 to 1 so multiply by 25 to get degrees
    # Then call randians to get radians
    steering_angle = radians(steering_angle*25)

    # Convert speed in mph to ft/sec
    speed *= 1.4666666

    # Convert shift to ft (shift given in fraction of image which is 6 ft wide)
    s *= lens_width
    
    r = calc_r(steering_angle,speed,s,t)
    print('r',r)
    return asin(wheelbase/r)

def convert_steering_to_rads():
    pass

#==========================
# Shift and translate image
#==========================
from scipy.ndimage.interpolation import shift,rotate

def shift_image(image_array,s_orig,speed,steering_angle):
    # Note: We need to figure out how to convert pixels to feet!
    # Question: should we crop the sides as well?
    # Definitely crop top and bottom every time so that's fine
    # I think we crop top and bottom and interpolate sides (yup!)
    # -.6 to .6 meters (roughly 1/3 of the image)
    # We could use the size of car for conversion from pixels to feet
    # figure out slope


    slope = 1/4
    new_image = shift(image_array,(0,s_orig,0))
    x_max = image_array.shape[1]
    y_max = image_array.shape[0]
    s = int(s_orig*x_max)
    if s > 0:
        slope = 1/4
        for x in range(s):
            for y in range(int((s-x)*slope),y_max):
                new_image[y][x] = new_image[y-int((s-x)*slope),s]
    else:
        s *= -1
        x_start = x_max - s        
        for x in range(x_start,x_max):
            for y in range(int((x-x_start)*slope),y_max):
                new_image[y][x] = new_image[y-int((x-x_start)*slope),x_start-1]
    print('old',steering_angle)
    new_steering_angle = get_new_steering_angle(steering_angle,speed,s_orig,0)
    print('new',new_steering_angle)
    return (new_image, new_steering_angle)

def random_augment(image_array,s_range,t_range):
    pass

def crop_top(image_array,t):
    y_max = image_array.shape[0]
    return image_array[int(y_max*t):,:,:]

def draw_steering_angle(image_array,steering_angle,s):
    # Steering angle passed in range -1 to 1 so multiply by 25 to get degrees
    # Then call randians to get radians
    steering_angle = radians(steering_angle*25)
    center = int(image_array.shape[1]/2)
    length = int(1/2*image_array.shape[0])
    width = max(int(1/20*image_array.shape[1]/2),1)
    new_image = image_array
    y_max = image_array.shape[0]
    x_max = image_array.shape[1]
    # s is the shift in fractional form so we multiply by num pixels
    s = int(s*x_max)
    num_colors = image_array.shape[2]
    if(steering_angle == 0):
        for y in range(0,length):
            for x in range(center - width,center + width):                
                new_image[y_max - y - 1][x+s] = [200]*num_colors
        return new_image

    slope = 1/tan(steering_angle)
    if(slope > 0):
        for x in range(center,center + int(length*sin(steering_angle))):
            y = int((x - center)*slope)
            for dx in range(0,width):
                for dy in range(0,width):
                    if y_max - y - 1 + dy < y_max:
                        new_image[y_max- y - 1+dy][x+s+dx] = [200]*num_colors
    else:        
        slope *= -1
        for x in range(center + int(length*sin(steering_angle)), center+1):            
            y = int((center - x)*slope)
            for dx in range(0,width):
                for dy in range(0,width):
                    if y_max - y - 1 + dy < y_max:
                        new_image[y_max- y - 1+dy][x+s+dx] = [200]*num_colors
        
    return new_image

#=======================================
# Reflect image and negate steering angle
#======================================

from read_data import img_from_file
from scipy import misc
from PIL import Image
from copy import deepcopy
if __name__ == "__main__":
    
    in_file_name = sys.argv[1]
    mid_file_name = 'mid_file.jpg'
    out_file_name = 'augment_test.jpg'
    image_array = misc.imread(in_file_name,mode = 'RGB')
    s = -1/8
    steering_angle = .5
    speed = 20
    mid_array = deepcopy(image_array)
    mid_array = draw_steering_angle(mid_array,steering_angle,0)
    mid_img = Image.fromarray(mid_array, 'RGB')    
    mid_img.save(mid_file_name)
    mid_img.show()
    (out_image,new_steering_angle) = shift_image(image_array,s,speed,steering_angle)
    out_image = crop_top(out_image,1/16)
    out_image = draw_steering_angle(out_image,new_steering_angle,s)
    misc.imsave(out_file_name, out_image)
    img_out = Image.open(out_file_name)
    img_out.show()
