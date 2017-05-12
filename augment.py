from math import cos,tan,sqrt,sin,asin,radians,degrees
import sys
import numpy as np
#==========================================================
# Calculate steering angle based on shifted/translate image
#==========================================================

# From ALVINN: time used to calculate lookahead distance l (in seconds)
#l_time = 2
#Changed lookahead to 20 ft because this is the lowest r_p can be and we don't want
# negative squareroot!
#tune this

#From udacity buick (in feet)
#tune this
wheelbase = 9.33

# tune this
lens_width = 8 #ft (estimated based on car width)
# r_p = previous turning radius
# d_p = A to T distance
# d = B to T distance
# s = shift in feet
# l = lookahead distance

def calc_r_p(steering_angle):
    if steering_angle == 0:
        #if going straight then turning radius is infinite so return big number
        return 10000000
    return (wheelbase/sin(steering_angle))

def calc_r(steering_angle,s,t):
    # find if we are shifting with or against d_p
    # This determines the sign of s
    # Similar idea for theta
    
    r_p = calc_r_p(steering_angle)
    l = 20
    sign = 1
#    print('s',s)
#    print('l',l)
#    print('r_p',r_p)
    d_p = (abs(r_p) - sqrt(r_p**2-l**2))*(r_p/abs(r_p))
    # ALVINN wrong here again!!
    d = d_p + s
#    print('d_p',d_p)
#    print('d',d)
    return (l**2 + d**2)/(2*d)

def get_new_steering_angle(steering_angle,s,t):
#    print('old', steering_angle)
    # Steering angle passed in range -1 to 1 so multiply by 25 to get degrees
    # Then call randians to get radians
    # Also, we're looking vertically insteal of horizontally so
    # multiply by -1
    steering_angle = -1*radians(steering_angle*25)
    
    # Convert shift to ft (shift given in fraction of image which is 6 ft wide)
    s *= lens_width
    
    r = calc_r(steering_angle,s,t)
#    print('r',r)
    
    angle = asin(wheelbase/r) # in radians
    # Convert to between 1 and -1
    # Convert back to vertical angle by multiplying by -1
    new_steering_angle = -1*degrees(angle)/25
#    print('new',new_steering_angle)
    return new_steering_angle

#==========================
# Shift and translate image
#==========================
from scipy.ndimage.interpolation import shift,rotate

# blur_slope is the slope at which to blur the side for interpolation
def shift_image(image_array,steering_angle,s_orig,blur_slope):
    # Definitely crop top and bottom every time
    # -.6 to .6 meters (roughly 1/3 of the image)

    x_max = image_array.shape[1]
    s = int(s_orig*x_max)
    new_image = shift(image_array,(0,s,0))

    y_max = image_array.shape[0]
    if s > 0:
        for x in range(s):
            for y in range(int((s-x)*blur_slope),y_max):
                new_image[y][x] = new_image[y-int((s-x)*blur_slope),s]
    else:
        s *= -1
        x_start = x_max - s        
        for x in range(x_start,x_max):
            for y in range(int((x-x_start)*blur_slope),y_max):
                new_image[y][x] = new_image[y-int((x-x_start)*blur_slope),x_start-1]
    new_steering_angle = get_new_steering_angle(steering_angle,s_orig,0)
    return (new_image, new_steering_angle)

def crop_top(image_array,blur_slope,shift_range):
    # crop down to blur
    aspect_ratio = image_array.shape[1]/image_array.shape[0]
    t = shift_range*blur_slope*aspect_ratio
    y_max = image_array.shape[0]
    return image_array[int(y_max*t):,:]

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
    #    num_colors = image_array.shape[2]
    num_colors = 1
    if(steering_angle == 0):
        for y in range(0,length):
            for x in range(center - width,center + width + 1):                
                new_image[y_max - y - 1][x+s] = 200 #[200]*num_colors
        return new_image

    slope = 1/tan(steering_angle)
    if(slope > 0):
        for x in range(center,center + int(length*sin(steering_angle)) + 1):
            y = int((x - center)*slope)
            for dx in range(0,width):
                for dy in range(0,width):
                    if y_max - y - 1 + dy < y_max:
                        new_image[y_max- y - 1+dy][x+s+dx] = 256 #[200]*num_colors
    else:        
        slope *= -1
        for x in range(center + int(length*sin(steering_angle)), center+2):            
            y = int((center - x)*slope)
            for dx in range(0,width):
                for dy in range(0,width):
                    if y_max - y - 1 + dy < y_max:
                        new_image[y_max- y - 1+dy][x+s+dx] = 256 #[200]*num_colors

    return new_image

#=======================================
# Test image augmentation functions
#======================================
from read_data import img_from_file,read_data_test,raw_img_from_file
def test_augment(sample_size,shift_range = 1/3):
    blur_slope = 1/2
    (test_images,test_steering_angles) = read_data_test('data_short',shape = (160,320,3),mode = 'colored')
    num_images = test_images.shape[0]
    step = num_images//sample_size
    j = 1
 
    for i in np.random.randint(0,num_images,sample_size):
        print('random image: ',i)
        in_file_name = 'data_short/IMG/center_2017_05_10_22_53_00_685.jpg'
        #image_array = raw_img_from_file('data_short','center_2017_05_10_22_53_00_685.jpg')
        image_array = test_images[i]
        steering_angle = test_steering_angles[i]
        steering_angle = -.15
        mid_file_name = 'mid_file.png'
        out_file_name = 'test_output/augment_test' + str(j) + '.jpg'
        j += 1
        mid_array = deepcopy(image_array)
        mid_array = draw_steering_angle(mid_array,steering_angle,0)
        misc.imsave(mid_file_name, mid_array)
        mid_img = Image.open(mid_file_name)
        mid_img.show()

        # Will need to do this in generator!
        s = 1*shift_range #np.random.uniform(-1*shift_range, shift_range)        
        (out_image,new_steering_angle) = shift_image(image_array,steering_angle,s,blur_slope)
        out_image = crop_top(out_image,blur_slope,shift_range)
        out_image = draw_steering_angle(out_image,new_steering_angle,s)
        misc.imsave(out_file_name, out_image)
        img_out = Image.open(out_file_name)
        img_out.show()
        
def test_left_right(which = 'left'):
    # Note this is the main logic
    if which == 'left':
        image_cols = [1]
        s = -1/4
    else:
        # Right
        s = 1/4
        image_cols = [2]

        
    (test_images,test_steering_angles) = read_data_test('data_short',shape = (160,320,3),image_cols = image_cols, mode = 'colored')
    num_images = test_images.shape[0]
    sample_size = 1
    j = 1
    #in_file_name = sys.argv[1]
    #image_array = misc.imread(in_file_name,mode = 'RGB')
    for i in np.random.randint(0,num_images,sample_size):        
        print('random image: ',i)
        print('which = ',which)
        image_array = test_images[i]
        steering_angle = test_steering_angles[i]
        mid_file_name = 'mid_file.png'
        out_file_name = 'test_output/augment_test' + str(j) + '.jpg'
        j += 1
        mid_array = deepcopy(image_array)
#        mid_array = draw_steering_angle(mid_array,steering_angle,0)
        misc.imsave(mid_file_name, mid_array)
        mid_img = Image.open(mid_file_name)
        mid_img.show()
        out_image = image_array

        # Note: also main logic
        new_steering_angle = get_new_steering_angle(steering_angle,s,0)
        
        out_image = crop_top(out_image,1/16)
#        out_image = draw_steering_angle(out_image,new_steering_angle,s)
        misc.imsave(out_file_name, out_image)
        img_out = Image.open(out_file_name)
        img_out.show()
        
#=======================================
# Reflect image and negate steering angle
#======================================

from scipy import misc
from PIL import Image
from copy import deepcopy
if __name__ == "__main__":
    test_augment(1,1/3)
    #test_left_right('left')
