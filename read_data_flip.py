### Script to read image data and steering angle into numpy ndarray
from numpy import genfromtxt
from scipy import misc
import numpy as np
from pandas import read_csv
import pandas as pd
def read_data(folder_name):
    my_data = read_csv(folder_name + '/driving_log.csv',usecols = [0,3])
    my_data.columns = ['image','steering']    
    get_img = lambda file_name: img_from_file(folder_name,file_name.rsplit('/', 1)[-1])
    images = np.ndarray(shape=(my_data.shape[0],30,32), dtype=int)
    for i in range(my_data['image'].shape[0]):
        images[i] = get_img(my_data['image'][0])
    #print (my_data['steering'].shape)
    return flipedSet(images,my_data['steering'].as_matrix())

def transform_image(image_array):
    return np.array([misc.imresize(image_array[:,:,-1],(30,32)).flatten().tolist()])
    
def img_from_file(folder_name,file_name):
    raw_img =  misc.imread(folder_name + '/IMG/' + file_name,mode = 'RGB')
    return misc.imresize(raw_img[:,:,-1],(30,32))

def flipedSet(img, steringAngs):
    y = img.shape[0]
    newImages = np.ndarray(shape=(y,30,32), dtype=int)
    nAngs = np.ndarray(shape = steringAngs.shape, dtype = int)
    for x in range(y):
        newImages[x] = np.fliplr(img[x])
        #newImages[x] = newImages[x].flatten()
        nAngs[x] =  - steringAngs[x]

    #print (newImages.shape)
    img = np.concatenate((img, newImages))
    #print (img[0])
    #print (img[y+1])
    steringAngs = np.concatenate((steringAngs, nAngs ))
    #print (img.shape)
    #print(steringAngs.shape)
    flt_img = flatimg(img)
    
    print (flt_img.shape)
    return (flt_img, steringAngs)

def flatimg(imgs):
    newImgs =  np.ndarray(shape=(imgs.shape[0],960), dtype=int)
    for x in range(imgs.shape[0]):
        newImgs[x] = imgs[x].flatten()
    
    return newImgs

if __name__ == "__main__":
    data = read_data('data_5_7_2017_JWM2')
    #print(data)
    #print(data.shape)
    
