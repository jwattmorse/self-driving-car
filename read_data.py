### Script to read image data and steering angle into numpy ndarray
from numpy import genfromtxt
from scipy import misc
import numpy as np
from pandas import read_csv
import pandas as pd
def read_data(folder_name):
    my_data = read_csv(folder_name + '/driving_log.csv',usecols = [2,3])
    my_data.columns = ['image','steering']
    get_img = lambda file_name: img_from_file(folder_name,file_name.rsplit('/', 1)[-1])
    images = np.ndarray(shape=(my_data.shape[0],960), dtype=int)
    for i in range(my_data['image'].shape[0]):
        images[i] = get_img(my_data['image'][0])
    return (images,my_data['steering'].as_matrix())

def transform_image(image_array):
    return np.array([misc.imresize(image_array[:,:,-1],(30,32)).flatten().tolist()])
    
def img_from_file(folder_name,file_name):
    raw_img =  misc.imread(folder_name + '/IMG/' + file_name,mode = 'RGB')
    return misc.imresize(raw_img[:,:,-1],(30,32)).flatten()

if __name__ == "__main__":
    data = read_data('data_4_27_17')
    print(data)
    print(data.shape)
    
