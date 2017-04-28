### Script to read image data and steering angle into numpy ndarray
from numpy import genfromtxt
from scipy import misc
import numpy as np
from pandas import read_csv

def read_data(folder_name):
    my_data = read_csv(folder_name + '/driving_log.csv',usecols = [0,1])    
    get_img = lambda file_name: img_from_file(folder_name,file_name.rsplit('/', 1)[-1])
    my_data['image'] = my_data['image'].apply(get_img)
    return (my_data['image'].as_matrix(),my_data['steering'].as_matrix())

def img_from_file(folder_name,file_name):
    raw_img =  misc.imread(folder_name + '/IMG/' + file_name,mode = 'RGB')
    return misc.imresize(raw_img[:,:,-1],(30,32))

if __name__ == "__main__":
    data = read_data('data_4_27_17')
    print(data)
    print(data.shape)
    
