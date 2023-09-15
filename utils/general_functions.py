'''
this if for general use functions and if run as main will split the data to
train and dev
'''
import os
from PIL import Image
import numpy as np
import base64
from io import BytesIO


def read_base64_image_from_str(string:str)->np.ndarray:
    '''
    this function take string of encoded image of base 64 and change it to numpy
    :param string: string contain encoded image, dtype: str
    :return: numpy array of the encoded image, dtype: ndarray
    '''
    image_as_pil = Image.open(BytesIO(base64.b64decode(string)))
    numpy_image = np.array(image_as_pil)
    return numpy_image
def read_base64_image_from_path(path:str)->np.ndarray:
    '''
    this function take the file path of encoded image of base 64 and change it to numpy
    :param path: string contain path to encoded image file, dtype: str
    :return: numpy array of the encoded image, dtype: ndarray
    '''
    with open(path, 'rb') as image_file:
        string_data = image_file.read()
        numpy_image=read_base64_image_from_str(string_data)
        return  numpy_image
def char_to_hot_vector(char:str,embedding:dict):
    hot_vector_len=len(embedding.keys())
    hot_vector=[0 for _ in range(hot_vector_len)]
    hot_vector[embedding[char]]=1
    return hot_vector
def split_train_dev(path:str):
    classes=os.listdir(path)
    for c in classes:
        for im_path in os.listdir(path+"/"+c)[0:int(len(os.listdir(path+"/"+c))*5/100)]:
            os.system(f"mv '{path+'/'+c+'/'+im_path}' '{path.replace('Train','Dev')+'/'+c+'/'+im_path}'")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
if __name__ == '__main__':
    split_data=False
    if split_data:
        split_train_dev("/media/res12/30aa9699-51c5-4590-9aa2-decf88416771/Personal/Dataset/Train")

