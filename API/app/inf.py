import cv2
import numpy as np
import torch
from PIL import Image
import base64
from io import BytesIO
import time

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)


def initialize_model(model_path:str):
    """
    This function initializes the model

    Parameters
    ==========

    model_path : str
        path of the saved model

    Returns
    =======
    model : pytorch model

    """
    model = torch.jit.load(model_path)
    model.to('cpu')
    torch.set_flush_denormal(True)
    return model

def read_base64_image_from_str(string:str)->np.ndarray:
    '''
    this function take string of encoded image of base 64 and change it to numpy
    :param string: string contain encoded image, dtype: str
    :return: numpy array of the encoded image, dtype: ndarray
    '''
    image_as_pil = Image.open(BytesIO(base64.b64decode(string)))
    numpy_image = np.array(image_as_pil)
    numpy_image=cv2.cvtColor(numpy_image,cv2.COLOR_RGB2BGR)
    return numpy_image

def inference_model(im, model):
    '''
    inference with the model an rgb image
    :param im: input image as rgb ( after decoding the string image ), dtype=np.ndarray
    :param model: the model as torch script, jit model
    :return:
    the predicted class, dtype=str 
    the confidence, dtype=str
    '''
    image_shape=32
    rev_embedding_vector = {
        0: 'ا',
        1: 'ب',
        2: 'ت',
        3: 'ث',
        4: 'ج',
        5: 'ح',
        6: 'خ',
        7: 'د',
        8: 'ذ',
        9: 'ر',
        10: 'ز',
        11: 'س',
        12: 'ش',
        13: 'ص',
        14: 'ض',
        15: 'ط',
        16: 'ظ',
        17: 'ع',
        18: 'غ',
        19: 'ف',
        20: 'ق',
        21: 'ك',
        22: 'ل',
        23: 'لا',
        24: 'م',
        25: 'ن',
        26: 'ه',
        27: 'و',
        28: 'ي',

    }
    im=cv2.resize(im,dsize=(image_shape,image_shape))
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    im = np.array(im)
    im = torch.from_numpy((im[None, ...])).to("cpu").float()
    with torch.jit.optimized_execution(False):
        with torch.no_grad():
            pred= model(im)
            index=np.argmax(np.array(pred),axis=-1)[0]
            conf=np.array(pred[0][index])
            return rev_embedding_vector[np.argmax(np.array(pred),axis=-1)[0]],str(conf)
