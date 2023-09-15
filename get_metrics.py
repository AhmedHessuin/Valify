'''
this function is main for the generating metrics for the model in json format
this will report Precision, Recall  and F1 score for each class
'''
import time
import cv2
import numpy as np
import torch
import glob
import os
import json
######## this is for the torch script run ##########
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
def get_images(image_path:str="")->list[str]:
    '''
    this function get the images paths in the directory under the image_path inpt
    :param image_path: path to the folder contain images, dtype=str
    :return:
    files: list of images paths, dtype=list[str]
    '''
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(image_path, '*.{}'.format(ext))))
    return files

device='cpu'
imgs_path="Dataset/Dev"
image_shape=32
model =torch.jit.load("checkpoints/model_last.ts")
model.to(device)
torch.set_flush_denormal(True)
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
TP={}
FP={}
FN={}
for key in rev_embedding_vector:
    TP[rev_embedding_vector[key]]=0
    FP[rev_embedding_vector[key]]=0
    FN[rev_embedding_vector[key]]=0

for dir in os.listdir(imgs_path):
    for img_path in get_images(imgs_path+f"/{dir}"):
        im = cv2.imread(img_path)
        im=cv2.resize(im,dsize=(image_shape,image_shape))
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        im = np.array(im)
        im = torch.from_numpy((im[None, ...])).to("cpu").float()
        with torch.jit.optimized_execution(False):
            with torch.no_grad():
                t1=time.time()
                pred= model(im)
                im = cv2.imread(img_path)
                # print(rev_embedding_vector[np.argmax(np.array(pred),axis=-1)[0]])
                if rev_embedding_vector[np.argmax(np.array(pred),axis=-1)[0]]==dir:
                    TP[dir]+=1
                else:
                    FP[rev_embedding_vector[np.argmax(np.array(pred),axis=-1)[0]]]+=1
                    FN[dir]+=1

report={}
for key in TP.keys():
    print(key)
    P=TP[key]/(TP[key]+FP[key])
    R=TP[key]/(TP[key]+FN[key])
    F =(2*P*R)/(P+R)
    print(P,R,F)
    report[key]={"Precision":P,"Recall":R,"F1":F}

with open("Metrics.json", "w",encoding="utf-8") as outfile:
    json.dump(report, outfile,ensure_ascii=False)


