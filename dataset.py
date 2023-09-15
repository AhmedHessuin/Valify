'''
this is the main file for the dataset genteration supporting balanced batch
'''
from torch.utils.data import Dataset
import glob
import time
import os
from utils import utils_blindsr as blindsr
import cv2
import imgaug.augmenters as iaa
import numpy as np
import random
from torch.utils.data import DataLoader


#####GENERATOR CORE FUNCTIONS ########
def get_images(image_path:str="")->list[str]:
    '''
    take the path to folder contain images directly under it, images format is
    jpg png jpeg JPG, and return the paths of these images
    :param image_path: path to the containing folder, dtype=str
    :return:
    files: list of paths, dtype=list[str]
    '''
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(image_path, '*.{}'.format(ext))))
    return files


class ClassificationData(Dataset):
    '''
    dataset class for classification support balanced batch this is the main dataset class
    where i does the augmentation and loading the images from paths
    '''
    def __init__(self, data_path:str, image_height:int=50, image_width:int=50, augment:bool=True,check_balanced_batch:bool=False,use_bsrgan:bool=False):
        '''
        this is the init function for the dataset class
        :param data_path: path to the dataset contain classes folder, each class in a folder, dtype=str
        :param image_height: the input height for the data generator, dtype=int
        :param image_width: the input width for the data generator, dtype=int
        :param augment: use augment or no, dtype=bool
        :param check_balanced_batch: check the balanced batch this is like verpose=1, dtype=bool
        :param use_bsrgan: use bsrgan augmentation this is heavy augmentation but the input shape is small, dtype=bool
        '''
        self.check_balanced_batch=check_balanced_batch
        self.use_bsrgan=use_bsrgan
        self.embedding_vector = {
            "ا": 0,
            "ب": 1,
            "ت": 2,
            "ث": 3,
            "ج": 4,
            "ح": 5,
            "خ": 6,
            "د": 7,
            "ذ": 8,
            "ر": 9,
            "ز": 10,
            "س": 11,
            "ش": 12,
            "ص": 13,
            "ض": 14,
            "ط": 15,
            "ظ": 16,
            "ع": 17,
            "غ": 18,
            "ف": 19,
            "ق": 20,
            "ك": 21,
            "ل": 22,
            "لا": 23,
            "م": 24,
            "ن": 25,
            "ه": 26,
            "و": 27,
            "ي": 28,
        }
        self.rev_embedding_vector = {
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
        self.class_iterator = {
            'ا': 0,
            'ب': 0,
            'ت': 0,
            'ث': 0,
            'ج': 0,
            'ح': 0,
            'خ': 0,
            'د': 0,
            'ذ': 0,
            'ر': 0,
            'ز': 0,
            'س': 0,
            'ش': 0,
            'ص': 0,
            'ض': 0,
            'ط': 0,
            'ظ': 0,
            'ع': 0,
            'غ': 0,
            'ف': 0,
            'ق': 0,
            'ك': 0,
            'ل': 0,
            'لا': 0,
            'م': 0,
            'ن': 0,
            'ه': 0,
            'و': 0,
            'ي': 0,

        }
        self.balanced_batch_paths = {}
        self.paths = []
        for key in self.embedding_vector.keys():
            self.balanced_batch_paths[key] = get_images(data_path + f"/{key}")
            self.paths.extend(get_images(data_path + f"/{key}"))
        self.height = image_height
        self.width = image_width
        self.augment = augment
        self.safe_image = "Dataset/ا/0.png"# safe image to load when there is an issue in the pipeline
        self.number_of_classes = 29
        self.iterator = 0

    def __gen_image__(self, path)->np.ndarray:
        '''
        this method is for loading the image and walk with it throw the pipeline (load,resize,augment)
        :param path: path of the image, dtype=str
        :return:
        org_image: image after preprocessing for train and augmentation, dtype=np.ndarray
        '''
        org_image = cv2.imread(path)
        if len(org_image.shape) == 2:
            org_image = cv2.cvtColor(org_image, cv2.COLOR_GRAY2RGB)
        elif len(org_image.shape) == 3:
            if org_image.shape[-1] == 1:
                org_image = cv2.cvtColor(org_image, cv2.COLOR_GRAY2RGB)
            elif org_image.shape[-1] == 3:
                pass
        org_image = cv2.resize(org_image, dsize=(self.width, self.height))

        if self.augment:
            if random.random() > 0.5:
                augments = [
                    # --------------#
                    iaa.WithColorspace(
                        to_colorspace="HSV",
                        from_colorspace="RGB",
                        children=iaa.WithChannels(
                            0,
                            iaa.Add((0, 50))
                        )
                    ),
                    iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                    iaa.UniformColorQuantizationToNBits(nb_bits=(2, 8)),
                    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                    iaa.Grayscale(alpha=(0.5, 1.0)),
                    iaa.Invert(1),
                    iaa.MultiplyHue((0.5, 1.5)),
                    iaa.AddElementwise((-40, 40)),  # new
                    iaa.JpegCompression(compression=(70, 99)),  # new

                    iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5)),
                    iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                       children=iaa.WithChannels(0, iaa.Add((0, 50)))),
                    iaa.SaltAndPepper(0.1),
                ]
                images_x = []
                images_x.append(org_image)
                random_color_augment_lst = [color_indx for color_indx in range(len(augments))]
                random_color_augment = random.choice(random_color_augment_lst)
                augmenter = augments[random_color_augment]
                augmented = augmenter(images=images_x)
                org_image = augmented[0]
            if self.use_bsrgan:
                img_lq, img_hq = blindsr.degradation_bsrgan(org_image, sf=1, lq_patchsize=50)
                org_image = img_lq
            if random.random() > 0.5:
                augs_geo = [
                    iaa.ShearX((-2, 2)),
                    iaa.ShearY((-2, 2)),
                ]
                aug = random.choice(augs_geo)
                aug = aug.to_deterministic()
                org_image = aug(images=[org_image])[0]

        org_image = np.transpose(org_image, (2, 0, 1))
        return org_image

    def __getitem__(self, idx:int)->[np.ndarray,int]:
        '''
        this is the main method that the generator use with the multi threads to load data
        :param idx: the index of the image in the dataset
        :return:
        im: the image, dtype=np.array
        label: the label for this image, dtype=int
        '''
        try:
            im_fn = self.paths[idx]
            im = self.__gen_image__(im_fn)
            class_name = im_fn.split("/")[-2]

            if self.check_balanced_batch:
                os.makedirs(f"/media/res12/30aa9699-51c5-4590-9aa2-decf88416771/Personal/Dataset_check/{class_name}",
                            exist_ok=True)
                os.system(
                    f"cp '{im_fn}'  '/media/res12/30aa9699-51c5-4590-9aa2-decf88416771/Personal/Dataset_check/{class_name}/'")
            label = self.embedding_vector[im_fn.split("/")[-2]]
            return im, label

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("image idx ", idx)
            print("image name ", im_fn)
            print("falling back to the safe image")

            im_fn = self.safe_image
            im = self.__gen_image__(im_fn)
            label = 0
            return im, label

    def __len__(self)->int:
        '''
        this method is used when the dataset generator want to run till the end of the dataset
        this method will return the max number of the training class data
        :return:
        max_value: the number of training images for the most dominate class, dtype=int
        '''
        max_value=0
        for key in self.balanced_batch_paths.keys():
            max_value=max(max_value,len(self.balanced_batch_paths[key]))
        return max_value

    def get_data_len(self)->dict[str:int]:
        '''
        get the total number of training images for each class
        :return:
        total_len: the total number of training images for each class, dtype=dict[str:int]
        '''
        total_len = {}
        for key in self.balanced_batch_paths.keys():
            total_len[key] = len(self.balanced_batch_paths[key])
        return total_len

    def get_max_len(self)->int:
        '''
        this method will return the max number of the training class data
        :return:
        max_value: the number of training images for the most dominate class, dtype=int
        '''
        max_value=0
        for key in self.balanced_batch_paths.keys():
            max_value=max(max_value,len(self.balanced_batch_paths[key]))
        return max_value

    def get_embedding(self)->dict[str:int]:
        '''
        get the embedding dictionary for the labels and classes
        :return:
        self.embedding_vector: dictionary contain the class and it's label like {"A":1,....}
        '''
        return self.embedding_vector


class ClassificationData_paths_only(Dataset):
    '''
    this is the classification dataset that generate paths only this is used for balanced batch
    idx generation to save time instead of make the full pipeline to get the index for the sampler
    '''

    def __init__(self, data_path):
        '''
        :param data_path: path to the dataset contain classes folder, each class in a folder, dtype=str
        '''
        self.embedding_vector = {
            "ا": 0,
            "ب": 1,
            "ت": 2,
            "ث": 3,
            "ج": 4,
            "ح": 5,
            "خ": 6,
            "د": 7,
            "ذ": 8,
            "ر": 9,
            "ز": 10,
            "س": 11,
            "ش": 12,
            "ص": 13,
            "ض": 14,
            "ط": 15,
            "ظ": 16,
            "ع": 17,
            "غ": 18,
            "ف": 19,
            "ق": 20,
            "ك": 21,
            "ل": 22,
            "لا": 23,
            "م": 24,
            "ن": 25,
            "ه": 26,
            "و": 27,
            "ي": 28,
        }
        self.rev_embedding_vector = {
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
        self.class_iterator = {
            'ا': 0,
            'ب': 0,
            'ت': 0,
            'ث': 0,
            'ج': 0,
            'ح': 0,
            'خ': 0,
            'د': 0,
            'ذ': 0,
            'ر': 0,
            'ز': 0,
            'س': 0,
            'ش': 0,
            'ص': 0,
            'ض': 0,
            'ط': 0,
            'ظ': 0,
            'ع': 0,
            'غ': 0,
            'ف': 0,
            'ق': 0,
            'ك': 0,
            'ل': 0,
            'لا': 0,
            'م': 0,
            'ن': 0,
            'ه': 0,
            'و': 0,
            'ي': 0,

        }
        self.balanced_batch_paths = {}
        self.paths = []
        for key in self.embedding_vector.keys():
            self.balanced_batch_paths[key] = get_images(data_path + f"/{key}")
            self.paths.extend(get_images(data_path + f"/{key}"))
        self.safe_image = "Dataset/ا/0.png"
        self.number_of_classes = 29
        self.iterator = 0

    def __getitem__(self, idx:int)->str:
        '''
        this method is for the generator to get the training label but in this Class it's
        used for generating the path of the image, given the idx
        :param idx: the index of the image in the dataset

        :return:
        im_fn: the full path of the image, dtype=str
        '''
        try:
            im_fn = self.paths[idx]
            return im_fn

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("image idx path", idx)
            print("image name ", im_fn)
            print("falling back to the safe image")

            im_fn = self.safe_image
            return im_fn

    def __len__(self)->int:
        '''
        total len of the training dataset
        :return: 
        len(self.paths): total data len, dtype=int
        '''
        return len(self.paths)

    def get_data_len(self)->dict[str:int]:
        '''
        get the total number of training images for each class
        :return:
        total_len: the total number of training images for each class, dtype=dict[str:int]
        '''
        total_len = {}
        for key in self.balanced_batch_paths.keys():
            total_len[key] = len(self.balanced_batch_paths[key])
        return total_len


def batch_sampler(data_source, batch_size:int, epochs:int, embedding:dict[str:int]):
    '''
    this is the balanced batch sampler this is used to create a Sampler for the dataset
    :param data_source: classification datatset for paths only, dtype=ClassificationData 
    :param batch_size: must be the number of classes * number, dtype=int
    :param epochs: numper of epochs expected the generator to yield better use more number
    of epochs than you need for safer run and control the number of epochs in the main train 
    loop, dtype=int
    :param embedding: the dictionary that contain the class and it's label, dtype=dict[str:int]
    :return: 
    '''
    class_indices = {}
    for key in embedding.keys():
        class_indices[key] = []

    for ind, path in enumerate(data_source):
        class_indices[path.split("/")[-2]].append(ind)
        if ind+1==len(data_source):break
    for e in range(epochs):
        class_iterator = {}
        for key in class_indices.keys():
            random.shuffle(class_indices[key])
            class_iterator[key] = 0

        max_data = max([len(class_indices[key]) for key in embedding.keys()])


        for i in range(0, max_data,batch_size//len(embedding.keys())):# batch_size // len(embedding.keys())):
            total = []
            while len(total)!=batch_size:
                for key in class_indices.keys():
                    iter = class_iterator[key] % len(class_indices[key])
                    class_iterator[key] += 1
                    total.append(class_indices[key][iter])
            for v in total:
                yield v
                total=[]

if __name__ == "__main__":
    data_dir = "Dataset/Dev"
    batch_size=29*4
    train_dataset = ClassificationData(data_dir, image_height=50, image_width=50, augment=True)
    train_dataset_paths = ClassificationData_paths_only(data_dir)
    train_sampler = batch_sampler(train_dataset_paths, batch_size=batch_size,
                                  embedding=train_dataset.get_embedding(),epochs=1)

    train_loader = DataLoader(train_dataset, shuffle=False, num_workers=24, batch_size=batch_size,
                              prefetch_factor=50, sampler=train_sampler)
    print(len(train_dataset))


    print("ok data set main")
    counter = 0
    t1 = time.time()
    for run_idx, (img, label) in enumerate(train_loader):
        print(img.dtype)
        print(label)
        print(run_idx)
        img = img.numpy().transpose(0, 2, 3, 1)  # torch.transpose(pred_geo,(0,2,3,1))
        print(img.shape)
        cv2.imshow("img,",img[0])
        cv2.imshow("img1,",img[1])
        cv2.imshow("img2,",img[2])
        cv2.imshow("img3,",img[3])
        cv2.waitKey()
    print(time.time() - t1)
    print(run_idx)
