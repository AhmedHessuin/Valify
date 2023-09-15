'''
this file for creating the model and if run as main will perform FLOPs calculation and model summary
if run as main will require a GPU for the "from torchsummary import summary"
'''
import torch
import torch.nn as nn
from torchsummary import summary
from ptflops import get_model_complexity_info
import torch.nn.functional as F


class MyNetwork(nn.Module):
    '''
    mynetwork is a customer Network this is simple network
    '''

    def __init__(self,number_of_classes:int=29,image_shape:int=32):
        '''
        this is the init for custom pytorch model used for classification
        :param number_of_classes: number of class you expect the model to predict, dtype=int
        :param image_shape: input image shape, dtype=int
        '''
        super(MyNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 8,(3,3),(2,2),(1,1))# output shape with 32 ---> 16
        self.conv2 = nn.Conv2d(8, 16,(3,3),(2,2),(1,1))# output shape with 16 --->8
        self.conv3 = nn.Conv2d(16, 32,(3,3),(2,2),(1,1))# output shape with 8 --->4
        self.fc1 = nn.Linear(32 * (image_shape//8) * (image_shape//8) , 128)  # 4*4 from image dimension
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, number_of_classes)



    def forward(self, x):
        x=x/255 # normalize the color by /255
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=torch.flatten(x, 1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x),dim=1)# the output is softmax this will affect the loss i use
        return x

class LossFunc(nn.Module):
    '''
    customer loss function made in shape as pytorch model 
    this is like a wraper for NLLLoss but with torch.log(model_output) to be and end to end loss
    function 
    '''
    def __init__(self):
        super(LossFunc, self).__init__()
        self.loss=nn.NLLLoss()
        return

    def forward(self,y_pred,y_true) :
        '''
        
        :param y_pred: torch tensor of the model output, dtype=torch.tensor
        :param y_true: torch tensor of the expetected output from the model, dtype=torch.tensor
        :return: 
        '''
        return self.loss(torch.log(y_pred),y_true)

if __name__ == "__main__":
    model = MyNetwork()
    model.to("cuda")
    print("="*10 ,"Model Summary ","="*10)
    summary(model,(3,32,32))
    print("="*10 ,"==============","="*10)

    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

    print("="*10 ,"Model Complex ","="*10)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
