'''
this is the main training script for the model
'''
import os
import numpy as np
# this section need to run before torch
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from dataset import ClassificationData,ClassificationData_paths_only,batch_sampler
from model import MyNetwork
from torch.utils.data import DataLoader
import os
from  progress.bar import Bar
from torchsummary import summary
from model import LossFunc
import time
import math
from utils.general_functions import  get_lr


def global_config():
    '''
    this function set deterministic algorithm and the global seed for torch, numpy and the workspace
    Config
    :return:
    '''
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(5)
    np.random.seed(5)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    print("DeterministicReport",torch.are_deterministic_algorithms_enabled())

def train(model, optimizer, train_loader, epoch_num,
          criterion_reg, device,len_of_dataset,writer,global_train_counter):
    '''
    this is the main train loop for 1 epoch
    :param model: pytorch model, dtype=nn.Module
    :param optimizer:pytorch optimizer, dtype=torch.optim.*
    :param train_loader:training data loader, dtype= torch.utils.data.DataLoader
    :param epoch_num: number of this epoch, dtype= int
    :param criterion_reg: the loss for this training, dtype=nn.Module
    :param device:run gpu or cpu, dtype=str
    :param len_of_dataset:number of iterations in a single epoch, dtype=int
    :param writer:summary writer, dtype=torchsummary.summary
    :param global_train_counter: number of iterations the model trained with so far, dtype=int
    :return:
    total_loss/len_of_dataset: the total loss divider by number of iterations this is like the 
    avg loss for the epoch, dtype=float 
    global_train_counter: number of iterations the model trained with so far, dtype=int
    '''
    print(f"Training Epoch {epoch_num}")
    model.train()
    total_loss = 0
    iterator=0
    with Bar('', fill='■', suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds',max=len_of_dataset) as bar:
        for i, (img,label) in enumerate(train_loader):
            if device=="cuda":
                img,label=img.to(device), label.to(device)

            pred= model(img)
            class_loss= criterion_reg(pred,label)
            writer.add_scalar("Loss/train", class_loss, global_train_counter)
            writer.add_images("Data/train",img,global_train_counter)

            global_train_counter+=1

            ####################
            optimizer.zero_grad()
            class_loss.backward()
            optimizer.step()
            bar.next()
            total_loss += class_loss.cpu().detach().numpy()#Sec:%(eta)ds -
            bar.suffix = f'%(index)d/%(max)d - %(percent).1f%% - Time:' \
                         f'%(eta_td)s -' \
                         f'Batch Loss {round(float(class_loss), 4)} - '
            iterator+=1

            if  i ==len_of_dataset-1:
                return total_loss/len_of_dataset,global_train_counter

def validate(model, val_loader, epoch_num, criterion_reg, device,len_of_dataset,writer,global_dev_counter):
    '''
    this is the main valid loop for 1 epoch
    :param model: pytorch model, dtype=nn.Module
    :param val_loader:validate data loader, dtype= torch.utils.data.DataLoader
    :param epoch_num: number of this epoch, dtype= int
    :param criterion_reg: the loss for this training, dtype=nn.Module
    :param device:run gpu or cpu, dtype=str
    :param len_of_dataset:number of iterations in a single epoch, dtype=int
    :param writer:summary writer, dtype=torchsummary.summary
    :param global_train_counter: number of iterations the model trained with so far, dtype=int
    :return:
    total_loss/len_of_dataset: the total loss divider by number of iterations this is like the 
    avg loss for the epoch, dtype=float 
    '''
    print(f"Validate Epoch {epoch_num}")
    model.eval()
    total_loss = 0
    iterator=0
    with Bar('', fill='■', suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds',max=len_of_dataset) as bar:
        for i, (img,label) in enumerate(val_loader):
            if device=="cuda":
                img,label=img.to(device), label.to(device)

            pred= model(img)
            class_loss = criterion_reg(pred,label)

            ####################
            bar.next()
            total_loss += class_loss.cpu().detach().numpy()
            bar.suffix = f'%(index)d/%(max)d - %(percent).1f%% - Time:' \
                         f'%(eta_td)s -' \
                         f'Batch Loss {round(float(class_loss), 4)} - '
            iterator+=1
            if  i ==len_of_dataset-1:
                writer.add_scalar("Loss/dev", total_loss/len_of_dataset, epoch_num)
                writer.add_images("Data/dev", img,epoch_num)

                return total_loss/len_of_dataset


def main():
    global_config()
    writer = SummaryWriter()
    global_train_counter = 0
    ######### Learning Parameters #########
    lr = 1e-3
    continue_training=False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Selected Device",device)
    #######################################
    ######### Checkpoint Config ###########
    checkpoints_file = 'checkpoints'
    os.makedirs(checkpoints_file, exist_ok=True)
    last_checkpoint = f'{checkpoints_file}/model_last.pth'
    #######################################
    ######### Data Set Config ############
    data_dir = "Dataset/Train"
    num_epochs = 10
    batch_size=29*3
    image_shape=32

    train_dataset = ClassificationData(data_dir, image_height=image_shape, image_width=image_shape, augment=True)
    train_dataset_paths = ClassificationData_paths_only(data_dir)
    train_sampler = batch_sampler(train_dataset_paths, batch_size=batch_size,
                                  embedding=train_dataset.get_embedding(),epochs=num_epochs*2)

    train_loader = DataLoader(train_dataset, shuffle=False, num_workers=24, batch_size=batch_size,
                              prefetch_factor=50, sampler=train_sampler)
    start_epoch = 0
    num_of_iterations=math.ceil(train_dataset.get_max_len()/batch_size)
    best_loss=100000
    #######################################
    ########## Validate Config ############
    dev_dir = "Dataset/Dev"
    dev_dataset = ClassificationData(dev_dir, image_height=image_shape,
                                     image_width=image_shape,augment=False)
    dev_loader = DataLoader(dev_dataset, shuffle=True, num_workers=15, batch_size=batch_size,
                            prefetch_factor=50)
    num_of_iterations_dev=math.ceil(dev_dataset.get_max_len()/batch_size)


    ############ Model Config #############
    model = MyNetwork(image_shape=image_shape)
    model.to(device)
    summary(model,(3,image_shape,image_shape))
    loss_function=LossFunc().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)#
    scheduler=lr_scheduler.LinearLR(optimizer,1,0.05,total_iters=num_epochs)
    #######################################
    ########## Continue Training ##########
    if continue_training:
        print("Continue Training ")
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint['state_dic'])
        start_epoch=checkpoint['epoch']+1
        print(f"start training from epoch {checkpoint['epoch']+1}")
    #######################################

    for epoch in range(start_epoch,num_epochs):
        epoch_time=time.time()
        total_loss,global_train_counter=train(model, optimizer, train_loader,
                                              epoch, loss_function, device,
                                              num_of_iterations,writer,global_train_counter)


        total_loss_val=validate(model, dev_loader,
                                epoch, loss_function, device,
                                num_of_iterations_dev,writer,epoch)

        ############ Save Check Point ##########
        checkpoint = {'optim_dic': optimizer.state_dict(),
                      'state_dic': model.state_dict(),
                      'epoch': epoch,
                      'train_loss': total_loss,
                      'train_loss_val': total_loss_val,
                      }
        torch.save(checkpoint, f'{checkpoints_file}/model_last.pth')
        print("\nlast epoch time ", time.time() - epoch_time)

        if total_loss_val < best_loss:
            total_loss_val = best_loss
            checkpoint = {'optim_dic': optimizer.state_dict(),
                          'state_dic': model.state_dict(),
                          'epoch': epoch,
                          'train_loss': total_loss,
                          'val_total_loss':total_loss_val
                          }
            torch.save(checkpoint, f'{checkpoints_file}/class_best.pth')


        ##########################################
        scheduler.step()


        writer.add_scalar("Hyper/lr",get_lr(optimizer) , epoch)
    writer.close()
if __name__ == "__main__":
    main()
