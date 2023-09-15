'''
this file is used to export the torch script model based on
the inpu shape and the .pth file
'''
from model import MyNetwork
import torch

####### config for the torch script ######
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
##########################################
device='cuda' # loading device
model_path="checkpoints/class_best.pth"
imgs_path="Dataset/Dev"
image_shape=32
model = MyNetwork(image_shape=image_shape)
model.to(device)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dic'])
model.to('cpu')# final device 
model.eval()
torch.set_flush_denormal(True)# denormal weights removing for Floating points issue
example_weight = torch.rand(1, 3, image_shape, image_shape)
example_forward_input = torch.rand(1, 3, image_shape, image_shape)
torch.set_flush_denormal(True)# denormal weights removing for Floating points issue
model.eval()
model = torch.jit.script(model, example_forward_input)
model.eval()
model.save("checkpoints/model_last.ts")
