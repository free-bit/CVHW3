#!/usr/bin/env python3

# ---- hyper-parameters ----
# You should tune these hyper-parameters using:
# (i) your reasoning and observations, 
# (ii) by tuning it on the validation set, using the techniques discussed in class.
# You definitely can add more hyper-parameters here.
# batch_size = 16
# max_num_epoch = 100
# hps = {'lr':0.001} # hyperparameters

# ---- options ----
DEVICES = {False: 'cpu', True: 'cuda'}
# DEVICE_ID = 'cpu' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
LOAD_CHECKPOINT = False

# --- imports ---
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from the3utils import *

# ---- utility functions -----
def get_loaders(batch_size,device):
    data_root = 'ceng483-s19-hw3-dataset' 
    train_set = HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader

def get_pad_count(n, s, f):
    return int(((n*s)-s-n+f) / 2)

# ---- ConvNet -----
# TODO: Construct network
class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        # Adding layers
        self.layers = []
        # Get padding needed
        pad = get_pad_count(80, 1, ARGS.kernelsize)
        print(pad)
        # Starting conv Layer: (input image channel: 1, output channel: 1*K, square kernel: FxF)
        clayer = nn.Conv2d(in_channels=1, out_channels=ARGS.kernelcount, kernel_size=ARGS.kernelsize, 
                           bias=True, padding=pad)
        relu = nn.ReLU()
        self.layers.extend([clayer, relu])
        for i in range(1, ARGS.clayers-1):
            # Conv Layer: (input image channel: K, output channel: K*K, square kernel: FxF)
            clayer = nn.Conv2d(in_channels=ARGS.kernelcount, out_channels=ARGS.kernelcount, kernel_size=ARGS.kernelsize, 
                               bias=True, padding=pad)
            # Apply ReLU
            relu = nn.ReLU()
            self.layers.extend([clayer, relu])
        clayer = nn.Conv2d(in_channels=ARGS.kernelcount, out_channels=3, kernel_size=ARGS.kernelsize, 
                           bias=True, padding=pad)
        self.layers.append(clayer)
        # Place layers in a sequence 
        self.layers = nn.Sequential(*self.layers)
        print(self.layers)

    def forward(self, grayscale_image):   
        x = self.layers(grayscale_image)
        return x

def train():
    # ---- training code -----
    device = torch.device(DEVICES[ARGS.gpu])
    print('Device: ' + str(device))
    net = ColorNet().to(device=device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=ARGS.learnrate) # learning rate
    train_loader, val_loader = get_loaders(ARGS.batchsize, device)

    if LOAD_CHECKPOINT:
        print('Loading the model from the checkpoint')
        model.load_state_dict(os.path.join(LOG_DIR, 'checkpoint.pt'))

    print('Training begins:')
    for epoch in range(100):#ARGS.maxepoch):  
        running_loss = 0.0 # training loss of the network
        for iteri, data in enumerate(train_loader, 0):
            inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.

            optimizer.zero_grad() # zero the parameter gradients

            # do forward, backward, SGD step
            preds = net(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            # print loss
            running_loss += loss.item()
            print_n = 100 # feel free to change this constant
            if iteri % print_n == (print_n-1):    # print every print_n mini-batches
                print('[%d, %5d] network-loss: %.3f' %
                    (epoch + 1, iteri + 1, running_loss / 100))
                running_loss = 0.0
                # note: you most probably want to track the progress on the validation set as well (needs to be implemented)

            if (iteri==0) and VISUALIZE: 
                visualize_batch(inputs,preds,targets)

        print('Saving the model, end of epoch %d' % (epoch+1))
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        torch.save(net.state_dict(), os.path.join(LOG_DIR, 'checkpoint.pt'))
        visualize_batch(inputs,preds,targets,os.path.join(LOG_DIR, 'example.png'))

    print('Finished training!')

def test():
    pass

def get_current_config():
    global ARGS
    """Return a string indicating current parameter configuration"""
    config = vars(ARGS)
    message = "\nRunning with the following parameter settings:\n"
    separator = "-" * (len(message)-2) + "\n"
    lines = ""
    for item, key in config.items():
        lines += "- {}: {}\n".format(item, key)
    return (message + separator + lines + separator) 

def show_current_config():
    """Print current parameter configuration"""
    print(get_current_config())

def main():
    global ARGS
    torch.multiprocessing.set_start_method('spawn', force=True)
    ARGS = arg_handler()
    # If required args are parsed properly
    if ARGS:
        show_current_config()
        train()

if __name__ == "__main__":
    main()