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
import time

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
    padding = ((n*s)-s-n+f) / 2
    assert padding.is_integer()
    return int(padding)

# ---- ConvNet -----
# TODO: Construct network
class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        # Adding layers
        self.layers = []
        # Get padding needed
        pad = get_pad_count(80, 1, ARGS.kernelsize)
        # Starting conv Layer: (input image channel: 1, output channel: K, square kernel: FxF)
        clayer = nn.Conv2d(in_channels=1, out_channels=ARGS.kernelcount, kernel_size=ARGS.kernelsize, 
                           bias=True, padding=pad)
        relu = nn.ReLU()
        self.layers.extend([clayer, relu])
        for i in range(1, ARGS.clayers-1):
            # Conv Layer: (input image channel: K, output channel: K, square kernel: FxF)
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

class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial''' 
  def __init__(self):
    self.reset()

  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def test(test_loader, net, device, criterion, epoch):
    with torch.no_grad():
        net.eval()

        losses = AverageMeter()

        # already_saved_images = False
        for iteri, data in enumerate(test_loader, 1):
            inputs, targets = data

            # Use GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Run net and record loss
            preds = net(inputs) # throw away class predictions
            loss = criterion(preds, targets)
            losses.update(loss.item(), inputs.size(0))

            correct_cnt = (preds == targets).float().sum()

            # Save images to file
            # if save_images and not already_saved_images:
                # already_saved_images = True
                # for j in range(min(len(preds), 10)): # save at most 5 images
                # save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                # save_name = 'img-{}-epoch-{}.jpg'.format(i * test_loader.batch_size + j, epoch)
                # to_rgb(inputs[j].cpu(), ab_input=preds[j].detach().cpu(), save_path=save_path, save_name=save_name)

            # Print net accuracy -- in the code below, val refers to both value and validation
            print_n = 25
            # Print every print_n mini-batches
            if (not iteri % print_n):
                print('Validating: [E: %d, I: %3d] Loss %.4f' % (epoch, iteri, loss))
        return losses.avg

# def set_learning_rate(last_five_losses):
#     std = np.std(last_five_losses)
#     print("Top five deviation:", std)

def print_lr(optimizer):
    print(next(iter(optimizer.param_groups))['lr'])

def train(train_loader, net, device, criterion, optimizer, epoch):
    net.train()
    losses = AverageMeter()
    # Training loss of the network
    running_loss = 0.0
    for iteri, data in enumerate(train_loader, 1):
        inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.
        
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad() # zero the parameter gradients

        # Forward, backward, SGD step
        preds = net(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))

        # Print loss
        print_n = 100 # feel free to change this constant
        if (not iteri % print_n):    # print every print_n mini-batches
            print('Training: [E: %d, I: %3d] Loss: %.4f' % (epoch, iteri, losses.avg))
            losses.reset()
            # print_lr(optimizer)

        if (iteri==0) and VISUALIZE: 
            visualize_batch(inputs, preds, targets)

    print('Saving the model, end of epoch %d' % (epoch))
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    torch.save(net.state_dict(), os.path.join(LOG_DIR, 'checkpoint.pt'))
    visualize_batch(inputs, preds, targets, os.path.join(LOG_DIR, 'example.png'))

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
        # Construct network
        seed = 5 # TODO: to the argparse later
        torch.manual_seed(seed)
        device = torch.device(DEVICES[ARGS.gpu])
        print('Device: ' + str(device))
        net = ColorNet().to(device=device)
        # Mean Squared Error
        criterion = nn.MSELoss()
        # Optimizer: Stochastic Gradient Descend with initial learning rate
        optimizer = optim.SGD(net.parameters(), lr=ARGS.learnrate)
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                         patience=2, min_lr=0.0001, verbose=True)
        # val_freq is the frequency of running tests on validation data in terms of # of epochs 
        val_freq = 3
        factor = 0.1
        patience = 2
        min_lr = 0.0001
        # TODO: Validation frequency to the argparse:
        # TODO: Scheduler params to the argparse:

        train_loader, val_loader = get_loaders(ARGS.batchsize, device)
        if LOAD_CHECKPOINT:
            print('Loading the net from the checkpoint')
            net.load_state_dict(os.path.join(LOG_DIR, 'checkpoint.pt'))
        # TODO: implement pipeline for train+validation and test
        run_full = (ARGS.pipe == "full")
        run_train = (ARGS.pipe == "train")
        run_test = (ARGS.pipe == "test")
        if (run_train or run_full):
            # Traning mode
            print('Training started.')
            for epoch in range(1, 100): #TODO: autoset epoch later
                train(train_loader, net, device, criterion, optimizer, epoch)
                print('Validation started.')
                val_loss = test(val_loader, net, device, criterion, epoch)
                scheduler.step(val_loss)
                print('Validation loss: %.4f' % val_loss)
                print('Validation finished!')
            print('Training finished!')

        elif (run_test or run_full):
            pass #TODO:

if __name__ == "__main__":
    main()