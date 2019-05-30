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

def test(val_loader, model, criterion, epoch=0, save_images=None):
    with torch.no_grad():
        model.eval()

        # Prepare value counters and timers
        batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

        end = time.time()
        already_saved_images = False
        for iteri, data in enumerate(val_loader, 0):
            data_time.update(time.time() - end)
            inputs, targets = data

            # Use GPU
            if ARGS.gpu: 
                inputs, targets = inputs.cuda(), targets.cuda()

            # Run model and record loss
            preds = model(inputs) # throw away class predictions
            loss = criterion(preds, targets)
            losses.update(loss.item(), inputs.size(0))

            # Save images to file
            # if save_images and not already_saved_images:
                # already_saved_images = True
                # for j in range(min(len(preds), 10)): # save at most 5 images
                # save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                # save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                # to_rgb(inputs[j].cpu(), ab_input=preds[j].detach().cpu(), save_path=save_path, save_name=save_name)

            # Record time to do forward passes and save images
            batch_time.update(time.time() - end)
            end = time.time()

            # Print model accuracy -- in the code below, val refers to both value and validation
            if iteri % 25 == 0:
                print('Validate: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    iteri, len(val_loader), batch_time=batch_time, loss=losses))

        return losses.avg

# def set_learning_rate(last_five_losses):
#     std = np.std(last_five_losses)
#     print("Top five deviation:", std)

def print_lr(optimizer):
    print(next(iter(optimizer.param_groups))['lr'])

def train():
    # TODO: Validation frequency to the argparse:
    val_freq = 3
    # TODO: Scheduler params to the argparse:
    factor = 0.1
    patience = 2
    min_lr = 0.0001
    # ---- training code -----
    device = torch.device(DEVICES[ARGS.gpu])
    print('Device: ' + str(device))
    net = ColorNet().to(device=device)
    net.train()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=ARGS.learnrate) # learning rate
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=0.0001) # learning rate scheduler
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
                print('[%d, %5d] network-loss: %.5f' %
                    (epoch + 1, iteri + 1, running_loss / 100))
                running_loss = 0.0
                # print_lr(optimizer)

            if (iteri==0) and VISUALIZE: 
                visualize_batch(inputs,preds,targets)

        print('Saving the model, end of epoch %d' % (epoch+1))
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        torch.save(net.state_dict(), os.path.join(LOG_DIR, 'checkpoint.pt'))
        visualize_batch(inputs,preds,targets,os.path.join(LOG_DIR, 'example.png'))
        # val_freq is the frequency of running tests on validation data in terms of # of epochs 
        if (val_freq == 1) or (epoch+1) % val_freq == 0:
            print("Testing the trained model on the validation set...")
            val_loss = test(val_loader, net, criterion)
            print('Finished validation!')
            print("Average validation loss:", val_loss)

    print('Finished training!')

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