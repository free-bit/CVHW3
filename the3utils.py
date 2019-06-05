# These are utility functions / classes that you probably dont need to alter.

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from utils import read_image

norm_vfunc = np.vectorize(lambda x: (2*x)/255 - 1)
denorm_vfunc = np.vectorize(lambda x: (255*x + 255) / 2)

def tensorshow(tensor,cmap=None):
    img = transforms.functional.to_pil_image(tensor/2+0.5)
    if cmap is not None:
        plt.imshow(img,cmap=cmap)
    else:
        plt.imshow(img)

# Disable multiple occurences of the same flag
class UniqueStore(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        if getattr(namespace, self.dest, self.default) is not None:
            parser.error(option_string + " appears several times.")
        setattr(namespace, self.dest, values)

# Custom format for arg Help print
class CustomFormatter(argparse.HelpFormatter):
    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=100, # Modified
                 width=100):
        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                parts.extend(action.option_strings)
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append('%s' % option_string)
                parts[-1] += ' %s' % args_string
            return ', '.join(parts)

def check_positive_int(value):
    try:
        value = int(value)
        assert (value > 0)
    except Exception as e:
        raise argparse.ArgumentTypeError("Positive integer is expected but got: {}".format(value))
    return value

def check_positive_float(value):
    try:
        value = float(value)
        assert (value > 0)
    except Exception as e:
        raise argparse.ArgumentTypeError("Positive float is expected but got: {}".format(value))
    return value

def check_non_negative(value):
    try:
        value = int(value)
        assert (value >= 0)
    except Exception as e:
        raise argparse.ArgumentTypeError("Non negative integer is expected but got: {}".format(value))
    return value


# def check_epoch(value):
#     try:
#         value = int(value)
#         assert (value > 0) and (value < 100)
#     except Exception as e:
#         raise argparse.ArgumentTypeError("Positive integer less than 100 is expected but got: {}".format(value))
#     return value  

def check_lr(value):
    try:
        value = float(value)
        assert (value >= 0.0001) and (value <= 0.1)
    except Exception as e:
        raise argparse.ArgumentTypeError("Float between 0.0001 & 0.1 is expected but got: {}".format(value))
    return value

# Handles cmd args
def arg_handler():
    parser = argparse.ArgumentParser(description='Image Colorization with PyTorch', 
                                     formatter_class=CustomFormatter, 
                                     add_help=False)
    # Optional flags
    parser.add_argument("-h", "--help", help="Help message", action="store_true")
    parser.add_argument("--gpu", help="Use GPU (default: False)", action="store_true", default=False)
    parser.add_argument("--maxepoch",  help="Specify max number of epochs for training (default: 100)",
                        type=check_positive_int, default=100, metavar="EPOCH")
    parser.add_argument("--valfreq", help="Specify validation frequency in terms of epochs (default: 1)",
                        type=check_positive_int, default=1, metavar="FREQ")
    # parser.add_argument("--factor", help="Specify learning rate decaying factor (default: 0.1)",
    #                     type=check_positive_int, default=0.1)
    # parser.add_argument("--lrpatience", help="Specify patience for learning rate in terms of epochs (default: 0)",
    #                     type=check_non_negative, default=0, metavar="EPOCHS")
    # parser.add_argument("--minlr", help="Specify minimum possible learning rate (default: 0.0001)",
    #                     type=check_lr, default=0.0001)
    parser.add_argument("--seed", help="Specify seed for pseudorandom initialization (default: 10)",
                        type=int, default=10)
    parser.add_argument("--checkpoint", help="Specify checkpoint for learning to start (default: -)",
                        type=str, default="")
    parser.add_argument("--earlystop", help="Enable early stop (default: False)", action="store_true", default=False)
    parser.add_argument("--epatience", help="Specify patience for early stop in terms of epochs (default: 5)",
                        type=check_non_negative, default=5)
    parser.add_argument("--tanh", help="Add tanh activation at the end (default: False)", 
                        action="store_true", default=False)
    parser.add_argument("--batchnorm", help="Add batch norm layer after each intermediate conv layer (default: False)", 
                        action="store_true", default=False)

    # Required flags
    enable_exec = ("-h" not in sys.argv)
    group = parser.add_argument_group(title='required arguments')
    group.add_argument("-p", "--pipe",  help="Specify pipeline execution mode", type=str,
                       choices=['train', 'test', 'full'], required=enable_exec, action=UniqueStore)
    group.add_argument("-bs", "--batchsize",  help="Specify batch size (e.g. 16)", type=check_positive_int, 
                       metavar="BATCH", required=enable_exec)
    # group.add_argument
    group.add_argument("-cl", "--clayers",  help="Specify number of convolutional layers (e.g. 1, 2, 4)", 
                       type=check_positive_int, metavar="CONV", required=enable_exec)
    group.add_argument("-ks", "--kernelsize",  help="Specify kernel size (e.g. 3, 5)", type=check_positive_int, 
                       metavar="SHAPE", required=enable_exec)
    group.add_argument("-kc", "--kernelcount",  help="Specify number of kernels (e.g. 2, 4, 8)", type=check_positive_int,
                       metavar="COUNT", required=enable_exec)
    group.add_argument("-lr", "--learnrate",  help="Specify learning rate (e.g. in range (0.0001, 0.1))", 
                       type=check_positive_float, metavar="LR", required=enable_exec)

    args = parser.parse_args()

    # Print help if -h is used
    if args.help:
        parser.print_help()
        return

    return args

class HW3ImageFolder(torchvision.datasets.ImageFolder):
    """A version of the ImageFolder dataset class, customized for the super-resolution task"""

    def __init__(self, root, device):
        super(HW3ImageFolder, self).__init__(root, transform=None)
        self.device = device

    def prepimg(self,img):
        return (transforms.functional.to_tensor(img)-0.5)*2 # normalize tensorized image from [0,1] to [-1,+1]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (grayscale_image, color_image) where grayscale_image is the decolorized version of the color_image.
        """
        color_image,_ = super(HW3ImageFolder, self).__getitem__(index) # Image object (PIL)
        grayscale_image = torchvision.transforms.functional.to_grayscale(color_image)
        return self.prepimg(grayscale_image).to(self.device), self.prepimg(color_image).to(self.device)

def visualize_batch(inputs,preds,targets,save_path=''):
    inputs = inputs.cpu()
    preds = preds.cpu()
    targets = targets.cpu()
    plt.clf()
    bs = inputs.shape[0]
    for j in range(bs):
        plt.subplot(3,bs,j+1)
        assert(inputs[j].shape[0]==1)
        tensorshow(inputs[j],cmap='gray')
        plt.subplot(3,bs,bs+j+1)
        tensorshow(preds[j])
        plt.subplot(3,bs,2*bs+j+1)
        tensorshow(targets[j])
    if save_path is not '':
        plt.savefig(save_path)
    else:
        plt.show(block=True)

def save_stats(filename, stats, **kwargs):
    path = kwargs.get('path', '')
    with open(path + filename, "w+") as file:
        for key, value in stats.items():
            file.write("{}: {}\n".format(key, value))

def load_stats(path):
    stats = {}
    regex = r'(\w*): (.*)'
    with open(path, "r") as file:
        lines = "".join(file.readlines())
        params = re.findall(regex, lines)
        for key, value in params:
            stats[key] = eval(value)
        return stats

def draw_train_val_plots(train_losses, val_losses, **kwargs):
    # Epoch lengths
    l_train = len(train_losses)
    l_val = len(val_losses)
    if (not l_train) or (not l_val):
        print("WARNING: No data to plot loss vs epoch!")
        return

    combined = kwargs.get("combined", True)
    save = kwargs.get("save", True)
    show = kwargs.get("show", True)

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Loss vs Epoch Graph')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")

    try:
        # Plot for training
        train_epochs = range(1, l_train+1)
        prepare_plot(ax, train_losses, train_epochs, 'blue', 'train')
        # If separate figures desired
        if not combined:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        # Plot for validation
        val_freq = int(l_train / l_val)
        val_epochs = range(val_freq, l_train+1, val_freq)
        prepare_plot(ax, val_losses, val_epochs, 'orange', 'validation')
    except Exception as e:
        print("WARNING: " + str(e))
        return

    if save:
        path = kwargs.get("path", None)
        if path:
            fig.savefig(path + "/" + 'l_vs_e_plot.png')
        else:
            fig.savefig('l_vs_e_plot.png')
            print("WARNING: Invalid path to save, plot saved under current directory!")
    if show:
        plt.show()

def draw_accuracy_plot(accuracies, train_epoch, **kwargs):
    l_acc = len(accuracies)
    if (not l_acc):
        print("WARNING: No data to plot accuracy vs epoch!")
        return

    save = kwargs.get("save", True)
    show = kwargs.get("show", True)

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Accuracy vs Epoch Graph')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")

    eval_freq = int(train_epoch / l_acc)
    eval_epochs = range(eval_freq, train_epoch+1, eval_freq)

    try:
        prepare_plot(ax, accuracies, eval_epochs, 'red', 'accuracy')
    except Exception as e:
        print("WARNING: " + str(e))
        return

    if save:
        path = kwargs.get("path", None)
        if path:
            fig.savefig(path + "/" + 'a_vs_e_plot.png')
        else:
            fig.savefig('a_vs_e_plot.png')
            print("WARNING: Invalid path to save, plot saved under current directory!")
    if show:
        plt.show()

def prepare_plot(ax, losses, epochs, color, label):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(epochs, losses, "o-", color='tab:'+color, label=label)
    ax.grid(True)
    ax.legend()

def write_preds(path, preds, type):
    # Change the order of the axes
    preds = preds.permute(0, 2, 3, 1).numpy()
    preds = denorm_vfunc(preds)
    filename = "estimations_" + type
    print('\nSaving estimations to file {}...'.format(filename))
    np.save(path + "/" + filename, preds)
    print('Saved!')

def get_file_paths(top_folder, sub_folder, sub_sub_folder):
    """
    Find all files under given path in the form of: 
    top_folder
        - sub_folder: 
            - images
    Returns list of file paths
    """
    file_paths = []
    foldername = top_folder + "/" + sub_folder + "/" + sub_sub_folder + "/"
    images = os.listdir(foldername)
    # images = sorted(images, key=lambda x: int(os.path.splitext(x)[0]))
    image_paths = [foldername + image for image in images]
    return image_paths

def write_image_paths(image_paths):
    with open("img_names.txt", "w+") as file:
        file.write('\n'.join(image_paths))

def evaluate(preds, targets):
    preds, targets = denorm_vfunc(preds.cpu().numpy()), denorm_vfunc(targets.cpu().numpy())
    im_size = np.prod(preds.shape[1:])
    accs_per_image = (np.abs(targets - preds) < 12).sum(axis=(1,2,3)) / im_size
    avg_acc = np.mean(accs_per_image)
    return avg_acc