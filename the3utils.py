# These are utility functions / classes that you probably dont need to alter.

import argparse
import sys

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

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
                 width=200):
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

def check_positive(value):
    try:
        value = int(value)
        assert (value > 0)
    except Exception as e:
        raise argparse.ArgumentTypeError("Positive integer is expected but got: {}".format(value))
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
        assert (value > 0.0001) and (value < 0.1)
    except Exception as e:
        raise argparse.ArgumentTypeError("Float between 0.0001 & 0.1 is expected but got: {}".format(value))
    return value

# Handles cmd args
def arg_handler():
    parser = argparse.ArgumentParser(description='Image Colorization with Torch', 
                                     formatter_class=CustomFormatter, 
                                     add_help=False)
    # Optional flags
    parser.add_argument("-h", "--help", help="Help message", action="store_true")
    parser.add_argument("--gpu", help="Use GPU", action="store_true", default=False)

    # Required flags
    enable_exec = ("-h" not in sys.argv)
    group = parser.add_argument_group(title='required arguments')
    group.add_argument("-bs", "--batchsize",  help="Specify batch size", type=check_positive, 
                       metavar="BATCH", required=enable_exec)
    # group.add_argument("-me", "--maxepoch",  help="Specify max number of epochs", type=check_epoch, 
    #                    metavar="EPOCH", required=enable_exec)
    group.add_argument("-cl", "--clayers",  help="Specify number of convolutional layers (e.g. 1, 2, 4)", 
                       type=check_positive, metavar="CONV", required=enable_exec)
    group.add_argument("-ks", "--kernelsize",  help="Specify kernel size (e.g. 3, 5)", type=check_positive, 
                       metavar="SHAPE", required=enable_exec)
    group.add_argument("-kc", "--kernelcount",  help="Specify number of kernels (e.g. 2, 4, 8)", type=check_positive,
                       metavar="COUNT", required=enable_exec)
    group.add_argument("-lr", "--learnrate",  help="Specify learning rate (e.g. in range (0.0001, 0.1))", 
                       type=check_lr, metavar="RATE", required=enable_exec)

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


