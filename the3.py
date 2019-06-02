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
DATA_ROOT = 'ceng483-s19-hw3-dataset'
FOLDERS = {'train': 'train', 'validation': 'val', 'test': 'test'}

# --- imports ---
from copy import deepcopy
import time

from the3utils import *

# ---- utility functions -----
def get_loaders(batch_size, device, **kwargs):
    load_train = kwargs.get('load_train', False) 
    load_test = kwargs.get('load_test', False)
    loaders = {}
    if load_train:
        # indices = range(100) # TODO: For sanity check
        train_set = HW3ImageFolder(root=os.path.join(DATA_ROOT, 'train'), device=device)
        # train_set = torch.utils.data.Subset(train_set, indices) # TODO: For sanity check
        loaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_set = HW3ImageFolder(root=os.path.join(DATA_ROOT, 'val'), device=device)
        # val_set = torch.utils.data.Subset(val_set, indices) # TODO: For sanity check
        loaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    if load_test:
        test_set = HW3ImageFolder(root=os.path.join(DATA_ROOT, 'test'), device=device)
        loaders['test'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return loaders

def get_pad_count(n, s, f):
    padding = ((n*s)-s-n+f) / 2
    assert padding.is_integer()
    return int(padding)

# CNN
class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        # Adding layers
        self.layers = []
        # Get padding needed
        pad = get_pad_count(80, 1, ARGS.kernelsize)
        layer_count = ARGS.clayers
        kernel_count = ARGS.kernelcount if layer_count > 1 else 3
        kernel_size = ARGS.kernelsize
        # First layer: (input image channel: 1, output channel: K or 3, square kernel: FxF)
        clayer = nn.Conv2d(in_channels=1, out_channels=kernel_count, kernel_size=kernel_size, 
                           bias=True, padding=pad)
        relu = nn.ReLU()
        self.layers.extend([clayer, relu])
        # Mid layers
        for i in range(0, layer_count-2): # Total-First-Last
            # Conv Layer: (input image channel: K, output channel: K, square kernel: FxF)
            clayer = nn.Conv2d(in_channels=kernel_count, out_channels=kernel_count, kernel_size=kernel_size, 
                               bias=True, padding=pad)
            # Apply ReLU
            relu = nn.ReLU()
            self.layers.extend([clayer, relu])
        # Last layer
        if layer_count > 1:
            clayer = nn.Conv2d(in_channels=kernel_count, out_channels=3, kernel_size=kernel_size, 
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

def test(test_loader, net, device, criterion, **kwargs):
    epoch = kwargs.get('epoch', None)
    save = kwargs.get('save', False)
    type = kwargs.get('type', "validation")
    eval = kwargs.get('eval', False)
    all_preds = torch.Tensor() if save or eval else None
    net.eval()
    with torch.no_grad():

        losses = AverageMeter()
        acc = None

        # Print net loss
        print_freq = 25

        # already_saved_images = False
        for iteri, (inputs, targets) in enumerate(test_loader, 1):
            # Use GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Run net and record loss
            preds = net(inputs)
            if save or eval:
                all_preds = torch.cat((all_preds, preds.cpu()), dim=0)
            loss = criterion(preds, targets)
            losses.update(loss.item(), inputs.size(0))

            # Save images to file
            # if save_images and not already_saved_images:
                # already_saved_images = True
                # for j in range(min(len(preds), 10)): # save at most 5 images
                # save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                # save_name = 'img-{}-epoch-{}.jpg'.format(i * test_loader.batch_size + j, epoch)
                # to_rgb(inputs[j].cpu(), ab_input=preds[j].detach().cpu(), save_path=save_path, save_name=save_name)

            # Print every print_freq mini-batches
            if epoch and (not iteri % print_freq):
                print('Validating: [E: %d, I: %3d] Loss %.4f' % (epoch, iteri, loss))
        if save:
            write_preds(LOG_DIR, all_preds, type)
        if eval:
            file_list = get_file_paths(DATA_ROOT, FOLDERS[type], "images")
            acc = evaluate(all_preds, file_list)
        return losses.avg, acc

def print_lr(optimizer):
    print(next(iter(optimizer.param_groups))['lr'])

# Train for one epoch
def train(train_loader, net, device, criterion, optimizer, epoch):
    net.train()
    # Keep average training loss for the entire epoch
    epoch_loss = AverageMeter()
    # Keep average training loss for each n iteration
    losses = AverageMeter()
    # Print net loss
    print_freq = 100
    inputs = None
    preds = None
    targets = None
    for iteri, (inputs, targets) in enumerate(train_loader, 1):        
        inputs, targets = inputs.to(device), targets.to(device)
        # Clear gradients
        optimizer.zero_grad()
        # Forward, backward, SGD step
        preds = net(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))

        if (not iteri % print_freq):    # print every print_freq mini-batches
            print('Training: [E: %d, I: %3d] Loss: %.4f' % (epoch, iteri, losses.avg))
            epoch_loss.update(losses.avg)
            losses.reset()
            # print_lr(optimizer)

        if (iteri==0) and VISUALIZE: 
            visualize_batch(inputs, preds, targets)

    # Visualize results periodically
    draw_freq = 10 # TODO: argparse
    if (epoch % draw_freq == 0):
        visualize_batch(inputs, preds, targets, os.path.join(LOG_DIR, 'example_e{}.png'.format(epoch)))

    # Return the average loss for this epoch
    return epoch_loss.avg

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

def generate_log_dir():
    global ARGS, LOG_DIR
    config = deepcopy(vars(ARGS))
    ignore_keys = ["help", "gpu", "maxepoch", "valfreq", "factor", "lrpatience", "minlr", "seed", "pipe"]
    for key in ignore_keys:
        config.pop(key, None)
    rest = ""
    for item, key in config.items():
        rest += "_{}{}".format(item, key)
    LOG_DIR = LOG_DIR + rest + "/"
    config_file = LOG_DIR + "config.txt"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    with open(config_file, "w+") as file:
        file.write("Command used to run: " + " ".join(sys.argv) + "\n")
        file.write(get_current_config())

def main():
    global ARGS
    torch.multiprocessing.set_start_method('spawn', force=True)
    ARGS = arg_handler()
    # If required args are parsed properly
    if ARGS:
        # Main args
        pipe = ARGS.pipe
        useGPU = ARGS.gpu
        max_epoch = ARGS.maxepoch
        batch_size = ARGS.batchsize
        lr = ARGS.learnrate

        # Side args
        # val_freq: the frequency of performing validation in terms of epochs 
        # factor: decaying factoe
        # lr_patience: wait for lr improment in terms of epochs
        # min_lr: minimum possible learning rate
        # seed: for pseudorandom initialization
        val_freq = ARGS.valfreq
        # factor = ARGS.factor
        # lr_patience = ARGS.lrpatience
        # min_lr = ARGS.minlr
        seed = ARGS.seed
        # epoch_patience = 2

        show_current_config()

        # Construct network
        torch.manual_seed(seed)
        device = torch.device(DEVICES[useGPU])
        print('Device: ' + str(device))
        net = ColorNet().to(device=device)
        # Mean Squared Error
        criterion = nn.MSELoss()
        # Optimizer: Stochastic Gradient Descend with initial learning rate
        optimizer = optim.SGD(net.parameters(), lr=lr)
        # Learning rate scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, 
                                                        #  patience=lr_patience, min_lr=min_lr, verbose=True)
        run_full = (pipe == "full")
        run_train = run_full or (pipe == "train")
        run_test = run_full or (pipe == "test")
        # Get loaders as dict
        loaders = get_loaders(ARGS.batchsize, device, load_train=run_train, load_test=run_test)
        if LOAD_CHECKPOINT:
            print('Loading the net from the checkpoint')
            net.load_state_dict(os.path.join(LOG_DIR, 'checkpoint.pt')) #TODO: Update
        # Traning mode
        if (run_train):
            # Initialization
            generate_log_dir()
            train_loader = loaders['train']
            val_loader = loaders['val']
            train_losses = []
            val_losses = []
            accuracies = []
            print('Training started.')
            print('Validation will be done after every {} epoch(s)'.format(val_freq))
            try:
                for epoch in range(1, max_epoch): 
                    #TODO: autoset epoch later
                    train_loss = train(train_loader, net, device, criterion, optimizer, epoch)
                    print('Average train loss for current epoch: %.4f' % train_loss)
                    train_losses.append(train_loss)
                    # Perform validation periodically
                    if (val_freq == 1) or (epoch % val_freq == 0):
                        print('Validation started.')
                        val_loss, acc = test(val_loader, net, device, criterion, epoch=epoch, eval=True)
                        val_losses.append(val_loss)
                        accuracies.append(acc)
                        # scheduler.step(val_loss)
                        print('Average validation loss: %.4f' % val_loss)
                        print('Achieved accuracy: %.4f' % acc)
                        print('Validation finished!')
                    # If more than 2 epochs have passed and current loss is lower than the previous
                    if len(train_losses)<2 or (train_losses[-1] < train_losses[-2]):
                        print('Model improved. Saving current state at the end of epoch %d' % (epoch))
                        torch.save(net.state_dict(), os.path.join(LOG_DIR, 'checkpoint_e{}.pt'.format(epoch)))
            except KeyboardInterrupt:
                print("Keyboard interrupt, stoping execution")
            finally:
                print('Training finished!')
                draw_train_val_plots(train_losses, val_losses, path=LOG_DIR)
                draw_accuracy_plot(accuracies, len(train_losses), path=LOG_DIR)
                # print('Saving training data...')
                # save_stats("data_train.txt", train_losses)
                # save_stats("data_val.txt", val_losses)
                # print('Saved.')

        elif (run_test):
            test_loader = loaders['test']
            print('Test started.')
            test_loss, acc = test(test_loader, net, device, criterion, eval=True, type="test")
            print('Test finished!')

if __name__ == "__main__":
    main()

# TODO: Add Early Stopping feature