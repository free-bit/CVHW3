#!/usr/bin/env python3

# ---- options ----
DEVICES = {False: 'cpu', True: 'cuda'}
# DEVICE_ID = 'cpu' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
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
        train_set = HW3ImageFolder(root=os.path.join(DATA_ROOT, FOLDERS['train']), device=device)
        # train_set = torch.utils.data.Subset(train_set, indices) # TODO: For sanity check
        loaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_set = HW3ImageFolder(root=os.path.join(DATA_ROOT, FOLDERS['validation']), device=device)
        # val_set = torch.utils.data.Subset(val_set, indices) # TODO: For sanity check
        loaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    if load_test:
        test_set = HW3ImageFolder(root=os.path.join(DATA_ROOT, FOLDERS['test']), device=device)
        loaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
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
        self.layers.append(clayer)

        # Apply batch norm if specified
        if ARGS.batchnorm:
            batchnorm = nn.BatchNorm2d(kernel_count)
            self.layers.append(batchnorm)   

        # Do NOT add ReLU if there is single layer
        if (layer_count > 1): 
            relu = nn.ReLU()
            self.layers.append(relu)
        
        # Mid layers
        for i in range(0, layer_count-2): # Total-First-Last
            # Conv Layer: (input image channel: K, output channel: K, square kernel: FxF)
            clayer = nn.Conv2d(in_channels=kernel_count, out_channels=kernel_count, kernel_size=kernel_size, 
                               bias=True, padding=pad)
            self.layers.append(clayer)

            # Apply batch norm if specified
            if ARGS.batchnorm:
                batchnorm = nn.BatchNorm2d(kernel_count)
                self.layers.append(batchnorm)  
                
            # Apply ReLU
            relu = nn.ReLU()
            self.layers.append(relu)

        # Last layer
        if layer_count > 1:
            clayer = nn.Conv2d(in_channels=kernel_count, out_channels=3, kernel_size=kernel_size, 
                            bias=True, padding=pad)
            self.layers.append(clayer)

            # Apply batch norm if specified
            if ARGS.batchnorm:
                batchnorm = nn.BatchNorm2d(3)
                self.layers.append(batchnorm)  

        # Add tanh layer if specified
        if ARGS.tanh:
            tanh = nn.Tanh()
            self.layers.append(tanh)

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
    epoch = kwargs.get('epoch', 1)
    save = kwargs.get('save', False)
    type = kwargs.get('type', "validation")
    eval = kwargs.get('eval', False)
    visualize = kwargs.get('visualize', False)
    info_type = "Testing" if type == "test" else "Validating"
    all_preds = torch.Tensor() if save else None
    net.eval()
    with torch.no_grad():
        # Keep validation loss for the entire epoch
        val_loss = AverageMeter()
        # Keep validation loss for each n iteration
        iter_loss = AverageMeter()
        acc = None

        # Print net loss
        print_freq = 25

        # already_saved_images = False
        for iteri, (inputs, targets) in enumerate(test_loader, 1):
            # Use GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Run net and record loss
            preds = net(inputs)

            if save:
                all_preds = torch.cat((all_preds, preds.cpu()), dim=0)

            if eval:
                acc = evaluate(preds, targets)

            loss = criterion(preds, targets)
            iter_loss.update(loss.item(), inputs.size(0))
            val_loss.update(loss.item(), inputs.size(0))

            # Save images to file
            # if save_images and not already_saved_images:
                # already_saved_images = True
                # for j in range(min(len(preds), 10)): # save at most 5 images
                # save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                # save_name = 'img-{}-epoch-{}.jpg'.format(i * test_loader.batch_size + j, epoch)
                # to_rgb(inputs[j].cpu(), ab_input=preds[j].detach().cpu(), save_path=save_path, save_name=save_name)

            # Print every print_freq mini-batches if validating/testing
            if (not iteri % print_freq):
                print('- %s: [E: %d, I: %3d] Loss %.4f' % (info_type, epoch, iteri, iter_loss.avg))
                iter_loss.reset()
                if (visualize):
                    visualize_batch(inputs, preds, targets, os.path.join(LOG_DIR, type+'_i{}.png'.format(iteri)))
        if save:
            write_preds(LOG_DIR, all_preds, type)
        return val_loss.avg, acc

def print_lr(optimizer):
    print(next(iter(optimizer.param_groups))['lr'])

# Train for one epoch
def train(train_loader, net, device, criterion, optimizer, epoch):
    net.train()
    # Keep training loss for the entire epoch
    epoch_loss = AverageMeter()
    # Keep training loss for each n iteration
    iter_loss = AverageMeter()
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
        iter_loss.update(loss.item(), inputs.size(0))
        epoch_loss.update(loss.item(), inputs.size(0))

        # Print every print_freq mini-batches
        if (not iteri % print_freq):
            print('- Training: [E: %d, I: %3d] Loss: %.4f' % (epoch, iteri, iter_loss.avg))
            iter_loss.reset()

        if (iteri==0) and VISUALIZE: 
            visualize_batch(inputs, preds, targets)

    # Visualize results periodically
    draw_freq = 10 # TODO: argparse
    if (draw_freq == 1) or (epoch % draw_freq == 0):
        visualize_batch(inputs, preds, targets, os.path.join(LOG_DIR, 'train_e{}.png'.format(epoch)))

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

def override_config(params):
    global ARGS
    config = vars(ARGS)
    ignore_keys = ["help", "gpu", "seed", "pipe"]
    for key, value in params:
        if key not in ignore_keys:
            target_type = type(config[key])
            config[key] = target_type(value)

def show_current_config():
    """Print current parameter configuration"""
    print(get_current_config())

def generate_log_dir():
    global ARGS, LOG_DIR
    config = deepcopy(vars(ARGS))
    ignore_keys = ["help", "gpu", "maxepoch", "valfreq", "factor", "lrpatience", "minlr", "seed",
                   "checkpoint", "earlystop", "wpatience", "mpatience", "tanh", "batchnorm", "pipe"]
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

# LR Scheduling related:
# val_freq: the frequency of performing validation in terms of epochs 
# factor: decaying factor
# lr_patience: wait for lr improment in terms of epochs
# min_lr: minimum possible learning rate
# seed: for pseudorandom initialization

def main():
    global ARGS
    torch.multiprocessing.set_start_method('spawn', force=True)
    ARGS = arg_handler()
    # If required args are parsed properly
    if ARGS:
        # Args
        pipe = ARGS.pipe
        useGPU = ARGS.gpu
        checkpoint_path = ARGS.checkpoint
        seed = ARGS.seed
        max_epoch = ARGS.maxepoch
        batch_size = ARGS.batchsize
        lr = ARGS.learnrate
        val_freq = ARGS.valfreq
        early_stop = ARGS.earlystop
        worse_patience = ARGS.wpatience
        no_max_patience = ARGS.mpatience
        if not early_stop and ("--wpatience" in sys.argv) or ("--mpatience" in sys.argv):
            print("WARNING: Early stop is disabled, patience values will not be used.")

        # factor = ARGS.factor
        # lr_patience = ARGS.lrpatience
        # min_lr = ARGS.minlr

        # Construct network
        torch.manual_seed(seed)
        device = torch.device(DEVICES[useGPU])
        print('Device: ' + str(device))
        net = ColorNet().to(device=device)

        # Start from checkpoint if desired
        start_epoch = 1
        if checkpoint_path:
            check_rxp = r'^(.*)(checkpoint_e(\d+).pt)$'
            found = re.search(check_rxp, checkpoint_path)
            path = found.group(1)
            entire = found.group(2)
            start_epoch = int(re.search(check_rxp, checkpoint_path).group(3)) + 1 # Previous one is already completed
            print('\nLoading the net from file: "{}"...'.format(entire))
            net.load_state_dict(torch.load(checkpoint_path))
            print('Loaded from epoch: {}.'.format(start_epoch - 1))

        show_current_config()

        # Mean Squared Error
        criterion = nn.MSELoss()
        # Optimizer: Stochastic Gradient Descend with initial learning rate
        optimizer = optim.SGD(net.parameters(), lr=lr)
        # Learning rate scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, 
                                                        #  patience=lr_patience, min_lr=min_lr, verbose=True)
        # Determine pipeline execution
        run_full = (pipe == "full")
        run_train = run_full or (pipe == "train")
        run_test = run_full or (pipe == "test")

        # Get loaders as dict
        loaders = get_loaders(batch_size, device, load_train=run_train, load_test=run_test)

        generate_log_dir()

        # Training mode
        if (run_train):
            # Initialization
            train_loader = loaders['train']
            val_loader = loaders['val']
            train_losses = []
            val_losses = []
            accuracies = []
            print('Validation will be done after every {} epoch(s)'.format(val_freq))
            epoch = 1
            worse_count = 0
            no_max_count = 0
            max_accuracy = -1
            try:
                for epoch in range(start_epoch, max_epoch + 1): 
                    #TODO: autoset epoch later
                    print('\nTraining started:')
                    train_loss = train(train_loader, net, device, criterion, optimizer, epoch)
                    print('* Training loss (AVG) for current epoch: %.4f' % train_loss)
                    train_losses.append(train_loss)

                    # Perform validation periodically
                    if (val_freq == 1) or (epoch % val_freq == 0):
                        print('\nValidation started:')
                        val_loss, acc = test(val_loader, net, device, criterion, epoch=epoch, eval=True)
                        val_losses.append(val_loss)
                        accuracies.append(acc)
                        # scheduler.step(val_loss)
                        print('* Validation loss (AVG): %.4f' % val_loss)
                        print('Validation finished!')
                        print('\n* Achieved accuracy (AVG): %.4f' % acc)
                    
                        is_max = acc > max_accuracy
                        is_better = True
                        
                        # When at least 2 epochs passed from the start, perform actual check for better model
                        if epoch >= start_epoch + 1:
                            # Model is better if current accuracy is higher than that of previous state
                            is_better = (accuracies[-1] > accuracies[-2])
                        
                        # Save the model if it is better and maximum up until now
                        if is_max:
                            # Update maximum accuracy
                            max_accuracy = acc
                            if is_better:
                                print('\nSaving current state at the end of epoch %d' % (epoch))
                                torch.save(net.state_dict(), os.path.join(LOG_DIR, 'checkpoint_e{}.pt'.format(epoch)))

                        # If early stopping enabled...
                        if early_stop:
                            # ...and model gets better continue
                            if is_better:
                                worse_count = 0
                            # ...and model gets worse more than patience stop early
                            else:
                                worse_count += 1
                                if (worse_count > worse_patience):
                                    print("\nPerformance got worse for more than {} epoch(s), terminating at epoch: {}...\n"
                                            .format(worse_patience, epoch))
                                    break
                            # ...and model achieves max accuracy continue
                            if is_max:
                                no_max_for = 0
                            # ...and model can not achieve max accuracy more than patience stop early
                            else:
                                no_max_for += 1
                                if (no_max_for > no_max_patience):
                                    print("No max seen for more than {} epoch(s), terminating at epoch: {}...\n"
                                        .format(no_max_patience, epoch))
                                    break
                         
            except KeyboardInterrupt:
                print("\nKeyboard interrupt, stoping execution...\n")
                
            finally:
                print('Training finished!')
                print('Saving training data...')
                draw_train_val_plots(train_losses, val_losses, path=LOG_DIR, show=False)
                draw_accuracy_plot(accuracies, len(train_losses), path=LOG_DIR, show=False)
                stats = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'accuracies': accuracies,
                    'total_epoch': epoch,
                    'max_accuracy': np.max(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_loss': np.max(val_losses),
                    'min_loss': np.min(val_losses),
                    'best_accuracy_epoch': np.argmax(accuracies) + 1,
                    'best_loss_epoch': (np.argmin(val_losses) + 1) * val_freq,
                }
                save_stats("stats.txt", stats, path=LOG_DIR)
                print('Saved.')

        elif (run_test):
            test_loader = loaders['test']
            print('Test started.')
            test_loss, acc = test(test_loader, net, device, criterion, eval=True, save=True, type="test")
            print('* Validation loss (AVG): %.4f' % test_loss)
            if acc:
                print('* Achieved accuracy (AVG): %.4f' % acc)
            print('Test finished!')

if __name__ == "__main__":
    main()