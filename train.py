from utils import save_model, load_model, data_stats

import torch
import time

import segnet

from dataset import Crop_Dataset
from dataset.transforms import Normalize, Compose, ToTensor

import sys
import argparse

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

# Collates into a 4th tensor dimension rather than tuples:
def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

def train(net, optimizer, criterion, dataloader, max_epochs = 50, device='cpu', start_epoch = 0, 
        print_epoch = 5, print_iteration = 20, net_out_base = 'NET'):
    net.train()
    t_start = time.time()
    print('[Time: %s]' % (time.strftime("%H:%M:%S", time.gmtime())))
    running_loss_epoch = 0.0
    print_big_iteration_count = 0
    for epoch in range(start_epoch, max_epochs):
        running_loss_iteration = 0.0
        print_iteration_count = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
                      
            running_loss_epoch += loss.item()
            running_loss_iteration += loss.item()
            print_iteration_count += 1
            print_big_iteration_count += 1
            if i % print_iteration == (print_iteration-1):
                t_current = time.time()
                print('[Time: %s Running: %.5f s] [%d, %5d] loss %.3f' % (time.strftime("%H:%M:%S", time.gmtime()), t_current-t_start, epoch+1, i+1, running_loss_iteration/print_iteration_count))
                running_loss_iteration = 0.0
                print_iteration_count = 0

        if epoch % print_epoch == (print_epoch-1):
            save_path =  './{}_e{}.pt'.format(net_out_base, epoch+1)
            save_model(net = net, optimizer = optimizer, epoch = epoch, iteration = i, PATH = save_path)
            print('[Time: %s Running: %.5f s] [%d]        loss %.3f, model saved to %s' % (time.strftime("%H:%M:%S", time.gmtime()), t_current-t_start, epoch+1, running_loss_epoch/print_big_iteration_count, save_path))
            running_loss_epoch = 0.0
            print_big_iteration_count = 0



def main():
    train_parser = argparse.ArgumentParser()

    train_parser.add_argument("-d", "--data-folder", type=str, required=True, metavar=('PATH'), help="Path to the dataset.")    
    
    train_parser.add_argument("-gt", "--ground-truth", type=str, required = True, choices = ['full', 'partial'], help='Whether full or partial annotations are used in training. This is important for data normalisation (always done from the test set).')
    
    train_parser.add_argument("-s", "--sample-size", nargs=2, type=int, default=(384,512), required=False, help="Input images are divided into samples of dimensions HEIGHTxWIDTH", metavar=('HEIGHT', 'WIDTH'))
    train_parser.add_argument("-spi", "--samples-per-image", nargs=2, type=int, required=False, metavar = ('COL_SAMPLES', 'ROW_SAMPLES'), help="Take at most COL_SAMPLESxROW_SAMPLES from the image")

    
    train_parser.add_argument("-e", "--epochs", type=int, required = False, default = 10, help = "Number of training epochs.")
    train_parser.add_argument("-pi", "--print-iteration", type=int, required = False, default = 10, help = "How often the current loss is displayed (in iterations).")
    train_parser.add_argument("-se", "--save-epoch", type=int, required = False, default = 1, help = "How often the current model is saved (in epochs).")
       
    train_parser.add_argument("-n", "--net-out-name", type=str, required = True, help = "Prefix of the saved model filenames.", metavar=('FILENAME_PREFIX'))
    
    train_parser.add_argument('-m','--model', type=str, required=False, help = "Path to the pre-trained model file.", metavar=('PATH'))   
    train_parser.add_argument('-t', '--training_mode', type=str, choices=['continue', 'finetune'], required = ('--model' in ' '.join(sys.argv)) or ('-m' in ' '.join(sys.argv)),
                            help='Mode in which to continue training.\n' +
                            '`continue` loads the optimiser and continues training with saved settings.\n'+
                            '`finetune` loads the model only and reinitialises the optimiser with finetuning settings.')
    
    args = vars(train_parser.parse_args())
  
    # Get the dataset statistics:
    
    print("Calculating datasets stats (from training set).")
  
    sample_size = tuple(args['sample_size'])
    if not args['samples_per_image']:
        samples_per_image = None
    else:
        samples_per_image = tuple(args['samples_per_image'])
    
    class_weights, d_mean, d_std = data_stats(
                                            args['data_folder'],
                                            partial_truth = (args['ground_truth']=='partial'),
                                            sample_size=sample_size,
                                            samples_per_image = samples_per_image,
                                    )
    
    # Network setup and loading

    print('Setting up the network.')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #net, criterion = net_setup(device, class_weights)
    
    net = segnet.SegNetBasic(in_channels = 4, num_classes = 3, final_batch_norm = False)
    net.to(device)
    
    class_weights.to(device)
    if args['ground_truth'] == 'full':
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)    
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index = Crop_Dataset.get_ignore_index())
    criterion.to(device)
    
    # if training from scratch or continuing training, set optimiser to regular settings

    start_epoch = 0
    if not args['model'] or args['training_mode']=='continue': 
        optimizer = segnet.utils.init_SGD_optimizer(net)

        if args['model']: # equivalent if args['trainig_mode']=='continue'
            net, optimizer, start_epoch = load_model(args['model'], net, optimizer)
        #print(optimizer.state_dict())
    else: #if args['training_mode']='finetune':
        net, _, _ = load_model(args['model'], net)
        optimizer = segnet.utils.init_SGD_optimizer(net, fine_tune=True)
           
    print("Setting up the training dataset.")
        
    # Set up the dataloader for a normalised dataset:
    
    my_transform = Compose([ToTensor(), Normalize(d_mean, d_std)])   
    train_batch = 4
    
    crops_train_norm = Crop_Dataset(root=args['data_folder'], train=True,
                                     sample_size=sample_size,
                                     samples_per_image = samples_per_image,
                                     transforms=my_transform,
                                     partial_truth = (args['ground_truth']=='partial'))
    trainloader = torch.utils.data.DataLoader(crops_train_norm, batch_size=train_batch,shuffle=True, collate_fn = collate_fn)

    # Model training

    print('Commencing training on device {}.'.format(device))

    net_out_base = args['net_out_name']
    max_epochs = args['epochs']
    print_epoch = args['save_epoch']
    print_iteration = args['print_iteration']
    
    train(
        net = net,
        optimizer = optimizer,
        criterion = criterion,
        dataloader = trainloader,
        device = device,
        start_epoch = start_epoch,
        print_epoch = print_epoch,
        max_epochs = max_epochs,
        net_out_base = net_out_base,
        print_iteration = print_iteration
    )

if __name__=="__main__":
    main()
