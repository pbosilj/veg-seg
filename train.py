from utils import save_model, load_model, data_stats

import torch
import time

import segnet

from dataset import Crop_Dataset
from dataset.transforms import Normalize, Compose, ToTensor

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

    train_parser.add_argument("-d", "--data-folder", type=str, required=True)
       
    train_parser.add_argument('-m','--model', type=str, required=False, help = "Path to the pre-trained model file.")
    
    train_parser.add_argument("-e", "--epochs", type=int, required = False, default = 10, help = "Number of training epochs.")
    train_parser.add_argument("-pi", "--print-iteration", type=int, required = False, default = 10, help = "How often the current loss is displayed (in iterations).")
    train_parser.add_argument("-pe", "--print-epoch", type=int, required = False, default = 1, help = "How often the current model is saved (in epochs).")
    train_parser.add_argument("-n", "--net-out-name", type=str, required = True, help = "Prefix of the saved model filenames.")
    
    args = vars(train_parser.parse_args())
  
    # Get the dataset statistics:
    
    print("Calculating datasets stats.")
  
    class_weights, d_mean, d_std = data_stats(args['data_folder'])
    
    # Network setup and loading

    print('Setting up the network.')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #net, criterion = net_setup(device, class_weights)
    
    net = segnet.SegNetBasic(in_channels = 4, num_classes = 3, final_batch_norm = False)
    net.to(device)
    
    class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)    
    criterion.to(device)
    
    optimizer = segnet.utils.init_SGD_optimizer(net)

    if args['model']:
        net, optimizer, start_epoch = load_model(args['model'], net, optimizer)

    print("Setting up the training dataset.")
        
    # Set up the dataloader for a normalised dataset:
    
    my_transform = Compose([ToTensor(), Normalize(d_mean, d_std)])   
    train_batch = 4
    
    crops_train_norm = Crop_Dataset(root=args['data_folder'], train=True,
                                     sample_size=(384,512), # un hard-code this
                                     transforms=my_transform)
    trainloader = torch.utils.data.DataLoader(crops_train_norm, batch_size=train_batch,shuffle=True, collate_fn = collate_fn)

    # Model training

    print('Commencing training on device {}.'.format(device))

    start_epoch = 0
    net_out_base = args['net_out_name']
    max_epochs = args['epochs']
    print_epoch = args['print_epoch']
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
