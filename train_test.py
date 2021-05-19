import torch
import torchvision
import torch.optim as optim

import time
import math
from itertools import islice

#from segnet.segnet_basic import SegNetBasic
from segnet import SegNetBasic
import segnet
#from crop_datasets import Crop_Dataset
from dataset import Crop_Dataset
from dataset.transforms import Normalize, Compose, Resize, ToTensor

import numpy as np
from skimage import io

import argparse



def save_model(net, optimizer, epoch, iteration, PATH):
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
#        'iteration': iteration
        }, PATH)

def load_model(PATH, net, optimizer, max_iteration=None):
    
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print("Loaded epoch {}".format(epoch+1))

    return net, optimizer, epoch+1 #, iteration
    
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



def test(net, criterion, dataloader, device='cpu', save_images = False, out_image_path = "prediction", truth_image_path="truth"):
    net.eval()
    
    total_loss = 0.0
    
    t_total_duration = 0
    
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            t_start = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            t_duration = time.time() - t_start
            loss = criterion(outputs, labels)
            total_loss += loss
            t_total_duration += t_duration

            print('[Batch size %d, classification time: %.5f] [%d] loss %.3f' % (dataloader.batch_size, t_duration, i, loss))
            if save_images:
                for bi, (predicted_image, truth_image) in enumerate(zip(outputs, labels)):
                    out_image = predicted_image.detach().cpu().numpy()
                    out_image = np.argmax(out_image, axis = 0).astype(dtype='uint8')
                    out_truth = truth_image.detach().cpu().numpy().astype(dtype='uint8')

                    out_image = (out_image*127).astype('uint8')
                    out_truth = (out_truth*127).astype('uint8')

                    io.imsave(fname = './{}_i{}_b{}.png'.format(out_image_path, i, bi), arr = out_image)
                    io.imsave(fname = './{}_i{}_b{}.png'.format(truth_image_path, i, bi), arr = out_truth)
                    
    dlen = len(dataloader.dataset)
    print('[Average classification time: %.5f per sample] average loss %.3f' % (t_total_duration/dlen, total_loss/dlen))            

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data-folder", type=str, required=True)
    #parser.add_argument("-d", "--data-folder", type=str, required=False, default="/home/pbosilj/Data/CA17/carrots_labelled")
    
    mode_parser = parser.add_subparsers(dest = "mode", help = "Train or test model.")
    test_parser = mode_parser.add_parser("test", description = "Train the model.")

    test_parser.add_argument('-m','--model', type=str, required=True, help = "Path to the pre-trained model file.")
      
    train_parser = mode_parser.add_parser("train", description = "Evaluate the model only.")
    
    train_parser.add_argument('-m','--model', type=str, required=False, help = "Path to the pre-trained model file.")
    
    train_parser.add_argument("-e", "--epochs", type=int, required = False, default = 10, help = "Number of training epochs.")
    train_parser.add_argument("-pi", "--print-iteration", type=int, required = False, default = 10, help = "How often the current loss is displayed (in iterations).")
    train_parser.add_argument("-pe", "--print-epoch", type=int, required = False, default = 1, help = "How often the current model is saved (in epochs).")
    train_parser.add_argument("-n", "--net-out-name", type=str, required = True, help = "Prefix of the saved model filenames.")
    
    args = vars(parser.parse_args())
  
    # Get the dataset statistics:
    
    print("Calculating datasets stats.")
  
    carrots_train = Crop_Dataset(root=args['data_folder'], train=True, sample_size=(384,512), transforms=ToTensor())
    
    class_weights = 1.0/carrots_train.get_class_probability()
    print("\tClass weights: {}".format(class_weights))
    
    # Class weights: tensor([ 1.3001, 12.8161,  6.5436])
    #class_weights = [ 1.3001, 12.8161,  6.5436]
    
    single_loader = torch.utils.data.DataLoader(carrots_train, batch_size=len(carrots_train), num_workers=1)
    data, labels = next(iter(single_loader))

    d_mean = data.mean(axis=(0,2,3))
    d_std = data.std(axis=(0,2,3))
    
    print("\tDataset mean: {} std: {}".format(d_mean, d_std))
    
    # Dataset mean: tensor([0.6142, 0.5552, 0.4108, 0.5264]) std: tensor([0.1480, 0.1209, 0.0966, 0.1800])
    #d_mean = [0.6142, 0.5552, 0.4108, 0.5264]
    #d_std = [0.1480, 0.1209, 0.0966, 0.1800]
    
    # Set up the dataloader for a normalised dataset:
    
    my_transform = Compose([ToTensor(), Normalize(d_mean, d_std)])
   
    train_batch = 4
    
    if args['mode'] == 'train':
        print("Loading train data.")
        carrots_train_norm = Crop_Dataset(root=args['data_folder'], train=True, sample_size=(384,512), transforms=my_transform)
        trainloader = torch.utils.data.DataLoader(carrots_train_norm, batch_size=train_batch,shuffle=True, collate_fn = collate_fn)



    print('Setting up the network.')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #net, criterion = net_setup(device, class_weights)
    
    net = SegNetBasic(in_channels = 4, num_classes = 3, final_batch_norm = False)
    net.to(device)
    
    class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)    
    criterion.to(device)
    
    optimizer = segnet.utils.init_SGD_optimizer(net)

    start_epoch = 0

    if args['model']:
        net, optimizer, start_epoch = load_model(args['model'], net, optimizer)

    if args['mode'] == 'train':
        net_out_base = args['net_out_name']
        max_epochs = args['epochs']
        print_epoch = args['print_epoch']
        print_iteration = args['print_iteration']

    
        print('Commencing training on device {}.'.format(device))
        
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
 
    print("Loading testing data.")
    
    carrots_test_norm = Crop_Dataset(root=args['data_folder'], train=False, sample_size=(384,512), transforms=my_transform)
    testloader = torch.utils.data.DataLoader(carrots_test_norm, batch_size=1, shuffle=False)
 
    print('Commencing testing on device {}.'.format(device))

    test(
        net = net,
        criterion = criterion,
        dataloader = testloader,
        device = device,
        save_images = True
    )                           


if __name__=="__main__":
    main()
