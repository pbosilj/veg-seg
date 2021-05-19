from utils import load_model, data_stats

import torch
import time
import numpy as np
from skimage import io

import segnet

from dataset import Crop_Dataset
from dataset.transforms import Normalize, Compose, ToTensor

import argparse

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
    test_parser = argparse.ArgumentParser()

    test_parser.add_argument("-d", "--data-folder", type=str, required=True)
    
    test_parser.add_argument('-m','--model', type=str, required=True, help = "Path to the pre-trained model file.")
       
    args = vars(test_parser.parse_args())
  
    # Get the dataset statistics:
    
    print("Calculating datasets stats.")
  
    class_weights, d_mean, d_std = data_stats(args['data_folder'])    
    
    # Network setup:
   
    print('Setting up the network.')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = segnet.SegNetBasic(in_channels = 4, num_classes = 3, final_batch_norm = False)
    net.to(device)   
        
    class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)    
    criterion.to(device)

    net, _, _  = load_model(args['model'], net)

    # Set up the dataloader for a normalised dataset:    

    print("Setting up the testing dataset.")

    my_transform = Compose([ToTensor(), Normalize(d_mean, d_std)])  
    carrots_test_norm = Crop_Dataset(root=args['data_folder'], train=False, sample_size=(384,512), transforms=my_transform)
    testloader = torch.utils.data.DataLoader(carrots_test_norm, batch_size=1, shuffle=False)
    
    # Inference
 
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
