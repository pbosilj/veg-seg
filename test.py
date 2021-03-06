from utils import load_model, data_stats

import torch
import time
import numpy as np

from skimage import io
from shutil import copyfile

from sklearn.metrics import cohen_kappa_score

import segnet

from dataset import Crop_Dataset
from dataset.transforms import Normalize, Compose, ToTensor

import argparse

def test(net, criterion, dataloader, device='cpu', save_images = False, out_image_path = "prediction", truth_image_path="truth", verbose = False):
    net.eval()
    
    total_loss = 0.0
    
    t_total_duration = 0
    
    all_labels = np.array([])
    all_outputs = np.array([])
    
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
            
            out = outputs.detach().cpu().numpy()
            out = np.argmax(out, axis = 1).astype(dtype='uint8')
            truth = labels.detach().cpu().numpy().astype(dtype='uint8')
            
            kappa = cohen_kappa_score(out.flatten(), truth.flatten())
            all_outputs = np.append(all_outputs, out.flatten())
            all_labels = np.append(all_labels, truth.flatten())
            
            if verbose:
                print('\t\t[Batch size %d, classification time: %.5f] [%d] loss: %.3f, kappa: %.3f' % (dataloader.batch_size, t_duration, i, loss, kappa))
            if save_images:
                for bi, (predicted_image, truth_image) in enumerate(zip(outputs, labels)):
                    out_image = predicted_image.detach().cpu().numpy()
                    out_image = np.argmax(out_image, axis = 0).astype(dtype='uint8')
                    out_truth = truth_image.detach().cpu().numpy().astype(dtype='uint8')

                    out_image = (out_image*127).astype('uint8')
                    out_truth = (out_truth*127).astype('uint8')

                    io.imsave(fname = './{}_i{}_b{}.png'.format(out_image_path, i, bi), arr = out_image)
                    io.imsave(fname = './{}_i{}_b{}.png'.format(truth_image_path, i, bi), arr = out_truth)
                    

    # Kappa calculation is really slow as sklearn does not allow calculating direclty from confusion matrix...
    # ... and so this is the entire test set concatenated.
    kappa = cohen_kappa_score(all_outputs, all_labels)

    dlen = len(dataloader.dataset)    
    print('\t[Average classification time: %.5f per sample] average loss: %.3f, kappa: %.5f' % (t_total_duration/dlen, total_loss/dlen, kappa))            

    return kappa

def main():
    test_parser = argparse.ArgumentParser()

    test_parser.add_argument("-d", "--data-folder", type=str, required=True, metavar=('PATH'), help="Path to the dataset.")    
    
    test_parser.add_argument("-gt", "--ground_truth", type=str, required = True, choices = ['full', 'partial'], help='Whether full or partial annotations are used in training. This is important for data normalisation (always done from the test set).')
    
    test_parser.add_argument("-s", "--sample-size", nargs=2, type=int, default=(384,512), required=False, help="Input images are divided into samples of dimensions HEIGHTxWIDTH", metavar=('HEIGHT', 'WIDTH'))
    test_parser.add_argument("-spi", "--samples-per-image", nargs=2, type=int, required=False, metavar = ('COL_SAMPLES', 'ROW_SAMPLES'), help="Take at most COL_SAMPLESxROW_SAMPLES from the image")
    
    test_parser.add_argument('-m','--model', type=str, required=True, help = "Path to the pre-trained model file(s).", nargs='+', metavar=('PATH'))
    
    test_parser.add_argument('-v', '--verbose', required=False, action='store_true', help = "Print per-image accuracy and kappa scores.")
    test_parser.add_argument('-si', '--save-images', required=False, action='store_true', help = 'Save output and ground truth images for visual inspection.')
    test_parser.add_argument('-cb', '--copy-best', required=False, type=str, metavar=('FILENAME'), help = 'Copy the best model to FILENAME.')
       
    args = vars(test_parser.parse_args())
   
    # Sort by (int) epoch number is model with filenames like model_eNUM.pt. Evaluate models with custom names first.
       
    args['model'] = sorted(args['model'], key=lambda x: int(x.split("_")[-1][1:-3]) if x.split("_")[-1][1:-3].isnumeric() else -1)
     
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

    print("Setting up the testing dataset.")

    my_transform = Compose([ToTensor(), Normalize(d_mean, d_std)])  
    carrots_test_norm = Crop_Dataset(root=args['data_folder'], train=False, sample_size=sample_size, samples_per_image = samples_per_image, transforms=my_transform)
    testloader = torch.utils.data.DataLoader(carrots_test_norm, batch_size=1, shuffle=False) # can add partial_truth = (args['ground_truth']=='partial') but test set always loads full truth
    
    # Network setup:
   
    print('Setting up the network.')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = segnet.SegNetBasic(in_channels = 4, num_classes = 3, final_batch_norm = False)
    net.to(device)   
        
    class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)    
    criterion.to(device)

    best_model = -1
    best_kappa = -1

    for i, model in enumerate(args['model']):
        net, _, _  = load_model(model, net)

        # Set up the dataloader for a normalised dataset:    
        
        # Inference
     
        print('Commencing testing of {} on device {}.'.format(model, device))

        kappa = test(
                    net = net,
                    criterion = criterion,
                    dataloader = testloader,
                    device = device,
                    save_images = args['save_images'],
                    verbose = args['verbose'],
                )                           
        
        if best_model == -1 or kappa > best_kappa:
            best_model = i
            best_kappa = kappa
            
    print("\nBest model {} with kappa over test set {:.5f}".format(args['model'][best_model], best_kappa))
    
    if args['copy_best']:
        print('Best model copied to {}'.format(args['copy_best']))
        copyfile(args['model'][best_model], args['copy_best'])


if __name__=="__main__":
    main()
