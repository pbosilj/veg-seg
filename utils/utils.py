import torch

from dataset.transforms import ToTensor
from dataset import Crop_Dataset

def save_model(net, optimizer, epoch, iteration, PATH):
    print("Saving trained model at epoch {}".format(epoch+1))
    
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
#        'iteration': iteration
        }, PATH)

def load_model(PATH, net, optimizer=None, max_iteration=None):
    
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['net_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print("Loaded pre-trained model at epoch {}".format(epoch+1))

    return net, optimizer, epoch+1
    
def data_stats(DATA_PATH):
    crops_train = Crop_Dataset(root=DATA_PATH, train=True, 
                               sample_size=(384,512), # un-hardcode
                               transforms=ToTensor())
    
    class_weights = 1.0/crops_train.get_class_probability()
    print("\tClass weights: {}".format(class_weights))
    
    single_loader = torch.utils.data.DataLoader(crops_train, batch_size=len(crops_train), num_workers=1)
    data, labels = next(iter(single_loader))

    d_mean = data.mean(axis=(0,2,3))
    d_std = data.std(axis=(0,2,3))
    
    print("\tDataset mean: {} std: {}".format(d_mean, d_std))
    
    return class_weights, d_mean, d_std
