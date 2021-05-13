import torch
import torchvision

from torchvision.datasets import VisionDataset

from typing import Any, Callable, List, Optional, Tuple

from vegseg_transforms import Normalize, Compose, Resize, ToTensor

import os

import traceback

from get_image_size import get_image_size

from skimage import io
import numpy as np

CROP_CLASSES = {'background': [0,0,0],
                'crop': [255,0,0],
                'weed': [0,0,255],
                'unlabelled': [0,255,0]}

NUM_CLASSES = len(CROP_CLASSES) - 1


#class Crop_Dataset(torch.utils.data.Dataset):
class Crop_Dataset(VisionDataset):
    """
    A data loader for crop datasets (CA17, ON17)
    """
    def __init__(self, 
                 root: str,
                 sample_size: Optional[Tuple[int, int]] = (384,512),
                 train: bool = True,
                 partial_truth: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 samples_per_image: Optional[Tuple[int, int]] = None):
                 
        super(Crop_Dataset, self).__init__(root, transform=transform, target_transform=target_transform, transforms=transforms)
               
        self.root = root.rstrip('/')
        self.sample_size = sample_size

        self.directories = next(os.walk(self.root))[1]
        self.directories = [directory.rstrip('/') for directory in self.directories]
        self.directories.sort(key=int)
               
        self.train = train
        self.partial_truth = partial_truth
        self.transform = transform

        sample_files = next(os.walk(os.path.join(self.root, self.directories[0])))[2]
        
        self.nir_file     = [s for s in sample_files if 'depth'   in s][0]
        self.rgb_file     = [s for s in sample_files if 'rgb'     in s][0]
        self.truth_file   = [s for s in sample_files if 'truth'   in s][0]
        self.partial_file = [s for s in sample_files if 'partial' in s][0]

       
        self.image_size = get_image_size(os.path.join(root, self.directories[0], self.truth_file))[::-1]
        self.max_samples_per_image = (int(self.image_size[0] / self.sample_size[0]),int(self.image_size[1] / self.sample_size[1]))
        if samples_per_image == None or samples_per_image[0] > self.max_samples_per_image[0] or samples_per_image[1] > self.max_samples_per_image[1]:
            self.samples_per_image = self.max_samples_per_image
        else:
            self.samples_per_image = samples_per_image
        self.loaded_image = -1
 
        """
            The train/test subsets are determined according to the first of the following rules that succeeds:
            - Look for 'train.txt' in root. If found and not empty, the listed samples are used in training, and the rest as a test set.
            - Look for 'test.txt' in root. If found and not empty, the unlisted samples are uesd in training, and the rest as a test set.
            - If neither 'train.txt' nor 'test.txt' is found in the root, use the first 80% for training and the rest as a test set.
        """

        self.directories.sort(key=int)
        train_list = []
        test_list = []
        try:
            with open(os.path.join(root, 'train.txt'), 'r') as train:
                train_list = train.readlines()
        except IOError:
            print("File 'train.txt' does not exist in root. Looking for 'test.txt'")

        if len(train_list) == 0:
            try:
                with open(os.path.join(root, 'test.txt'), 'r') as test:
                    test_list = test.readlines()
            except IOError:
                print("File 'test.txt' does not exist in root. Defaulting to 80-20 split.")

        if len(train_list) == 0 and len(test_list) == 0:
            testing_samples = int(len(self.directories)/5)
            train_list = self.directories[:-testing_samples]
            test_list = self.directories[-testing_samples:]
        elif len(train_list) == 0:
            test_list = [x.rstrip() for x in test_list if not x.rstrip() == ""]
            test_list.sort(key=int)
            train_list = [x for x in self.directories if not x in test_list]
        else:
            train_list = [x.rstrip() for x in train_list if not x.rstrip() == ""]
            train_list.sort(key=int)
            test_list = [x for x in self.directories if not x in train_list]

        if self.train == True:
            self.directories = train_list
        else:
            self.directories = test_list
        
        self.counts = self.__compute_class_probability()
 
    def __len__(self):
        return len(self.directories)*self.samples_per_image[0]*self.samples_per_image[1]
        #return len(self.directories)*self.max_samples_per_image[0]*self.max_samples_per_image[1]
  
    def __getitem__(self, index):
        
        img_index = int(index / (self.samples_per_image[0]*self.samples_per_image[1]))
        img_subindex = index % (self.samples_per_image[0]*self.samples_per_image[1])
        sample_coordinates = (int(img_subindex / self.samples_per_image[1]), img_subindex % self.samples_per_image[1])
        if not self.loaded_image == img_index:
            #print('Rgb image: {}'.format(os.path.join(self.root, self.directories[img_index],   self.rgb_file)))
            rgb_image        = io.imread(os.path.join(self.root, self.directories[img_index],   self.rgb_file))
            #bgr to rgb:
            #rgb_image = rgb_image[:, :, ::-1]
            nir_image        = io.imread(os.path.join(self.root, self.directories[img_index],   self.nir_file))
 
            #print("rgb max: {} min: {}".format(np.amax(rgb_image), np.amin(rgb_image)))
            #print("nir max: {} min: {}".format(np.amax(nir_image), np.amin(nir_image)))
 
            if self.train and self.partial_truth:
                truth_rgb_image = io.imread(os.path.join(self.root, self.directories[img_index], self.partial_file))
            else:
                truth_rgb_image = io.imread(os.path.join(self.root, self.directories[img_index], self.truth_file))
    
            if nir_image.dtype == 'uint16':
                nir_image = (nir_image/256).astype('uint8')
                
            #print("rgb max: {} min: {}".format(np.amax(rgb_image), np.amin(rgb_image)))
            #print("nir max: {} min: {}".format(np.amax(nir_image), np.amin(nir_image)))

                
            #rgb_image = np.swapaxes(np.swapaxes(rgb_image, 0, 2), 1, 2)
    
            self.truth_image = np.zeros(nir_image.shape, dtype='uint8') 
   
            self.truth_image[np.all(truth_rgb_image == CROP_CLASSES['crop'], axis=2)] = 1
            self.truth_image[np.all(truth_rgb_image == CROP_CLASSES['weed'], axis=2)] = 2
            self.truth_image[np.all(truth_rgb_image == CROP_CLASSES['unlabelled'], axis=2)] = 3
 
            self.image_data = np.concatenate([rgb_image, nir_image[:, :, np.newaxis]], axis = 2)

            self.loaded_image = img_index

        #print("Image data shape: {}".format(self.image_data.shape))        
        #print("total max: {} min: {}".format(np.amax(self.image_data), np.amin(self.image_data)))
        image_sample_coordinates = ((sample_coordinates[0]*self.sample_size[0], (sample_coordinates[0]+1)*self.sample_size[0]), 
                                    (sample_coordinates[1]*self.sample_size[1], (sample_coordinates[1]+1)*self.sample_size[1]))
        data_sample  = self.image_data[image_sample_coordinates[0][0]:image_sample_coordinates[0][1], image_sample_coordinates[1][0]:image_sample_coordinates[1][1], :]
        truth_sample = self.truth_image[image_sample_coordinates[0][0]:image_sample_coordinates[0][1], image_sample_coordinates[1][0]:image_sample_coordinates[1][1]]
        
        #print(data_sample.shape)
        #print(truth_sample.shape)
        
        
        #print("Data sample shape: {}".format(data_sample.shape))
        #print("Data sample max: {} min: {}".format(np.amax(data_sample), np.amin(data_sample)))

        #print(data_sample.dtype, truth_sample.dtype)
        #if self.transform:
            #data_sample, truth_sample = self.transform((data_sample, truth_sample))
        #    data_sample = self.transform(data_sample)
        #    truth_sample = self.transform(truth_sample)
        
        #if self.transform is not None:
        #    data_sample = self.transform(data_sample)

        #if self.target_transform is not None:
        #    truth_sample = self.target_transform(truth_sample)
            
            
        if self.transforms is not None:
            data_sample, truth_sample = self.transforms(data_sample, truth_sample)

        #print("Samples {} {} ".format(data_sample.dtype, truth_sample.dtype))
        #print("Data sample shape: {}".format(data_sample.shape))
        #print(torch.max(data_sample), torch.min(data_sample))
        #print("Truth sample shape: {}".format(truth_sample.shape))
        #print(torch.max(truth_sample), torch.min(truth_sample))
        return data_sample, truth_sample

    def __compute_class_probability(self):
        
        counts = dict.fromkeys([k for k in CROP_CLASSES.keys() if not k == 'unlabelled'], 0)

        for img_path in self.directories:
            # think if this needs to be processed differently when partial truth is used. I think not.
            truth_rgb_image = io.imread(os.path.join(self.root, img_path, self.truth_file))
            
            # make sure you consider only the samples that are used
            truth_rgb_image = truth_rgb_image[:self.sample_size[0]*self.samples_per_image[0]][:self.sample_size[1]*self.samples_per_image[1]]
            
            for class_name, class_rgb in CROP_CLASSES.items():
                if class_name == 'unlabelled':
                    continue
                counts[class_name] += np.sum(np.all(truth_rgb_image == class_rgb, axis=2))

        return counts

    # Only for the portion currently used (train/test)
    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)
        
        return torch.Tensor(p_values)

def __save_image(image, path = './image.png'):
    np_image = image.numpy()
    #print("Saving image with shape {} max {} min {}".format(np_image.shape, np.amax(np_image), np.amin(np_image)))
    if np_image.dtype == 'float32':
        np_image = (np_image*255).astype('uint8')
    else:
        # high-contrast:
        np_image = (np_image*255/np.amax(np_image)).astype('uint8')
        # input-contrast
        #np_image = np_image.astype('uint8')
    if np.ndim(np_image) == 3:
        np_image = np.squeeze(np.transpose(np_image,(1,2,0)))
    io.imsave(fname = path, arr = np_image)


def main():
    transform = ToTensor()
    carrots_data_full = Crop_Dataset(
                            root = "/home/pbosilj/Data/CA17/carrots_labelled",
                            train = True,
                            partial_truth = False,
                            transforms = transform
                        )   
    probabilities = carrots_data_full.get_class_probability()
    for class_name, class_probability in zip([k for k in CROP_CLASSES.keys() if not k == 'unlabelled'], probabilities):
        print('Probability {:5.2f}%% for class {}'.format(class_probability*100, class_name))

    full_loader = torch.utils.data.DataLoader(
                                    carrots_data_full,
                                    batch_size=4,
                                    shuffle=True,
                                    num_workers=1
                                )
    
    print('Output: images with full annotations + ground truth')
    test_iterations = 2
    for i, data in enumerate(full_loader, 0):
        images, labels = data
        for bi, (image, label) in enumerate(zip(images, labels)):
            input_path = 'rgbn_i{}_bi{}.png'.format(i, bi)
            label_path = 'label_i{}_bi{}.png'.format(i, bi)
            
            np_image = image.numpy()
            np_target = label.numpy()
            #print("Image  max: {} min: {}".format(np.amax(np_image), np.amin(np_image)))
            #print("Target max: {} min: {}".format(np.amax(np_target), np.amin(np_target)))
            #print("Unique labels: {}".format(np.unique(np_target)))
            
            print('Saving input image at {}'.format(input_path))
            __save_image(image, path=input_path)
            print('Saving label image at {}'.format(label_path))
            __save_image(label, path=label_path)

        if i >= test_iterations:
            break

if __name__=="__main__":
    main()
