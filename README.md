# Semantic segmentation of crops and weeds

This repo contains pytorch implementations reproducing the results from the paper:<br>
Petra Bosilj, Erchan Aptoula, Tom Duckett, and Grzegorz Cielniak: “Transfer learning between crop types for semantic segmentation of crops versus weeds in precision agriculture”, _Journal of Field Robotics_ (2019)

It relies on the original data used in the paper, which can be found [here](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/crop-vs-weed-discrimination-dataset/). If you find this data or code helpful in your research, you can cite our paper (bibtex entry can be found at the bottom of this file).

The list of required python packages can be found in `requirements.txt`.

## Running the training, inference and evaluation

The implementation of both the training and testing protocol is in `train_test.py`.

### Training

To **train a model**, run the command with the following options:
```
> python ./train_test.py train --help
usage: train_test.py train [-h] [-m MODEL] [-e EPOCHS] [-pi PRINT_ITERATION] [-pe PRINT_EPOCH] -n NET_OUT_NAME

Evaluate the model only.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to the pre-trained model file.
  -e EPOCHS, --epochs EPOCHS
                        Number of training epochs.
  -pi PRINT_ITERATION, --print-iteration PRINT_ITERATION
                        How often the current loss is displayed (in iterations).
  -pe PRINT_EPOCH, --print-epoch PRINT_EPOCH
                        How often the current model is saved (in epochs).
  -n NET_OUT_NAME, --net-out-name NET_OUT_NAME
                        Prefix of the saved model filenames.
```

**Example:**<br>
This example will run the training for 100 epochs, displaying the loss every 10 iterations and saving the model every epoch into files with a prefix `SegNetBasic_CA17_FS`.
```
> python ./train_test.py -d /path/to/carrots train -e 100 -pi 10 -pe 1 -n SegNetBasic_CA17_FS
```
To continue training the last model produced by the above command, use (same freqency of displaying loss/saving model):
```
> python ./train_test.py -d /path/to/carrots train -e 200 -pi 10 -pe 1 -n SegNetBasic_CA17_FS_test -m SegNetBasic_CA17_FS_e100_i80.pt

```
### Inference

To **perform inference on the test data**, and save the output images predicted by the network, run the command with the following options:
```
> python ./train_test.py test --help
usage: train_test.py test [-h] -m MODEL

Train the model.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to the pre-trained model file.
```

**Example:**<br>
Test the model saved in the file `SegNetBasic_CA17_FS_e120_i80.pt` (model after epoch 120 and all 80 iterations).
```
> python ./train_test.py -d /path/to/carrots test -m SegNetBasic_CA17_FS_e120_i80.pt
```

### Evaluation

To **compare the network output with the ground truth** images and calculate Cohen's Kappa, after performing inference on the test set (step above), run:
```
> python kappa3class.py -n 80 -p './'
```
(If you change the size of the samples cropped from the images, and consequently the size of train/test datasets, you will have to adjust `-n 80` to the new size of the train set.)

## Understanding the code

### Loading the data

The dataloader is implemented in `crop_datasets.py` and uses transformations defined in `vegseg_transforms.py`. It assumes a 80-20 data split (train: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]` test: `[17, 18, 19, 20]`) which can be changed by including `train.txt` or `text.txt` file in the data folder (_Note: custom train and test split setting is not tested).

Two dataset specific parameters are:
- `sample_size`: input images will be cut into tiled samples of this size. Default value `(384x512)` matches the input size of SegNetBasic (original `caffe` implementation)
- `samples_per_image`: number of samples to take from each image in x- and y-direction (if different from the maximum allowed by `sample_size`)

Examples of using this dataloader can be seen by running the file such as:
```
> python crop_dataset.py
```
and can also be found in `dataloader_demo.ipynb`. They also show how to access class probabilities calculated from the test data (which is used to set the weighting for the loss function).

_Note: currently has been tested only for loading the full ground truth, but partial ground truth (see paper or dataset) is expected._

### Initialising the network

The network is implemented in `segnet_basic.py`. However, for proper training behaviour it is important to **initialise the bias learning rates to x2 of other weight parameters learning rates**. It is also important to **apply weight decay only to the weights and not biases** if used. These details were encoded in the SegNetBasic architecture in `caffe` but have to be set up after net initialisation in `pytorch`.

In addition to the standard parameters `in_channels` and `num_classes`, the implementation allows adjusting the depth of the encoder-decoder through `depth` parameter (default 4 like in the original `caffe` implementation) and adding an additional batch normalisation layer following the classification layer with `filan_batch_norm` param (not used in the original implementation).

Examples of correctly setting the learning rate for an optimiser used with this implementation of SegNetBasic can be found in `set_segnet_lr.ipynb` and in the `net_setup()` function of `train_test.py`.


### Bibtex entry
```
@article{bosilj2019transfer,
    author = {Bosilj, Petra and Aptoula, Erchan and Duckett, Tom and Cielniak, Grzegorz},
    title = {Transfer learning between crop types for semantic segmentation of crops versus weeds in precision agriculture},
    journal = {Journal of Field Robotics},
    volume={37},
    number={1},
    pages={7--19},
    year={2020},
    publisher={Wiley}
}

```
