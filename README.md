# Semantic segmentation of crops and weeds

This repo contains pytorch implementations reproducing the results from the paper:<br>
Petra Bosilj, Erchan Aptoula, Tom Duckett, and Grzegorz Cielniak: “Transfer learning between crop types for semantic segmentation of crops versus weeds in precision agriculture”, _Journal of Field Robotics_ (2019)

It relies on the original data used in the paper, which can be found [here](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/crop-vs-weed-discrimination-dataset/). If you find this data or code helpful in your research, you can cite our paper (bibtex entry can be found at the bottom of this file).

The list of required python packages can be found in `requirements.txt`.

## Running the training, inference and evaluation

The implementation of both the training protocol is in `train.py` and inference+evaluation in `test.py`.

### Training

To **train a model**, run the command with the following options:
```
> python ./train.py --help
usage: train.py [-h] -d DATA_FOLDER -gt {full,partial} [-s HEIGHT WIDTH] [-spi COL_SAMPLES ROW_SAMPLES] [-e EPOCHS] [-pi PRINT_ITERATION] [-se SAVE_EPOCH] -n NET_OUT_NAME [-m MODEL] [-t {continue,finetune}]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data-folder DATA_FOLDER
  -gt {full,partial}, --ground-truth {full,partial}
                        Whether full or partial annotations are used in training. This is important for data normalisation (always done from the test set).
  -s HEIGHT WIDTH, --sample-size HEIGHT WIDTH
                        Input images are divided into samples of dimensions HEIGHTxWIDTH
  -spi COL_SAMPLES ROW_SAMPLES, --samples-per-image COL_SAMPLES ROW_SAMPLES
                        Take at most COL_SAMPLESxROW_SAMPLES from the image
  -e EPOCHS, --epochs EPOCHS
                        Number of training epochs.
  -pi PRINT_ITERATION, --print-iteration PRINT_ITERATION
                        How often the current loss is displayed (in iterations).
  -se SAVE_EPOCH, --save-epoch SAVE_EPOCH
                        How often the current model is saved (in epochs).
  -n NET_OUT_NAME, --net-out-name NET_OUT_NAME
                        Prefix of the saved model filenames.
  -m MODEL, --model MODEL
                        Path to the pre-trained model file.
  -t {continue,finetune}, --training_mode {continue,finetune}
                        Mode in which to continue training. `continue` loads the optimiser and continues training with saved settings. `finetune` loads the model only and reinitialises the optimiser with
                        finetuning settings.
```

**Example:**<br>
This example will run the training on carrots for 100 epochs, displaying the loss every 10 iterations and saving the model every epoch into files with a prefix `SegNetBasic_CA17_FS`.
```
> python ./train.py -d /path/to/carrots -gt full -e 100 -pi 10 -se 1 -n SegNetBasic_CA17_FS
```
To continue training the last model produced by the above command, use (same freqency of displaying loss/saving model):
```
> python ./train.py -d /path/to/carrots -gt full -e 200 -pi 10 -se 1 -n SegNetBasic_CA17_FS -m SegNetBasic_CA17_FS_e100.pt -t continue

```
Assuming the best performing model was in epoch 137 and saved in `SegNetBasic_CA17_FS_e_137.pt`, following will _finetune_ this network for onions using _partial annotations only_:
```
> python ./train.py -d /path/to/onions -gt partial -e 50 -pi 10 -se 1 -n SegNetBasic_ON17_partial_FT_CA17 -m SegNetBasic_CA17_FS_e137.pt -t finetune

```
Resuming training of a network which was fine-tuned from pre-initialised weights _is the same_ as continuing training of a randomly initialised model (example above):
```
> python ./train.py -d /path/to/onions -gt partial -e 100 -pi 10 -se 1 -n SegNetBasic_ON17_partial_FT_CA17 -m SegNetBasic_ON17_partial_FT_CA17_e50.pt -t continue

```

### Inference

To **perform inference and evaluation on the test data**, and save the output images predicted by the network, run the command with the following options:
```
> python ./test.py --help
usage: test.py [-h] -d DATA_FOLDER -gt {full,partial} [-s HEIGHT WIDTH] [-spi COL_SAMPLES ROW_SAMPLES] -m MODEL [-si]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_FOLDER, --data-folder DATA_FOLDER
  -gt {full,partial}, --ground_truth {full,partial}
                        Whether full or partial annotations are used in training. This is important for data normalisation (always done from the test set).
  -s HEIGHT WIDTH, --sample-size HEIGHT WIDTH
                        Input images are divided into samples of dimensions HEIGHTxWIDTH
  -spi COL_SAMPLES ROW_SAMPLES, --samples-per-image COL_SAMPLES ROW_SAMPLES
                        Take at most COL_SAMPLESxROW_SAMPLES from the image
  -m MODEL, --model MODEL
                        Path to the pre-trained model file.
  -si, --save-images    Save output and ground truth images for visual inspection
```

**Example:**<br>
Test the model saved in the file `SegNetBasic_CA17_FS_e120.pt` (model after epoch 120).
```
> python ./test.py -d /path/to/carrots -gt full -m SegNetBasic_CA17_FS_e120.pt
```
When testing a model trained with _partial ground truth_ it is important to specify this for testing, as _training set stats_ are used for data normalisation before inference.
```
> python ./test.py -d /path/to/onions -gt partial -m SegNetBasic_ON17_partial_FT_CA17_e75.pt
```

### Evaluation

To **compare the network output with the ground truth** images and calculate Cohen's Kappa, after performing inference on the test set (step above), run:
```
> python kappa3class.py -n 80 -p './'
```
(If you change the size of the samples cropped from the images, and consequently the size of train/test datasets, you will have to adjust `-n 80` to the new size of the train set.)

## Understanding the code

### Loading the data

The dataloader is implemented in `dataset/crop_datasets.py` and uses transformations defined in `dataset/transforms.py`. It assumes a 80-20 data split (train: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]` test: `[17, 18, 19, 20]`) which can be changed by including `train.txt` or `text.txt` file in the data folder containing folder names for training or testing (other set determined automatically).

Two dataset specific parameters are:
- `sample_size Optional[Tuple[int, int]]`: input images will be cut into tiled samples of this size. Default value `(384x512)` matches the input size of SegNetBasic (original `caffe` implementation)
- `samples_per_image: Optional[Tuple[int, int]]`: number of samples to take from each image in x- and y-direction (if different from the maximum allowed by `sample_size`)
- `partial_truth: bool`: load partial grount truth, with certain pixels unlabelled, for training (ignored while testing)

Examples of using this dataloader can be seen by running the file such as:
```
> python crop_dataset.py
```
and can also be found in `dataloader_demo.ipynb`. They also show how to access class probabilities calculated from the test data (which is used to set the weighting for the loss function). Unlabelled data is ignored while calculating class probabilities.

_Note: the dataloader has now been tested for loading the partial ground truth, but the training configuration does not process it yet. Need to include ignore_index in the optimizer._

### Initialising the network

The network is implemented in `segnet/segnet_basic.py`. However, for proper behavour when training (from scratch) it is important to **initialise the bias learning rates to x2 of other weight parameters learning rates**. It is also important to **apply weight decay only to the weights and not biases** if used. For proper fine-tuning behaviour, **conv-classifier learning rates should additionally be increased by x10**. These details were encoded in the SegNetBasic architecture in `caffe` but have to be set up after net initialisation in `pytorch`. They are encapsulated in `segnet.utils.init_SGD_optimizer()`.

In addition to the standard parameters `in_channels` and `num_classes`, the implementation allows adjusting the depth of the encoder-decoder through `depth` parameter (default 4 like in the original `caffe` implementation) and adding an additional batch normalisation layer following the classification layer with `final_batch_norm` param (not used in the original implementation).

Step-by-step explanation of how the learning rate and optimiser are through `segnet.utils.init_SGD_optimizer()` can be found in `set_segnet_lr.ipynb`.


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
