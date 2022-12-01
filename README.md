# Compression Supports Spatial Deep Learning



## Abstract
In the last decades, the domain of spatial computing became more and more data-driven, especially when using remote sensing-based images. Furthermore, the satellites provide huge amounts of images, so the number of available datasets is increasing. This leads to the need for large storage requirements and high computational costs when estimating the label scene classification problem using deep learning. This consumes and blocks important hardware recourses, energy, and time. In this paper, the use of aggressive compression algorithms will be discussed to cut the wasted transmission and resources for selected land cover classification problems. To compare the different compression methods and the classification performance, the satellite image patches are compressed by two methods. The first method is the image quantization of the data to reduce the bit depth. Second is the lossy and lossless compression of images with the use of image file formats, such as JPEG and TIFF. The performance of the classification is evaluated with the use of convolutional neural networks like VGG16. The experiments indicated that not all remote sensing image classification problems improve their performance when taking the full available information into account. Moreover, compression can set the focus on specific image features, leading to fewer storage needs and a reduction of computing time with comparably small costs in terms of quality and accuracy. All in all, quantization and embedding into file formats do support convolutional neural networks to estimate the labels of images, by strengthening the features.

![Compression in deep learning - experiment pipline](assets/comp-in-dl.png)

*Figure 1: The pipeline of the experiment is to evaluate the impact of the two selected compression methods and their parameters in convolutional neural networks.*

## Requirements
This project does use public datasets:
1. EuroSAT - [Source](https://github.com/phelber/eurosat), [Paper no. 1](https://doi.org/10.1109/JSTARS.2019.2918242), [Paper no. 2](https://doi.org/10.1109/IGARSS.2018.8519248)
2. RSI-CB256 - [Source](https://github.com/lehaifeng/RSI-CB), [Paper](https://doi.org/10.3390/s20061594)
3. AID - [Source](https://captain-whu.github.io/AID/), [Paper](https://doi.org/10.1109/TGRS.2017.2685945)

Those datasets need to be downloaded and split into three subsets, one training set (folder name: train), one validation set (folder name: val_0), and one test set (folder name: val_1). The path to those datasets needs to be adjusted in `./set_env.sh`.

Additionally, Docker should be available to run perform the training and deploy the models.

## Run Experiments
This section shows the steps to pre-process the data, train, and deploy the data.  

### Step 1: Data pre-processing 
During the pre-processing, the data is quantization followed by an embedding into selected file formats. 

Note that the data needs to be downloaded manually, before performing the following steps. 

**Quantization**:
To quantize the data use the file `quantize.py`
```console
user@hostnmame:~$ source ./*.sh <dataset name> <data root path> "
```

details see [_data/README.md](_data/README.md)

```console
user@hostnmame:~$ source ./*.sh <in> <output> <quality factor>
```

**in**: input path to the dataset folder (e.g.,`./aid/`) </br>
**out**: path to the output location </br>
**quality factor**: quality parameter (note that the range of this parameter might change, documentation: [https://imagemagick.org/script/command-line-options.php#quality](https://imagemagick.org/script/command-line-options.php#quality)). </br>

please note that a large amount of storage might be required.

### Step 2: Training the models

The models are trained using the scripts in the folder `./01_train/`. 
In order to train a model with specific parameters, a config file is required, this does look like the file is ./01_train/config/sample.json:

To train all configurations in one folder, the file `train.sh` starts a docker container and trains the model using TensorFlow.
 
```console
user@hostnmame:~$ source ./train.sh <path>
```

details see [01_train/README.md](01_train/README.md)


### Step 3: Deploy to custom hardware (Xilinx ZCU102)
This step deploys the trained models of a hardware accelerator, specifically an FPGA from type Xilinx ZCU102 using the scripts in the folder `./02_deploy/`.

To deploy a model TensorFlow model to FPGA , run the script `deploy.sh`
```console
user@hostnmame:~$ source ./deploy.sh <dataset path> <model path> <target name>
```

details see [02_deploy/README.md](02_deploy/README.md)


## Citation
This project has been accepted to IEEE JSTARS and is currently in the publication process. 

Bibtex will be added as soon as the paper is online available. 