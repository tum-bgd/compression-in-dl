During the pre-processing, the data is quantization followed by an embedding into selected file formats. 

Note that the data needs to be downloaded manually, before performing the following steps. 

**Quantization**:
To quantize the data use the file `quantize.py`
```console
user@hostnmame:~$ source ./*.sh <dataset name> <data root path> "
```
**data root path**: the path where the dataset is located. </br>
**dataset name**: The name of the dataset folder. </br>

This step quantizes the dataset to 1,2,3,4,5,6,7,8 bit per channel. 

**Image Embedding**:
Next, the images need to be embedded into several file formats. Converters for the file formats JPEG, PNG, TIFF, PPM, and BMP are provided. The scripts can be used with
```console
user@hostnmame:~$ source ./*.sh <in> <output> <quality factor>
```
**in**: input path to the dataset folder (e.g.,`./aid/`) </br>
**out**: path to the output location </br>
**quality factor**: quality parameter (note that the range of this parameter might change, documentation: [https://imagemagick.org/script/command-line-options.php#quality](https://imagemagick.org/script/command-line-options.php#quality)). </br>

please note that a large amount of storage might be required.
