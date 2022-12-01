# Deploy to custom hardware

This section is about deploying the trained models of a hardware accelerator, specifically an FPGA from type Xilinx ZCU102.

The deployment is done using the following steps:
1. Inference the model to be checked if the model is valid. 
2. Quantize the model using VitisAI.
3. Compile the quantized model for a defined DPU. 
3. Finally, the target application for the FPGA is created. 

To deploy a model TensorFlow model to FPGA , run the script `deploy.sh`
```console
user@hostnmame:~$ source ./deploy.sh <dataset path> <model path> <target name>
```
**dataset path**: path to the datasets, which has been used to train the model. </br>
**model path**: path to the saved model. </br>
**target name**: the name used to save the output application for the FPGA. </br>
