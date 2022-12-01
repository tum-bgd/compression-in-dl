# Training the models 

The models are trained using the scripts in the folder `./01_train/`. 
In order to train a model with specific parameters, a config file is required, this does look like the file is ./01_inferance/config/sample.json:
```json
{
    "name": "aid-reference-km-8",
    "model": "vgg16_vitis",
    "weights": "imagenet",
    "dataset": "aid-reference-km-8",
    "img_dest_size": [224, 224],
    "inference_script": "gpu",
    "batch_size":128,
    "save_model": false,
    "loss": "categorical_crossentropy",
    "metrics": ["acc"],
    "top_layers":[
        {"layer":"Flatten", "param":{}},
        {"layer":"Dense", "param":{"units":4096, "activation":"relu"}},
        {"layer":"Dropout", "param":{"rate":0.5}},
        {"layer":"Dense", "param":{"units":4096, "activation":"relu"}},
        {"layer":"Dropout", "param":{"rate":0.5}},
        {"layer":"Dense", "param":{"activation":"softmax"}}
    ],
    "train": [
          {"epochs": 10, "base_model_is_trainable": true }
    ],
    "optimizer": [{"opt": "Adam", "param": {"learning_rate": 1e-4}}]
}
```

To train all configurations in one folder, the file `train.sh` starts a docker container and trains the model using TensorFlow.
 
```console
user@hostnmame:~$ source ./train.sh <path>
```
**path**: path to config files
