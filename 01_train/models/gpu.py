import os
import sys
import json
import pickle
import datetime
import importlib
import multiprocessing
from uuid import uuid4
import numpy as np
import subprocess as sp
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Non python standard libs
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from numpy import trapz


def train(**kwargs):
    model_identifier = str(uuid4())
    print("-----------------------------------------")
    print("Model Paramater", kwargs["name"].upper())
    print("-----------------------------------------")
    print("  MODEL:", kwargs["model"])
    print("  DATASET:", kwargs["dataset"])
    print("  JOB_UUID:", model_identifier)

#    if "img_dest_size" not in kwargs:
#        print("  IMGCONFIG", kwargs["img_dest_size"])
    
    print("  BATCH_SIZE:", kwargs["batch_size"])

    print("  TRAIN_CONFIG:", kwargs["train"])
    print("  OPT_CONFIG:", kwargs["optimizer"])
    

    data_path_out = "./"
    if os.environ.get('LOG_PATH') is not None:
        data_path_out = os.environ.get("LOG_PATH")

    data_path_out += "/" + kwargs["name"]
    
    if os.path.exists(data_path_out) is False and data_path_out != "":
        os.makedirs(data_path_out)

    print("  Save data to:",  data_path_out)

    if "save_model" not in kwargs:
        kwargs["save_model"] = False
    print("  save_model:", kwargs["save_model"])

    if kwargs["save_model"]:
        kwargs["save_model_path"] = data_path_out + "/saved_model"
        print("  Save models to:", kwargs["save_model_path"])
    
    if "weights" in kwargs:
        print("  WEIGHTS:", kwargs["weights"])

        
    print("-----------------------------------------")
    # **********************
    # * Create Data Folders*
    # **********************
    if "save_model_path" in kwargs and os.path.exists(kwargs["save_model_path"]) is False:
        os.makedirs(kwargs["save_model_path"]) 

    # *************************
    # * Create Data Generators*
    # *************************

    if os.path.exists("datasets/%s.py" % (kwargs["dataset"])):
        mod = importlib.import_module("datasets.%s" %(kwargs["dataset"]))
        globals()["dg"] = getattr(mod,"default_data_generators") 
        
    else:
        mod = importlib.import_module("datasets" )
        globals()["dg"] = getattr(mod,"fallback_data_generators")
    
    batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 2

    if kwargs["dataset"] != "TEST":
        dataset = "".join(kwargs["dataset"].split("-")[0:-3])
        f_format = kwargs["dataset"].split("-")[-3]
        k = kwargs["dataset"].split("-")[-1]
        path = os.environ.get(dataset.lower())
        datasource = f"{path}/{f_format}/km-{k}"
    else:
        datasource = "../..//rs_data/AID/data_split/"
    
    generator_train, generator_valid, generator_test = dg(datasource, batch_size, kwargs["img_dest_size"])
    
    assert len(generator_train.class_indices) == len(generator_valid.class_indices) == len(generator_test.class_indices)
    number_of_classes = len(generator_train.class_indices)

    img, _ = generator_train.next() 
    no_bands = img.shape[-1]

    img_input_shape = tuple(kwargs["img_dest_size"]) + (no_bands,)

    # *************************
    # * Model                 *
    # *************************
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():

        mod = importlib.import_module("CNN.%s" %(kwargs["model"].lower()))
        globals()["base_model"] = getattr(mod,kwargs["model"].lower()) 

        new_inputs = tf.keras.layers.Input(shape=img_input_shape)
        x = base_model(new_inputs, return_tensor=True, include_top=False) # HACK: TODO: Put parameter (include top to the json file)

        conv = tf.keras.Model(new_inputs, x)

        if "weights" in kwargs:
            weight_path =  os.environ.get("WEIGHT_PATH")
    
            conv.load_weights(weight_path + kwargs["weights"] + "_" + kwargs["model"].split("_")[0] + "_notop" +".h5")

            conv._layers.pop()

        if isinstance(kwargs["top_layers"], str):
            x = globals()["Dense"](**{"activation":kwargs["top_layers"], "units":number_of_classes})(x)
        else:
            for layer_config in kwargs["top_layers"]:
                kwargs["top_layers"][-1]["param"]["units"] = number_of_classes
                x = globals()[layer_config["layer"]](**layer_config["param"])(x)

        model = tf.keras.Model(new_inputs, x)
            
        model.summary()
        kwargs["input_layer"] = "{ips}".format(ips=(model.inputs))
        kwargs["output_layer"] = "{ops}".format(ops=(model.outputs))
        print("Model Inputs:", kwargs["input_layer"])
        print("Model Outputs:", kwargs["output_layer"])

        model._name = kwargs["name"]

        use_metrics = []

        for m in kwargs["metrics"]:
            if m.islower() is False:
                use_metrics.append(globals()[m]())
            else:
                use_metrics.append(m)

    # *************************
    # * Training              *
    # *************************
    ov_start = datetime.datetime.now()
    
    opt_id = 0
    history = None
    epochs = [0, 0]
    for elem in kwargs["train"]:
        if "base_model_is_trainable" in elem:
            trainable = elem["base_model_is_trainable"]
        else:
            trainable = False
            
        conv.trainable = trainable 

        opt = globals()[kwargs["optimizer"][opt_id]["opt"]](**kwargs["optimizer"][opt_id]["param"])
        
        if opt_id < (len(kwargs["optimizer"])-1):
            opt_id += 1

        model.compile(loss=kwargs["loss"], optimizer=opt, metrics=kwargs["metrics"])

        epochs[0] = epochs[1]
        epochs[1] = epochs[0] + elem["epochs"]

        print("Start training...")
        start_time = datetime.datetime.now()
        tmp = model.fit(
            generator_train,
            epochs=epochs[1],
            initial_epoch=epochs[0],
            validation_data=generator_valid,
            workers=multiprocessing.cpu_count(),
            verbose=1
        )
        training = datetime.datetime.now() - start_time
        print("Elapsed training time:", str(training))

        if history is not None:
            for elem in history.history:
                history.history[elem] = history.history[elem] + tmp.history[elem]
                print(elem)
        else:
            history = tmp

    # *************************
    # * Stats                 *
    # *************************
    auc = trapz(history.history['loss'], dx=1) - trapz(history.history['val_loss'], dx=1)
    if len(history.history['loss']) <= 3:
        auclast3 = auc
    else:
        auclast3 = trapz(history.history['val_loss'][-3:], dx=1) - trapz(history.history['loss'][-3:], dx=1)

    for key in history.history:
        kwargs[ "history_" + key ] = history.history[key]

    # *************************
    # * Evaluate Model        *
    # *************************
    print("Evaluate Model")
    ov_end = datetime.datetime.now()
    
    model.compile(optimizer=opt, loss=kwargs["loss"], metrics=use_metrics)
    
    score = model.evaluate(generator_test)

    data_train_str = ""
    if "weights" in kwargs:
        data_train_str += kwargs["weights"] + "_"
    data_train_str += kwargs["name"]

    file_name = kwargs["name"] + "-" + kwargs["model"] + "-" + data_train_str + "-" + model_identifier

    if(kwargs["save_model"]):
        p = kwargs["save_model_path"] + "/" + file_name + ".h5"

        model.save(p)

    # **********************
    # * Create Logfile     *
    # **********************

    kwargs["training_time"] = str(training)
    kwargs["score"] = score
    kwargs["AUC_last3_loss"] = auclast3
    kwargs["uuid"] = model_identifier
    kwargs["finished"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open(data_path_out + '/logfile_' + file_name + ".json", 'w') as outfile:
        json.dump(kwargs, outfile, indent=4, sort_keys=True)
        