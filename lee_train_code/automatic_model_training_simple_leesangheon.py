import os
import sys
import locale
import numpy as np
import torch
from pathlib import Path
import uuid
import yaml
import datasets
import scipy
from tqdm import tqdm


def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding


target_word = 'hey_thomas'
custom_model_dir = "my_custom_model"


# Load default YAML config file for training
config_path = os.path.join(os.path.dirname(__file__), "custom_model.yml")
config = yaml.load(open(config_path, 'r').read(), yaml.Loader)

# Modify values in the config and save a new version
number_of_examples = 1000 # @param {type:"slider", min:100, max:50000, step:50}
number_of_training_steps = 16500  # @param {type:"slider", min:0, max:50000, step:100}
false_activation_penalty = 3500  # @param {type:"slider", min:100, max:5000, step:50}
config["target_phrase"] = [target_word]
config["model_name"] = config["target_phrase"][0].replace(" ", "_")
config["n_samples"] = number_of_examples
config["n_samples_val"] = max(500, number_of_examples//10)
config["steps"] = number_of_training_steps
config["target_accuracy"] = 0.5
config["target_recall"] = 0.25
config["output_dir"] = f"./{custom_model_dir}"
config["max_negative_weight"] = false_activation_penalty
config["piper_sample_generator_path"] = "/home/data/openwakeword/piper-sample-generator"

config["background_paths"] = ['/home/data/openwakeword/background_sound/audioset_16k', '/home/data/openwakeword/background_sound/fma']  # multiple background datasets are supported
config["false_positive_validation_data_path"] = "/home/data/openwakeword/validation_set_features.npy"
config["feature_data_files"] = {"ACAV100M_sample": "/home/data/openwakeword/ACAV100M_2000_hrs_16bit_sample/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}

with open('./my_model.yaml', 'w') as file:
    documents = yaml.dump(config, file)

# Generate clips
os.system(f"{sys.executable} openwakeword/train.py --training_config my_model.yaml --generate_clips")

# Step 2: Augment the generated clips

os.system(f"{sys.executable} openwakeword/train.py --training_config my_model.yaml --augment_clips")

# Step 3: Train model

os.system(f"{sys.executable} openwakeword/train.py --training_config my_model.yaml --train_model")

print("Now use the google colab to convert to tflite")
print("https://colab.research.google.com/drive/1WU7C4xsdr05EMMLdmC7yLjWshbKzPL-V#scrollTo=L9vFF06rWYwP")
print("upload the onnx file to get the tflite file")