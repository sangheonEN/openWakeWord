print("FIRST RUN 'wipe_before_rerun.sh' to remove all generated/training artifacts")
print("THEN SET YOUR TARGET WORD HERE and make sure it sounds like what you want (test_generation.wav)")
print("look in my_custom_model for the finished onnx after it all runs.")
print("if it fails and you ahve to re-start, you may need to delete my_custom_model/positive_features_train.npy")
print("open colab https://colab.research.google.com/drive/1WU7C4xsdr05EMMLdmC7yLjWshbKzPL-V#scrollTo=L9vFF06rWYwP and upload the onnx file to get the tflite file")

target_word = 'hey_thomas'

import os
import sys

# 현재 파일 위치 기준으로 piper-sample-generator 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "piper-sample-generator"))

from generate_samples import generate_samples

# sys.path.append("piper_sample_generator/")
# from generate_samples import generate_samples

lang_model  = os.path.join(os.path.dirname(__file__), "piper-sample-generator", "models", "en_US-libritts_r-medium.pt")

custom_model_dir = "my_custom_model"

def text_to_speech(text):
    generate_samples(text = text,
                max_samples=1,
                length_scales=[1.1],
                noise_scales=[0.7], noise_scale_ws = [0.7],
                output_dir = './', batch_size=1, auto_reduce_batch_size=True,
                file_names=["test_generation.wav"],
                model = lang_model
                )

text_to_speech(target_word)

# os.system("afplay test_generation.wav")

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# Imports
import sys


import numpy as np
import torch
import sys
from pathlib import Path
import uuid
import yaml
import datasets
import scipy
from tqdm import tqdm

## Download all data

## Download MIR RIR data (takes about ~2 minutes)
output_dir = "./mit_rirs"
"""
import os
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    rir_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav")]}).cast_column("audio", datasets.Audio())
    # Save clips to 16-bit PCM wav files
    for row in tqdm(rir_dataset):
        name = row['audio']['path'].split('/')[-1]
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
"""
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    # Get all wav files
    wav_files = list(Path("./MIT_environmental_impulse_responses/16khz").glob("*.wav"))
    
    # Create dataset with file paths as strings
    rir_dataset = datasets.Dataset.from_dict({"audio": [str(f) for f in wav_files]})
    
    # Cast to Audio type - this will load the actual audio data
    rir_dataset = rir_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    
    # Save clips to 16-bit PCM wav files
    for i, row in enumerate(tqdm(rir_dataset)):
        # Debug: print the structure
        if i == 0:  # Only print for the first item
            print("Row keys:", row.keys())
            print("Audio type:", type(row['audio']))
            print("Audio content:", row['audio'])
        
        # Get the original filename from the path
        original_path = wav_files[i]  # Use the original path from our list
        name = original_path.name
        
        # Handle different possible audio data structures
        if isinstance(row['audio'], dict) and 'array' in row['audio']:
            # Standard structure
            audio_array = row['audio']['array']
            sample_rate = row['audio']['sampling_rate']
        else:
            # Try to access the audio data directly
            import soundfile as sf
            audio_array, sample_rate = sf.read(str(original_path))
        
        # Write the audio array to file
        scipy.io.wavfile.write(
            os.path.join(output_dir, name), 
            sample_rate, 
            (audio_array * 32767).astype(np.int16)
        )
## Download noise and background audio (takes about ~3 minutes)

# Audioset Dataset (https://research.google.com/audioset/dataset/index.html)
# Download one part of the audioset .tar files, extract, and convert to 16khz
# For full-scale training, it's recommended to download the entire dataset from
# https://huggingface.co/datasets/agkphysics/AudioSet, and
# even potentially combine it with other background noise datasets (e.g., FSD50k, Freesound, etc.)

# Install soundfile if you haven't already
# uv pip install soundfile

import soundfile as sf

if not os.path.exists("audioset"):
    os.mkdir("audioset")

    fname = "bal_train09.tar"
    out_path = f"audioset/{fname}"
    link = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/" + fname
    
    # Download using subprocess
    import subprocess
    subprocess.run(["curl", "-L", "-o", out_path, link])
    
    # Extract the tar file
    subprocess.run(["tar", "-xf", fname], cwd="audioset")

    output_dir = "./audioset_16k"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get all FLAC files
    flac_files = list(Path("audioset/audio").glob("**/*.flac"))
    
    # Process each file using soundfile
    for flac_file in tqdm(flac_files):
        try:
            # Read the FLAC file using soundfile
            audio_array, sample_rate = sf.read(str(flac_file))
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                from scipy import signal
                num_samples = int(len(audio_array) * 16000 / sample_rate)
                audio_array = signal.resample(audio_array, num_samples)
                sample_rate = 16000
            
            # Convert to int16
            audio_array = (audio_array * 32767).astype(np.int16)
            
            # Save as WAV
            output_name = flac_file.name.replace(".flac", ".wav")
            output_path = os.path.join(output_dir, output_name)
            scipy.io.wavfile.write(output_path, sample_rate, audio_array)
        except Exception as e:
            print(f"Error processing {flac_file}: {e}")
            continue

"""
MANUALLY DOWNLOAD FMA
# Create directory
mkdir -p fma_data
cd fma_data

# Download the audio files (8GB for small dataset)
curl -L -o fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip

# Download metadata
curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

# Extract files
unzip fma_small.zip
unzip fma_metadata.zip

cd ..


LOAD THE FMA:
# Process FMA dataset
fma_dir = "./fma_data/fma_small"
if os.path.exists(fma_dir):
    output_dir = "./fma_16k"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Get all MP3 files from FMA
    mp3_files = list(Path(fma_dir).glob("**/*.mp3"))
    
    # Process first 10,000 files (or adjust as needed)
    for mp3_file in tqdm(mp3_files[:10000]):
        try:
            # Read MP3 using soundfile
            audio_array, sample_rate = sf.read(str(mp3_file))
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                from scipy import signal
                num_samples = int(len(audio_array) * 16000 / sample_rate)
                audio_array = signal.resample(audio_array, num_samples)
            
            # Convert to int16
            audio_array = (audio_array * 32767).astype(np.int16)
            
            # Save as WAV
            output_name = mp3_file.stem + ".wav"
            output_path = os.path.join(output_dir, output_name)
            scipy.io.wavfile.write(output_path, 16000, audio_array)
        except Exception as e:
            print(f"Error processing {mp3_file}: {e}")
            continue


"""

# Generate synthetic audio as FMA substitute
output_dir = "./fma"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
    print("Generating synthetic audio files...")
    # Generate 100-500 synthetic audio files (adjust number as needed)
    num_files = 200  # Much smaller than 8000, but enough for training
    
    for i in tqdm(range(num_files)):
        # Vary the duration between 5-30 seconds
        duration = np.random.uniform(5, 30)
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # Generate different types of synthetic audio
        audio_type = np.random.choice(['noise', 'tone', 'mixed'])
        
        if audio_type == 'noise':
            # White noise
            audio = np.random.randn(samples) * 0.1
        elif audio_type == 'tone':
            # Pure tone with random frequency
            freq = np.random.uniform(100, 2000)  # Hz
            t = np.linspace(0, duration, samples)
            audio = 0.3 * np.sin(2 * np.pi * freq * t)
        else:
            # Mixed: noise + tone
            freq = np.random.uniform(200, 1000)
            t = np.linspace(0, duration, samples)
            audio = 0.1 * np.random.randn(samples) + 0.2 * np.sin(2 * np.pi * freq * t)
        
        # Add some envelope to make it more realistic
        envelope = np.ones_like(audio)
        fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio = audio * envelope
        
        # Convert to int16
        audio = np.clip(audio, -1, 1)  # Ensure within range
        audio = (audio * 32767).astype(np.int16)
        
        # Save as WAV
        output_path = os.path.join(output_dir, f"synth_audio_{i:05d}.wav")
        scipy.io.wavfile.write(output_path, sample_rate, audio)

# training set (~2,000 hours from the ACAV100M Dataset)
# See https://huggingface.co/datasets/davidscripka/openwakeword_features for more information
#if not os.path.exists("./openwakeword_features_ACAV100M_2000_hrs_16bit.npy"):
#    os.system("wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")

# validation set for false positive rate estimation (~11 hours)
#if not os.path.exists("validation_set_features.npy"):
#    os.system("wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy")

# @title  { display-mode: "form" }
# @markdown # 3. Train the Model
# @markdown Now that you have verified your target wake word and downloaded the data,
# @markdown the last step is to adjust the training paramaters (or keep
# @markdown the defaults below) and start the training!

# @markdown Each paramater controls a different aspect of training:
# @markdown - `number_of_examples` controls how many examples of your wakeword
# @markdown are generated. The default (1,000) usually produces a good model,
# @markdown but between 30,000 and 50,000 is often the best.

# @markdown - `number_of_training_steps` controls how long to train the model.
# @markdown Similar to the number of examples, the default (10,000) usually works well
# @markdown but training longer usually helps.

# @markdown - `false_activation_penalty` controls how strongly false activations
# @markdown are penalized during the training process. Higher values can make the model
# @markdown much less likely to activate when it shouldn't, but may also cause it
# @markdown to not activate when the wake word isn't spoken clearly and there is
# @markdown background noise.

# @markdown With the default values shown below,
# @markdown this takes about 30 - 60 minutes total on the normal CPU Colab runtime.
# @markdown If you want to train on more examples or train for longer,
# @markdown try changing the runtime type to a GPU to significantly speedup
# @markdown the example generating and model training.

# @markdown When the model finishes training, you can navigate to the `my_custom_model` folder
# @markdown in the file browser on the left (click on the folder icon), and download
# @markdown the [your target wake word].onnx or  <your target wake word>.tflite files.
# @markdown You can then use these as you would any other openWakeWord model!

# Load default YAML config file for training
import yaml
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
config["piper_sample_generator_path"] = "./piper_sample_generator"
config["model"] = lang_model

config["background_paths"] = ['./audioset_16k', './fma']  # multiple background datasets are supported
# config["false_positive_validation_data_path"] = "validation_set_features.npy"
# config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}

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