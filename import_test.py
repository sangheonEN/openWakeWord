import os
import collections
import numpy as np
from numpy.lib.format import open_memmap
from pathlib import Path
from tqdm import tqdm
import openwakeword
import openwakeword.data
import openwakeword.utils
import openwakeword.metrics

import scipy
import datasets
import matplotlib.pyplot as plt
import torch
from torch import nn
import IPython.display as ipd

print("import done?")

"""
https://github.com/sangheonEN/openWakeWord/blob/main/notebooks/training_models.ipynb

# Install requirements (it's recommended that you do this in a new virtual environment)

pip install openwakeword
pip install speechbrain
pip install datasets
pip install scipy matplotlib

- adding pip lib

pip install ipython
pip install acoustics
pip install mutagen
pip install torch-audiomentations
pip install audiomentations
pip install pronouncing

"""