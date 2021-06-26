import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime
import inspect

import sys
sys.path.append('C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\RitiCode\\Scene Classification Project\\Image Recognition Classes')
import training

strMainPath = "C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\Robotics\\ImageAI\\Project"
strDataset = "scenes2"
image_size = (150, 150)
batch_size = 32
intEpochs = 50

objCreateModel = training.RitModel(strMainPath, strDataset, image_size, batch_size, intEpochs)
objCreateModel.fnLoadDatasets()
objCreateModel.fnFinishModel()
