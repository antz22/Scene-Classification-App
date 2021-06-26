import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import sys
sys.path.append('C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\RitiCode\\Scene Classification Project\\Image Recognition Classes')
import predicting

strImagePath = "C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\RitiCode\\Scene Classification Project\\samples\\ntFmJUZ8tw3ULD3tkBaAtf.jpg"
image_size = (150, 150)
strModelPath = 'C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\RitiCode\\Scene Classification Project\\scenes2_model-1624665688.json'
strWeightsPath = "C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\RitiCode\\Scene Classification Project\\scenes2_model-1624665688.h5"

objFinalModel = predicting.TrainedModel(strImagePath, image_size, strModelPath, strWeightsPath)
objFinalModel.fnLoadAndCompile()
objFinalModel.fnPredict()