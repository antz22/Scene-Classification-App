from django.shortcuts import render

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import sys
from .forms import ImageForm
sys.path.append('C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\RitiCode\\Scene Classification Project\\Image Recognition Classes')
import predicting


def index(request):
    """Process images uploaded by users"""


    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            # scene = classify('C:/Users/suchi/Dropbox (Sandipan.com)/Creative/RitiCode/Scene Classification Project/Image Recognition Classes/django_scene_classification_app/django_scene_classification_app' + img_obj.image.url)
            scene = classify('../django_scene_classification_app' + img_obj.image.url)
            img_obj.title = scene
            return render(request, 'core/index.html', {'form': form, 'img_obj': img_obj, 'scene': scene})


    else:
        form = ImageForm()
    return render(request, 'core/index.html', {'form': form})

# def classified(request):
#     return render(request, "core/classified.html", {
#         "classification": classify('C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\RitiCode\\Scene Classification Project\\samples\\ntFmJUZ8tw3ULD3tkBaAtf.jpg'),
#     })


def classify(strImagePath):
    # strImagePath = "C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\RitiCode\\Scene Classification Project\\samples\\ntFmJUZ8tw3ULD3tkBaAtf.jpg"
    image_size = (150, 150)
    strModelPath = 'C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\RitiCode\\Scene Classification Project\\scenes2_model-1624665688.json'
    strWeightsPath = "C:\\Users\\suchi\\Dropbox (Sandipan.com)\\Creative\\RitiCode\\Scene Classification Project\\scenes2_model-1624665688.h5"

    objFinalModel = predicting.TrainedModel(strImagePath, image_size, strModelPath, strWeightsPath)
    objFinalModel.fnLoadAndCompile()
    return objFinalModel.fnPredict()