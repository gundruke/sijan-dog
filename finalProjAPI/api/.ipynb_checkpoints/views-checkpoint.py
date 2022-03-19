# Import necessary libraries
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
from django.core.files.storage import default_storage
import  os
import requests
import json
import pickle 

from keras.utils import np_utils
import numpy as np
from glob import glob

import random
random.seed(8675309)

import cv2

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.preprocessing import image                  
from tqdm import tqdm


from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

with open("pickle/dog-names.pkl", "rb") as f:
    dog_names = pickle.load(f)
dog_breeds = len(dog_names)

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

img_width, img_height = 224, 224

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(img_width, img_height))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


ResNet50_model = ResNet50(weights='imagenet')

bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')
train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3 = bottleneck_features['test']


inception_model = Sequential()
inception_model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))
inception_model.add(Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
inception_model.add(Dropout(0.4))
inception_model.add(Dense(dog_breeds, activation='softmax'))


inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
inception_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')

def extract_InceptionV3(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))




# top_N defines how many predictions to return
top_N = 1

def predict_breed(path):
    
    # load image using path_to_tensor
    print('Loading image...')
    image_tensor = path_to_tensor(path)
    
    # obtain bottleneck features using extract_InceptionV3
    print('Extracting bottleneck features...')
    bottleneck_features = extract_InceptionV3(image_tensor)
    
    # feed into top_model for breed prediction
    print('Feeding bottlenneck features into top model...')
    prediction = inception_model.predict(bottleneck_features)[0]
    
    # sort predicted breeds by highest probability, extract the top N predictions
    breeds_predicted = [dog_names[idx] for idx in np.argsort(prediction)[::-1][:top_N]]
    confidence_predicted = np.sort(prediction)[::-1][:top_N]
    
    print('Predicting breed...')
    # take prediction, lookup in dog_names, return value
    return breeds_predicted, confidence_predicted


def make_prediction(path, multiple_breeds = False):
    breeds, confidence = predict_breed(path)
    #img = mpimg.imread(path)
    
    # since the dog detector worked better, and we don't have 
    # access to softmax probabilities from dog and face detectors
    # we'll first check for dog detection, and only if there are no dogs
    # detected we'll check for humans
    if dog_detector(path):
        return "dog", breeds[0], confidence[0]
        
    elif face_detector(path):
        return "human", breeds[0], confidence[0]
    else:
        return "None", "None", 0.0
        
        

# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error": "0",
        "message": "Successful",
    }
    return Response(return_data)


@api_view(["POST"])
def predict(request):
    try:
        f=request.FILES["imageFile"] # here you get the file needed

        if not f:
            predictions =  {
                "error": "1",
                "result": "Please supply an image in body as imageFile"
            }
            return Response(predictions)
        
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        
        type_species, dog_breeds, breeds_confidence = make_prediction(file_name_2)
        
        try:
            # Delete the file so that the next file can have same name
            os.remove(file_name_2)
        except:
            pass
        
        predictions = {
            "error": "0",
            "type": type_species,
            "breed": dog_breeds,
            "confidence": breeds_confidence,
            "message" : "prediction success",
            "file_name" : file_name_2
        }
            
        print(type_species)
        print(dog_breeds)
        print(breeds_confidence)
        print(type(type_species))
        print(type(dog_breeds))
        print(type(breeds_confidence))
            
        
        
        
    except Exception as e:
        predictions = {
            "error": "2",
            "type": "None",
            "breed": "None",
            "confidence": 0.0,
            "message" : str(e),
            "file_name" : file_name_2
        }

    return Response(predictions)
