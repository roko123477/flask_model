from flask import Flask, render_template, request
import random
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
import tensorflow as tf
import keras.backend as K
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, BatchNormalization, Dropout, Flatten, Dense, Activation, MaxPool2D, Conv2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization, Reshape, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Conv2D, LSTM, Dense, Flatten, Dropout, Activation
from PIL import Image
from PIL import UnidentifiedImageError

# from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
import itertools
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, jsonify, render_template
import requests 
import urllib
import numpy as np
# 
import tensorflow as tf
BASE_MODEL='Xception' # ['VGG16', 'RESNET52', 'InceptionV3', 'Xception', 'DenseNet169', 'DenseNet121','InceptionResNetV2']

if BASE_MODEL=='VGG16':
    from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
# elif BASE_MODEL=='InceptionResNetV2':
#     from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input
# elif BASE_MODEL=='InceptionV3':
#     from keras.applications.inception_v3 import InceptionV3 as PTModel, preprocess_input
elif BASE_MODEL=='Xception':
    from keras.applications.xception import Xception as PTModel
    
from tensorflow.keras.layers import Conv2D, LSTM, Dense, Flatten, Dropout, Activation
from keras.applications import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input
from keras.applications import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications import MobileNet
from keras.applications import InceptionResNetV2

def create_model(input_shape, n_out):
    pretrain_model = PTModel(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape)
    input_tensor = Input(input_shape)
    # c0 = BatchNormalization()(input_tensor)
    c1= pretrain_model(input_tensor)
    c1 = Conv2D(128,3, activation='relu')(c1)
    c1 = Flatten()(c1)
    c1 = Dropout(0.4)(c1)
    c1 = Dense(256, activation='relu')(c1)
    c1 = Dropout(0.4)(c1)
    output = Dense(n_out, activation='softmax')(c1)
    model = Model(input_tensor, output)

    return model

vgg16_model=create_model((248,248,3),5)
vgg16_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
vgg16_model.summary()
vgg16_model.load_weights("./models/xception_weights.h5")


# 


app = Flask(__name__)
img_w, img_h = 248, 248
# img_w1, img_h1 = 224, 224
target_img = os.path.join(os.getcwd() , 'static/images')

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(tf.keras.layers.RepeatVector(
#     10
# ))  # Repeat vector for LSTM input
# model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))
# model.summary()

# # Load the model weights
# model.load_weights("./models/mymodel.h5")

# model.compile(optimizer='Adamax', loss = tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

@app.route('/')
def index_view():
    return "flask running"
#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(img,width,height):
    # img = cv2.imread(filename)
    img = cv2.resize(img, (width, height))
    img = np.reshape(img, [1, width, height, 3])
    img = img/255.0
    return img

# import json 
@app.route('/predict',methods=['GET','POST'])
def predict():
    
    data = request.get_json() 
      
    #     file = request.files['file']
    #     if file and allowed_file(file.filename): #Checking file format
    #         filename = file.filename
    file_path = data[0]['url']
    print(file_path);
    # return json.dumps({"newdata":file_path}) 
    #         file.save(file_path)
    req = urllib.request.urlopen(file_path)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img = read_image(img,img_w,img_h) #prepressing method
    class_prediction=vgg16_model.predict(img) 
    classes_x=np.argmax(class_prediction,axis=1)
    if classes_x == 0:
      pred_class = "No Dr"
    elif classes_x == 1:
      pred_class = "Mild Dr"
    elif classes_x == 2:
      pred_class = "Moderate Dr"
    elif classes_x == 3:
      pred_class = "Proliferative Dr"
    else:
      pred_class = "Severe Dr"   
            #'fruit' , 'prob' . 'user_image' these names we have seen in predict.html.
    data={'pred_class':pred_class,'user_image':file_path,'prediction': class_prediction.tolist()}
    return jsonify(data)
            # return render_template('predict.html', fruit = fruit,prob=class_prediction, user_image = file_path)
        # else:
        #     return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)