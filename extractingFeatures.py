import numpy as np
import csv
import os
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

# set paths
# set categories/examplars directory
main_path = '~/ImageSet/test'
# set a new directory to store features extracted.txt (set a new directory for each new layer)
newpath = '~/block1_conv1' #change for each new layer
# set a name to the new file (only identify the layer)
table = '_block1_conv1_features.txt' #change for each new layer
# number of features extracted from each layer (e.g. from block1_conv1 the output has 224*224*64 = 3211264 dimentions)
dimention = 3211264 #change for each new block
# load model, select layer to extract features from, and flat layer outputs
base_model = load_model('vgg16_80Tools.h5')
base_model.summary()

# rebuild NN, select layer to extract features from, and flat layer outputs
#new_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output) #use for full connected layers
modelpart = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_conv1').output)
modelpart.summary()
new_model = Sequential()
new_model.add(modelpart)
new_model.add(Flatten())
new_model.summary()

# extract features from each examplar of each category
categories = os.listdir(main_path)
for cat in categories:
        print("---")
        folderpath = os.path.join(main_path, cat)
        exemplars = [f for f in listdir(folderpath) if isfile(join(folderpath, f))]
        print(exemplars)
        cont = 0
        features_extracted = np.array([[]], dtype=int).reshape(0, dimention)
        for exep in exemplars:
                img_path = os.path.join(folderpath, exep)
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                block_conv_features = new_model.predict(x)
                features_extracted = np.concatenate((features_extracted, block_conv_features), axis=0)

        filename = cat + table
        newfolderpath = os.path.join(newpath, filename)
        print(newfolderpath)
        # write a .txt with each examplar in a row and features in columns
        with open(newfolderpath, 'w') as f:
                csv.writer(f, delimiter=' ').writerows(features_extracted)
                
