from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import warnings
import csv
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

model = load_model('/home/proactionlab/PycharmProjects/80Tools_VGG-16/vgg16_80Tools.h5')

# Define a new Model that will take an image as input, and will output
# intermediate representations for all layers except the first layer.
layer_outputs = [layer.output for layer in model.layers[1:]]
visual_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

# Read your image
#img = load_img('/media/proactionlab/Storage/NeuralNet_80Tools/ImageSet/test/abre_caricas/abre_caricas1.bmp')
img_path = os.path.join('/media/proactionlab/Storage/NeuralNet_80Tools/ImageSet/test/abre_caricas', 'abre_caricas1.bmp')
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#x = img_to_array(img)
#x = x.reshape((1,) + x.shape) # add one extra dimension to the front
#x /= 255. # rescale by 1/255.

# run your image through the network; make a prediction
feature_maps = visual_model.predict(x)

# Plotting intermediate representations for your image

# Collect the names of each layer except the first one for plotting
layer_names = [layer.name for layer in model.layers[1:]]

# Plotting intermediate representation images layer by layer
for layer_name, feature_map in zip(layer_names, feature_maps):
    if len(feature_map.shape) == 4: # skip fully connected layers
        # number of features in an individual feature map
        n_features = feature_map.shape[-1]
        # The feature map is in shape of (1, size, size, n_features)
        size = feature_map.shape[1]
        # Tile our feature images in matrix `display_grid
        display_grid = np.zeros((size, size * n_features))
        # Fill out the matrix by looping over all the feature images of your image
        for i in range(n_features):
            # Postprocess each feature of the layer to make it pleasible to your eyes
            x = feature_map[0, :, :, i]
            #x -= x.mean()
            #x /= x.std()
            #x *= 64
            #x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            # We'll tile each filter into this big horizontal grid
            display_grid[:, i * size : (i + 1) * size] = x
        # Display the grid
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
