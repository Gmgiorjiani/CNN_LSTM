# Fine tune with Keras - VGG16

# Imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import PIL
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Getting our train, valid and test sets

train_path = '~/ImageSet/train'
valid_path = '~/ImageSet/valid'
test_path = '/media/proactionlab/Storage/NeuralNet_80Tools/ImageSet/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['class1', 'class2','class3', ..., 'classN'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['class1', 'class2','class3', ..., 'classN'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['class1', 'class2','class3', ..., 'classN'], batch_size=10, shuffle=False)

imgs, labels = next(train_batches)
print(imgs.shape)

# importing pre-trained VGG-16 model from Keras, and transforming it from
# type .model to type .sequential which will allow us to modify the model in the way we want it
vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()
type(vgg16_model)

# remove the last dense softmax layer which could output 1000 classes probability to
# include a 2 nodes dense layer
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

model.summary()

# Next, we’ll iterate over each of the layers in our new Sequential model and set them to be non-trainable. This
# freezes the weights and other trainable parameters in each layer so that they will not be trained or updated when
# we later pass in our images of cats and dogs.
for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=80, activation='softmax'))

model.summary()

# now we will train our pre-trained VGG-16 model using our train set
# use a pretreined model helps to speed up the training as the loss function and accuracy
# starts with an already good value and just get better within few epochs
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=15,
          verbose=2
          )

test_imgs, test_labels = next(test_batches)
# plotImages(test_imgs)
print(test_labels)

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

#model.save("vgg16_80Tools.h5")
print("Saved model to disk")

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.show()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

cm_plot_labels = ['class1', 'class2','class3', ..., 'classN']
#plt.show(cm)
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')



