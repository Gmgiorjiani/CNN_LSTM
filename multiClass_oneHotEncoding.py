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

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("Num GPUs Available: ", len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Getting our train, valid and test sets

train_path = '/media/proactionlab/Storage/NeuralNet_80Tools/ImageSet/new_train_augmented'
valid_path = '/media/proactionlab/Storage/NeuralNet_80Tools/ImageSet/new_valid'
test_path = '/media/proactionlab/Storage/NeuralNet_80Tools/ImageSet/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['abre_caricas', 'afia-lapis',
                                                                                'agrafador', 'agulha', 'alicate',
                                                                                'apagador', 'apito', 'balde',
                                                                                'batedeira', 'berbequim', 'boia',
                                                                                'bola_de_basquete', 'borracha',
                                                                                'borrifador', 'broca', 'buzina',
                                                                                'cabide', 'cana_de_pesca', 'canivete',
                                                                                'carimbo', 'carrinho_compras',
                                                                                'castical', 'chave', 'chave_inglesa',
                                                                                'chavena', 'clip', 'colher',
                                                                                'colher_pau', 'copo', 'corta_unhas',
                                                                                'dardos', 'descascador', 'desentupidor',
                                                                                'enxada', 'escova_cabelo',
                                                                                'escova_dentes', 'esfregona',
                                                                                'esponja', 'espremedor', 'faca',
                                                                                'fosforo', 'furador', 'garfo',
                                                                                'garrafa', 'guardanapo', 'isqueiro',
                                                                                'jarro', 'lanterna', 'lapis', 'leque',
                                                                                'lima', 'lupa', 'manipulo_de_porta',
                                                                                'maquina_de_barbear', 'martelo',
                                                                                'moedor_de_pimenta', 'mola_de_roupa',
                                                                                'pa', 'parafuso', 'peso', 'piao',
                                                                                'pinca', 'pincel', 'prego',
                                                                                'quebra_nozes', 'ralador', 'raquete',
                                                                                'rato_de_pc', 'remo', 'rolha',
                                                                                'rolo_de_massa', 'saco_de_pasteleiro',
                                                                                'secador', 'seringa', 'taco_de_golf',
                                                                                'tampa_de_garrafa', 'tesoura', 'tigela',
                                                                                'varinha_magica', 'vassoura'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['abre_caricas', 'afia-lapis',
                                                                                'agrafador', 'agulha', 'alicate',
                                                                                'apagador', 'apito', 'balde',
                                                                                'batedeira', 'berbequim', 'boia',
                                                                                'bola_de_basquete', 'borracha',
                                                                                'borrifador', 'broca', 'buzina',
                                                                                'cabide', 'cana_de_pesca', 'canivete',
                                                                                'carimbo', 'carrinho_compras',
                                                                                'castical', 'chave', 'chave_inglesa',
                                                                                'chavena', 'clip', 'colher',
                                                                                'colher_pau', 'copo', 'corta_unhas',
                                                                                'dardos', 'descascador', 'desentupidor',
                                                                                'enxada', 'escova_cabelo',
                                                                                'escova_dentes', 'esfregona',
                                                                                'esponja', 'espremedor', 'faca',
                                                                                'fosforo', 'furador', 'garfo',
                                                                                'garrafa', 'guardanapo', 'isqueiro',
                                                                                'jarro', 'lanterna', 'lapis', 'leque',
                                                                                'lima', 'lupa', 'manipulo_de_porta',
                                                                                'maquina_de_barbear', 'martelo',
                                                                                'moedor_de_pimenta', 'mola_de_roupa',
                                                                                'pa', 'parafuso', 'peso', 'piao',
                                                                                'pinca', 'pincel', 'prego',
                                                                                'quebra_nozes', 'ralador', 'raquete',
                                                                                'rato_de_pc', 'remo', 'rolha',
                                                                                'rolo_de_massa', 'saco_de_pasteleiro',
                                                                                'secador', 'seringa', 'taco_de_golf',
                                                                                'tampa_de_garrafa', 'tesoura', 'tigela',
                                                                                'varinha_magica', 'vassoura'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['abre_caricas', 'afia-lapis',
                                                                                'agrafador', 'agulha', 'alicate',
                                                                                'apagador', 'apito', 'balde',
                                                                                'batedeira', 'berbequim', 'boia',
                                                                                'bola_de_basquete', 'borracha',
                                                                                'borrifador', 'broca', 'buzina',
                                                                                'cabide', 'cana_de_pesca', 'canivete',
                                                                                'carimbo', 'carrinho_compras',
                                                                                'castical', 'chave', 'chave_inglesa',
                                                                                'chavena', 'clip', 'colher',
                                                                                'colher_pau', 'copo', 'corta_unhas',
                                                                                'dardos', 'descascador', 'desentupidor',
                                                                                'enxada', 'escova_cabelo',
                                                                                'escova_dentes', 'esfregona',
                                                                                'esponja', 'espremedor', 'faca',
                                                                                'fosforo', 'furador', 'garfo',
                                                                                'garrafa', 'guardanapo', 'isqueiro',
                                                                                'jarro', 'lanterna', 'lapis', 'leque',
                                                                                'lima', 'lupa', 'manipulo_de_porta',
                                                                                'maquina_de_barbear', 'martelo',
                                                                                'moedor_de_pimenta', 'mola_de_roupa',
                                                                                'pa', 'parafuso', 'peso', 'piao',
                                                                                'pinca', 'pincel', 'prego',
                                                                                'quebra_nozes', 'ralador', 'raquete',
                                                                                'rato_de_pc', 'remo', 'rolha',
                                                                                'rolo_de_massa', 'saco_de_pasteleiro',
                                                                                'secador', 'seringa', 'taco_de_golf',
                                                                                'tampa_de_garrafa', 'tesoura', 'tigela',
                                                                                'varinha_magica', 'vassoura'], batch_size=10,
                                                                                shuffle=False)

imgs, labels = next(train_batches)

plt.imshow(imgs[[1]])
plt.show()

print(labels[[1]])
# importing pre-trained VGG-16 model from Keras, and transforming it from
# type .model to type .sequention which will allow us to modify the model in the way we wanna it

vgg16_model = tf.keras.applications.vgg16.VGG16()
vgg16_model.summary()
type(vgg16_model)
