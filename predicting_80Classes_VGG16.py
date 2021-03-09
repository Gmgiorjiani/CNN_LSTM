# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import warnings
import csv
warnings.simplefilter(action='ignore', category=FutureWarning)

# load model
model = load_model('vgg16_80Tools.h5')
# summarize model
model.summary()
# set test images paths
test_path = '/media/proactionlab/Storage/NeuralNet_80Tools/ImageSet/test'

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
                                                                                'varinha_magica', 'vassoura'], batch_size=800, shuffle=False)

test_imgs, test_labels = next(test_batches)
print(test_labels)

predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

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
    with open('tmp_file3.txt', 'w') as f:
        csv.writer(f, delimiter=' ').writerows(cm)

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

cm_plot_labels = ['abre_caricas', 'afia-lapis',
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
                  'varinha_magica', 'vassoura']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')