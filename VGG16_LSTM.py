import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN, Reshape, TimeDistributed
from keras import regularizers, optimizers
import pandas as pd
from TimeDistributedImageDataGenerator import TimeDistributedImageDataGenerator
from tensorflow.keras.models import load_model

df=pd.read_csv('~/allLabels_shortTrainer.csv')
columns=['label1', 'label2', 'label3', ..., 'labelN']

num_labels = 855
batch_size = 10

datagen = TimeDistributedImageDataGenerator(time_steps = 20)
test_datagen=TimeDistributedImageDataGenerator(time_steps = 20)
train_generator=datagen.flow_from_dataframe(

dataframe=df[:64000],
directory="~/images/train",
x_col="Filenames",
y_col=columns,
color_mode='rgb',
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))

valid_generator=datagen.flow_from_dataframe(
dataframe=df[64000:76160],
directory="~/images/valid",
x_col="Filenames",
y_col=columns,
color_mode='rgb',
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode="raw",
target_size=(224,224))

test_generator=datagen.flow_from_dataframe(
dataframe=df[76162:],
directory="~/images/test",
x_col="Filenames",
color_mode='rgb',
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(224,224))

# Building the model
model = Sequential()
model.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape=(None,20, 224, 224, 3))))
model.add(TimeDistributed(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(2,2))))
model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(2,2))))
model.add(TimeDistributed(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(2,2))))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(2,2))))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(2,2))))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(units=4096,activation="relu")))
model.add(TimeDistributed(Dense(units=4096,activation="relu")))
model.add(LSTM(units=855, activation="sigmoid"))
model.compile(optimizers.Adam(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])

imgs, labels = next(train_generator)
print(imgs.shape)

model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(train_generator),
                    validation_data=valid_generator,
                    validation_steps=len(valid_generator),
                    epochs=45,

model.summary()

model.save("vgg16_attractor_80tools.h5")
print("Saved model to disk")



