import tensorflow as tf
import datetime

import keras
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, BatchNormalization
from keras.utils.np_utils import to_categorical

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Q1 et Q2 create model, create callbacks, load the best model
model = Sequential()
model.add(BatchNormalization(input_shape=(28*28,)))  # nb d'entrées égale à la dimension des images, on remplace le 400 par 28*28
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="model/best_model",
    monitor='val_categorical_accuracy',
    mode='max',
    verbose=1,
    save_best_only=True)




x_train = x_train.reshape(60000, 28*28)        # nb_dimension est 28*28 = 784 puisque ce sont des images d'une dimension de 28*28 pixels
x_test = x_test.reshape(10000, 28*28)

y_train = to_categorical(y_train, 10)           # le nb_classes est égale à 10 puisque ce sont les 10 chiffres possible
y_test = to_categorical(y_test, 10)

model.fit(x=x_train,
          y=y_train,
          epochs=20,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback, model_checkpoint_callback])




best_model = keras.models.load_model("model/best_model")

score = best_model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
