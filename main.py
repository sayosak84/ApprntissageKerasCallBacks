import tensorflow as tf
import datetime

from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, BatchNormalization
from keras.utils.np_utils import to_categorical

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()





#model = create_model()
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


x_train = x_train.reshape(60000, 28*28)        # nb_dimension est 28*28 = 784 puisque ce sont des images d'une dimension de 28*28 pixels
x_test = x_test.reshape(10000, 28*28)

y_train = to_categorical(y_train, 10)           # le nb_classes est égale à 10 puisque ce sont les 10 chiffres possible
y_test = to_categorical(y_test, 10)

model.fit(x=x_train,
          y=y_train,
          epochs=20,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])



EPOCHS = 10
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# The model weights (that are considered the best) are loaded into the model.
modelBest = model.load_weights(checkpoint_filepath)

