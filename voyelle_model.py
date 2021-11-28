import numpy as np
import tensorflow as tf
import datetime

import keras
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, BatchNormalization
from keras.utils.np_utils import to_categorical
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split




# Function to get the unique elements in an array (here used to get the unique 'voyelles' in the dataset)
def unique(array):
	print(np.unique(array))


# Managing Data

dataset = pd.read_csv('voyelles.txt', sep='\t', usecols=['voyelle', 'F1', 'F2', 'F3', 'F4', 'Z1', 'Z2', 'f0'])

dataset = dataset.replace('--undefined--', np.nan)             # replace the unkown values by np.nan
print(dataset['F4'][66672])                                    # check the value
simple_impute = SimpleImputer(strategy='median')               # Initialise simpleImputer with a median strategy
X = simple_impute.fit_transform(dataset[dataset.columns[1:]])  # replace the np.nan values by the median => X values
Y = dataset[dataset.columns[0]]                                # get the Y values => voyelles dataset

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, train_size=0.80)            # get train dataset 80%
x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.50, train_size=0.50)  # get test and validation datasets 50% of 20%


print(unique(Y))
# List of voyelles ['@' 'E' 'a' 'c' 'e' 'i' 'o' 'u' 'x' 'y']
# So 10 classes

# Encode the characters
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_train = to_categorical(y_train, 10)

y_test = label_encoder.fit_transform(y_test)
y_test = to_categorical(y_test, 10)

y_valid = label_encoder.fit_transform(y_valid)
y_valid = to_categorical(y_valid, 10)

# Create the  binary_accuracy model
model_binary_accuracy = Sequential()
model_binary_accuracy.add(BatchNormalization(input_shape=(7,)))      # 7 values in the input set
model_binary_accuracy.add(Dense(200, activation='sigmoid'))
model_binary_accuracy.add(Dense(200, activation='sigmoid'))
model_binary_accuracy.add(Dense(10, activation='softmax'))           # 10 values in the output set => nb_classes
model_binary_accuracy.compile(optimizer='adam',
							  loss='binary_crossentropy',
							  metrics=['binary_accuracy'])           # binary_accuracy : Calculates how often predictions match binary labels

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath="model/best_model_binary_accuracy_voyelles",
	monitor='val_binary_accuracy',
	mode='max',
	verbose=1,
	save_best_only=True)

model_binary_accuracy.fit(x=x_train,
						  y=y_train,
						  epochs=20,
						  validation_data=(x_valid, y_valid),
						  callbacks=[model_checkpoint_callback, tensorboard_callback])

best_model_binary_accuracy = keras.models.load_model("model/best_model_binary_accuracy_voyelles")

score = best_model_binary_accuracy.evaluate(x_test, y_test, verbose=0)
print('Test score for best_model_binary_accuracy :', score[0])
print('Test accuracy for best_model_binary_accuracy :', score[1])

#Create the accuracy model
model_accuracy = Sequential()
model_accuracy.add(BatchNormalization(input_shape=(7,)))      # 7 values in the input set
model_accuracy.add(Dense(200, activation='sigmoid'))
model_accuracy.add(Dense(200, activation='sigmoid'))
model_accuracy.add(Dense(10, activation='softmax'))           # 10 values in the output set => nb_classes
model_accuracy.compile(optimizer='adam',
					   loss='binary_crossentropy',
					   metrics=['accuracy'])                  # accuracy : Calculates how often predictions equal labels

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath="model/best_model_accuracy_voyelles",
	monitor='val_accuracy',
	mode='max',
	verbose=1,
	save_best_only=True)

model_accuracy.fit(x=x_train,
				   y=y_train,
				   epochs=20,
				   validation_data=(x_valid, y_valid),
				   callbacks=[model_checkpoint_callback, tensorboard_callback])

best_model_accuracy = keras.models.load_model("model/best_model_accuracy_voyelles")

score = best_model_accuracy.evaluate(x_test, y_test, verbose=0)
print('Test score best_model_accuracy:', score[0])
print('Test accuracy best_model_accuracy:', score[1])


#Create the accuracy model
model_accuracy = Sequential()
model_accuracy.add(BatchNormalization(input_shape=(7,)))      # 7 values in the input set
model_accuracy.add(Dense(200, activation='sigmoid'))
model_accuracy.add(Dense(200, activation='sigmoid'))
model_accuracy.add(Dense(10, activation='softmax'))           # 10 values in the output set => nb_classes
model_accuracy.compile(optimizer='adam',
					   loss='binary_crossentropy',
					   metrics=['categorical_accuracy'])                  # categorical_accuracy : Calculates how often predictions match one-hot labels.

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath="model/best_model_categorical_accuracy_voyelles",
	monitor='val_categorical_accuracy',
	mode='max',
	verbose=1,
	save_best_only=True)

model_accuracy.fit(x=x_train,
				   y=y_train,
				   epochs=20,
				   validation_data=(x_valid, y_valid),
				   callbacks=[model_checkpoint_callback, tensorboard_callback])

best_model_accuracy = keras.models.load_model("model/best_model_categorical_accuracy_voyelles")

score = best_model_accuracy.evaluate(x_test, y_test, verbose=0)
print('Test score best_model_categorical_accuracy:', score[0])
print('Test accuracy best_model_categorical_accuracy:', score[1])
