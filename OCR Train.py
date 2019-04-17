import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import LambdaCallback
import cv2


def inflate_answers(answers):   # Increases emnist y-set values by 9 to accomadate mnist set
    output = answers
    for i in range(len(output)):
        output[i] += 9
    return output


def preprocess_emnist(images):   # Normalizes emnist
    images = np.array(images)
    images = images.reshape(images.shape[0], 1, 28, 28)
    make_it_right(images)
    images = images.astype('float32')
    images /= 255
    return images


def make_it_right(letters):   # Rotates 90 degrees and flips vertically, to orient emnist properly
    output = letters
    for i in output:
        for n in range(len(i)):
            i[n] = np.rot90(i[n])
            i[n] = np.flipud(i[n])
    return output


def multiply_mnist(images, number_of_times):   # Adds set number of transformation sets
    output = images
    if number_of_times - 1 != 0:
        for i in range(0, (number_of_times - 1)):
            output = np.concatenate((output, images))
    return output


def multiply_answers(answers, number_of_times):   # Adds y value array to itself a certain number of times
    output = answers
    if number_of_times - 1 != 0:
        for i in range(0, (number_of_times - 1)):
            output = np.concatenate((output, answers))
    return output


def preprocess_mnist(images):   # Normalizes mnist
    images = np.array(images)
    images = images.reshape(images.shape[0], 1, 28, 28)
    images = images.astype('float32')
    images /= 255
    return images


# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess and transform MNIST data
X_train = preprocess_mnist(X_train[0:10])
X_test = preprocess_mnist(X_test[0:10])
X_train = multiply_mnist(X_train, 2)
X_test = multiply_mnist(X_test, 2)
y_train = multiply_answers(y_train[0:10], 2)
y_test = multiply_answers(y_test[0:10], 2)

# Preprocess MNIST class labels
y_train = np_utils.to_categorical(y_train, 37)
y_test = np_utils.to_categorical(y_test, 37)

# Load EMNIST data
emnist = spio.loadmat('emnist-letters.mat')
x_trainE = emnist['dataset'][0][0][0][0][0][0]
x_testE = emnist['dataset'][0][0][1][0][0][0]
y_trainE = emnist['dataset'][0][0][0][0][0][1]
y_testE = emnist['dataset'][0][0][1][0][0][1]

# Prepocess and transform EMNIST data
x_trainE = preprocess_emnist(x_trainE[0:10])
x_testE = preprocess_emnist(x_testE[0:10])
x_trainE = multiply_mnist(x_trainE, 1)
x_testE = multiply_mnist(x_testE, 1)
y_trainE = multiply_answers(y_trainE[0:10], 1)
y_testE = multiply_answers(y_testE[0:10], 1)
y_trainE = inflate_answers(y_trainE)
y_testE = inflate_answers(y_testE)

# Preprocess EMNIST class labels
y_trainE = np_utils.to_categorical(y_trainE, 37)
y_testE = np_utils.to_categorical(y_testE, 37)


# Combine all training and test data
x_train_combined = np.array(np.concatenate((X_train, x_trainE)))
x_test_combined = np.array(np.concatenate((X_test, x_testE)))
y_train_combined = np.array(np.concatenate((y_train, y_trainE)))
y_test_combined = np.array(np.concatenate((y_test, y_testE)))

# Print training and test data sizes
print('x_train_combined shape: (%s, %s, %s, %s)'
      % (x_train_combined.shape[0], x_train_combined.shape[1],
         x_train_combined.shape[2], x_train_combined.shape[3])
      )   # (244800, 1, 28, 28)
print('y_train_combined shape: (%s, %s)'
      % (y_train_combined.shape[0], y_train_combined.shape[1])
      )   # (244800, 37)
print('x_test_combined shape: (%s, %s, %s, %s)'
      % (x_test_combined.shape[0], x_test_combined.shape[1],
         x_test_combined.shape[2], x_test_combined.shape[3])
      )
print('y_test_combined shape: (%s, %s)'
      % (y_test_combined.shape[0], y_test_combined.shape[1])
      )


# Defining model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28), padding="same", data_format="channels_first"))
print('model output shape: (%s, %s, %s, %s)'
      % (model.output_shape[0], model.output_shape[1],
         model.output_shape[2], model.output_shape[3])
      )   # (None, 32, 26, 26)
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", data_format="channels_first"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(.5))
model.add(Dense(37, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Define callbacks and information storage locations
val_acc_list = []
val_loss_list = []
callbacks_list = [
    LambdaCallback(on_epoch_end=lambda epoch,
                   logs: val_loss_list.append(model.evaluate(x_test_combined, y_test_combined, verbose=0)[0])),
    LambdaCallback(on_epoch_end=lambda epoch,
                   logs: val_acc_list.append(model.evaluate(x_test_combined, y_test_combined, verbose=0)[1]))
]

# Fit model on training data
history = model.fit(x_train_combined, y_train_combined,
                    batch_size=32, epochs=10, verbose=1, callbacks=callbacks_list)

# Print history object's keys
history_keys = str(history.history.keys())
history_keys = history_keys[10:(len(history_keys) - 1)]
print("Model history keys: %s" % history_keys)

# Save model weights
model.save_weights('OCR.h5')

# Evaluate model on test data
score = model.evaluate(x_test_combined, y_test_combined, verbose=0)
print("Score: %s" % score)

# Print accuracy graph
plt.plot(history.history['acc'])
plt.plot(val_acc_list)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Print loss graph
plt.plot(history.history['loss'])
plt.plot(val_loss_list)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Print loss and accuracy lists
print('Model accuracy list: %s' % history.history['acc'])
print('Model loss list: %s' % history.history['loss'])
print('Model validation accuracy list: %s' % val_acc_list)
print('Model validation loss list: %s' % val_loss_list)

# Write loss and accuracy lists to file
file = open('model_history.txt', 'w')
file.write('Model accuracy list: %s' % history.history['acc'])
file.write('\n')
file.write('Model loss list: %s' % history.history['loss'])
file.write('\n')
file.write('Model validation accuracy list: %s' % val_acc_list)
file.write('\n')
file.write('Model validation loss list: %s' % val_loss_list)
file.write('\n')
file.close()
