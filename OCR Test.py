import numpy as np
from scipy import io as spio
<<<<<<< HEAD
=======
from random import shuffle
>>>>>>> 575a4b83d9eaf458dfa244f948f372c8796d44a3
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
from matplotlib import pyplot as plt
import cv2


def convert_to_result(confidences_list):   # Converts array of confidences to class labels
    output = []
    for confidences in confidences_list:
        max_value = [0]
        max_index = 0
        for i in range(len(confidences)):
            if confidences[i] > max_value:
                max_value = confidences[i]
                max_index = i
        if max_index == 0:
            output.append('0')
        elif max_index == 1:
            output.append('1')
        elif max_index == 2:
            output.append('2')
        elif max_index == 3:
            output.append('3')
        elif max_index == 4:
            output.append('4')
        elif max_index == 5:
            output.append('5')
        elif max_index == 6:
            output.append('6')
        elif max_index == 7:
            output.append('7')
        elif max_index == 8:
            output.append('8')
        elif max_index == 9:
            output.append('9')
        elif max_index == 10:
            output.append('A')
        elif max_index == 11:
            output.append('B')
        elif max_index == 12:
            output.append('C')
        elif max_index == 13:
            output.append('D')
        elif max_index == 14:
            output.append('E')
        elif max_index == 15:
            output.append('F')
        elif max_index == 16:
            output.append('G')
        elif max_index == 17:
            output.append('H')
        elif max_index == 18:
            output.append('I')
        elif max_index == 19:
            output.append('J')
        elif max_index == 20:
            output.append('K')
        elif max_index == 21:
            output.append('L')
        elif max_index == 22:
            output.append('M')
        elif max_index == 23:
            output.append('N')
        elif max_index == 24:
            output.append('O')
        elif max_index == 25:
            output.append('P')
        elif max_index == 26:
            output.append('Q')
        elif max_index == 27:
            output.append('R')
        elif max_index == 28:
            output.append('S')
        elif max_index == 29:
            output.append('T')
        elif max_index == 30:
            output.append('U')
        elif max_index == 31:
            output.append('V')
        elif max_index == 32:
            output.append('W')
        elif max_index == 33:
            output.append('X')
        elif max_index == 34:
            output.append('Y')
        elif max_index == 35:
            output.append('Z')
        else:
            output.append('Blank')
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


def preprocess_mnist(images):   # Normalizes mnist
    images = np.array(images)
    images = images.reshape(images.shape[0], 1, 28, 28)
    images = images.astype('float32')
    images /= 255
    return images


def show_images(images):   # Shows test images
    for i in range(len(images)):
        plt.imshow(images[i][0], cmap='gray')
        plt.show()
    return None


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


def preprocess_mnist(images):   # Normalizes mnist
    images = np.array(images)
    images = images.reshape(images.shape[0], 1, 28, 28)
    images = images.astype('float32')
    images /= 255
    return images


# Defining model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28), padding="same", data_format="channels_first"))
print(model.output_shape)   # (None, 32, 28, 28)
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", data_format="channels_first"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(.5))
model.add(Dense(37, activation="softmax"))

# Define test images
# Load pre-shuffled MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess and transform MNIST data
mSet = preprocess_mnist(X_test)

# Load EMNIST data
emnist = spio.loadmat('emnist-letters.mat')
eSet = emnist['dataset'][0][0][1][0][0][0]

# Prepocess and transform EMNIST data
eSet = preprocess_emnist(eSet)

# Shuffle
np.random.shuffle(mSet)
np.random.shuffle(eSet)

# Combine all training and test data
images = np.array(np.concatenate((mSet[0:5], eSet[0:5])))

# Compile the model
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Load model weights
model.load_weights('OCR.h5', by_name=False)

# Make predictions on test images
p = model.predict(images, batch_size=images.shape[0], verbose=0)
print(p)
print(convert_to_result(p))
show_images(images)
