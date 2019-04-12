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


def slice_im_generator(images, number_of_times):   # Create random 28 x 28 image slices (1/4) from input images
    output = rand_slice(images[random.randint(0, len(images)-1)])
    i = 1
    while i < number_of_times:
        output = np.concatenate((output, rand_slice(images[random.randint(0, len(images)-1)])))
        i += 1
    return output


def rand_slice(image):   # Returns 1 random (28x28 reshaped) quarter form 28x28 image
    template_q = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    q1 = template_q.copy()
    for n1 in range(0, 14):
        q1[n1] = image[0][n1][0:14]
    q2 = template_q.copy()
    for n2 in range(0, 14):
        q2[n2] = image[0][n2][14:28]
    q3 = template_q.copy()
    for n3 in range(0, 14):
        q3[n3] = image[0][n3+14][0:14]
    q4 = template_q.copy()
    for n4 in range(0, 14):
        q4[n4] = image[0][n4+14][14:28]
    q1 = cv2.resize(q1, (28, 28))
    q2 = cv2.resize(q2, (28, 28))
    q3 = cv2.resize(q3, (28, 28))
    q4 = cv2.resize(q4, (28, 28))
    q1 = q1.reshape(1, 1, 28, 28)
    q2 = q2.reshape(1, 1, 28, 28)
    q3 = q3.reshape(1, 1, 28, 28)
    q4 = q4.reshape(1, 1, 28, 28)
    output = np.concatenate((q1, q2, q3, q4))
    output = np.array(output[random.randint(0, 3)])
    output = output.reshape(1, 1, 28, 28)
    return output


def blank_generator(original, number_of_times):   # randomly generates blanks with placeholder dimension
    new_blank = np.array(original.astype('float32'))
    blanks_list = np.array(original.astype('float32'))
    i = 1
    while i <= number_of_times:
        num = random.uniform(0.0, 0.5)
        var_max = random.uniform(0.0, 0.5)
        i = i + 1
        for n in new_blank:
            for a in n:
                for b in range(len(a)):
                    a[b] = (num + random.uniform(0.0, var_max))
            blanks_list = np.concatenate((blanks_list, new_blank))
    blanks_list = blanks_list[1:len(blanks_list)]
    return blanks_list.reshape(blanks_list.shape[0], 1, 28, 28)


def create_blank_y(number_of_times):   # generates 'blank' class labels
    output = []
    for i in range(number_of_times):
        output.append(36)
    return np.array(output)


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
    output = mnist_transformer(images)
    if number_of_times - 1 != 0:
        for i in range(0, (number_of_times - 1)):
            output = np.concatenate((output, mnist_transformer(images)))
    return output


def multiply_answers(answers, number_of_times):   # Adds y value array to itself a certain number of times
    output = answers
    if number_of_times - 1 != 0:
        for i in range(0, (number_of_times - 1)):
            output = np.concatenate((output, answers))
    return output


def mnist_transformer(images):    # Generates transformed mnist images at random
    images_copy = np.array(images)
    for i in images_copy:
        for n in i:
            background_num = random.uniform(0.65, 0.82)
            background_varmax = random.uniform(0.0, 0.13)
            letter_num = random.uniform(0.0, 0.1)
            letter_varmax = random.uniform(0.0, 0.01)
            for a in n:
                for b in range(len(a)):
                    if a[b] == 0.0:
                        a[b] = background_num + random.uniform(0.0, background_varmax)
                    else:
                        a[b] = 1 - a[b]
                        if a[b] + (letter_num + letter_varmax) >= background_num:
                            a[b] = background_num - (letter_num + letter_varmax)
                        a[b] = a[b] + (letter_num + random.uniform(0.0, letter_varmax))
    return images_copy


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
x_trainE = multiply_mnist(x_trainE, 2)
x_testE = multiply_mnist(x_testE, 2)
y_trainE = multiply_answers(y_trainE[0:10], 2)
y_testE = multiply_answers(y_testE[0:10], 2)
y_trainE = inflate_answers(y_trainE)
y_testE = inflate_answers(y_testE)

# Preprocess EMNIST class labels
y_trainE = np_utils.to_categorical(y_trainE, 37)
y_testE = np_utils.to_categorical(y_testE, 37)

# Template for static blanks
template_image = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])

# Create and process static blanks
x_trainB = blank_generator(template_image, 10)
x_testB = blank_generator(template_image, 10)

# Create and preprocess static blank class labels
y_trainB = create_blank_y(10)
y_testB = create_blank_y(10)
y_trainB = np_utils.to_categorical(y_trainB, 37)
y_testB = np_utils.to_categorical(y_testB, 37)

# Create and process slice blanks
x_trainS = np.concatenate((slice_im_generator(x_trainE, 5), slice_im_generator(X_train, 5)))
x_testS = np.concatenate((slice_im_generator(x_testE, 5), slice_im_generator(X_test, 5)))

# Create and preprocess slice blank class labels
y_trainS = create_blank_y(10)
y_testS = create_blank_y(10)
y_trainS = np_utils.to_categorical(y_trainS, 37)
y_testS = np_utils.to_categorical(y_testS, 37)

# Combine all training and test data
x_train_combined = np.array(np.concatenate((X_train, x_trainE, x_trainB, x_trainS)))
x_test_combined = np.array(np.concatenate((X_test, x_testE, x_testB, x_testS)))
y_train_combined = np.array(np.concatenate((y_train, y_trainE, y_trainB, y_trainS)))
y_test_combined = np.array(np.concatenate((y_test, y_testE, y_testB, y_testS)))

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
print('y_train_combined shape: (%s, %s)'
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
model.save_weights('OCR_M.h5')

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
file = open('model_history_M.txt', 'w')
file.write('Model accuracy list: %s' % history.history['acc'])
file.write('\n')
file.write('Model loss list: %s' % history.history['loss'])
file.write('\n')
file.write('Model validation accuracy list: %s' % val_acc_list)
file.write('\n')
file.write('Model validation loss list: %s' % val_loss_list)
file.write('\n')
file.close()
