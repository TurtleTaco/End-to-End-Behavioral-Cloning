import csv
import cv2
from matplotlib import pyplot as plt
import numpy as np
import keras
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split

__load_X_train_example__ = __load_y_train_example__ = __load_X_valid_example__ = __load_y_valid_example__ = True
__train_model__ = True
camera_correction = 0.2
validation_percentage = 0.2
train_batch_size = 128
valid_batch_size = 128
X_train = []
y_train = []
X_valid = []
y_train = []

# Because of random split, update either X_train or y_train needs to update both array
if __load_X_train_example__ == False or __load_y_train_example__ == False or __load_X_valid_example__ == False or __load_y_valid_example__ == False:
    lines = []
    with open("./data_new/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    # Split train and validation sets
    train_lines, validation_line = train_test_split(lines, test_size=validation_percentage)
    # Further splits validation data into the actual validation data used in training and a small portion for testing
    validation_line, test_line = train_test_split(validation_line, test_size=0.01)

if __load_X_train_example__ == False and __load_y_train_example__ == False:
    print("Loading test data from scratch")
    test_images = []
    test_measurements = []
    for line in test_line:
        if line[0] == "center":
            continue
        else:
            source_path = line[0]
            test_measurements.append(float(line[3]))
            local_path = "./data_new/IMG/" + source_path.split('/')[-1]
            BGR_image = cv2.imread(local_path)
            test_images.append(cv2.cvtColor(BGR_image, cv2.COLOR_BGR2RGB))

    print("Loading training data from scratch")
    images = []
    measurements = []
    for line in train_lines:
        if line[0] == "center":
            # first line is column name, skip
            continue
        source_path = []
        for i in range(3):
            source_path.append(line[i])
            if i == 0:
                measurements.append(float(line[3])) # center
            elif i == 1:
                measurements.append(float(line[3]) + camera_correction) # left
            elif i == 2:
                measurements.append(float(line[3]) - camera_correction) # right

        for path in source_path:
            local_path = "./data_new/IMG/" + path.split('/')[-1]
            BGR_image = cv2.imread(local_path)
            images.append(cv2.cvtColor(BGR_image, cv2.COLOR_BGR2RGB))

    X_test = np.array(test_images)
    np.save("X_test_example.npy", X_test)
    y_test = np.array(test_measurements)
    np.save("y_test_example.npy", y_test)
    print("Finished loading X_test and y_test from scratch")

    X_train = np.array(images)
    np.save("X_train_example.npy", X_train)
    print("Finished loading X_train from scratch")

    y_train = np.array(measurements)
    np.save("y_train_example_" + str(camera_correction) + ".npy", y_train)
    print("Finished loading y_train from scratch")

if __load_X_valid_example__ == False and __load_y_valid_example__ == False:
    print("Loading valid data from scratch")
    images = []
    measurements = []
    for line in validation_line:
        if line[0] == "center":
            # first line is column name, skip
            continue
        source_path = []
        for i in range(3):
            source_path.append(line[i])
            if i == 0:
                measurements.append(float(line[3]))  # center
            elif i == 1:
                measurements.append(float(line[3]) + camera_correction)  # left
            elif i == 2:
                measurements.append(float(line[3]) - camera_correction)  # right

        for path in source_path:
            local_path = "./data_new/IMG/" + path.split('/')[-1]
            BGR_image = cv2.imread(local_path)
            images.append(cv2.cvtColor(BGR_image, cv2.COLOR_BGR2RGB))

    X_valid = np.array(images)
    np.save("X_valid_example.npy", X_valid)
    print("Finished loading X_valid from scratch")

    y_valid = np.array(measurements)
    np.save("y_valid_example_" + str(camera_correction) + ".npy", y_valid)
    print("Finished loading y_valid from scratch")

# Or load from *.npy
if __load_X_train_example__ == True and __load_y_train_example__ == True:
    # load from npy
    print("Loading from X_test_example.npy")
    X_test = np.load("X_test_example.npy")
    print("  Loaded from X_test_example.npy")
    print("Loading from y_test_example.npy")
    y_test = np.load("y_test_example.npy")
    print("  Loaded from y_test_example.npy")

    print("Loading from X_train_example.npy")
    X_train = np.load("X_train_example.npy")
    print("  Loaded from X_train_example.npy")
    print("Loading from y_train_example_" + str(camera_correction) + ".npy")
    y_train = np.load("y_train_example_" + str(camera_correction) + ".npy")
    print("  Loaded from y_train_example_"  + str(camera_correction) + ".npy")

if __load_X_valid_example__ == True and __load_y_valid_example__ == True:
    # load from npy
    print("Loading from X_valid_example.npy")
    X_valid = np.load("X_valid_example.npy")
    print("  Loaded from X_valid_example.npy")
    print("Loading from y_valid_example_" + str(camera_correction) + ".npy")
    y_valid = np.load("y_valid_example_" + str(camera_correction) + ".npy")
    print("  Loaded from y_valid_example_"  + str(camera_correction) + ".npy")

# plt.imshow(X_train[0])
# plt.show()
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

def generator(input_samples, label_samples, batch_size=128):
    num_samples = len(input_samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            data_batch = input_samples[offset:offset+batch_size]
            lable_batch = label_samples[offset:offset+batch_size]

            yield sklearn.utils.shuffle(data_batch, lable_batch)

def augmentation(datas, labels, datas_aug, labels_aug):
    for data, label in zip(datas, labels):
        datas_aug.append(data)
        labels_aug.append(label)
        flipped_data = cv2.flip(data, 1)
        flipped_label = float(label * -1.0)
        datas_aug.append(flipped_data)
        labels_aug.append(flipped_label)


# data augmentation on X_train, y_train and X_valid, y_valid
X_train_aug = []
y_train_aug = []
X_valid_aug = []
y_valid_aug = []

augmentation(X_train, y_train, X_train_aug, y_train_aug)
augmentation(X_valid, y_valid, X_valid_aug, y_valid_aug)

X_train_array = np.array(X_train_aug)
y_train_array = np.array(y_train_aug)
X_valid_array = np.array(X_valid_aug)
y_valid_array = np.array(y_valid_aug)

print(X_train_array.shape)
print(y_train_array.shape)

if __train_model__:
    # compile and train the model using the generator function
    train_generator = generator(X_train_array, y_train_array, batch_size = train_batch_size)
    validation_generator = generator(X_valid_array, y_valid_array, batch_size = valid_batch_size)

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit_generator(train_generator, samples_per_epoch= len(X_train)/train_batch_size, validation_data=validation_generator, nb_val_samples=len(X_valid), nb_epoch=5)

    model.save("model.h5")

    accuracy = model.evaluate(X_test, y_test, batch_size=1, verbose=1)
    print("Test accuracy:", accuracy)

