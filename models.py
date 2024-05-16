from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def get_model_1(sizes, image_size):
    model = Sequential()

    model.add(Conv2D(sizes[0], (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)))

    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Conv2D(sizes[1], (3, 3), activation="relu"))

    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Flatten())

    model.add(Dense(sizes[2], activation="relu"))
    model.add(Dense(sizes[1], activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    return model


def get_model_2(sizes, image_size):
    model = Sequential()

    model.add(Conv2D(sizes[0], (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)))

    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Conv2D(sizes[1], (3, 3), activation="relu"))

    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Conv2D(sizes[2], (3, 3), activation="relu"))

    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Conv2D(sizes[2], (3, 3), activation="relu"))

    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Flatten())

    model.add(Dense(sizes[2], activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    return model


def get_model_3(sizes, image_size):
    model = Sequential()

    model.add(Conv2D(sizes[0], (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)))

    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Conv2D(sizes[1], (3, 3), activation="relu"))

    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Conv2D(sizes[2], (3, 3), activation="relu"))

    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Flatten())

    model.add(Dense(sizes[3], activation="relu"))

    model.add(Dense(sizes[2], activation="relu"))

    model.add(Dense(sizes[3], activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    return model
