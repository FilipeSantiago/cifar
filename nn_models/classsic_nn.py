import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Model


class ClassicNN:

    def __init__(self):
        pass

    def get_conv_layer(filters, x, dropout=0.4, kernel_size=(3, 3)):
        x = layers.Conv2D(filters, kernel_size=kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
        return x

    def get_nn_model(self):

        x = self.get_conv_layer(64, auto_input, dropout=0.3)
        x = self.get_conv_layer(64, x, dropout=0)
        x = layers.MaxPooling2D((2, 2))(x)

        x = self.get_conv_layer(128, x)
        x = self.get_conv_layer(128, x, dropout=0)
        x = layers.MaxPooling2D((2, 2))(x)

        x = self.get_conv_layer(256, x)
        x = self.get_conv_layer(256, x, dropout=0)
        x = layers.MaxPooling2D((2, 2))(x)

        # x = get_conv_layer(512, x)
        # x = get_conv_layer(512, x)
        # x = get_conv_layer(512, x)
        # x = get_conv_layer(512, x, dropout=0)
        # x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # x = get_conv_layer(512, x)
        # x = get_conv_layer(512, x)
        # x = get_conv_layer(512, x)
        # x = get_conv_layer(512, x, dropout=0)
        # x = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Flatten()(x)

        x = layers.Dense(512, activation='relu')(x)

        x = layers.Dense(10, activation='softmax')(x)

        return x