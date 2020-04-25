import tensorflow as tf
from tensorflow import keras

from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Deconvolution2D, UpSampling2D, Dropout
from keras.models import Model
from keras.models import load_model


class DNN:
    def __init__(self, model=None):
        self.model = model

    def summary(self):
        return self.model.summary()

    def fit(self, x, y, epochs, callbacks=[], validation_data=None):
        self.model.fit(x, y, epochs=epochs, callbacks=callbacks, validation_data=validation_data)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def load_from_path(self, path: str):
        self.model = load_model(path)
        return self.model

    def save_to_path(self, path: str):
        self.model.save(path)

    def compile(self, optimizer, metrics):
        pass

    def layer_shapes(self):
        return '\n'.join(list(map(lambda x: x.input_shape[1:], self.model.layers)))

    def get_keras_model(self):
        return self.model

    def get_layers(self):
        return self.model.layers


class DenoisingAutoEncoder(DNN):
    def __init__(self, path=None, shape=(None, None, None), optimizer='adam', loss='mean_squared_error',
                 metrics=['acc']):
        """
        :param path: path to model if exists
        :param metrics: matrics for evaluation
        :param shape: shape of input
        :param optimizer: adam optimizer is the default
        :param loss:

        """

        super().__init__()
        inputs = Input(shape=shape)
        conv1 = Conv2D(filters=20, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        max1 = MaxPool2D((2, 2))(conv1)

        drop = Dropout(rate=0.05)(max1)

        conv2 = Conv2D(filters=20, kernel_size=(3, 3), activation='relu', padding='same')(drop)
        max2 = MaxPool2D((2, 2))(conv2)
        conv3 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(max2)

        deconv3 = Deconvolution2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(conv3)

        upsampling1 = UpSampling2D((2, 2))(deconv3)
        deconv2 = Deconvolution2D(filters=20, kernel_size=(3, 3), activation='relu', padding='same')(upsampling1)
        upsampling2 = UpSampling2D((2, 2))(deconv2)

        # deconv2 = Deconvolution2D(filters=20,kernel_size=(3,3), activation='relu') (upsampling1)
        inv_input = Deconvolution2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same')(
            upsampling2)  # probably you will have a problem with padding
        encoder = Model(inputs, max2)
        super().__init__(Model(inputs, inv_input))
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        if path:
            self.load_from_path(path)

    def compile(self, optimizer=None, metrics=None):
        optimizer = optimizer if optimizer else self.optimizer
        metrics = metrics if metrics else self.metrics
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)


class Encoder(DNN):
    def __init__(self, path):
        super().__init__(self.load_from_path(path))


class Decoder(DNN):
    def __init__(self, path):
        super(Decoder, self).__init__(self.load_from_path(path))


class CNN(DNN):
    def __init__(self, path: str, input_shape=(28, 28, 1), num_classes=10):

        cnn = [
            Input(input_shape),
            Conv2D(filters=10, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPool2D((2, 2)),
            Conv2D(filters=10, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPool2D((2, 2)),
            Conv2D(filters=10, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPool2D((2, 2)),
            Conv2D(filters=5, kernel_size=(3, 3), padding='same', activation='relu'),

            Flatten(),
            Dense(units=1000, activation='relu'),
            Dense(units=1000, activation='relu'),
            Dense(units=num_classes, activation='softmax')
        ]
        x = cnn[0]
        for layer in cnn:
            x = layer(x)
        super().__init__(Model(x, cnn[0]))
        if path:
            self.load_from_path(path=path)


class AnyDNN(DNN):
    def __init__(self, path: str):
        super(AnyDNN, self).__init__()
        self.load_from_path(path)


if __name__ == '__main__':
    path = '../Models/encoder_ann.h5'
    model = load_model(path)

