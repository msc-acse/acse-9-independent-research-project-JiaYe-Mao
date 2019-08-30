import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Model
import matplotlib.pyplot as plt
import math

from keras.models import load_model


from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Concatenate, Lambda

from keras.callbacks import EarlyStopping

from keras.layers.core import Flatten, Reshape
from keras.preprocessing.image import ImageDataGenerator



class Autoencoder_interpolation():
    def __init__(self, nLatent, img_rows, img_cols, ndomain):
        self.nLatent = nLatent
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.dd_num = ndomain

        self.nsplt = int(math.log(self.dd_num, 2))

        self.size = [int(img_rows[i] / 16 * img_cols[i] / 16 * 8) for i in range(ndomain)]


        self.partition = [0]
        for i in range(ndomain):
            self.partition.append(sum(self.size[:i + 1]))
        print(self.partition)
        self.channels = 2
        self.img_shape = [None for i in range(self.dd_num)]
        for i in range(self.dd_num):
            self.img_shape[i] = (self.img_rows[i], self.img_cols[i], self.channels)

        self.autoencoder_model, self.encoder_model, self.decoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer='Nadam')
        self.encoder_model.compile(loss='mse', optimizer='Nadam')
        # self.decoder_model.compile(loss='mse', optimizer='Nadam')
        self.autoencoder_model.summary()

    def build_model(self):
        input_layer = [Input(shape=self.img_shape[i]) for i in range(self.dd_num)]

        # encoder
        hidden = [None for i in range(self.dd_num)]

        for i in range(self.dd_num):
            hidden[i] = Conv2D(4, (5, 5), strides=(2, 2), activation='elu', padding='same')(input_layer[i])  #
            hidden[i] = Conv2D(8, (5, 5), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Conv2D(8, (3, 3), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Conv2D(8, (3, 3), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Flatten()(hidden[i])

        concat_layer = hidden[0]
        if (self.dd_num > 1):
            concat_layer = Concatenate()([hidden[i] for i in range(self.dd_num)])

        concat_layer = Dense(2*self.nLatent, activation='elu')(concat_layer)
        encoded = Dense(self.nLatent, activation='elu')(concat_layer)

        # decoder
        h = Dense(2 * self.nLatent, activation='elu')(encoded)
        h = Dense(self.partition[-1], activation='elu')(h)

        output_layer = [None for i in range(self.dd_num)]

        partition = self.partition
        for i in range(self.dd_num):
            a = self.partition[i]
            b = self.partition[i + 1]
            hidden[i] = Lambda(lambda x: x[:, partition[i]:partition[i + 1]])(h)

        for i in range(self.dd_num):
            hidden[i] = Reshape((int(self.img_rows[i] / 16), int(self.img_cols[i] / 16), 8))(hidden[i])

            hidden[i] = Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Conv2DTranspose(4, (5, 5), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            output_layer[i] = Conv2DTranspose(2, (5, 5), strides=(2, 2), activation='elu', padding='same')(hidden[i])

        autoencoder = Model(inputs=[input_layer[i] for i in range(self.dd_num)],
                            outputs=[output_layer[i] for i in range(self.dd_num)])

        start_index = -int((len(autoencoder.layers) - 1) / 2)
        if self.dd_num == 1:
            start_index -= 1

        decoded_output = [None for i in range(self.dd_num)]

        encoded_input = Input(shape=(self.nLatent,))
        decoded_output_combined = autoencoder.layers[start_index](encoded_input)
        decoded_output_combined = autoencoder.layers[start_index + 1](decoded_output_combined)

        for i in range(self.dd_num):
            a = self.partition[i]
            b = self.partition[i + 1]
            decoded_output[i] = Lambda(lambda x: x[:, partition[i]:partition[i + 1]])(decoded_output_combined)

        start_index = start_index + 2 + self.dd_num
        while (start_index < 0):
            for i in range(self.dd_num):
                decoded_output[i] = autoencoder.layers[start_index](decoded_output[i])
                start_index += 1

        return autoencoder, Model(inputs=[input_layer[i] for i in range(self.dd_num)],
                                  outputs=encoded), \
                             Model(inputs=encoded_input, outputs=[decoded_output[i] for i in range(self.dd_num)])

    def decoder(self):
        encoded_input = Input(shape=(self.nLatent,))
        # decoder
        h = Dense(2 * self.nLatent, activation='elu')(encoded_input)
        h = Dense(self.partition[-1], activation='elu')(h)

        output_layer = [None for i in range(self.dd_num)]

        hidden = [None for i in range(self.dd_num)]

        partition = self.partition
        for i in range(self.dd_num):
            a = self.partition[i]
            b = self.partition[i + 1]
            hidden[i] = Lambda(lambda x: x[:, partition[i]:partition[i + 1]])(h)


        output_layer = [None for i in range(self.dd_num)]

        # partition = self.partition
        # for i in range(self.dd_num):
        #     a = self.partition[i]
        #     b = self.partition[i + 1]
        #     hidden[i] = Lambda(lambda x: x[:, partition[i]:partition[i + 1]])(h)

        for i in range(self.dd_num):
            hidden[i] = Reshape((int(self.img_rows[i] / 16), int(self.img_cols[i] / 16), 8))(hidden[i])
            hidden[i] = Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            hidden[i] = Conv2DTranspose(4, (5, 5), strides=(2, 2), activation='elu', padding='same')(hidden[i])
            output_layer[i] = Conv2DTranspose(2, (5, 5), strides=(2, 2), activation='elu', padding='same')(hidden[i])

        return Model(inputs=encoded_input, outputs=[output_layer[i] for i in range(self.dd_num)])


    def train_model(self, x_train, epochs, batch_size=20):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=10,
                                       verbose=1,
                                       mode='auto')
        history = self.autoencoder_model.fit(x_train, x_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=None)
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds


if __name__ == '__main__':

    ndomain = int(input("Please input the number of domains"))

    data = []
    for i in range(ndomain):
        data.append(np.load("interpolated_matrix_" + str(i) + ".npy"))

    print(data[0].shape)

    ae = Autoencoder_interpolation(32, img_rows=[data[i].shape[1] for i in range(ndomain)],
                                   img_cols=[data[i].shape[2] for i in range(ndomain)], ndomain=ndomain)

    ae.train_model(data, epochs=1000, batch_size=128)

    # ae.autoencoder_model.save('conv_interpolation_r32_1000_sb.h5')  # creates a HDF5 file 'my_model.h5'

    ae.encoder_model.save('encoder_dd.h5')
    ae.decoder_model.save_weights('decoder_dd.h5')





