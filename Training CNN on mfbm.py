import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import SpectralSynthesis as ss

import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

def my_generator():
    i = 0
    while True:
        #initialise variables
        H = np.random.random()
        S = 3*np.random.random()
        noise_pct = np.random.random()*0.01
        #generate large field and cut down
        field = ss.fBm(E=E,exp=True,H=H,sigma = S,N=N_p*4, centred = False)
        field = field[int((N_p*4-N_p)/2):int((N_p*4+N_p)/2),int((N_p*4-N_p)/2):int((N_p*4+N_p)/2)]
        #standardise field
        m_1_field = np.mean(field)
        s_1_field = np.std(field)
        m_2_field = 0
        s_2_field = 1/4

        field *= s_2_field/s_1_field
        field += (m_2_field-m_1_field*s_2_field/s_1_field)
        #generate noise field, standardise and add to fbm field
        noise = ss.fBm(E=E,exp=False, H = -1, N=N_p)
        m_1_noise = np.mean(noise)
        s_1_noise = np.std(noise)
        m_2_noise = 0
        s_2_noise = s_2_field*noise_pct
        noise *= s_2_noise/s_1_noise
        noise += (m_2_noise-m_1_noise*s_2_noise/s_1_noise)
        field += noise
        #standardise field
        m_1_field = np.mean(field)
        s_1_field = np.std(field)
        m_2_field = 0
        s_2_field = 1/4
        field *= s_2_field/s_1_field
        field += (m_2_field-m_1_field*s_2_field/s_1_field)
        #truncate field
        field[np.where(field>1.)] = 1.
        field[np.where(field<-1.)] = -1.
        x = (field-np.min(field))/(np.max(field)-np.min(field))
        
        label = np.array([H,S])
        yield x.reshape(N_p, N_p, 1), label


def my_input_fn(epochs, batch_size):
    dataset = tf.data.Dataset.from_generator(lambda: my_generator(),output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape((None, None, None)), tf.TensorShape((2))))

    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    return dataset

layer_size = 256
kernel_size = (3,3)
epochs = 100
E=2
N_p=64
total_items = 20000
batch_size = 32


NAME = "5-conv-{}-channels-5-dense-{}-epochs".format(layer_size,epochs)

inp =  Input(shape = (N_p,N_p,1))

conv1 = Conv2D(layer_size, kernel_size, activation = 'relu')(inp)
pool1 = MaxPooling2D(pool_size = (2,2))(conv1)

conv2 = Conv2D(layer_size, kernel_size, activation = 'relu')(pool1)
pool2 = MaxPooling2D(pool_size = (2,2))(conv2) 

conv3 = Conv2D(layer_size, kernel_size, activation = 'relu')(pool2)
pool3 = MaxPooling2D(pool_size = (2,2))(conv3)

conv4 = Conv2D(layer_size, kernel_size, activation = 'relu')(pool3)
pool4 = MaxPooling2D(pool_size = (2,2))(conv4)

# conv5 = Conv2D(layer_size, kernel_size, activation = 'relu')(pool4)
# pool5 = MaxPooling2D(pool_size = (2,2))(conv5)

flat1 = Flatten()(pool4)

dense1 = Dense(layer_size)(flat1)
dense2 = Dense(layer_size)(dense1)
dense3 = Dense(layer_size)(dense2)
dense4 = Dense(layer_size)(dense3)
dense5 = Dense(layer_size)(dense4)

out = Dense(2)(dense5)

model = Model(inp, out)

model.compile(loss='mse', optimizer = 'rmsprop', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
num_batches = total_items//batch_size
dataset = my_input_fn(epochs, batch_size)
model.fit(dataset, epochs=epochs, steps_per_epoch=num_batches, validation_data = dataset, validation_steps=50)

model.save('model64px_large.h5')
print('SAVED')

