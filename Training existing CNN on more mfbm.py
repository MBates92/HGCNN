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
epochs = 50
E=2
N_p=128
total_items = 20000
batch_size = 32

loaded_model = tf.keras.models.load_model('model128px_small')

num_batches = total_items//batch_size
dataset = my_input_fn(epochs, batch_size)
loaded_model.fit(dataset, epochs=epochs, steps_per_epoch=num_batches, validation_data = dataset, validation_steps=50)

loaded_model.save('model128px_large')
print('SAVED')

