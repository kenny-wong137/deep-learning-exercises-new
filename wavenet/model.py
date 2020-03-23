import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
FILENAME = 'jena_climate_2009_2016.csv.zip'
START_TIME = datetime(year=2009, month=1, day=1)
TEST_START = datetime(year=2016, month=1, day=1)
END_TIME   = datetime(year=2017, month=1, day=1)
TIME_DELTA = timedelta(minutes=90)
STEPS_PER_DAY = 16
STEPS_OFFSET_BETWEEN_SEQS = 2

def load_data():
    zip_path = tf.keras.utils.get_file(origin=URL, fname=FILENAME, extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    
    df = pd.read_csv(csv_path)
    df.drop_duplicates('Date Time', inplace=True)
    df.set_index('Date Time', inplace=True)
    
    mean = df['T (degC)'].mean()
    stdev = df['T (degC)'].std()

    train_values = []
    test_values  = []
    
    current_time = START_TIME
    
    while current_time < END_TIME:
        try:
            temperature = df.loc[current_time.strftime('%d.%m.%Y %H:%M:%S')]['T (degC)']
            temperature = (temperature - mean) / stdev
            temperature = np.clip(temperature, -5.0, 5.0)
        except KeyError:
            temperature = float('nan')  # a handful of readings are missing from the dataset
        
        if current_time < TEST_START:
            train_values.append(temperature)
        else:
            test_values.append(temperature)
            
        current_time += TIME_DELTA
    
    train_values = np.array(train_values).astype('float32')
    test_values = np.array(test_values).astype('float32')
    
    return train_values, test_values


def make_sequences(values, steps_per_seq, batch_size, num_epochs):    
    sequences = []
    for start_index in range(0, len(values) - steps_per_seq, STEPS_OFFSET_BETWEEN_SEQS):
        seq = values[start_index : start_index + steps_per_seq]
        if np.sum(np.isnan(seq)) == 0:
            sequences.append(seq)
    sequences = np.expand_dims(np.array(sequences), axis=-1)
    
    dataset = tf.data.Dataset.from_tensor_slices(sequences)
    return dataset.shuffle(10000).batch(batch_size, drop_remainder=True).repeat(num_epochs)


class ResidualDilatedConv1D(tf.keras.layers.Layer):
    
    def __init__(self, model_dim, dilation_factor):
        super(ResidualDilatedConv1D, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters=model_dim, kernel_size=2, activation='relu',
                                           padding='causal', dilation_rate=dilation_factor)
        self.dropout = tf.keras.layers.Dropout(rate=0.3)
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs, training):
        return self.norm(inputs + self.dropout(self.conv(inputs), training=training))


def build_wavenet_model(model_dim, max_dilation_factor):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=model_dim, activation='relu', input_shape=(None, 1)))
    
    dilation_factor = 1
    while dilation_factor <= max_dilation_factor:
        model.add(ResidualDilatedConv1D(model_dim, dilation_factor))
        dilation_factor *= 2
    # Total lookback across all layers is (2 * max_dilation_factor - 1) timesteps.
    # Hence burn-in period (start and end inclusive) should be (2 * max_dilation_factor) timesteps.
    
    model.add(tf.keras.layers.Dense(units=model_dim, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation=None))

    return model


def build_rnn_model(model_dims):  # an alternative model, for comparison only
    return tf.keras.models.Sequential([
                tf.keras.layers.Dense(units=model_dims, activation='relu', input_shape=(None, 1)),
                tf.keras.layers.LSTM(units=model_dims, return_sequences=True),
                tf.keras.layers.Dense(units=model_dims, activation='relu'),
                tf.keras.layers.Dense(units=1, activation=None)
           ])


@tf.function
def train_step(model, optimizer, inputs, targets,
               batch_size, num_burn_in_steps, num_steps_in_seq):
    # The inputs and targets tensors both have (num_steps_in_seq - 1) timesteps.
    # The first timestep with a full convolutional history is the (num_burn_in_steps - 1)th timestep.
    weights = tf.concat([tf.zeros(shape=(batch_size, num_burn_in_steps - 1, 1)),
                         tf.ones(shape=(batch_size, num_steps_in_seq - num_burn_in_steps, 1))],
                        axis=1)
    
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.MeanAbsoluteError()(targets, predictions, sample_weight=weights)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss


@tf.function
def generate(model, burn_in_values,
             demo_size, num_burn_in_steps, num_steps_in_seq):    
    values_per_step = tf.unstack(burn_in_values, axis=1) \
                        + [tf.zeros(shape=(demo_size, 1))
                           for _ in range(num_steps_in_seq - num_burn_in_steps)]
    
    for step in range(num_burn_in_steps, num_steps_in_seq):
        inputs = tf.stack(values_per_step[:-1], axis=1)
        predictions = model(inputs, training=False)
        values_per_step[step] = predictions[:, step - 1, :]
    
    return tf.stack(values_per_step, axis=1)


def display_predictions(real_values, predicted_values, demo_size, num_burn_in_steps,
                        num_steps_in_seq, save_path_prefix):    
    for sample_id in range(demo_size):
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(0, num_steps_in_seq) / STEPS_PER_DAY,
                 real_values[sample_id, :, 0],
                 color='k')
        plt.plot(np.arange(num_burn_in_steps - 1, num_steps_in_seq) / STEPS_PER_DAY,
                 predicted_values[sample_id, num_burn_in_steps - 1:, 0],
                 color='r')
        plt.ylim(-4.0, 4.0)
        
        if save_path_prefix is not None:
            plt.savefig('{}_{:02}.png'.format(save_path_prefix, sample_id))
        plt.show()
        plt.clf()


def run(model_type, days_per_seq, num_burn_in_days, model_dims, num_epochs, batch_size, demo_size,
        save_path_prefix):
    '''
    Train autoregressive WaveNet model to forecast temperatures.
    
    :param model_type: name of model ('wavenet' or 'rnn')
    :param days_per_seq: number of days in each time series chunk
    :param num_burn_in_days: number of days to use as burn-in
    :param model_dims: number of filters in the causal dilated convolutional layers /
                       number of units in dense layers
    :param num_epochs: number of training epochs
    :param batch_size: size of training batch
    :param demo_size: number of predictions to display at demo time
    :param save_path_prefix: path to output predictions to, images will be called
                        '{save_prefix}_{id}.png'
    '''
    train_values, test_values = load_data()
    
    num_steps_in_seq = STEPS_PER_DAY * days_per_seq
    train_dataset = make_sequences(train_values, num_steps_in_seq, batch_size, num_epochs)
    test_dataset = make_sequences(test_values, num_steps_in_seq, demo_size, 1)
    
    num_burn_in_steps = num_burn_in_days * STEPS_PER_DAY
    max_dilation_factor = 1
    while 2 * max_dilation_factor <= num_burn_in_steps // 2:
        max_dilation_factor *= 2
    
    if model_type == 'wavenet':
        model = build_wavenet_model(model_dims, max_dilation_factor)
    elif model_type == 'rnn':
        model = build_rnn_model(model_dims)
    else:
        raise ValueError('Unrecognised model name: {}'.format(model_type))
    
    # Training
    optimizer = tf.keras.optimizers.Adam()
    losses = []
    
    for batch_id, data in enumerate(train_dataset):
        inputs = data[:, :-1, :]
        targets = data[:, 1:, :]  # shifted by 1 timestep (auto-regressive training)
        
        loss = train_step(model, optimizer, inputs, targets,
                          batch_size, num_burn_in_steps, num_steps_in_seq)
        losses.append(loss)
        
        if (batch_id + 1) % 1000 == 0:
            ave_loss = sum(losses) / len(losses)
            print('After {} batches: ave error = {:.5f}'.format(batch_id + 1, ave_loss))
            losses = []
    
    # Predicting
    real_values = next(iter(test_dataset))
    burn_in_values = real_values[:, :num_burn_in_steps, :]
    predictions = generate(model, burn_in_values, demo_size, num_burn_in_steps, num_steps_in_seq)
    display_predictions(real_values, predictions, demo_size, num_burn_in_steps,
                        num_steps_in_seq, save_path_prefix)
    
    
if __name__ == '__main__':
    model_type = sys.argv[1]
    save_path_prefix = sys.argv[2]
    
    run(model_type=model_type,
        days_per_seq=6,
        num_burn_in_days=4,
        model_dims=32,
        num_epochs=50,
        batch_size=64,
        demo_size=16,
        save_path_prefix=save_path_prefix)
    