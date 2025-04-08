#!pip install -U efficientnet
#print(tf.__version__)

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, LSTM, TimeDistributed, Reshape
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from mypackages.seismic import npseismic as npseis  # Custom seismic processing library
from mypackages.learningSort import massiveDataMethods as massMethods  # Custom data handling methods

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Configure GPU memory usage
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # Limit GPU memory usage to 90%
config.gpu_options.allow_growth = True  # Allow dynamic memory allocation
session = InteractiveSession(config=config)

# Define the main directory for saving results
mainDirect = '/DATA/edgar/google-drive/PhD/Semestre 8/data science 2/training/supervised/experiment0/'

# Load the test dataset from TFRecord files
tfrecordsPath = '/DATA/edgar/opendTectRoot/clearSeismic/No8Samples/Well test/'
test = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Load the training dataset from TFRecord files
tfrecordsPath = '/DATA/edgar/opendTectRoot/clearSeismic/No8Samples/train/'
train = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Calculate the number of samples in the training dataset
sample_count = train.cardinality().numpy()

# Automatically compute batch size, warmup epochs, and learning rate using a custom method
batch_size, warmup_epoch, lrm = massMethods.supervisedLEGWautomaticScaled(sample_count, 90)

# Adjust warmup epochs if necessary
if warmup_epoch > 1:
    warmup_epoch = warmup_epoch - 1

# Compute the number of warmup batches
warmup_batches = math.floor(warmup_epoch * sample_count / batch_size)

# Create a warmup learning rate scheduler
warm_up_lr = massMethods.WarmUpLearningRateScheduler(warmup_batches, init_lr=lrm)

# Define buffer size and batch size for datasets
BATCH_SIZE = batch_size
BUFFER_SIZE = 10000

# Prepare the training dataset
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# Prepare the test dataset
test_dataset = test.batch(BATCH_SIZE).repeat()

# Define LSTM model parameters
trainBATCH_SIZE = batch_size
testBATCH_SIZE = batch_size
UNITS = 1024
drop = 0.75  # Dropout rate
recu_drop = 0.0  # Recurrent dropout rate

# Define a function to track the learning rate during training
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.learning_rate
    return lr

# Define the supervised LSTM model architecture
SupervisedPretModel = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=(53, 18), dropout=drop, recurrent_dropout=recu_drop),
    tf.keras.layers.LSTM(units=UNITS, return_sequences=True, dropout=drop, recurrent_dropout=recu_drop),
    tf.keras.layers.LSTM(units=UNITS, dropout=drop, recurrent_dropout=recu_drop),
    tf.keras.layers.Dense(UNITS),
    tf.keras.layers.Dense(1)
])

# Compile the supervised model
opt = tf.keras.optimizers.Adam(lrm)
lr_metric = get_lr_metric(opt)
SupervisedPretModel.compile(
    optimizer=opt,
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=[lr_metric]
)

# Define callbacks for training
cbks = [
    warm_up_lr,
    tf.keras.callbacks.LearningRateScheduler(lambda epoch: (lrm) * (np.exp(-epoch / 25))),
    tf.keras.callbacks.CSVLogger(mainDirect + 'supervised.csv', append=True, separator=',')
]

# Train the supervised model
stepsPerEpoch = math.floor(train.cardinality().numpy() / trainBATCH_SIZE)
stepsTest = math.floor(test.cardinality().numpy() / testBATCH_SIZE)
eff_history = SupervisedPretModel.fit(
    train_dataset,
    validation_data=test_dataset,
    steps_per_epoch=stepsPerEpoch,
    validation_steps=stepsTest,
    epochs=225,
    verbose=1,
    callbacks=cbks
)

# Save the trained model weights
SupervisedPretModel.save_weights(mainDirect + 'supervised.h5')