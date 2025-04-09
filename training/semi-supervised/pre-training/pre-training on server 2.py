#!pip install -U efficientnet
#print(tf.__version__)

# Import necessary libraries
import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math
import numpy as np
from mypackages.seismic import npseismic as npseis
from mypackages.learningSort import massiveDataMethods as massMethods

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Configure GPU memory usage
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Multi-worker setup for distributed training
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['192.0.0.1:xxxx', '192.0.0.2:xxxx'] # Worker IPs and ports
    },
    'task': {'type': 'worker', 'index': 1}  # Chief worker (index 0)
})

# Configure communication options for distributed training
communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL  # Use NCCL for faster communication
)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options
)
print('Number of devices: %d' % strategy.num_replicas_in_sync)  # Print the number of devices in sync

# Load the training dataset from TFRecord files
tfrecordsPath = '/home/edgar/opendTectRoot/clearSeismic/No8Samples/unlabeled/'
train = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Map the dataset to use the same data for input and output (unsupervised learning)
train = train.map(lambda image, label: (image, image))

# Split the dataset into training and testing sets
test = train.take(52920)  # Use 35% of the data for testing
train = train.skip(52920)  # Use the remaining data for training

# Define training parameters
batchSize = 2048
warmupEpoch = 16
lr = 0.001
mainDirect = '/home/edgar/google-drive/PhD/Semestre 8/data science 2/training/semi-supervised/experiment0/emiimas/'
namefile = 'pretrainingLayerWiseWell_n'
Epochs = 150

# Extract input shape from the dataset
timeSteps = train.element_spec[0].shape[0]
features = train.element_spec[0].shape[1]
inputShape = (timeSteps, features)

# Calculate the number of training and testing samples
sample_count = train.cardinality().numpy()
numExamTest = test.cardinality().numpy()

# Compute the number of warmup epochs
warmup_epoch = massMethods.minWarmupEpochs(sample_count, batchSize, warmupEpoch)

# Training batch size, set small value here for demonstration purpose.
batch_size = batchSize# * 2 servers then 512 to feed to each model.

# Compute the number of warmup batches
warmup_batches = math.floor(warmup_epoch * sample_count / batch_size)
 
# Create a learning rate scheduler for warmup
warm_up_lr = massMethods.WarmUpLearningRateScheduler(warmup_batches, init_lr=lr)

# Prepare the training and testing datasets
BATCH_SIZE = batch_size
BUFFER_SIZE = 10000
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
test_dataset = test.batch(BATCH_SIZE).repeat()

# Define LSTM model parameters
UNITS = 1024
drop = 0.75
recu_drop = 0.0

# Train models with different numbers of LSTM layers (1 to 5 layers)
for L in range(1,4,1): # 3 layers are used only
  print(f"Training model with {L} LSTM layers")
  with strategy.scope():
    def get_lr_metric(optimizer):
          def lr(y_true, y_pred):
              return optimizer.learning_rate
          return lr

    # Define the LSTM model based on the number of layers
    if L == 1:
      model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.Dense(UNITS),
        tf.keras.layers.Dense(features) 
      ])
    
    if L == 2:
      model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.Dense(UNITS),
        tf.keras.layers.Dense(features) 
      ])
    
      pretrainedModel = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.Dense(UNITS),
        tf.keras.layers.Dense(features) 
      ])
    
    if L == 3:
      model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.Dense(UNITS),
        tf.keras.layers.Dense(features) 
      ])
    
      pretrainedModel = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.Dense(UNITS),
        tf.keras.layers.Dense(features) 
      ])

    if L == 4:
      model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.Dense(UNITS),
        tf.keras.layers.Dense(features)
      ])
    
      pretrainedModel = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.Dense(UNITS),
        tf.keras.layers.Dense(features) 
      ])

    if L == 5:
      model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.Dense(UNITS),
        tf.keras.layers.Dense(features) 
      ])
    
      pretrainedModel = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=inputShape,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True,dropout=drop, recurrent_dropout=recu_drop),         
        tf.keras.layers.Dense(UNITS),
        tf.keras.layers.Dense(features) 
      ])


    # Compile the model
    opt = tf.keras.optimizers.Adam(lr)
    lr_metric = get_lr_metric(opt)
    model.compile(optimizer=opt, 
                loss=tf.keras.losses.MeanSquaredError(), 
                metrics=[lr_metric]) 

  # Define callbacks for training
  cbks = [warm_up_lr,
            tf.keras.callbacks.LearningRateScheduler(lambda epoch: (lr)*(np.exp(-epoch/25))),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1),
            tf.keras.callbacks.CSVLogger(mainDirect+'log_'+namefile+'.csv', append=True, separator=',')
            ]
  # Trained weights are loaded
  if L == 2:
    pretrainedModel.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=[lr_metric])
    pretrainedModel.load_weights(mainDirect+str(namefile)+'.h5')
    model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
    model.layers[2].set_weights(pretrainedModel.layers[1].get_weights())
    model.layers[3].set_weights(pretrainedModel.layers[2].get_weights())

    model.layers[0].trainable = False
    
  if L == 3:
    pretrainedModel.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=[lr_metric])
    pretrainedModel.load_weights(mainDirect+str(namefile)+'.h5')
    model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
    model.layers[1].set_weights(pretrainedModel.layers[1].get_weights())
    model.layers[3].set_weights(pretrainedModel.layers[2].get_weights())
    model.layers[4].set_weights(pretrainedModel.layers[3].get_weights())

    model.layers[0].trainable = False
    model.layers[1].trainable = False

  if L == 4:
    pretrainedModel.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=[lr_metric])
    pretrainedModel.load_weights(mainDirect+str(namefile)+'.h5')
    model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
    model.layers[1].set_weights(pretrainedModel.layers[1].get_weights())
    model.layers[2].set_weights(pretrainedModel.layers[2].get_weights())
    model.layers[4].set_weights(pretrainedModel.layers[3].get_weights())
    model.layers[5].set_weights(pretrainedModel.layers[4].get_weights())

    
    model.layers[0].trainable = False
    model.layers[1].trainable = False
    model.layers[2].trainable = False

  if L == 5:
    pretrainedModel.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=[lr_metric])
    pretrainedModel.load_weights(mainDirect+str(namefile)+'.h5')
    model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
    model.layers[1].set_weights(pretrainedModel.layers[1].get_weights())
    model.layers[2].set_weights(pretrainedModel.layers[2].get_weights())
    model.layers[3].set_weights(pretrainedModel.layers[3].get_weights())
    model.layers[5].set_weights(pretrainedModel.layers[4].get_weights())
    model.layers[6].set_weights(pretrainedModel.layers[5].get_weights())

    model.layers[0].trainable = False
    model.layers[1].trainable = False
    model.layers[2].trainable = False
    model.layers[3].trainable = False
  
  # Train the model
  stepsPerEpoch = math.floor(sample_count/batch_size)
  stepsTest = math.floor(numExamTest/batch_size)
  eff_history = model.fit(train_dataset,epochs=Epochs, 
                        verbose=1,
                        steps_per_epoch=stepsPerEpoch,  
                        validation_data=test_dataset,
                        validation_steps=stepsTest, 
                        callbacks=cbks
            )
  
  # Save the model weights
  model.save_weights(mainDirect+str(namefile)+'.h5')

  # Plot training history  
  npseis.plot_train_history(eff_history,
                   'LSTM Model Training and Validation Loss')
