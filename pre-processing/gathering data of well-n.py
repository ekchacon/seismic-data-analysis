#!pip install -U efficientnet
#print(tf.__version__)

# Import necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math
import numpy as np

# Define the directory containing the TFRecord files
direct = '/home/edgar/opendTectRoot/clearSeismic/No8Samples/'

# Load labeled training datasets for the "well-n" well
# Top train dataset
tfrecordsPath = direct + 'well-n/Top train/'
test_top = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Right train dataset
tfrecordsPath = direct + 'well-n/Right train/'
test_bottom = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Bottom train dataset
tfrecordsPath = direct + 'well-n/Bottom train/'
test_left = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Left train dataset
tfrecordsPath = direct + 'well-n/Left train/'
test_right = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Combine the labeled datasets into a single dataset
test = tf.data.experimental.sample_from_datasets(
    [test_top, test_bottom, test_left, test_right], 
    weights=[0.25, 0.25, 0.25, 0.25]  # Equal weights for all datasets
)

# Calculate the total cardinality (number of samples) of the combined dataset
test_cardinality = (
    test_top.cardinality().numpy() +
    test_bottom.cardinality().numpy() +
    test_left.cardinality().numpy() +
    test_right.cardinality().numpy()
)

# Ensure the combined dataset has the correct cardinality
test = test.apply(tf.data.experimental.assert_cardinality(test_cardinality))

# Save the combined labeled dataset to a new TFRecord file
tfrecordsPath = direct + 'well-n/train/'
tf.data.experimental.save(test, tfrecordsPath)

# Load unlabeled datasets for the "well-n" well
# Left unlabeled dataset
tfrecordsPath = direct + 'well-n/Left unlabeled/'
test_top = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Top unlabeled dataset
tfrecordsPath = direct + 'well-n/Top unlabeled/'
test_bottom = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Right unlabeled dataset
tfrecordsPath = direct + 'well-n/Right unlabeled/'
test_left = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Bottom unlabeled dataset
tfrecordsPath = direct + 'well-n/Bottom unlabeled/'
test_right = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Combine the unlabeled datasets into a single dataset
test = tf.data.experimental.sample_from_datasets(
    [test_top, test_bottom, test_left, test_right], 
    weights=[0.25, 0.25, 0.25, 0.25]  # Equal weights for all datasets
)

# Calculate the total cardinality (number of samples) of the combined dataset
test_cardinality = (
    test_top.cardinality().numpy() +
    test_bottom.cardinality().numpy() +
    test_left.cardinality().numpy() +
    test_right.cardinality().numpy()
)

# Ensure the combined dataset has the correct cardinality
test = test.apply(tf.data.experimental.assert_cardinality(test_cardinality))

# Save the combined unlabeled dataset to a new TFRecord file
tfrecordsPath = direct + 'well-n/unlabeled/'
tf.data.experimental.save(test, tfrecordsPath)