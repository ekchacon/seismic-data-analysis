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

# -------------------- Labeled Data --------------------

# Load labeled datasets for each well
# well_n-1
tfrecordsPath = direct + 'well-n-1/train/'
well_n1 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n-3
tfrecordsPath = direct + 'well-n-3/train/'
well_n3 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n-33
tfrecordsPath = direct + 'well-n-33/train/'
well_n33 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n-39
tfrecordsPath = direct + 'well-n-39/train/'
well_n39 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n
tfrecordsPath = direct + 'well_n/train/'
well_n = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Combine labeled datasets into a single dataset
labeled = tf.data.experimental.sample_from_datasets(
    [well_n1, well_n3, well_n33, well_n39, well_n], 
    weights=[0.2, 0.2, 0.2, 0.2, 0.2]
)

# Calculate the total cardinality of the labeled dataset
labeled_cardinality = (
    well_n1.cardinality().numpy() +
    well_n3.cardinality().numpy() +
    well_n33.cardinality().numpy() +
    well_n39.cardinality().numpy() +
    well_n.cardinality().numpy()
)

# Ensure the labeled dataset has the correct cardinality
labeled = labeled.apply(tf.data.experimental.assert_cardinality(labeled_cardinality))

# Save the combined labeled dataset to a new TFRecord file
tfrecordsPath = direct + 'train/'
tf.data.experimental.save(labeled, tfrecordsPath)

# -------------------- Test Data --------------------

# Load test datasets for each well
# well_n-1
tfrecordsPath = direct + 'well-n-1/Well test/'
well_n1 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n-3
tfrecordsPath = direct + 'well-n-3/Well test/'
well_n3 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n-33
tfrecordsPath = direct + 'well-n-33/Well test/'
well_n33 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n-39
tfrecordsPath = direct + 'well-n-39/Well test/'
well_n39 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n
tfrecordsPath = direct + 'well_n/Well test/'
well_n = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Combine test datasets into a single dataset
test = tf.data.experimental.sample_from_datasets(
    [well_n1, well_n3, well_n33, well_n39, well_n], 
    weights=[0.2, 0.2, 0.2, 0.2, 0.2]
)

# Calculate the total cardinality of the test dataset
test_cardinality = (
    well_n1.cardinality().numpy() +
    well_n3.cardinality().numpy() +
    well_n33.cardinality().numpy() +
    well_n39.cardinality().numpy() +
    well_n.cardinality().numpy()
)

# Ensure the test dataset has the correct cardinality
test = test.apply(tf.data.experimental.assert_cardinality(test_cardinality))

# Save the combined test dataset to a new TFRecord file
tfrecordsPath = direct + 'Well test/'
tf.data.experimental.save(test, tfrecordsPath)

# -------------------- Unlabeled Data --------------------

# Load unlabeled datasets for each well
# well_n-1
tfrecordsPath = direct + 'well-n-1/unlabeled/'
well_n1 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n-3
tfrecordsPath = direct + 'well-n-3/unlabeled/'
well_n3 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n-33
tfrecordsPath = direct + 'well-n-33/unlabeled/'
well_n33 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n-39
tfrecordsPath = direct + 'well-n-39/unlabeled/'
well_n39 = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# well_n
tfrecordsPath = direct + 'well_n/unlabeled/'
well_n = tf.data.experimental.load(
    tfrecordsPath, 
    (tf.TensorSpec(shape=(53, 18), dtype=tf.float64, name=None),
     tf.TensorSpec(shape=(), dtype=tf.float64, name=None))
)

# Combine unlabeled datasets into a single dataset
unlabeled = tf.data.experimental.sample_from_datasets(
    [well_n1, well_n3, well_n33, well_n39, well_n], 
    weights=[0.2, 0.2, 0.2, 0.2, 0.2]
)

# Calculate the total cardinality of the unlabeled dataset
unlabeled_cardinality = (
    well_n1.cardinality().numpy() +
    well_n3.cardinality().numpy() +
    well_n33.cardinality().numpy() +
    well_n39.cardinality().numpy() +
    well_n.cardinality().numpy()
)

# Ensure the unlabeled dataset has the correct cardinality
unlabeled = unlabeled.apply(tf.data.experimental.assert_cardinality(unlabeled_cardinality))

# Save the combined unlabeled dataset to a new TFRecord file
tfrecordsPath = direct + 'unlabeled/'
tf.data.experimental.save(unlabeled, tfrecordsPath)