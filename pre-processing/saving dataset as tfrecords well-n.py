#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:22:13 2022

@author: est1

This script processes seismic data and well log data for a specific well (well-n),
generates training datasets, and saves them in TensorFlow's TFRecord format.
"""

# Import necessary libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import pathlib
from IPython.display import display
from segysak.segy import segy_loader, well_known_byte_locs, segy_writer
from segysak.segy import segy_loader, segy_header_scan
from mypackages.seismic import npseismic as npseis
from numpy import genfromtxt
import matplotlib.pyplot as plt

# Define parameters for the specific well (well-n)
_well_name = 'well-n'  # Name of the well
_path_sgy_smaller_cube = '/DATA/edgar/opendTectRoot/seismicAIrawDatasetClearSeismic/well-nPreStack.sgy'  # Path to the SEG-Y seismic cube
_syntheticdir = '/DATA/edgar/opendTectRoot/seismicAIrawDatasetClearSeismic/well-nAIandSynthNon-Zero.dat'  # Path to the synthetic well log data
_dirToSave = '/edgar/opendTectRoot/clearSeismic/No8Samples/'  # Directory to save the processed TFRecord files
_well_iline = 2287  # Inline index of the well location
_well_xline = 1166  # Crossline index of the well location
_startTime = 2121  # Start time for the range of interest
_stopTime = 2532  # Stop time for the range of interest
_root = 'DATA'  # Root directory for saving the TFRecord files

# Process the seismic data and well log data for the specified well
# This function extracts seismic data around the well location, preprocesses it,
# and saves the resulting dataset in TensorFlow's TFRecord format.
npseis.ringOfData(
    _well_name, 
    _path_sgy_smaller_cube, 
    _syntheticdir, 
    _dirToSave, 
    _well_iline, 
    _well_xline, 
    _startTime, 
    _stopTime, 
    _root
)