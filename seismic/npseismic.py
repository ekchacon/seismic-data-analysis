#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:25:52 2022

@author: est1
"""
import tensorflow as tf
from sklearn.preprocessing import MaxAbsScaler#Scaling features to a range
from scipy.interpolate import Rbf
import numpy as np
import pathlib
import pandas as pd
from segysak.segy import segy_loader
from segysak.segy import segy_loader, segy_header_scan
from mypackages.seismic import npseismic as npseis
from numpy import genfromtxt

def ringOfData(_well_name, _path_sgy_smaller_cube, _syntheticdir, _dirToSave, _well_iline, _well_xline, _startTime, _stopTime, _root):
    """
    Processes seismic data and well log data to generate training datasets for machine learning models.
    The function extracts seismic data around a well location, preprocesses it, and saves the resulting
    dataset in TensorFlow's TFRecord format.

    Parameters:
        _well_name (str): Name of the well.
        _path_sgy_smaller_cube (str): Path to the smaller SEG-Y seismic cube file.
        _syntheticdir (str): Path to the synthetic well log data file.
        _dirToSave (str): Directory to save the processed TFRecord files.
        _well_iline (int): Inline index of the well location.
        _well_xline (int): Crossline index of the well location.
        _startTime (float): Start time for the range of interest.
        _stopTime (float): Stop time for the range of interest.
        _root (str): Root directory for saving the TFRecord files.

    Returns:
        None
    """
    # Load the SEG-Y file
    path_sgy_smaller_cube = _path_sgy_smaller_cube
    segy_file = pathlib.Path(path_sgy_smaller_cube)
    with pd.option_context("display.max_rows", 100):
        display(segy_header_scan(segy_file))  # Display the SEG-Y header information

    # Load the seismic data into an xarray DataArray
    smaller_3d_gath = segy_loader(
        segy_file, iline=189, xline=193, cdpx=181, cdpy=185, offset=37
    )

    # Convert the seismic data to a DataFrame (if needed)
    smaller_3d_gath_df = smaller_3d_gath

    # Define the policy for extracting data around the well
    policy = np.array(
        [
            ['Well test', 0, 0, 0, 0],
            ['Top train', 0, 1, 0, 0],
            ['Right train', 1, -1, 0, 1],
            ['Bottom train', -1, -1, 0, 0],
            ['Left train', -1, 1, 0, 1],
            ['Left unlabeled', -2, 0, 1, 4],
            ['Top unlabeled', 3, 3, 1, 1],
            ['Right unlabeled', 3, -3, 1, 4],
            ['Bottom unlabeled', -3, -3, 1, 1]
        ],
        dtype=object
    )

    # Initialize the well location
    x = _well_xline
    y = _well_iline

    # Loop through each policy to process data
    for i in range(0, len(policy), 1):
        # Update the well location based on the policy
        x = x + policy[i][1]
        y = y + policy[i][2]

        # Extract the number of gathers and inlines to process
        gathers_each_side_well_xline = policy[i][3]
        inlines_each_side_well_iline = policy[i][4]

        # Extract seismic data (M6) around the updated well location
        well_iline = y
        well_xline = x
        cube_of_M6, time = npseis.getM6fromInlines(
            smaller_3d_gath_df, well_iline, well_xline, gathers_each_side_well_xline, inlines_each_side_well_iline
        )

        # Load the synthetic well log data
        syntheticdir = _syntheticdir
        synthetic = genfromtxt(syntheticdir, delimiter=None, skip_header=1)

        # Prepare inputs for preprocessing
        x1 = time
        traces_per_cdp = 18
        x2 = synthetic[:, 2]  # Time from the well log
        AI = synthetic[:, 3]  # Acoustic Impedance from the well log
        startTime = _startTime
        stopTime = _stopTime

        # Preprocess the seismic cube and well log data
        cube_preprocessed, label, xi = npseis.cubeandAIPreprocessing(
            x1, cube_of_M6, traces_per_cdp, x2, AI, startTime, stopTime
        )

        # Reduce the number of samples in the data
        num_samples_deleted = 8
        cube_preprocessedDel, label_deleted = npseis.reduceSamplesInTraces(
            cube_preprocessed, label, num_samples_deleted
        )

        #cube_preprocessedDel
        #label_deleted
        #traces_per_cdp = 18

        # Slice the data into windows for each CDP
        window_size = 53
        train_windowed, predictions = npseis.windowingCubeperCDP(
            cube_preprocessedDel, label_deleted, traces_per_cdp, window_size
        )

        # Prepare the data for TensorFlow
        x_train_multi = train_windowed
        y_train_multi = predictions
        train_well_n3 = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))

        # Define the path to save the TFRecord files
        tfrecordsPath = '/' + _root + _dirToSave + _well_name + '/' + policy[i][0] + '/'

        # Save the dataset as TFRecord files
        tf.data.experimental.save(train_well_n3, tfrecordsPath)

def windowingCubeperCDP(cube_preprocessed, label, traces_per_cdp, window_size):
    """
    Slices a preprocessed seismic data cube into windows for each Common Depth Point (CDP) 
    and associates each window with the corresponding label.

    Parameters:
        cube_preprocessed (numpy.ndarray): The preprocessed 3D seismic data cube (rows, columns, slices).
        label (numpy.ndarray): The corresponding labels (e.g., Acoustic Impedance values).
        traces_per_cdp (int): Number of traces per CDP.
        window_size (int): The size of the window to slice from the data.

    Returns:
        tuple:
            CDP_concat_per_inline (numpy.ndarray): A 3D array containing the sliced windows of the seismic data.
            label_concat_per_inline (numpy.ndarray): A 1D array containing the corresponding labels for each window.
    """
    # Get the number of inlines (slices) in the seismic cube
    num_inlines = len(cube_preprocessed[0, 0, :])

    # Calculate the number of CDPs in each inline
    num_cdps = cube_preprocessed[:, :, 0].shape[1] / traces_per_cdp

    # Initialize empty arrays to store the windows and labels
    CDP_concat_per_inline = np.empty((0, window_size, traces_per_cdp), float)  # 3D array for windows
    label_concat_per_inline = np.empty((0), float)  # 1D array for labels

    # Loop through each inline in the seismic cube
    for inline in range(0, num_inlines, 1):
        # Initialize the start and end trace indices for the first CDP
        start_trace = 0
        end_trace = traces_per_cdp

        # Loop through each CDP in the current inline
        for i in range(0, int(num_cdps), 1):
            # Extract the current CDP from the seismic cube
            CDP = cube_preprocessed[:, start_trace:end_trace, inline]

            # Prepare the dataset and labels for windowing
            dataset = CDP
            prediction = label
            start_index = 0
            end_index = None  # Use the entire dataset for windowing

            # Slice the dataset into windows and associate labels
            train_windowed, predictions = multiVariateDataset(dataset, prediction, start_index, end_index, window_size)

            # Concatenate the windows and labels for the current inline
            CDP_concat_per_inline = np.concatenate((CDP_concat_per_inline, train_windowed), axis=0)
            label_concat_per_inline = np.concatenate((label_concat_per_inline, predictions), axis=0)

            # Update the start and end trace indices for the next CDP
            start_trace = end_trace
            end_trace += traces_per_cdp

    # Return the concatenated windows and labels
    return CDP_concat_per_inline, label_concat_per_inline

def reduceSamplesInTraces(cube_preprocessed, label, num_samples_deleted):
    """
    Reduces the number of samples in the seismic data and corresponding labels by deleting rows 
    at regular intervals.

    Parameters:
        cube_preprocessed (numpy.ndarray): The preprocessed 3D seismic data cube.
        label (numpy.ndarray): The corresponding labels (e.g., Acoustic Impedance values).
        num_samples_deleted (int): The number of samples to delete between retained samples.

    Returns:
        tuple:
            cube_preprocessedDel (numpy.ndarray): The seismic data cube with reduced samples.
            label_deleted (numpy.ndarray): The labels with reduced samples.
    """
    # Get the total number of rows (samples) in the seismic data cube
    num_rows = len(cube_preprocessed)

    # Create an array of all row indices
    all_rows = np.arange(0, num_rows, 1)

    # Create an array of indices for rows to retain (fixed samples)
    samples_fixed = np.arange(0, num_rows, (num_samples_deleted + 1))

    # Identify the indices of rows to delete by finding the difference between all rows and fixed samples
    samples_deleted = np.setdiff1d(all_rows, samples_fixed)

    # Delete the identified rows from the seismic data cube
    cube_preprocessedDel = np.delete(cube_preprocessed, samples_deleted, axis=0)

    # Delete the corresponding rows from the labels
    label_deleted = np.delete(label, samples_deleted, axis=0)

    # Return the reduced seismic data cube and labels
    return cube_preprocessedDel, label_deleted

def cubeandAIPreprocessing(x1, cube_of_M6, traces_per_cdp, x2, AI, startTime, stopTime):
    """
    Preprocesses a 3D seismic cube and well log data by normalizing, interpolating, 
    and combining them into a preprocessed seismic cube.

    Parameters:
        x1 (numpy.ndarray): Seismic time array.
        cube_of_M6 (numpy.ndarray): The 3D seismic data cube (rows, columns, slices).
        traces_per_cdp (int): Number of traces per Common Depth Point (CDP).
        x2 (numpy.ndarray): Well log time array.
        AI (numpy.ndarray): Acoustic Impedance (AI) data from the well log.
        startTime (float): Start time for the range of interest.
        stopTime (float): Stop time for the range of interest.

    Returns:
        tuple:
            cube_of_M6_preprocessed (numpy.ndarray): The preprocessed 3D seismic data cube.
            label (numpy.ndarray): Normalized Acoustic Impedance data for the specified range.
            xi (numpy.ndarray): Interpolated time array for the specified range.
    """
    # Get the number of inlines (slices) in the seismic cube
    num_inlines = len(cube_of_M6[0, 0, :])

    # Extract the first inline (slice) from the cube to preprocess as a CDP
    CDP = cube_of_M6[:, :, 0]

    # Preprocess the first inline (CDP) and well log data
    iline_preprocessed, label, xi = CDPAndAIpreProcessing(x1, CDP, x2, AI, startTime, stopTime)

    # Initialize an empty 3D array to store the preprocessed seismic cube
    cube_of_M6_preprocessed = np.empty((len(iline_preprocessed[:, 0]), len(iline_preprocessed[0]), 0), float)

    # Loop through all inlines in the seismic cube
    for inline in range(0, num_inlines, 1):
        # Extract the current inline (slice) from the cube
        CDP = cube_of_M6[:, :, inline]

        # Preprocess the current inline (CDP) and well log data
        iline_preprocessed, label, xi = CDPAndAIpreProcessing(x1, CDP, x2, AI, startTime, stopTime)

        # Reshape the preprocessed inline to add it as a slice to the cube
        iline_preprocessed = iline_preprocessed[:, :, np.newaxis]

        # Append the preprocessed inline as a new slice in the seismic cube
        cube_of_M6_preprocessed = np.concatenate((cube_of_M6_preprocessed, iline_preprocessed), axis=2)

    # Return the preprocessed seismic cube, normalized AI data, and interpolated time array
    return cube_of_M6_preprocessed, label, xi

def getM6fromInlines(smaller_3d_gath_df, well_iline, well_xline, gathers_each_side_well_xline, inlines_each_side_well_iline):
    """
    Extracts seismic data (M6) from multiple inlines and combines them into a 3D seismic cube.

    Parameters:
        smaller_3d_gath_df (xarray.DataArray): The 3D seismic gather data.
        well_iline (int): The central inline index for the extraction.
        well_xline (int): The central crossline index for the extraction.
        gathers_each_side_well_xline (int): The number of gathers to include on each side of the central crossline.
        inlines_each_side_well_iline (int): The number of inlines to include on each side of the central inline.

    Returns:
        tuple:
            cube_of_M6 (numpy.ndarray): A 3D array containing the seismic data cube (rows, columns, slices).
            time (numpy.ndarray): A 1D array containing the time values corresponding to the seismic data.
    """
    # Extract seismic data for the central inline
    iline_of_M6, time = getM6fromGathersfromInline(smaller_3d_gath_df, well_iline, well_xline, gathers_each_side_well_xline)

    # Initialize an empty 3D array to store the seismic cube
    cube_of_M6 = np.empty((len(iline_of_M6[:, 0]), len(iline_of_M6[0]), 0), float)

    # Define the range of inlines to process
    central_inline = well_iline
    start_inline = central_inline - inlines_each_side_well_iline
    end_inline = central_inline + inlines_each_side_well_iline

    # Loop through the range of inlines to extract seismic data
    for i in range(start_inline, end_inline + 1, 1):
        print('iline: ', i)  # Log the current inline being processed

        # Update the inline index
        well_iline = i

        # Extract seismic data for the current inline
        iline_of_M6, time = getM6fromGathersfromInline(smaller_3d_gath_df, well_iline, well_xline, gathers_each_side_well_xline)

        # Reshape the extracted inline data to add it as a slice to the cube
        iline_of_M6 = iline_of_M6[:, :, np.newaxis]

        # Append the inline data as a new slice in the seismic cube
        cube_of_M6 = np.concatenate((cube_of_M6, iline_of_M6), axis=2)

    # Return the seismic cube and the corresponding time values
    return cube_of_M6, time

def getM6fromGathersfromInline(smaller_3d_gath_df, inline, well_xline, gathers_each_side_xline):
    """
    Extracts seismic data (M6) from multiple gathers along a specific inline and a range of crosslines.

    Parameters:
        smaller_3d_gath_df (xarray.DataArray): The 3D seismic gather data.
        inline (int): The inline index to extract data from.
        well_xline (int): The central crossline index for the extraction.
        gathers_each_side_xline (int): The number of gathers to include on each side of the central crossline.

    Returns:
        tuple:
            iline_M6_gathers (numpy.ndarray): A 2D array containing the extracted seismic data for the inline.
            time (numpy.ndarray): A 1D array containing the time values corresponding to the seismic data.
    """
    # Initialize variables for the inline and crossline
    iline = inline
    xline = well_xline  # Start with the central crossline
    startOffset = 400  # Start offset to avoid zero traces

    # Extract traces and time for the central crossline
    tracesOfCDP, time = getTracesFromCDP(smaller_3d_gath_df, iline, xline, startOffset)

    # Initialize an empty array to store the seismic data for all gathers
    iline_M6_gathers = np.empty((len(tracesOfCDP), 0), float)

    # Define the range of crosslines to process
    central_gather = well_xline
    gathers_each_side = gathers_each_side_xline
    start_gather = central_gather - gathers_each_side
    end_gather = central_gather + gathers_each_side

    # Loop through the range of crosslines to extract seismic data
    for i in range(start_gather, end_gather + 1, 1):
        print('xline: ', i)  # Log the current crossline being processed

        # Update the crossline index
        xline = i

        # Extract traces and time for the current crossline
        tracesOfCDP, time = getTracesFromCDP(smaller_3d_gath_df, iline, xline, startOffset)

        # Concatenate the extracted traces to the seismic data array
        iline_M6_gathers = np.concatenate((iline_M6_gathers, tracesOfCDP), axis=1)

    # Return the seismic data and corresponding time values
    return iline_M6_gathers, time

def getTracesFromCDP(smaller_3d_gath, _iline, _xline, startOffset):
    """
    Extracts seismic traces from a 3D seismic gather for a specific inline, crossline, and offset range.

    Parameters:
        smaller_3d_gath (xarray.DataArray): The 3D seismic gather data.
        _iline (int): The inline index to extract traces from.
        _xline (int): The crossline index to extract traces from.
        startOffset (int): The starting offset for extracting traces.

    Returns:
        tuple:
            tracesOfGather (numpy.ndarray): A 2D array containing the extracted seismic traces.
            time (numpy.ndarray): A 1D array containing the time values corresponding to the traces.
    """
    # Extract the first trace for the given inline, crossline, and starting offset
    aTrace = smaller_3d_gath.data.sel(iline=_iline, xline=_xline, offset=startOffset).data
    traceLength = len(aTrace)  # Determine the length of the trace

    # Extract the time values from the seismic gather
    time = smaller_3d_gath.twt.values

    # Initialize an empty array to store the traces (rows: time, columns: traces)
    tracesOfGather = np.empty((traceLength, 0), float)

    # Loop through the offsets from startOffset to 2200 in steps of 100
    for i in range(startOffset, 2200, 100):
        print('Taking trace of', i)  # Log the current offset being processed

        # Extract the trace for the current offset
        aTrace = smaller_3d_gath.data.sel(iline=_iline, xline=_xline, offset=i).data

        # Reshape the trace to add it as a column to the matrix
        aTrace = aTrace[:, np.newaxis]

        # Append the trace to the tracesOfGather matrix
        tracesOfGather = np.concatenate((tracesOfGather, aTrace), axis=1)

    # Return the matrix of traces and the corresponding time values
    return tracesOfGather, time


def multiVariateDataset(dataset, prediction, start_index, end_index, window_size):
    """
    Creates a multivariate dataset for seismic data (for a CDP) by slicing the input dataset into windows
    and associating each window with a corresponding prediction value.

    Parameters:
        dataset (numpy.ndarray): The input dataset containing seismic data.
        prediction (numpy.ndarray): The target values corresponding to the dataset.
        start_index (int): The starting index for slicing the dataset.
        end_index (int or None): The ending index for slicing the dataset. If None, the entire dataset is used.
        window_size (int): The size of the window to slice from the dataset.

    Returns:
        tuple:
            data (numpy.ndarray): A 3D array containing the sliced windows of the dataset.
            predictions (numpy.ndarray): A 1D array containing the corresponding prediction values for each window.
    """
    data = []  # List to store the sliced windows of the dataset
    predictions = []  # List to store the corresponding prediction values

    # Adjust the start index to account for the window size
    start_index = start_index + window_size

    # If end_index is None, set it to the length of the dataset
    if end_index is None:
        end_index = len(dataset)

    # Loop through the dataset to create windows and associate predictions
    for i in range(start_index, end_index):
        # Get the indices for the current window
        indices = range(i - window_size, i)

        # Reshape the data from (window_size,) to (window_size, number of features)
        data.append(np.reshape(dataset[indices], (window_size, len(dataset[0]))))

        # Find the middle sample in the current window for the prediction
        input_list = list(indices)
        middle_sample_prediction = findMiddle(input_list)
        predictions.append(prediction[middle_sample_prediction])

    # Convert the lists to numpy arrays for output
    return np.array(data), np.array(predictions)

def findMiddle(input_list): #OK
    """
    Finds the middle element(s) of a list.

    Parameters:
        input_list (list): A list of elements.

    Returns:
        int or tuple:
            - If the list has an odd number of elements, returns the middle element.
            - If the list has an even number of elements, returns a tuple containing the two middle elements.
    """
    # Calculate the middle index of the list
    middle = float(len(input_list))/2
    
    # If the list length is odd, return the single middle element
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
         # If the list length is even, return the two middle elements as a tuple
        return (input_list[int(middle)], input_list[int(middle-1)])
    
    

def CDPAndAIpreProcessing(x1, CDP, x2, AI, startTime, stopTime):
    """
    Preprocesses seismic and well log data by normalizing and interpolating.

    Parameters:
        x1 (numpy.ndarray): Seismic time array.
        CDP (numpy.ndarray): Seismic data (CDP - Common Depth Point).
        x2 (numpy.ndarray): Well log time array.
        AI (numpy.ndarray): Acoustic Impedance (AI) data from the well log.
        startTime (float): Start time for the range of interest.
        stopTime (float): Stop time for the range of interest.

    Returns:
        tuple:
            train (numpy.ndarray): Interpolated and normalized seismic data.
            label (numpy.ndarray): Normalized Acoustic Impedance data for the specified range.
            xi (numpy.ndarray): Interpolated time array for the specified range.
    """
    
    # Reshape AI (Acoustic Impedance) data to a 2D array for normalization
    AI = AI[:, np.newaxis]
    
    # Normalize AI data using MaxAbsScaler
    scaler = MaxAbsScaler()
    AINorm = scaler.fit_transform(AI)
    y2 = np.array(AINorm[:, 0])  # Extract normalized AI as a 1D array
    
    # Normalize CDP (seismic data) using MaxAbsScaler
    scaler = MaxAbsScaler()
    CDPNorm = scaler.fit_transform(CDP)
    
    # Find the start index in x2 (well log time array) corresponding to startTime
    auxVec = np.array(np.where(x2 >= startTime))
    auxStart = auxVec[0, 0]

    # Find the stop index in x2 (well log time array) corresponding to stopTime
    auxVec = np.array(np.where(x2 >= stopTime))
    auxStop = auxVec[0, 0]
    
    # Calculate the length of the range of interest in the well log time array
    xilength = len(y2[auxStart:auxStop])
    
    # Extract the time range of interest from x2
    xi = x2[auxStart:auxStop]
    
    # Interpolate CDP data to match the time range of interest
    CDPinterpolated = np.empty((xilength, 0), float)  # Initialize an empty array for interpolated data
    for i in range(0, len(CDPNorm[0, :]), 1):
        y1 = CDPNorm[:, i]  # Extract a single trace from normalized CDP data
        rbfi = Rbf(x1, y1, function='cubic')  # Create a cubic Radial Basis Function interpolator
        yi = rbfi(xi)  # Interpolate the trace to match the time range of interest
        
        yi = yi[:, np.newaxis]  # Reshape to add as a column
        CDPinterpolated = np.concatenate((CDPinterpolated, yi), axis=1)  # Append the interpolated trace
    
    # Prepare the output: interpolated seismic data (train), normalized AI (label), and time array (xi)
    train = CDPinterpolated
    label = y2[auxStart:auxStop]
    
    return train, label, xi

