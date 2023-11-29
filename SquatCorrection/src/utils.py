import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
from .config import squat_result
import sys
from sys import platform
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import logging
import numpy as np
import argparse
import os
import glob
import csv
import pandas as pd
tf.disable_v2_behavior() #disable all tensorflow version 2 function behavior

def tensorflow_init():
    MODEL_NAME = 'inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    print("> ====== Loading detection frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:  # 使用 'rb' 以二進位模式打開檔案
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("> ====== Detection Inference graph loaded.")

    #Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    #Each box represents a part of the image where a partucular object was detected
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    #Each score represent how level of confidence for each of the object.
        #Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    return detection_graph, image_tensor, boxes, scores, classes, num_detections

def openpose_init():
    try:
        print("Attempting to import OpenPose...")
        if platform == "win32":
            sys.path.append(os.path.join(os.getcwd(), '..'))
            import Open_Pose.Release.pyopenpose as op
        else:
            path = os.path.join(os.getcwd(), '../Open_Pose/openpose')
            print("OpenPose path:", path)
            print(path)
            sys.path.append(path)
            import pyopenpose as op
        print("OpenPose imported successfully!")
    except ImportError as e:
        print("Something went wrong when importing OpenPose")
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = {
    "model_folder": "../OpenPose/models",
    "hand": False,  # Set to True if you want to detect hands
    "face": False,  # Set to True if you want to detect faces
    "number_people_max": 1,  # Set the maximum number of people to detect
    "keypoint_scale": 3
}

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    return datum, opWrapper, op

def detect_squat(frame, datum, opWrapper, op):
    # getting openpose keypoints
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    print(datum.poseKeypoints)

    # convert to csv
    pose_keypoints = datum.poseKeypoints

    # Define indices of keypoints to keep
    indices_to_keep = [1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]

    # Extract x and y values while excluding confidence
    numerical_values = []
    for idx in indices_to_keep:
        x, y, _ = pose_keypoints[0][idx]  # Unpack x, y, and confidence
        numerical_values.extend([x, y])

    # Output directory for CSV files
    output_directory = 'static/output/csv'
    # Save numerical values to a CSV file
    csv_filename = os.path.join(output_directory, 'pose_keypoints.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(numerical_values)

    print(f"Numerical values saved to {csv_filename}")

    # load model
    from keras.models import load_model
    lstm_model_path = os.path.join(os.getcwd(), "model_training/Model_9types_Squat_LSTM_final_v1.h5")
    lstm_model = load_model(lstm_model_path)

    #Load csv
    csv_path = 'static/output/csv/pose_keypoints.csv'
    df = pd.read_csv(csv_path, header=None, encoding='gb2312', sep=',')
    df = np.array(df)
    df = df.reshape(df.shape[0], 32, 1)

    #Start predicting
    prediction = np.argmax(lstm_model.predict(df), axis=-1)
    #most_common_prediction, count = np.unique(predictions, return_counts=True)
    #most_common_prediction = most_common_prediction[np.argmax(count)]
    #percentage = count[np.argmax(count)] / len(predictions) * 100

    # Default values
    squat_result['feetDistance'] = "Unknown"
    squat_result['kneeDistance'] = "Unknown"

    # Determine feetDistance
    if prediction in [0, 1, 2]:
        squat_result['feetDistance'] = "Standard"
    elif prediction in [3, 4, 5]:
        squat_result['feetDistance'] = "Narrow"
    elif prediction in [6, 7, 8]:
        squat_result['feetDistance'] = "Wide"

    # Determine kneeDistance
    if prediction in [0, 3, 6]:
        squat_result['kneeDistance'] = "Standard"
    elif prediction in [1, 4, 7]:
        squat_result['kneeDistance'] = "Narrow"
    elif prediction in [2, 5, 8]:
        squat_result['kneeDistance'] = "Wide"

    print(prediction)
    print("Feet Distance:", squat_result['feetDistance'])
    print("Knee Distance:", squat_result['kneeDistance'])

    return frame, prediction, squat_result['feetDistance'], squat_result['kneeDistance']
