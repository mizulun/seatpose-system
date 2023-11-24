import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
#from .config import shooting_result, previous, during_shooting, shooting_pose, shot_result
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
import json
import csv
tf.disable_v2_behavior() #disable all tensorflow version 2 function behavior

def tensorflow_init():
    MODEL_NAME = 'inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    print("> ====== Loading detection frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
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
            import OpenPose.Release.pyopenpose as op
        else:
            path = os.path.join(os.getcwd(), '../OpenPose/openpose')
            print("OpenPose path:", path)
            print(path)
            sys.path.append(path)
            import pyopenpose as op
        print("OpenPose imported successfully!")
    except ImportError as e:
        print("Something went wrong when importing OpenPose")
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../OpenPose/models"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    return datum, opWrapper, op

def json_to_csv():
    # Directory containing the JSON files
    directory = 'static/output/json'

    # Output directory for CSV files
    output_directory = 'static/output/csv'

    # List to store the extracted values
    data = []

    # Indices to remove from the keypoints list
    indices_to_remove = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 26,
                        29, 32, 35, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                        56, 59, 62, 65, 68, 71, 74]

    # Iterate through the JSON files in the directory to find the minimum value
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # Read the JSON file
            with open(file_path) as f:
                json_data = json.load(f)

            # Extract the desired value (the 25th index)
            value_at_25 = json_data['people'][0]['pose_keypoints_2d'][25]

            # Initialize variables to find the minimum value for each file
            min_value = float('inf')  # Initialize with positive infinity

            # Process the keypoints for the current file
            keypoints = json_data['people'][0]['pose_keypoints_2d']
        
            # Find the minimum value for the current file
            for value in keypoints:
                if value < min_value:
                    min_value = value

    # Iterate through the JSON files again and process them
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # Read the JSON file
            with open(file_path) as f:
                json_data = json.load(f)

            # Extract the desired values
            keypoints = json_data['people'][0]['pose_keypoints_2d']

        # Find nodes with a difference less than 0.02 from the minimum value
            if abs(value_at_25 - min_value) > 0.02:
                # Remove specific indices from the keypoints list
                keypoints = [val for i, val in enumerate(keypoints) if i not in indices_to_remove]
                # Append the values to the data list
                data.append(keypoints)

    # Save the data to a CSV file
    output_file = os.path.join(output_directory, 'output.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"Extraction complete. Data saved to {output_file}.")

