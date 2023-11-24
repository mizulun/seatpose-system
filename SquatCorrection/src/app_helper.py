import subprocess
import os
import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
#from .config import shooting_result, previous, during_shooting, shooting_pose, shot_result
import sys, logging
from sys import platform
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .utils import tensorflow_init, openpose_init, json_to_csv
from statistics import mean
import pandas as pd
#from openpyxl import load_workbook
import glob
import json
import csv
tf.disable_v2_behavior()

def getRealTimeStream():
    # load model
    from keras.models import load_model
    #cgan_model_path = os.path.join(os.getcwd(), "model_training/pix2pix_generator_v3.h5")
    #cgan_model = load_model(cgan_model_path)

    lstm_model_path = os.path.join(os.getcwd(), "model_training/Model_9types_Squat_LSTM_final_v1.h5")
    lstm_model = load_model(lstm_model_path)

    original_dir = os.getcwd()
    openpose_dir = 'OpenPose'
    os.chdir(openpose_dir)

    command = [
        'build2\\x64\\Debug\\OpenPoseDemo.exe', '--flir_camera', '--3d', '--write_json', '..\\static\\output\\json', '--keypoint_scale', '3', '--number_people_max', '1'
    ]

    subprocess.run(command)

    os.chdir(original_dir)

def openposeAnalysis(video_path, video_name):
    # load model
    from keras.models import load_model
    #cgan_model_path = os.path.join(os.getcwd(), "model_training/pix2pix_generator_v3.h5")
    #cgan_model = load_model(cgan_model_path)

    lstm_model_path = os.path.join(os.getcwd(), "model_training/Model_9types_Squat_LSTM_final_v1.h5")
    lstm_model = load_model(lstm_model_path)

    original_dir = os.getcwd()
    openpose_dir = 'OpenPose'
    os.chdir(openpose_dir)

    command = [
        'build2\\x64\\Debug\\OpenPoseDemo.exe', '--video', '..\\static\\uploads\\' + video_name, '--write_json', '..\\static\\output\\json', 
        '--keypoint_scale', '3', '--number_people_max', '1', '--write_video_with_audio', '..\\static\\output\\processed_videos\\output.mp4', 
    ]


    subprocess.run(command)

    os.chdir(original_dir)

    json_to_csv()

    #Load csv
    csv_path = 'static/output/csv/output.csv'
    df = pd.read_csv(csv_path, header=None, encoding='gb2312', sep=',')
    df = np.array(df)
    df = df.reshape(df.shape[0], 32, 1)

    #Start predicting
    predictions = np.argmax(lstm_model.predict(df), axis=-1)
    most_common_prediction, count = np.unique(predictions, return_counts=True)
    most_common_prediction = most_common_prediction[np.argmax(count)]
    percentage = count[np.argmax(count)] / len(predictions) * 100

    # Default values
    feetDistance = "Unknown"
    kneeDistance = "Unknown"

    # Determine feetDistance
    if most_common_prediction in [0, 1, 2]:
        feetDistance = "標準"
    elif most_common_prediction in [3, 4, 5]:
        feetDistance = "過窄"
    elif most_common_prediction in [6, 7, 8]:
        feetDistance = "過寬"

    # Determine kneeDistance
    if most_common_prediction in [0, 3, 6]:
        kneeDistance = "標準"
    elif most_common_prediction in [1, 4, 7]:
        kneeDistance = "過窄"
    elif most_common_prediction in [2, 5, 8]:
        kneeDistance = "過寬"

    print(predictions)
    print(most_common_prediction)
    print("Feet Distance:", feetDistance)
    print("Knee Distance:", kneeDistance)
    print("準確率： {:.2f}%".format(percentage))

    return feetDistance, kneeDistance, percentage

