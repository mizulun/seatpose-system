from asyncio import BoundedSemaphore
import subprocess
import os
import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
from .config import squat_result
import sys, logging
from sys import platform
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .utils import tensorflow_init, openpose_init, detect_squat
from statistics import mean
import pandas as pd
#from openpyxl import load_workbook
import glob
tf.disable_v2_behavior()

def openposeAnalysis(video_path, filename, folder_path):
    # load model
    from keras.models import load_model
    cgan_model_path = os.path.join(os.getcwd(), "model_training/pix2pix_generator_v5.h5")
    cgan_model = load_model(cgan_model_path)

    lstm_model_path = os.path.join(os.getcwd(), "model_training/Model_9types_Squat_LSTM_final_v1.h5")
    lstm_model = load_model(lstm_model_path)

    squat_result['feet_distance']
    SKIPS = 2

    datum, opWrapper, op = openpose_init()
    #detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    fig = plt.figure()
    #objects to store detection status

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.36
    
    print("Start getting data!!")

    # Reading selected file
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    trace = np.full((int(height), int(width), 3), 255, np.uint8)

    # config printer
    print("Image Size: {} x {}".format(width, height))
    print("Frame per Second:",fps)

    # sample output as video
        # note that the size must be accuracy
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('static/output/'+str(filename), fourcc, float(fps), (int(width),  int(height)))

    skip_count = 0
    #with tf.Session(graph=detection_graph, config=config) as sess:
    while True:
        ret, img = cap.read()
        if ret == False:
            print("end of video")
            break
        skip_count += 1
        if(skip_count < SKIPS):
            continue
        skip_count = 0
        output_frame, prediction, feetDistance, kneeDistance = detect_squat(img, datum, opWrapper, op)

        if (feetDistance != "Standard" or kneeDistance != "Standard"):
            print('write image')
            cv2.imwrite("C:\\Users\\USER\\SquatCorrection\\static\\output\\incorrect_images\\" + str(filename.split(".")[0]) + ".jpg", bones)
            cv2.imwrite("C:\\Users\\USER\\SquatCorrection\\static\\detections\\origin.jpg", cv2.resize(output_frame, (int(width),  int(height))))
            x = bones

        # sample output as video
        video = cv2.resize(output_frame, (int(width),  int(height)))
        # Add feetDistance and kneeDistance text to the video frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'Feet Distance: {feetDistance}  Knee Distance: {kneeDistance}'
        cv2.putText(video, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(video)

        prediction = cv2.resize(prediction, (0, 0), fx=0.83, fy=0.83)
        frame = cv2.imencode('.jpg', video)[1].tobytes()
        result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        yield result # output every frames to website by generator

