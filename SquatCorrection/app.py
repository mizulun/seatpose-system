from asyncio import streams
import os
import sys
import cv2

from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash, jsonify, abort
from werkzeug.utils import secure_filename
from PIL import Image

#from src.config import squat_result
from src.app_helper import getRealTimeStream
from src.app_helper import openposeAnalysis

app = Flask(__name__)

#設置文件上傳目錄
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# 添加一個secret key来啟用 Flask 的 flash 消息
app.secret_key = "super secret key" 

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/video_upload', methods=['GET'])
def video_upload():
    return render_template("upload.html")

@app.route('/realtime_analysis', methods=['GET'])
def realtime():
    stream = getRealTimeStream()
    return Response(stream, content_type='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/squat-correction', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        f = request.files['video']
        # create a secure filename
        filename = secure_filename(f.filename)
        print("filename", filename)
        # save file to /static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("filepath", filepath)
        f.save(filepath)
        session['video_path'] = filepath
        session['video_name'] = filename
        session['folder_path'] = app.config['UPLOAD_FOLDER']
        feetDistance, kneeDistance, percentage = openposeAnalysis(filepath, filename)
        return render_template("squat_analysis.html", feetDistance=feetDistance, kneeDistance=kneeDistance, percentage=percentage)

@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path', None)
    video_name = session.get('video_name', None)
    #feetDistance, kneeDistance, percentage = openposeAnalysis(video_path, video_name)

    #response_data = {
     #   'feetDistance': feetDistance,
      #  'kneeDistance': kneeDistance,
       # 'percentage': percentage
    #}
    #return response_data
    #return Response(stream,
                    #mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/result", methods=['GET', 'POST'])
def result():
    return render_template("result.html", squat_result=squat_result)

#disable caching
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)