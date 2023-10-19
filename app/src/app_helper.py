import subprocess
import os
import pandas as pd

def getVideoStream(video_path, filename, folder_path):
    # 調用OpenPose處理影片
    openpose_cmd = [
        'openpose.bin',
        '--video', video_path,
        '--write_json', 'json_output_folder',
        '--number_people_max', '1',
        '--keypoint_scale', '3'
    ]
    
    try:
        subprocess.run(openpose_cmd, check=True)
    except subprocess.CalledProcessError:
        # 處理錯誤情況
        pass

    # 呼叫 json_to_csv.py 來轉換 JSON 到 CSV
    json_to_csv_cmd = [
        'python',
        'C:\\Users\\Miley\\Squat-Correction\\json_to_csv.py',
        'json_output_folder',  # 指定 JSON 檔案的路徑
        'output.csv'  # 指定輸出的 CSV 檔案名稱和路徑
    ]
    
    try:
        subprocess.run(json_to_csv_cmd, check=True, cwd=folder_path)
    except subprocess.CalledProcessError:
        # 處理錯誤情況
        pass

    # 使用 LSTM 模型進行預測
    csv_path = os.path.join(folder_path, 'output.csv')
    lstm_model_path = 'C:\\Users\\Miley\\Squat-Correction\\Model_9types_Squat_LSTM_final_v1'  # 指定 LSTM 模型的路徑
    lstm_model = load_lstm_model(lstm_model_path)

    # 讀取 CSV 數據
    df = pd.read_csv(csv_path)

    # 進行預測
    predictions = lstm_model.predict(df)

    return predictions  # 返回預測結果