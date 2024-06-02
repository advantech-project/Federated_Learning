import config as cfg
import json
import numpy as np
import pandas as pd
import boto3
import os
import logging
from model import LSTM_model
from data import ts_train_test, read_data, preprocess_data



time_steps = 144
for_periods = 144

csv_names = ["2호남관", "2호북관", "5호동관", "하이테크센터"]
en_names = ['South of No. 2 building', 'North of No. 2 building', 'East of No. 5 building', 'HighTech'] 

def upload_model():
    for i in range(len(csv_names)):
        data = read_data(csv_names[i])

        data = preprocess_data(data)

        x_train, y_train, test_data = ts_train_test(data, time_steps, for_periods)

        model = LSTM_model(x_train)
        model.load_weights(f'Model/{csv_names[i]}.h5')
        weights = model.get_weights()
        weights_as_list = [w.tolist() for w in weights]
        # JSON 파일로 저장합니다.
        with open(f'{csv_names[i]}.json', 'w') as json_file:
            json.dump(weights_as_list, json_file)


    # 로깅 설정
    logging.basicConfig(level=logging.INFO)

    def upload_npy_to_s3(building_name, bucket_name):
        # JSON 파일 이름 설정
        file_name = f"{building_name}.json"
        
        # 파일이 존재하는지 확인
        if not os.path.isfile(file_name):
            logging.error(f"파일 {file_name}이(가) 존재하지 않습니다.")
            return
        
        # S3 클라이언트 생성
        try:
            s3 = boto3.client('s3')
            logging.info("S3 클라이언트 생성 성공")
        except Exception as e:
            logging.error(f"S3 클라이언트 생성 실패: {e}")
            return
        
        # 파일 업로드 시도
        try:
            s3.upload_file(file_name, bucket_name, file_name)
            logging.info("업로드 성공")
        except Exception as e:
            logging.error(f"업로드 실패: {e}")

    for i in range(len(csv_names)):
        building_name = f"{csv_names[i]}"
        bucket_name = "ecoala-bucket-socket"
        upload_npy_to_s3(building_name, bucket_name)
    
    print("Complete Upload Local Model")