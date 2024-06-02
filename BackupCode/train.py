import config as cfg
import json
import numpy as np
import pandas as pd
import boto3
from model import LSTM_model
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from data import ts_train_test, read_data, preprocess_data
from upload_weights import upload_model

import logging
import os
time_steps = 144
for_periods = 144

csv_names = ["2호남관", "2호북관", "5호동관", "하이테크센터"]
en_names = ["South of No.2", "North of No.2", "East of No.5", "HighTech Center"]


# 로깅 설정
logging.basicConfig(level=logging.INFO)

def download_npy_files_from_s3(bucket_name, prefix):
    download_dir = os.path.join(os.getcwd(), 'Download_Models')
    
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    # S3 클라이언트 생성
    s3 = boto3.client('s3')
    
    # S3 버킷에서 특정 접두사를 가지는 파일 목록 가져오기
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    # 파일 다운로드
    if 'Contents' in response:
        for obj in response['Contents']:
            file_name = obj['Key']
            if file_name.endswith('.json'):
                local_file_path = os.path.join(download_dir, os.path.basename(file_name))
                try:
                    s3.download_file(bucket_name, file_name, local_file_path)
                    logging.info(f"Downloaded '{file_name}' to '{local_file_path}'.")
                except Exception as e:
                    logging.error(f"An error occurred while downloading '{file_name}': {e}")
    else:
        logging.info(f"No files found with prefix '{prefix}' in bucket '{bucket_name}'.")

# 변수 설정
bucket_name = "ecoala-bucket-socket"
for i in range(len(csv_names)):
    prefix = f"{csv_names[i]}"  # 접두사를 지정

# 다운로드 함수 호출
    download_npy_files_from_s3(bucket_name, prefix)

for i in range(len(csv_names)):
    # JSON 파일에서 가중치를 불러옵니다.
    with open(f'Download_Models/{csv_names[i]}.json', 'r') as json_file:
        weights_as_list = json.load(json_file)

    loaded_weights = [np.array(w) for w in weights_as_list]
    
    model = load_model(f'Model/{csv_names[i]}.h5')
    model.set_weights(loaded_weights)
    model.save(f'Model/{csv_names[i]}.h5')

print("Complete Download Global Model")

for i in range(len(csv_names)):
    data = read_data(csv_names[i])
    data = preprocess_data(data)
    x_train, y_train, test_data = ts_train_test(data, time_steps, for_periods)
    model = load_model(f'Model/{csv_names[i]}.h5')
    model.compile(optimizer = SGD(learning_rate = 0.01, decay = 1e-7,
                                 momentum=0.9, nesterov=False), loss = 'mean_squared_error')

    model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)
    model.summary()
    model.save(f'Model/{csv_names[i]}.h5')

upload_model()

test_dataset = [[] for _ in range(len(csv_names))]
dataset = []

for i in range(len(csv_names)):
    data = read_data(csv_names[i])
    data = preprocess_data(data)
    dataset.append(data)

    x, y, test = test_data(data, 144, 144)

    test_dataset[i].append(test)

print("Test Dataset is Ready")

predict_result = [[] for _ in range(len(csv_names))]

for i in range(len(csv_names)):
    model = load_model(f'Model/{csv_names[i]}.h5')
    predict_result[i].append(model.predict(test_dataset[i]))

print("Complete")

dataset[0]['DateTime'] = pd.to_datetime(dataset[0]['DateTime'])
# training & test set 만들기
start_date = dataset[0].iloc[1]['DateTime']
end_date = dataset[0].iloc[-1]['DateTime']
end_date = end_date.replace(hour=23, minute=50, second=0, microsecond=0) - pd.Timedelta(days=1)

mask2_date = end_date - pd.Timedelta(days=1)
predicted_dates = pd.date_range(start=end_date, periods=144, freq='10T')
print(predicted_dates)

for i in range(len(csv_names)):
    df = pd.DataFrame({
        'DateTime':predicted_dates,
        'Prediction': predict_result[i]
    })
    df.to_csv(f'{csv_names[i]}_prediction.csv', index=False)

print("Save Prediction!")
