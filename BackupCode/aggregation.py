import numpy as np
import tensorflow as tf
from keras.layers import Dropout
from keras.models import Sequential, load_model
from keras.layers import LSTM, TimeDistributed, Dense, Flatten
from keras.optimizers import SGD
import random
import math
import json
import boto3
import os
import logging


alpha = 1

csv_names = ["하이테크센터", "로스쿨관", "1호관", "7호관", 
                "2호남관", "5호북관", "2호북관", "김현태인하드림센터", "5호동관",
                "5호남관", "인하드림센터", "9호관", "60주년기념관", "서호관"]

                
def LSTM_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(144, 3), activation='tanh'))
    model.add(Dropout(0.2))  # Dropout 추가

    model.add(LSTM(units=50, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))  # Dropout 추가

    model.add(TimeDistributed(Dense(units=1)))  # 각 시간 단계마다 독립적인 예측
    #model.add(Dense(units=144)) 

    # Compile
    model.compile(optimizer = SGD(learning_rate = 0.01, decay = 1e-7,
                                 momentum=0.9, nesterov=False), loss = 'mean_squared_error')

    return model

def aggregation(local_weights):
    average_weights = []

    # Iterate over each layer index
    for layer_index in range(len(local_weights[0])):
        # Stack the weights of this layer from all models
        layer_weights = np.array([model_weights[layer_index] for model_weights in local_weights])
        
        # Compute the mean of the weights for this layer
        layer_mean = np.mean(layer_weights, axis=0)
        
        # Append the mean weights for this layer to the list
        average_weights.append(layer_mean)

    return average_weights

    # Define the mutation function
def mutation(alpha, weights_delta, global_weights, num_clients, num_layer):
    mutated_model = [[] for _ in range(num_clients)]

    if num_clients % 2 == 1:
        mutated_model = [[tf.Variable(gp) for gp in global_weights] for _ in range(num_clients)]

    # Generate random vector
    random_vector = [-1.0] * (math.floor(num_clients // 2)) + [1.0] * (math.floor(num_clients // 2))
    random_vectors = [random_vector[:] for _ in range(num_layer)]

    # Shuffle random vector
    for i in range(num_layer):
        random.shuffle(random_vectors[i])
    for i in range(2 * math.floor(num_clients // 2)):
        mutated_weights = global_weights.copy()
        for j in range(num_layer):
            mutation_value = [alpha * random_vectors[j][i] * d for d in weights_delta[j]]
            mutated_weights[j] = tf.math.add(global_weights[j], mutation_value[j])
            mutated_model[i].append(mutated_weights[j])

    return mutated_model

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def download_npy_files_from_s3(bucket_name, prefix):
    download_dir = os.path.join(os.getcwd(), 'Local_Models')
    
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

weights = []

for i in range(len(csv_names)):
    # JSON 파일에서 가중치를 불러옵니다.
    with open(f'Local_Models/{csv_names[i]}.json', 'r') as json_file:
        weights_as_list = json.load(json_file)

    loaded_weights = [np.array(w) for w in weights_as_list]
    weights.append(loaded_weights)
    

global_weights = aggregation(weights)
global_weights = [tf.convert_to_tensor(w, dtype=tf.float32) for w in global_weights]
weights_delta = []

for i in range(len(weights)):
    tensor_weights = [tf.convert_to_tensor(w, dtype=tf.float32) for w in weights[i]]
    delta = [tf.math.subtract(gw, tw) for gw, tw in zip(global_weights, tensor_weights)]    
    weights_delta.append(delta)

mutation_model = mutation(alpha, weights_delta, global_weights, len(csv_names), len(global_weights))

random_csv_names = random.sample(csv_names, len(csv_names))

for i in range(len(csv_names)):
    model = LSTM_model()
    model.set_weights(mutation_model[i])
    weights = model.get_weights()
    weights_as_list = [w.tolist() for w in weights]
    # JSON 파일로 저장합니다.
    with open(f'Global_weights({random_csv_names[i]}).json', 'w') as json_file:
        json.dump(weights_as_list, json_file)

print("Complete Aggregation and Generate Global Model")