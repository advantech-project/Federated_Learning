import json
import numpy as np
import pandas as pd
import boto3
import os
import logging

csv_names = ["하이테크센터", "로스쿨관", "1호관", "7호관", 
                "2호남관", "5호북관", "2호북관", "김현태인하드림센터", "5호동관",
                "5호남관", "인하드림센터", "9호관", "60주년기념관", "서호관"]

              
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
    building_name = f"Global_weights({csv_names[i]})"
    bucket_name = "ecoala-bucket-socket"
    upload_npy_to_s3(building_name, bucket_name)

print("Complete Download Global Models")