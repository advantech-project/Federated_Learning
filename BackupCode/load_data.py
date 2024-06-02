import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pyodbc

# 데이터베이스 접속 정보
server = '165.246.21.10,10001'
database = 'EMS'
username = 'sw_stud'
password = 'SW@stud1!'
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# SQL Server에 접속
conn = pyodbc.connect(conn_str)

# 데이터 가져오기
def fetch_data(query):
    df = pd.read_sql(query, conn)
    return df       

def preprocess_building_data(building, df):
    scaler = MinMaxScaler()

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')

    if 'TimeStamp' in df.columns:
        df.drop('TimeStamp', axis=1, inplace=True)

    df['DayOfWeek'] = df['DateTime'].dt.dayofweek

    first_row_day_of_week = 1  # 화요일
    shift = first_row_day_of_week - df.loc[0, 'DayOfWeek']
    df['DayOfWeek'] = (df['DayOfWeek'] + shift) % 7

    df['Time'] = df['DateTime'].dt.time

    df.reset_index(drop=True, inplace=True)
    for i in range(1, len(df)-1):
        if df.loc[i, 'DataValue'] < df.loc[i-1, 'DataValue']:
            median_value = np.median([df.loc[i-1, 'DataValue'], df.loc[i+1, 'DataValue']])
            df.loc[i, 'DataValue'] = median_value

    df['PowerUsage'] = df['DataValue'].diff().fillna(0)
    df.loc[0, 'PowerUsage'] = 0  # 첫 번째 행 처리

    df['OriginalPowerUsage'] = df['PowerUsage']

    df['PowerUsage'] = scaler.fit_transform(df[['PowerUsage']])
    return df

def main():
    building_names = ["하이테크센터", "2호남관/4호관", "2호북관", "5호동관"]

    csv_names = ["하이테크센터", "2호남관", "2호북관", "5호동관"]

    queries = [f"SELECT * FROM Tech_All_KWH WHERE Building = '{building}'" for building in building_names]

    output_dir = '/home/ec2-user/environment/OriginCSV'

    for building, query, name in zip(building_names, queries, csv_names):
        df = fetch_data(query)
        processed_df = preprocess_building_data(building, df)
        output_path = f'{output_dir}/{name}.csv'
        processed_df.to_csv(output_path, index=False)
        print(f"Data for building {building} fetched, processed, and saved as {output_path}")

if __name__ == "__main__":
    main()
