import numpy as np
import pandas as pd

def ts_train_test(data, time_steps, for_periods):
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    
    start_date = data.iloc[1]['DateTime']
    end_date = data.iloc[-1]['DateTime'] - pd.Timedelta(days=1)
    # middle_date = start_date + (end_date - start_date) / 2 + pd.Timedelta(days=1*bound)
    # middle_date = middle_date.replace(hour=23, minute=50, second=0, microsecond=0)
    # middle_date = middle_date - pd.Timedelta(days=2)
    end_date = end_date.replace(hour=23, minute=50, second=0, microsecond=0)
    mask2_date = end_date - pd.Timedelta(days=1)

    print(start_date)
    print(end_date)
    print(mask2_date)

    mask1 = (data['DateTime'] >= start_date) & (data['DateTime'] <= end_date)  
    mask2 = (data['DateTime'] >= mask2_date) & (data['DateTime'] <= end_date)

    ts_train_scaled = data.loc[mask1,['DayOfWeek', 'Time', 'PowerUsage']].values
    ts_test_scaled = data.loc[mask2,['DayOfWeek', 'Time', 'PowerUsage']].values
    x_train = []
    y_train = []

    for i in range(time_steps, len(ts_train_scaled) - for_periods): # 4594번실행
        x_train.append(ts_train_scaled[i-time_steps:i, :])
        y_train.append(ts_train_scaled[i:i+for_periods,2])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    inputs = np.concatenate((ts_train_scaled[-time_steps:], ts_test_scaled[:for_periods]))
    x_test = []

    for i in range(time_steps, len(inputs) - for_periods + 1):
        x_test.append(inputs[i-time_steps:i])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    return x_train, y_train, x_test

def preprocess_data(data):
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['DayOfWeek'] = data['DayOfWeek'].astype(float)
    # ex 12:20:00은 13이 시간, 20이 분이므로 13*6 + 20//10 = 80이 된다.
    data['Time'] = data['Time'].apply(lambda x: int(x.split(':')[0]) * 6 + int(x.split(':')[1]) // 10).astype(float)

    # 필요한 열만 선택 (DateTime 포함)
    features = ['DateTime', 'DayOfWeek', 'Time', 'PowerUsage']
    data = data[features].copy()  # 슬라이스를 명시적으로 복사


    # 정규화
    data['DayOfWeek'] = data['DayOfWeek'] / 6.0  # 요일은 0에서 6 사이 값이므로
    data['Time'] = data['Time'] / 144.0  # 하루는 144개의 10분 단위로 구성

    return data

def read_data(building):
    df = pd.read_csv(f"OriginCSV/{building}.csv")
    return df



# def test_data(data, time_steps, for_periods, bound):
#     data['DateTime'] = pd.to_datetime(data['DateTime'])
    
#     start_date = data.iloc[1]['DateTime']
#     end_date = data.iloc[-1]['DateTime']
#     middle_date = start_date + (end_date - start_date) / 2 + pd.Timedelta(days=1*bound)    
#     middle_date = middle_date.replace(hour=23, minute=50, second=0, microsecond=0)
#     middle_date = middle_date - pd.Timedelta(days=3)
    
#     mask2_date = middle_date - pd.Timedelta(days=1)
#     test_date = mask2_date + pd.Timedelta(days=1)

#     mask1 = (data['DateTime'] >= start_date) & (data['DateTime'] <= middle_date)  
#     mask2 = (data['DateTime'] >= mask2_date) & (data['DateTime'] <= test_date)
#     print(f"mask2_date: {mask2_date}")
#     print(f"test_date: {test_date}")
#     ts_train_scaled = data.loc[mask1,['DayOfWeek', 'Time', 'PowerUsage']].values
#     ts_test_scaled = data.loc[mask2,['DayOfWeek', 'Time', 'PowerUsage']].values
#     x_train = []
#     y_train = []

#     for i in range(time_steps, len(ts_train_scaled) - for_periods): # 4594번실행
#         x_train.append(ts_train_scaled[i-time_steps:i, :])
#         y_train.append(ts_train_scaled[i:i+for_periods,2])

#     x_train, y_train = np.array(x_train), np.array(y_train)

#     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
#     y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
#     inputs = ts_test_scaled[:for_periods]

#     x_test = []

#     for i in range(time_steps, len(inputs) + 1):#len(inputs) - for_periods + 1):
#         x_test.append(inputs[i-time_steps:i])
        
#     # x_test.append(inputs)

#     x_test = np.array(x_test)
#     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

#     return x_train, y_train, x_test

def test_data(data, time_steps, for_periods):
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    
    start_date = data.iloc[1]['DateTime']
    end_date = data.iloc[-1]['DateTime']
    end_date = end_date.replace(hour=23, minute=50, second=0, microsecond=0) - pd.Timedelta(days=1)
    mask2_date = end_date - pd.Timedelta(days=1)

    mask1 = (data['DateTime'] >= start_date) & (data['DateTime'] <= end_date)  
    mask2 = (data['DateTime'] >= mask2_date) & (data['DateTime'] <= end_date)

    print(f"mask2_date: {mask2_date}")
    print(f"test_date: {end_date}")
    
    ts_train_scaled = data.loc[mask1,['DayOfWeek', 'Time', 'PowerUsage']].values
    ts_test_scaled = data.loc[mask2,['DayOfWeek', 'Time', 'PowerUsage']].values
    x_train = []
    y_train = []

    for i in range(time_steps, len(ts_train_scaled) - for_periods): # 4594번실행
        x_train.append(ts_train_scaled[i-time_steps:i, :])
        y_train.append(ts_train_scaled[i:i+for_periods,2])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    inputs = ts_test_scaled[:for_periods]

    x_test = []

    for i in range(time_steps, len(inputs) + 1):#len(inputs) - for_periods + 1):
        x_test.append(inputs[i-time_steps:i])
        
    # x_test.append(inputs)

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    return x_train, y_train, x_test

