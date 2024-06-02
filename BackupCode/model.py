import tensorflow as tf

from keras.layers import Dropout
from keras.models import Sequential, load_model
from keras.layers import LSTM, TimeDistributed, Dense, Flatten
from keras.optimizers import SGD

def LSTM_model(x_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]), activation='tanh'))
    model.add(Dropout(0.2))  # Dropout 추가

    model.add(LSTM(units=50, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))  # Dropout 추가

    model.add(TimeDistributed(Dense(units=1)))  # 각 시간 단계마다 독립적인 예측
    #model.add(Dense(units=144)) 

    return model
