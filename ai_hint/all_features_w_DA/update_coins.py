from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import joblib
import datetime
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import pyupbit

from config import Config


interval = 'minute1'
count = 60 * 24 * 365 * 2

timestep = 1
window_size = 5

cfg = Config()
col_list = cfg.used_cols
n_features = len(col_list)


def inverse_transform_predictions(preds, scaler, label_idx=-1):
    dummy = np.zeros((len(preds), scaler.n_features_in_))
    dummy[:, label_idx] = preds[:, 0]
    return scaler.inverse_transform(dummy)[:, label_idx]


def get_percentage(future, past):
    return (future - past) / past * 100


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(timestep, n_features)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


def get_data(coin):
    
    print('Loading data of new coin -', coin)

    df = pyupbit.get_ohlcv(coin, interval=interval, to=datetime.datetime.now(), count=cfg.trim_rows)
    df.reset_index().rename(columns={'index':'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f'New {coin} data loaded')
    df.to_csv(f'../../data/from_pyupbit/{coin}.csv', index=False)
    

def train(coin):
    get_data(coin)
    
    df = pd.read_csv(f'../../data/from_pyupbit/{coin}.csv')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    df['close_change'] = df['close'].diff().fillna(0)

    df['percentage_change'] = get_percentage(df['close'], df['close'].shift(1)).fillna(0)
    df['volatility'] = df['percentage_change'].rolling(window=window_size).std()
    df['avg_change_rate'] = df['percentage_change'].rolling(window=5).mean()

    df = df[col_list].fillna(0)

    scaled_data = scaler.fit_transform(df)
    joblib.dump(scaler, f'models/fewer/{coin}_scaler.pkl')

    X = []
    y = []
    for i in range(len(scaled_data) - timestep):
        X.append(scaled_data[i:(i + timestep), :])
        y.append(scaled_data[i + timestep, -1])

    X, y = np.array(X), np.array(y)

    train_size = int(len(df) * .8)
    X_train, X_val = X[:train_size], X[train_size:] # n_samples, timestep, n_features
    y_train, y_val = y[:train_size], y[train_size:]

    train_dates = df.index[:train_size]
    val_dates = df.index[train_size:]

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='loss', patience=10)

    print(f'{coin} train start')

    model.fit(X_train, y_train, batch_size=32, epochs=20,
            validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)
    model.save(f'models/fewer/lstm_{coin}.h5')

    train_predict = model.predict(X_train)
    val_predict = model.predict(X_val)

    train_predict_original = inverse_transform_predictions(train_predict, scaler)
    val_predict_original = inverse_transform_predictions(val_predict, scaler)
    y_train_original = inverse_transform_predictions(y_train.reshape(-1, 1), scaler)
    y_val_original = inverse_transform_predictions(y_val.reshape(-1, 1), scaler)

    train_score = np.sqrt(mean_squared_error(y_train, train_predict))
    val_score = np.sqrt(mean_squared_error(y_val, val_predict))
    print(f'{coin} Train RMSE: {train_score:.6f} Validation RMSE: {val_score:.6f}')

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 1, 1)
    plt.plot(train_dates, y_train_original, label='Actual', color = 'Blue')
    plt.plot(train_dates, train_predict_original, label='Predicted', color = 'Red')
    plt.title(f'{coin} Train Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.savefig(f'results/fewer/{coin}_train.png')

    plt.subplot(1, 1, 1)
    plt.plot(val_dates[:-1], y_val_original, label='Actual', color = 'Blue')
    plt.plot(val_dates[:-1], val_predict_original, label='Predicted', color = 'Red')
    plt.title(f'{coin} Validation Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    plt.tight_layout()
    plt.savefig(f'results/fewer/{coin}_val.png')

    last_1_day = scaled_data[-timestep:]
    X_test = last_1_day[-1, :].reshape(1, timestep, n_features)
    pred = model.predict(X_test)
    pred = inverse_transform_predictions(pred, scaler)
    print(f'{coin} Actual {scaler.inverse_transform(last_1_day)[-1, -1]:.6f} Pred {pred[0]:.6f}')
    print('.')
    