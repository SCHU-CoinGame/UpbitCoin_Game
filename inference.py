import tensorflow as tf
import os
import joblib
import pandas as pd
import numpy as np
import time
from threading import Thread
import pyupbit
import csv
import datetime


model_dir = 'ai_hint/models'
data_dir = 'data/from_pyupbit'

# coins = ['KRW-BTC', 'KRW-SXP', 'KRW-SUI', 'KRW-ARK', 'KRW-SHIB', 'KRW-UXLINK', 'KRW-XRP', 'KRW-SEI', 'KRW-HIFI']
coins = ['KRW-SXP']
last_times = {}  # csv 파일 마지막에 빈 줄 하나 있어야 함

models = []
scalers = []
data_paths = {}
model_paths = {}
scaler_paths = {}


def get_last_date(coin):
    filepath = os.path.join(data_dir, f'{coin}.csv')
    total_rows = sum(1 for _ in open(filepath))
    last_row = pd.read_csv(filepath, skiprows=range(1, total_rows-1))
    last_date = last_row['timestamp'].values[0]
    return last_date


def get_last_close(coin, count=10):
    filepath = os.path.join(data_dir, f'{coin}.csv')
    total_rows = sum(1 for _ in open(filepath))
    last_row = pd.read_csv(filepath, skiprows=range(1, total_rows-count))
    last_close = last_row['close'].values
    return last_close

for coin in coins:
    model_paths[coin] = os.path.join(model_dir, f'lstm_{coin}.h5')
    models.append(tf.keras.models.load_model(model_paths[coin]))
    
    scaler_paths[coin] = os.path.join(model_dir, f'{coin}_scaler.pkl')
    scalers.append(joblib.load(scaler_paths[coin]))
    
    data_paths[coin] = os.path.join(data_dir, f'{coin}.csv')
    
    last_times[coin] = get_last_date(coin)
    
    
# TODO: get data from DB
def get_data(coin, count=1, to=str(datetime.datetime.now()).split('.')[0][:-2]+'00'):
    return pyupbit.get_ohlcv(ticker=coin, interval='minute1', count=count, to=to)


def before_train():
    start_time = datetime.datetime.now()
    print('Before train', start_time)
    now = datetime.datetime.now()
    current_time = now.strftime('%Y-%m-%d %H:%M:00')
    now = datetime.datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    while any(last_times[coin][:-2]+'00' < current_time for coin in last_times.keys()):
        for i, coin in enumerate(coins):
            if current_time > last_times[coin]:
                diff_min = now - datetime.datetime.strptime(last_times[coin], '%Y-%m-%d %H:%M:%S')
                diff_min = diff_min.total_seconds() // 60
                this_time = get_data(coin=coin, count=int(diff_min), to=current_time).reset_index()
                this_time.to_csv(data_paths[coin], mode='a', header=not os.path.exists(data_paths[coin]), index=False)
                    
                X = this_time['close'].values.reshape(-1, 1)
                scalers[i].partial_fit(X)
                joblib.dump(scalers[i], scaler_paths[coin])
                scaled_data = scalers[i].transform(X)
                
                X = []
                y = []
                for j in range(len(scaled_data) - 1):
                    X.append(scaled_data[j:(j + 1), 0])
                    y.append(scaled_data[j + 1, 0])
                print(i, coin, len(X), len(y))
                    
                X, y = np.array(X), np.array(y)
                X = X.reshape(X.shape[0], X.shape[1], 1)
                models[i].fit(X, y, epochs=1, batch_size=1)
                models[i].save(model_paths[coin])
                
                last_times[coin] = current_time
                now = datetime.datetime.now()
                current_time = str(now).split('.')[0][:-2] + '00'
            print(f'{coin} loop', datetime.datetime.now() - start_time)
            start_time = datetime.datetime.now()


analysis = {
    'fluctuative': 'KRW-SXP',
    'up': 'KRW-SUI',
    'down': 'KRW-ARK',
    'flat': 'KRW-SHIB'
}


def analyze():
    analysis = {'std':std, ''}
    return analysis
    
    
# 1 min interval (train & analyze)
def train():
    while True:
        coin_an = {}
        for i, coin in enumerate(coins):
            with open(os.path.join(data_dir, f'{coin}.csv'), 'a') as f:
                wr = csv.writer(f)
                this_time = get_data(coin)
                wr.writerow(this_time)
            ten_closes = get_last_close(coin, 10)
            
            X = np.array(ten_closes)
            scalers[i].partial_fit(X)
            joblib.dump(scalers[i], scaler_paths[coin])
            scaled_data = scalers[i].transform(X.reshape(-1, 1))
            
            X = [scaled_data[-2]]
            y = [scaled_data[-1]]
            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            X = X.reshape(1, 1, 1)
            models[i].fit(X, y, epochs=1, batch_size=1)
            models[i].save(model_paths[coin])
            
            # TODO: analyze
            coin_an[coin] = analyze()
        
        most_fluc = max(coin_an, key=lambda x: coin_an[x]['fluctuative'])
        up = max(coin_an, key=lambda x: coin_an[x]['up'])
        down = min(coin_an, key=lambda x: coin_an[x]['up'])
        flat = min(coin_an, key=lambda x: coin_an[x]['std'])
        
        analysis['fluctuative'] = most_fluc
        analysis['up'] = up
        analysis['down'] = down
        analysis['flat'] = flat
        
        print(analysis)
        
        print('Trained and analyzed all coins')
        time.sleep(60)


def get_percentage(pred, data):
    return (pred - data) / pred * 100


req = {}  # 'close' of each coin in 


# predict
def on_message():
    percentages = {}
    response = {}
    for i, coin in enumerate(coins):
        req[coin] = get_data(coin)
        curr_price = req[coin]['close']
        
        scalers[i].transform(np.array(curr_price).reshape(-1, 1))
        X = X.reshape(1, 1, 1)
        
        pred = models[i].predict(X)
        pred = scalers[i].inverse_transform(pred)
        
        percentages[coin] = [0, get_percentage(pred, curr_price)]
    percentages = sorted(percentages, key=lambda x: x[1][1], reverse=True)
    
    for i in enumerate(percentages, 1):
        response[percentages[i][0]] = [i, percentages[i][1][1]]
    
    response['prediction_timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # TODO: attach analysis results
        
    print(percentages)
    print(response)
    
    return response

    
if __name__ == '__main__':
    before_train()
    train_thread = Thread(target=train)
    train_thread.start()
