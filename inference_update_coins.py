import tensorflow as tf
import os
import joblib
import pandas as pd
import numpy as np
import time
import pyupbit
import datetime
import requests
import json
import warnings
from threading import Thread

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

import config
from ai_hint.all_features_w_DA import update_coins

warnings.filterwarnings('ignore')


cfg = config.Config()

steady_coins = cfg.steady_coins
coins = cfg.coins
coin_dict = {}

tickers = pyupbit.get_tickers(fiat='KRW')
tickers = [ticker for ticker in tickers if not ticker in steady_coins]

model_dir = cfg.model_dir
data_dir = cfg.data_dir

timestep = 1

last_times = {}  # csv 파일 마지막에 빈 줄 하나 있어야 함

models = []
scalers = []
data_paths = {}
model_paths = {}
scaler_paths = {}

response = []

stop_train = 0
stop_inference = 0

second_train = 0
second_inference = 0


def inverse_transform_predictions(preds, scaler, label_idx=-1):
    dummy = np.zeros((len(preds), scaler.n_features_in_))
    dummy[:, label_idx] = preds[:, 0]
    return scaler.inverse_transform(dummy)[:, label_idx]


def get_last_date(coin):
    filepath = os.path.join(data_dir, f'{coin}.csv')
    total_rows = sum(1 for _ in open(filepath))
    last_row = pd.read_csv(filepath, skiprows=range(1, total_rows-1))
    last_date = last_row['timestamp'].values[0]
    return last_date


def get_last_row(coin, count=10):
    filepath = os.path.join(data_dir, f'{coin}.csv')
    total_rows = sum(1 for _ in open(filepath))
    last_row = pd.read_csv(filepath, skiprows=range(1, total_rows-count))
    return last_row
    
    
# TODO: get data from DB
def get_data(coin, count=1, to=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:00')):
    return pyupbit.get_ohlcv(ticker=coin, interval='minute1', count=count, to=to)


def get_percentage(future, current):
    return ((future - current) / current) * 100


def prepare():
    global last_times, models, scalers, data_paths, model_paths, scaler_paths, response, coin_dict
    
    last_times = {}

    models = []
    scalers = []
    data_paths = {}
    model_paths = {}
    scaler_paths = {}

    response = []
    coin_dict = {}

    for i, coin in enumerate(coins):
        # if os.path.exists(os.path.join(model_dir, f'lstm_{coin}.h5')):
        #     model_paths[coin] = os.path.join(model_dir, f'lstm_{coin}.h5')
        #     models.append(tf.keras.models.load_model(model_paths[coin]))
            
        #     scaler_paths[coin] = os.path.join(model_dir, f'{coin}_scaler.pkl')
        #     scalers.append(joblib.load(scaler_paths[coin]))
        # else:
        #     print('Model not found for', coin)
        
        model_paths[coin] = os.path.join(model_dir, f'lstm_{coin}.h5')
        models.append(tf.keras.models.load_model(model_paths[coin]))
        
        scaler_paths[coin] = os.path.join(model_dir, f'{coin}_scaler.pkl')
        scalers.append(joblib.load(scaler_paths[coin]))
        
        data_paths[coin] = os.path.join(data_dir, f'{coin}.csv')
            
        last_times[coin] = get_last_date(coin)
        
        response.append({'code':coin, 'rank':-1, 'prediction_timestamp':'', 'percentage':0.0, 'future':0.0, 'current':0.0,
                        'most_volatile':False, 'least_volatile':False, 'largest_drop':False, 'largest_rise':False,
                        'largest_spike':False, 'fastest_growth':False, 'fastest_decline':False, 'sell_up': 0.0, 'sell_down': 0.0})
        coin_dict[coin] = i


def before_train():
    prepare()
    # start_time = datetime.datetime.now()
    # print('Before train', start_time)
    # now = datetime.datetime.now()
    # current_time = now.strftime('%Y-%m-%d %H:%M:00')
    # now = datetime.datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
    # while any(last_times[coin][:-2]+'00' < current_time for coin in last_times.keys()):
    #     for i, coin in enumerate(coins):
    #         if current_time > last_times[coin]:
    #             diff_min = now - datetime.datetime.strptime(last_times[coin], '%Y-%m-%d %H:%M:%S')
    #             diff_min = diff_min.total_seconds() // 60
    #             df = get_data(coin=coin, count=int(diff_min), to=current_time).reset_index()
    #             df.to_csv(data_paths[coin], mode='a', header=not os.path.exists(data_paths[coin]), index=False)
                
    #             df['close_change'] = df['close'].diff().fillna(get_last_row(coin, 1)['close'] - df.iloc[-1]['close'])
                
    #             # fewer
    #             df['percentage_change'] = get_percentage(df['close'], df['close'].shift(1)).fillna(0)
    #             df['volatility'] = df['percentage_change'].rolling(window=5).std()
    #             df['avg_change_rate'] = df['percentage_change'].rolling(window=5).mean()
                
    #             X = df[cfg.used_cols].fillna(0)
    #             scalers[i].partial_fit(X)
    #             joblib.dump(scalers[i], scaler_paths[coin])
    #             scaled_data = scalers[i].transform(X)
                
    #             X = []
    #             y = []
    #             for j in range(len(scaled_data) - timestep):
    #                 X.append(scaled_data[j:(j + timestep), :])
    #                 y.append(scaled_data[j + timestep, -1])
    #             # print(i, coin, len(X), len(y))
                    
    #             if len(X) >= 1 and len(y) >= 1:
    #                 X, y = np.array(X), np.array(y)
    #                 models[i].fit(X, y, epochs=20, batch_size=1, verbose=0)
    #                 models[i].save(model_paths[coin])
                
    #             last_times[coin] = current_time
    #             now = datetime.datetime.now()
    #             current_time = str(now).split('.')[0][:-2] + '00'
    #         print(f'{coin} loop', datetime.datetime.now() - start_time)
    #         start_time = datetime.datetime.now()
            
            
def on_message(msg):
    url = cfg.url
    headers = {'Content-Type': 'application/json'}
    
    try:
        r = requests.post(url, data=msg, headers=headers)
        print(f'Status Code: {r.status_code}, Response: {r.text}')
    except requests.exceptions.RequestException as e:
        print(f'An error occurred: {e}')
    
    return response


def get_sellprice(percentage, current_price, coin_idx):
    config_tmp = config.Config()
    
    coins_w = config_tmp.coins_w
    coins_up_ratio = config_tmp.coins_up_ratio
    coins_down_ratio = config_tmp.coins_down_ratio
    
    percent = abs(percentage) * coins_w[coin_idx]
    sell_up = current_price + (current_price * percent * coins_up_ratio[coin_idx])
    sell_down = current_price - (current_price * percent * coins_down_ratio[coin_idx])
    
    return sell_up, sell_down, current_price


volatility = {}
price_change ={}
volume_change = {}
avg_change_rate = {}


def analyze_and_predict():
    
    percentages = {}

    while True:
        if stop_inference:
            break
        
        
        for i, coin in enumerate(coins):
            
            # TODO: analyze
            
            recent_rows = get_last_row(coin, cfg.rows+1)
            recent_closes = recent_rows['close'].values
            
            percentage_changes = [get_percentage(recent_closes[i+1], recent_closes[i]) for i in range(len(recent_closes) - 1)]
            volatility[coin] = np.std(percentage_changes)  # 변동성
            price_change[coin] = get_percentage(recent_closes[-1], recent_closes[0])  # 가격 변화율
            volume_change[coin] = get_percentage(recent_rows['volume'].iloc[-1], recent_rows['volume'].iloc[0])  # 거래량 변화
            avg_change_rate[coin] = sum(percentage_changes) / len(percentage_changes)  # 평균 변화율
            
            # TODO: predict
            
            recent_rows = recent_rows.iloc[-1]
            # recent_rows['close_change'] = recent_closes[-1] - recent_closes[-2]
            one_recent = pyupbit.get_current_price(coins[i])
            recent_rows['close_change'] = one_recent - recent_closes[-2]
            recent_rows['percentage_change'] = percentage_changes[-1]
            recent_rows['volatility'] = volatility[coin]
            recent_rows['avg_change_rate'] = avg_change_rate[coin]
            
            X = recent_rows[cfg.used_cols].fillna(0)
            X = scalers[i].transform(X.values.reshape(1, -1))
            X = tf.convert_to_tensor(X.reshape(X.shape[0], 1, X.shape[1]), dtype=tf.float32)
            
            pred = models[i].predict(X)
            pred = inverse_transform_predictions(pred, scalers[i])
            
            response[coin_dict[coin]]['future'] = pred[0]
            
            # percentages[coin] = get_percentage(pred[0], recent_closes[-1])
            percentages[coin] = get_percentage(pred[0], one_recent)
            
            # response[coin_dict[coin]]['sell_up'], response[coin_dict[coin]]['sell_down'] = get_sellprice(percentages[coin] / 100, recent_closes[-1], i)
            sell_up, sell_down, current = get_sellprice(percentages[coin] / 100, one_recent, i)
            response[coin_dict[coin]]['sell_up'], response[coin_dict[coin]]['sell_down'] = (sell_up, sell_down) if sell_up > sell_down else (sell_down, sell_up)
            response[coin_dict[coin]]['current'] = current
            
        sorted_percentages = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        
        pred_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for i, (coin, percentage) in enumerate(sorted_percentages):
            response[coin_dict[coin]]['percentage'] = percentage
            response[coin_dict[coin]]['rank'] = i + 1
            response[coin_dict[coin]]['prediction_timestamp'] = pred_time
        
        response[coin_dict[max(volatility, key=volatility.get)]]['most_volatile'] = True
        response[coin_dict[min(volatility, key=volatility.get)]]['least_volatile'] = True
        response[coin_dict[min(price_change, key=price_change.get)]]['largest_drop'] = True
        response[coin_dict[max(price_change, key=price_change.get)]]['largest_rise'] = True
        response[coin_dict[max(volume_change, key=volume_change.get)]]['largest_spike'] = True
        response[coin_dict[max(avg_change_rate, key=avg_change_rate.get)]]['fastest_growth'] = True
        response[coin_dict[min(avg_change_rate, key=avg_change_rate.get)]]['fastest_decline'] = True
        
        # TODO: send
        
        message = json.dumps(response)
        on_message(message)
        
        print()
        print(f'{"Code":<15} {"Future":<20} {"Current":<15} {"Percentage":<15} {"Rank":<5} {"Sell U":<20} {"Sell L":<20} {"Tags"}')
        print('-' * 110)
        for r in response:
            code = r['code']
            future = r['future']
            current = r['current']
            percentage = r['percentage']
            rank = r['rank']
            sell_up = r['sell_up']
            sell_down = r['sell_down']
            
            tags = [tag for tag in r if tag not in ['code', 'future', 'current', 'percentage', 'rank', 'prediction_timestamp', 'sell_up', 'sell_down'] and r[tag]]
            tags_str = ' '.join(tags)
    
            print(f'{code:<15} {future:<20.8f} {current:<15.3f} {percentage:<15.8f} {rank:<5} {sell_up:<20.8f} {sell_down:<20.8f} {tags_str}')
        print()
        
        print('Trained and analyzed all coins', datetime.datetime.now())
        time.sleep(cfg.predict_seconds)
        
        response[coin_dict[max(volatility, key=volatility.get)]]['most_volatile'] = False
        response[coin_dict[min(volatility, key=volatility.get)]]['least_volatile'] = False
        response[coin_dict[min(price_change, key=price_change.get)]]['largest_drop'] = False
        response[coin_dict[max(price_change, key=price_change.get)]]['largest_rise'] = False
        response[coin_dict[max(volume_change, key=volume_change.get)]]['largest_spike'] = False
        response[coin_dict[max(avg_change_rate, key=avg_change_rate.get)]]['fastest_growth'] = False
        response[coin_dict[min(avg_change_rate, key=avg_change_rate.get)]]['fastest_decline'] = False


def train_thread():
    while True:
        Thread(target=train, daemon=True).start()
        if stop_train:
            break
        time.sleep(cfg.train_seconds)


def train():
    train_start_time = datetime.datetime.now()
    print('Train started', train_start_time)
    for i, coin in enumerate(coins):
            
        last_time_dt = datetime.datetime.strptime(last_times[coin], '%Y-%m-%d %H:%M:00')
        now = datetime.datetime.now()
        # if last_time_dt >= now:
        #     continue
        
        diff_min = now - last_time_dt
        diff_min = diff_min.total_seconds() // 60
        
        now = now.strftime('%Y-%m-%d %H:%M:00')
        this_time = get_data(coin=coin, count=int(diff_min), to=now).reset_index()
        
        last_times[coin] = now
        
        # TODO: train
        
        if len(this_time) == 0:
            continue
        
        recent_rows = get_last_row(coin, cfg.rows)
        this_time.to_csv(data_paths[coin], mode='a', header=not os.path.exists(data_paths[coin]), index=False)

        combined_rows = pd.concat([recent_rows, this_time], ignore_index=True)
        this_time['close_change'] = this_time['close'].diff()[-len(this_time):]
        combined_rows['percentage_change'] = get_percentage(combined_rows['close'], combined_rows['close'].shift(1))
        combined_rows['volatility'] = combined_rows['percentage_change'].rolling(window=5).std()
        combined_rows['avg_change_rate'] = combined_rows['percentage_change'].rolling(window=5).mean()
        this_time[['percentage_change', 'volatility', 'avg_change_rate']] = combined_rows[
            ['percentage_change', 'volatility', 'avg_change_rate']].iloc[-len(this_time):].reset_index(drop=True)
        
        X = this_time[cfg.used_cols].fillna(0)
        scalers[i].partial_fit(X)
        joblib.dump(scalers[i], scaler_paths[coin])
        scaled_data = scalers[i].transform(X)
        
        X, y = [], []
        
        for idx in range(len(scaled_data) - timestep):
            X.append(scaled_data[idx:idx+timestep, :])
            y.append(scaled_data[idx + timestep, -1])
        
        X, y = np.array(X), np.array(y)
        if len(X) == 0:
            continue
        # print(X.shape, y.shape)
        
        models[i].fit(X, y, epochs=20, batch_size=1, verbose=0)
        models[i].save(model_paths[coin])
        
        print(f'{coin} trained', last_times[coin])
    print('Trained all coins', 'Took', datetime.datetime.now() - train_start_time)


def get_volumes():
    volumes = {}
    all_coins = tickers
    while True:
        failed = []
        for ticker in all_coins:
            try:
                querystring = {"markets": ticker}
                volume = requests.request("GET", cfg.upbit, params=querystring)
                volume = volume.json()
                volume = float(volume[0]['acc_trade_price_24h'])
                
                if volume >= cfg.min_volume:
                    volumes[ticker] = volume
            except Exception as e:
                failed.append(ticker)
                continue
        if len(failed) == 0:
            break
        all_coins = failed
            
    volumes = dict(sorted(volumes.items(), key=lambda x: x[1], reverse=False))
    volumes = list(volumes.keys())[:9 - len(steady_coins)]
    
    return volumes


def update_coins_thread():
    global stop_train, stop_inference, train_th, predict_th
    while True:
        time.sleep(cfg.update_seconds)
        
        # if datetime.datetime.now().hour != 4:
        #     continue
        try:
            print('Coin update started', datetime.datetime.now())
            
            stop_train = 1
            train_th.join()
            
            volumes = get_volumes()
            volumes = dict(sorted(volumes.items(), key=lambda x: x[1], reverse=False))
            volumes = list(volumes.keys())[:9 - len(steady_coins)]
            
            for coin in volumes:
                try:
                    update_coins.train(coin)
                except Exception as e:
                    print(f"Error updating coin {coin}: {e}")
                    continue
            
            updated_coins = steady_coins + list(volumes)
            
            global coins
            coins = updated_coins
            
            print('New coin list', coins)
            
            stop_inference = 1
            time.sleep(1)

            train_th = Thread(target=train_thread) 
            stop_train = 0
            train_th.start()
            
            predict_th.join(timeout=30)
            predict_th = Thread(target=analyze_and_predict)
            stop_inference = 0
            predict_th.start()
            
            print('Coin update finished', datetime.datetime.now())
        except Exception as e:
            print(f"Error in update_coins_thread: {e}")
            
            stop_train = 0
            stop_inference = 0
            
            if not train_th.is_alive():
                train_th = Thread(target=train_thread)
                train_th.start()
            if not predict_th.is_alive():
                predict_th = Thread(target=analyze_and_predict)
                predict_th.start()


Thread(target=update_coins_thread).start()
before_train()

train_th = Thread(target=train_thread)
train_th.start()

predict_th = Thread(target=analyze_and_predict)
predict_th.start()
