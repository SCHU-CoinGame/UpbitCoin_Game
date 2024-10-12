import tensorflow as tf
import os
import joblib
import pandas as pd
import numpy as np
import time
from threading import Thread
import pyupbit as pu
import datetime
import csv

model_dir = 'ai_hint/models'

coin = 'KRW-SXP'
model_path = os.path.join(model_dir, f'lstm_{coin}.h5')
scaler_path = os.path.join(model_dir, f'{coin}_scaler.pkl')
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
last = '2024-10-12 18:36:00'
data_path = 'ai_hint/data/KRW-SXP_test.csv'

def train_before():
    while(True):
        now = datetime.datetime.now()[:-2]+'00'
        
        data = pu.get_ohlcv('KRW-SXP', interval='minute1', to=last)
        with open(data_path, 'a') as f:
            wr = csv.writer(f)
            wr.writerow(f)
            
        data = data['close'].values.reshape(-1, 1)
        scaler.partial_fit(data)
        data = scaler.transform(data)
        
        data = data.reshape(1, 1, 1)
        model.fit(data, epochs=1, batch_size=1)
        
        if now >= last:
            training_thread = Thread(target=train)
            training_thread.start()
            break
    
    
def train():
    data = pu.get_current_price('KRW-SXP')
    
    start_time = time.time()
    
    data = np.array(data).reshape(-1, 1)
    
    scaler.partial_fit(data)
    data = scaler.transform(data)
    
    data = data.reshape(1, 1, 1)
    model.fit(data, epochs=1, batch_size=1)
    print('training time', time.time() - start_time)
    
    on_message(data)
    time.sleep(60)


def get_percentage(pred, data):
    changed = pred - data
    percentage = changed / pred * 100
    return percentage
    
    
def on_message(data):
    start_time = time.time()
    scaler.transform(np.array(data).reshape(-1, 1))
    data = data.reshape(1, 1, 1)
    
    pred = model.predict(data)
    pred = scaler.inverse_transform(pred)
    data = scaler.inverse_transform([[data]])
    
    percentage = get_percentage(pred[0][0], data[0][0])
    
    print('predicting time', time.time() - start_time)
    print(pred, percentage)

train_before()
