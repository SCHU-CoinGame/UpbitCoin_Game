import tensorboard as tf
import os
import joblib
import pandas as pd
import numpy as np
import jsonify
from flask import Flask, request, jsonify
import time
from flask_cors import CORS
from threading import Thread

app = Flask(__name__)
CORS(app)

# access_key = os.getenv('AWS_ACCESS_KEY_ID')
# secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
# s3 = boto3.client('s3', region_name='ap-northeast-2')
# bucket_name = 'aws-s3-bucket-fastcampus'
# prefix = 'dataLake_upbit/'
# response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
# parquet_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]

model_dir = 'ai_hint/models'
coins = ['KRW-BTC', 'KRW-SXP', 'KRW-SUI', 'KRW-ARK', 'KRW-SHIB', 'KRW-UXLINK', 'KRW-XRP', 'KRW-SEI', 'KRW-HIFI']
models = []
scalers = []

for coin in coins:
    model_path = os.path.join(model_dir, f'lstm_{coin}.h5')
    scaler_path = os.path.join(model_dir, f'{coin}_scaler.pkl')
    models.append(tf.keras.models.load_model(model_path))
    scalers.append(joblib.load(scaler_path))


# TODO: get data from DB
def get_data():
    for coin in coins:
        # obj = s3.get_object(Bucket=bucket_name, Key=prefix+coin+'.parquet')
        # df = pq.read_table(io.BytesIO(obj['Body'].read())).to_pandas()
        return data
    
    
def train():
    while True:
        data = get_data()
        
        # TODO: preprocess data
        
        for i, coin in enumerate(coins):
            model_path = os.path.join(model_dir, f'lstm_{coin}.h5')
            scaler_path = os.path.join(model_dir, f'{coin}_scaler.pkl')
            
            data = np.array(data).reshape(-1, 1)
            
            scalers[i].partial_fit(data)
            joblib.dump(scalers[i], scaler_path)
            data = scalers[i].transform(data)
            
            data = data.reshape(1, 1, 1)
            models[i].fit(data, epochs=1, batch_size=1)
            
            models[i].save(model_path)
            
            time.sleep(60)


def get_percentage(pred, data):
    changed = pred - data
    percentage = changed / pred * 100
    return percentage
    
    
@app.route('/predict', methods=['POST'])
def on_message():
    percentages = {}
    content = request.json
    for i, coin in enumerate(coins):
        curr_price = content[coin]['close']
        
        scalers[i].transform(np.array(curr_price).reshape(-1, 1))
        data = data.reshape(1, 1, 1)
        
        pred = models[i].predict(data)
        pred = scalers[i].inverse_transform(pred)
        
        percentages.get(coin, get_percentage(pred, curr_price))
    sorted(percentages.items(), key=lambda x: x[1], reverse=True)
    
    return jsonify(percentages)
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    training_thread = Thread(target=train)
    