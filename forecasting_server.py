import torch
import json
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from sets.prepare_sets import prepare_indicator_sets

epsilon = 1e-10
forecasting_model = torch.load('prediction_configs/model.pth', weights_only=False)
forecasting_model.eval()

config = json.load(open('prediction_configs/eurusd_m5_ma14_ws240_f6_set1.json', 'r'))
min_X_values = np.array(config['min_X_values'])
max_X_values = np.array(config['max_X_values'])
min_y_values = np.array(config['min_y_values'])
max_y_values = np.array(config['max_y_values'])

normalize_min = float(config['normalize_min'])
normalize_max = float(config['normalize_max'])


app = Flask(__name__)

def process_request(values):
    values = np.array(values)
    
    values = (values - min_X_values) / (max_X_values - min_X_values) * (normalize_max - normalize_min) + normalize_min
    values = np.expand_dims(values, axis=0)
    values = torch.tensor(values, dtype=torch.float32).to('cpu' if not torch.cuda.is_available() else 'cuda')
    prediction = forecasting_model(values).detach().cpu().numpy()

    original_values = (prediction - normalize_min) / (normalize_max - normalize_min)
    original_values = original_values * (max_y_values - min_y_values) + min_y_values

    return original_values.flatten()

def reconstruct_ma_from_log_returns(log_returns, last_ma, epsilon=1e-10):
    """
    Ricostruisce i valori successivi di 'ma' a partire da log_returns e l'ultimo valore di ma noto.

    :param log_returns: Array con i log_returns successivi
    :param last_ma: Ultimo valore noto di 'ma'
    :param epsilon: Valore piccolo per evitare divisioni per zero
    :return: Lista dei valori ricostruiti di 'ma'
    """
    reconstructed_ma = []
    current_ma = last_ma

    for log_return in log_returns:
        current_ma = np.exp(log_return) * (current_ma + epsilon) - epsilon
        reconstructed_ma.append(float(current_ma))

    return reconstructed_ma

@app.route('/forecast', methods=['POST'])
def process_arrays():
    try:
        data = request.get_json()

        if not data or 'high' not in data or 'low' not in data:
            return jsonify({'success': False, 'error': 'Invalid input, expected JSON with high and low', 'values': []}), 400
        
        if 'ma' not in data:
            return jsonify({'success': False, 'error': 'Invalid input, expected JSON with ma', 'values': []}), 400

        high = [float(x) for x in data['high']]
        low = [float(x) for x in data['low']]
        ma = int(data['ma'])
        ws = int(data['ws'])
        set_name = data['set_name']

        if len(high) != len(low):
            return jsonify({'success': False, 'error': 'Arrays must have the same length', 'values': []}), 400

        dataset = pd.DataFrame({'high': high, 'low': low})

        dataset['hl_avg'] = (dataset['high'] + dataset['low']) / 2
        dataset['ma'] = dataset['hl_avg'].rolling(window=ma).mean()
        dataset, columns, indicator_len = prepare_indicator_sets(dataset, set_name)

        del dataset['high']
        del dataset['low']
        del dataset['hl_avg']

        dataset['log_returns'] = np.log((dataset['ma'] + epsilon) / (dataset['ma'].shift(1) + epsilon))
        last_ma = dataset['ma'].iloc[-1]
        del dataset['ma']

        dataset = dataset[ma + indicator_len:]


        values = dataset[columns + ['log_returns']].values

        if values.shape[0] != ws:
            return jsonify({'success': False, 'error': 'Not enough data', 'values': []}), 400

        print(f"Returning {values.shape[0]} values")

        result = process_request(values)
        reconstructed_ma = reconstruct_ma_from_log_returns(result, last_ma)
        print(f"Reconstructed MA: {reconstructed_ma}")

        # Restituisci il risultato come array JSON
        return jsonify({'success': True, 'values': reconstructed_ma, 'error': None}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'values': []}), 500

if __name__ == '__main__':

    app.run(debug=True)