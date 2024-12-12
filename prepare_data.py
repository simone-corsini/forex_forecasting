import pandas as pd
import numpy as np
import zipfile
import h5py
import argparse
import os
import json


from rich.progress import Progress
from rich import print
from pathlib import Path

epsilon = 1e-10


def default_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

def get_data(X_values, y_values, window_size):
    X, y = [], []
    len_values = len(X_values)
    for i in range(window_size, len_values):
        X.append(X_values[i-window_size:i])
        y.append(y_values[i])

    return np.asarray(X), np.asarray(y)

def add_to_dataset(output_file, X, y):
    with h5py.File(output_file, 'a') as f:
        if "X" in f:
            dset_X = f['X']
            original_size_X = dset_X.shape[0]
            new_size_X = original_size_X + X.shape[0]
            dset_X.resize((new_size_X,) + dset_X.shape[1:])
            dset_X[original_size_X:new_size_X, :] = X.astype(np.float32)
        else:
            f.create_dataset('X', data=X, maxshape=(None,) + X.shape[1:], compression='gzip', compression_opts=9)

        if "y" in f:
            dset_y = f['y']
            original_size_y = dset_y.shape[0]
            new_size_y = original_size_y + y.shape[0]
            dset_y.resize((new_size_y,))
            dset_y[original_size_y:new_size_y]  = y.astype(np.float32)
        else:
            f.create_dataset('y', data=y, maxshape=(None,), compression='gzip', compression_opts=9)

        return f['X'].shape[0], f['y'].shape[0]
    
def split_and_normalize_data(output_file, output_scaler_file, X_all, y_all, train_size):
    split_point = int(len(X_all) * train_size)
    X_train = X_all[:split_point]
    y_train = y_all[:split_point]
    X_val = X_all[split_point:]
    y_val = y_all[split_point:]

    normalize_max = 0.95
    normalize_min = 0.05
    min_X_values = np.min(X_train, axis=(0, 1)).astype(np.float32)
    max_X_values = np.max(X_train, axis=(0, 1)).astype(np.float32)
    min_y_values = np.min(y_train, axis=(0)).astype(np.float32)
    max_y_values = np.max(y_train, axis=(0)).astype(np.float32)

    print(f"[green]Min X values: {min_X_values}")
    print(f"[green]Max X values: {max_X_values}")
    print(f"[green]Min y values: {min_y_values}")
    print(f"[green]Max y values: {max_y_values}")
    
    data = {
        'min_X_values': min_X_values,
        'max_X_values': max_X_values,
        'min_y_values': min_y_values,
        'max_y_values': max_y_values,
        'normalize_min': normalize_min,
        'normalize_max': normalize_max
    }

    with open(output_scaler_file, 'w') as file_scaler:
        json.dump(data, file_scaler, indent=4, default=default_serializer)

    X_train = (X_train - min_X_values) / (max_X_values - min_X_values) * (normalize_max - normalize_min) + normalize_min
    y_train = (y_train - min_y_values) / (max_y_values - min_y_values) * (normalize_max - normalize_min) + normalize_min
    X_val = (X_val - min_X_values) / (max_X_values - min_X_values) * (normalize_max - normalize_min) + normalize_min
    y_val = (y_val - min_y_values) / (max_y_values - min_y_values) * (normalize_max - normalize_min) + normalize_min

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)

    with h5py.File(output_file, 'w') as f:
        # Crea i dataset
        f.create_dataset('X_train', data=X_train, dtype='float32', compression='gzip', compression_opts=9)
        f.create_dataset('y_train', data=y_train, dtype='float32', compression='gzip', compression_opts=9)
        f.create_dataset('X_val', data=X_val, dtype='float32', compression='gzip', compression_opts=9)
        f.create_dataset('y_val', data=y_val, dtype='float32', compression='gzip', compression_opts=9)

        f['X_train'].attrs['min_X'] = min_X_values
        f['X_train'].attrs['max_X'] = max_X_values
        f['X_train'].attrs['min_y'] = min_y_values
        f['X_train'].attrs['max_y'] = max_y_values
        f['X_train'].attrs['normalize_min'] = normalize_min
        f['X_train'].attrs['normalize_max'] = normalize_max

        print(f"[green]X Train size: {f["X_train"].shape}")
        print(f"[green]y Train size: {f["y_train"].shape}")
        print(f"[green]X Val size: {f["X_val"].shape}")
        print(f"[green]y Val size: {f["y_val"].shape}")

def prepare_indicator_set1(dataset):
    ema_slow_period = 9
    ema_fast_period = 21
    bollinger_period = 20
    bolinger_deviation = 2

    columns = []
    dataset['ema_slow'] = dataset['hl_avg'].ewm(span=ema_slow_period, adjust=False).mean()
    dataset['ema_fast'] = dataset['hl_avg'].ewm(span=ema_fast_period, adjust=False).mean()
    dataset['ema_fast_slow'] = dataset['ema_fast'] - dataset['ema_slow']
    dataset['ema_fast_slow_diff_rel'] = dataset['ema_fast_slow'].diff()
    del dataset['ema_slow']
    del dataset['ema_fast']
    del dataset['ema_fast_slow']
    columns += ['ema_fast_slow_diff_rel']

    dataset['hl_diff'] = dataset['high'] - dataset['low']
    dataset['hl_diff_diff_rel'] = dataset['hl_diff'].diff()
    del dataset['hl_diff']
    columns += ['hl_diff_diff_rel']

    dataset['bollinger_ma'] = dataset['ma'].rolling(window=bollinger_period).mean()
    dataset['bollinger_std'] = dataset['ma'].rolling(window=bollinger_period).std()
    dataset['bollinger_upper'] = dataset['bollinger_ma'] + bolinger_deviation * dataset['bollinger_std']
    dataset['bollinger_lower'] = dataset['bollinger_ma'] - bolinger_deviation * dataset['bollinger_std']
    dataset['bollinger_width'] = dataset['bollinger_upper'] - dataset['bollinger_lower']
    dataset['deviation_upper'] = (dataset['ma'] - dataset['bollinger_upper']) / dataset['bollinger_width']
    dataset['deviation_lower'] = (dataset['ma'] - dataset['bollinger_lower']) / dataset['bollinger_width']
    dataset['bollinger_width_diff_rel'] = dataset['bollinger_width'].diff()
    dataset['deviation_upper_diff_rel'] = dataset['deviation_upper'].diff()
    dataset['deviation_lower_diff_rel'] = dataset['deviation_lower'].diff()
    del dataset['bollinger_ma']
    del dataset['bollinger_std']
    del dataset['bollinger_upper']
    del dataset['bollinger_lower']
    del dataset['bollinger_width']
    del dataset['deviation_upper']
    del dataset['deviation_lower']
    columns += ['bollinger_width_diff_rel', 'deviation_upper_diff_rel', 'deviation_lower_diff_rel']

    # del dataset['bolinger_ma']
    # del dataset['bolinger_std']
    # del dataset['bolinger_upper']
    # del dataset['bolinger_lower']

    # dataset['log_hl_diff'] = np.log((dataset['hl_diff'] + epsilon) / (dataset['hl_diff'].shift(1) + epsilon))
    # dataset['log_ema_fast_slow'] = np.log((dataset['ema_fast_slow'] + epsilon) / (dataset['ema_fast_slow'].shift(1) + epsilon))
    # dataset['log_bu_distance'] = np.log((dataset['bu_distance'] + epsilon) / (dataset['bu_distance'].shift(1) + epsilon))
    # dataset['log_bl_distance'] = np.log((dataset['bl_distance'] + epsilon) / (dataset['bl_distance'].shift(1) + epsilon))
    # del dataset['hl_diff']
    # del dataset['ema_fast_slow']
    # del dataset['bu_distance']
    # del dataset['bl_distance']

    return dataset, columns, max(ema_slow_period, ema_fast_period, bollinger_period)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processa un file di input per generare un file HDF5.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Il percorso del file di input"
    )
    parser.add_argument('--output_folder', type=str, help="Cartella di output")
    parser.add_argument('--ma_periods', type=int, default=14, help="Numero di periodi per la media mobile dei Log Returns")
    parser.add_argument('--window_size', type=int, default=256, help="Dimensione della finestra di osservazione")
    args = parser.parse_args()
    
    input_file = args.input_file
    output_folder = args.output_folder
    if output_folder is None:
        output_folder = Path(input_file).parent

    output_filename = Path(input_file).stem

    output_file = output_folder / f"{output_filename}_ma{args.ma_periods}_ws{args.window_size}_set1.h5"
    output_scaler_file = output_folder / f"{output_filename}_ma{args.ma_periods}_ws{args.window_size}_set1.json"

    if os.path.exists(output_file):
        os.remove(output_file)

    if os.path.exists(output_scaler_file):
        os.remove(output_scaler_file)

    x_shape, y_shape = 0, 0

    X_all, y_all = None, None

    with Progress() as progress:
        with zipfile.ZipFile(args.input_file, 'r') as origin:
            file_list = origin.namelist()

            np.random.shuffle(file_list)

            task = progress.add_task("[green]Processing...", total=len(file_list))
            for file in file_list:
                with origin.open(file) as f:
                    progress.update(task, description=f"[green]Processing {file} - X: {x_shape}, y: {y_shape}")
                    dataset = pd.read_csv(f, usecols=['timestamp', 'high', 'low'], index_col=['timestamp'], parse_dates=['timestamp'])

                    dataset = dataset[dataset.index < '2024-01-01']
                    #dataset = dataset[dataset.index >= '2023-01-01']
                    dataset = dataset[dataset.index >= '2009-05-01']

                    dataset['hl_avg'] = dataset['high'] + dataset['low'] / 2
                    dataset['ma'] = dataset['hl_avg'].rolling(window=args.ma_periods).mean()
                    dataset, columns, indicator_len = prepare_indicator_set1(dataset)

                    del dataset['high']
                    del dataset['low']
                    del dataset['hl_avg']

                    dataset['log_returns'] = np.log((dataset['ma'] + epsilon) / (dataset['ma'].shift(1) + epsilon))
                    del dataset['ma']

                    dataset = dataset[args.ma_periods + indicator_len:]

                    if dataset.shape[0] <= args.window_size:
                        progress.update(task, advance=1)
                        continue

                    X, y = get_data(dataset[columns + ['log_returns']].values, dataset['log_returns'].values, args.window_size)

                    if X_all is None:
                        X_all = X
                        y_all = y
                    else:
                        X_all = np.append(X_all, X, axis=0)
                        y_all = np.append(y_all, y, axis=0)

                    x_shape = X_all.shape[0]
                    y_shape = y_all.shape[0]

                    # x_shape, y_shape = add_to_dataset(output_file, X, y)
                    progress.update(task, advance=1)
            
            progress.update(task, completed=True)
            progress.remove_task(task)

        if (np.isnan(X_all).any() or np.isnan(y_all).any()):
            print("[red]Nan values found")
            exit(1)
        

        task = progress.add_task("[green]Split train/val and normailze[/green]", total=None)
        split_and_normalize_data(output_file, output_scaler_file, X_all, y_all, 0.8)
        progress.update(task, completed=True)
        progress.remove_task(task)

