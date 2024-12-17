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

from sets.prepare_sets import prepare_indicator_sets

epsilon = 1e-10

def default_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

def get_data(X_values, y_values, window_size, future_size):
    X, y = [], []
    len_values = len(X_values)
    for i in range(window_size, len_values - future_size + 1):
        X.append(X_values[i-window_size:i])
        y.append(y_values[i:i+future_size] if future_size > 1 else y_values[i])

    out_X = np.asarray(X)
    out_y = np.expand_dims(np.asarray(y), axis=-1)

    assert out_X.shape[1:] == (window_size, len(X_values[0]))
    assert out_y.shape[1:] == (future_size, 1)

    return out_X, out_y

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
            dset_y.resize((new_size_y,) + y.shape[1:])
            dset_y[original_size_y:new_size_y, ...] = y.astype(np.float32)
        else:
            f.create_dataset('y', data=y, maxshape=(None,) + (y.shape[1:] if len(y.shape) > 1 else ()), compression='gzip', compression_opts=9)

        return f['X'].shape[0], f['y'].shape[0]

def split_and_normalize_data(output_file, output_scaler_file, train_size, fearures):
    with h5py.File(output_file, 'a') as f:
        X_all = f['X'][:]
        y_all = f['y'][:]
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
        print(f"[green]Features: {fearures}")

        data = {
            'min_X_values': min_X_values,
            'max_X_values': max_X_values,
            'min_y_values': min_y_values,
            'max_y_values': max_y_values,
            'normalize_min': normalize_min,
            'normalize_max': normalize_max,
            'features': fearures
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
        f['X_train'].attrs['features'] = fearures

        del f['X']
        del f['y']

        print(f"[green]X Train size: {f['X_train'].shape}")
        print(f"[green]y Train size: {f['y_train'].shape}")
        print(f"[green]X Val size: {f['X_val'].shape}")
        print(f"[green]y Val size: {f['y_val'].shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processa un file di input per generare un file HDF5.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Il percorso del file di input"
    )
    parser.add_argument('--set_name', type=str, help="Nome del set di features da utilizzare", default='set1')
    parser.add_argument('--output_folder', type=str, help="Cartella di output")
    parser.add_argument('--ma_periods', type=int, default=14, help="Numero di periodi per la media mobile dei Log Returns")
    parser.add_argument('--window_size', type=int, default=256, help="Dimensione della finestra di osservazione")
    parser.add_argument('--future_size', type=int, default=1, help="Dimensione della finestra di osservazione futura")
    args = parser.parse_args()

    input_file = args.input_file
    output_folder = args.output_folder
    if output_folder is None:
        output_folder = Path(input_file).parent

    output_filename = Path(input_file).stem

    output_file = output_folder / f"{output_filename}_ma{args.ma_periods}_ws{args.window_size}_f{args.future_size}_{args.set_name}.h5"
    output_scaler_file = output_folder / f"{output_filename}_ma{args.ma_periods}_ws{args.window_size}_f{args.future_size}_{args.set_name}.json"

    if os.path.exists(output_file):
        os.remove(output_file)

    if os.path.exists(output_scaler_file):
        os.remove(output_scaler_file)

    x_shape, y_shape = 0, 0

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
                    dataset, columns, indicator_len = prepare_indicator_sets(dataset, args.set_name)

                    del dataset['high']
                    del dataset['low']
                    del dataset['hl_avg']

                    dataset['log_returns'] = np.log((dataset['ma'] + epsilon) / (dataset['ma'].shift(1) + epsilon))
                    del dataset['ma']

                    dataset = dataset[args.ma_periods + indicator_len:]

                    if dataset.shape[0] <= args.window_size + args.future_size:
                        progress.update(task, advance=1)
                        continue

                    X, y = get_data(dataset[columns + ['log_returns']].values, dataset['log_returns'].values, args.window_size, args.future_size)

                    x_shape, y_shape = add_to_dataset(output_file, X, y)
                    progress.update(task, advance=1)

            progress.update(task, completed=True)
            progress.remove_task(task)

        task = progress.add_task("[green]Split train/val and normalize[/green]", total=None)
        split_and_normalize_data(output_file, output_scaler_file, 0.8, columns + ['log_returns'])
        progress.update(task, completed=True)
        progress.remove_task(task)
