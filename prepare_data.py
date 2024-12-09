import pandas as pd
import numpy as np
import zipfile
import h5py
import argparse
import os


from rich.progress import Progress
from rich import print
from pathlib import Path

def get_data(values, window_size):
    X, y = [], []
    len_values = len(values)
    for i in range(window_size, len_values):
        X.append(values[i-window_size:i])
        y.append(values[i])
    X, y = np.asarray(X), np.asarray(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y

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
    
def split_and_normalize_data(output_file, train_size):
    with h5py.File(output_file, 'a') as f:
        X = f['X'][:]
        y = f['y'][:]
        
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        # Determina gli split
        split_index = int(len(X) * train_size)
        train_indices = indices[:split_index]
        val_indices = indices[split_index:]

        normalize_max = 0.95
        normalize_min = 0.05
        min_values = np.min(X[train_indices], axis=(0, 1)).astype(np.float32)
        max_values = np.max(X[train_indices], axis=(0, 1)).astype(np.float32)

        print(f"[green]Min values: {min_values}")
        print(f"[green]Max values: {max_values}")

        # Normalizza i dati
        X = (X - min_values) / (max_values - min_values) * (normalize_max - normalize_min) + normalize_min
        y = (y - min_values) / (max_values - min_values) * (normalize_max - normalize_min) + normalize_min

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Crea i dataset
        f.create_dataset('X_train', data=X[train_indices], dtype='float32', compression='gzip', compression_opts=9)
        f.create_dataset('y_train', data=y[train_indices], dtype='float32', compression='gzip', compression_opts=9)
        f.create_dataset('X_val', data=X[val_indices], dtype='float32', compression='gzip', compression_opts=9)
        f.create_dataset('y_val', data=y[val_indices], dtype='float32', compression='gzip', compression_opts=9)

        f['X_train'].attrs['min'] = min_values
        f['X_train'].attrs['max'] = max_values
        f['X_train'].attrs['normalize_min'] = normalize_min
        f['X_train'].attrs['normalize_max'] = normalize_max

        # Cancella i dataset originali
        del f['X']
        del f['y']

        print(f"[green]X Train size: {f["X_train"].shape}")
        print(f"[green]y Train size: {f["y_train"].shape}")
        print(f"[green]X Val size: {f["X_val"].shape}")
        print(f"[green]y Val size: {f["y_val"].shape}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processa un file di input per generare un file HDF5.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Il percorso del file di input"
    )
    parser.add_argument('--output_folder', type=str, help="Cartella di output")
    parser.add_argument('--ma_periods', type=int, default=14, help="Numero di periodi per la media mobile")
    parser.add_argument('--window_size', type=int, default=256, help="Dimensione della finestra di osservazione")
    args = parser.parse_args()
    
    input_file = args.input_file
    output_folder = args.output_folder
    if output_folder is None:
        output_folder = Path(input_file).parent

    output_filename = Path(input_file).stem

    output_file = output_folder / f"{output_filename}_ma{args.ma_periods}_ws{args.window_size}.h5"

    if os.path.exists(output_file):
        os.remove(output_file)

    x_shape, y_shape = 0, 0

    with Progress() as progress:
        with zipfile.ZipFile(args.input_file, 'r') as origin:
            file_list = origin.namelist()

            task = progress.add_task("[green]Processing...", total=len(file_list))
            for file in file_list:
                with origin.open(file) as f:
                    progress.update(task, description=f"[green]Processing {file} - X: {x_shape}, y: {y_shape}")
                    dataset = pd.read_csv(f, usecols=['timestamp', 'high', 'low'], index_col=['timestamp'], parse_dates=['timestamp'])

                    dataset = dataset[dataset.index < '2024-01-01']
                    dataset = dataset[dataset.index >= '2009-05-01']

                    dataset['hl_avg'] = dataset['high'] + dataset['low'] / 2
                    del dataset['high']
                    del dataset['low']

                    dataset['ma'] = dataset['hl_avg'].rolling(window=args.ma_periods).mean()
                    dataset['log_returns'] = np.log(dataset['ma'] / dataset['ma'].shift(1))

                    dataset = dataset[args.ma_periods:]

                    if dataset.shape[0] <= args.window_size:
                        progress.update(task, advance=1)
                        continue

                    X, y = get_data(dataset['log_returns'].values, args.window_size)

                    x_shape, y_shape = add_to_dataset(output_file, X, y)
                    progress.update(task, advance=1)
            
            progress.update(task, completed=True)
            progress.remove_task(task)

        task = progress.add_task("[green]Split train/val and normailze[/green]", total=None)
        split_and_normalize_data(output_file, 0.8)
        progress.update(task, completed=True)
        progress.remove_task(task)

