import h5py
import argparse
import os
import json
import numpy as np

from pathlib import Path

def default_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processa un file di input per generare un file HDF5.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Il percorso del file di input"
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_folder = Path(input_file).parent

    output_filename = Path(input_file).stem

    output_file = output_folder / f"{output_filename}.json"

    if os.path.exists(output_file):
        os.remove(output_file)

    with h5py.File(input_file, 'r') as f:
        min_values = f['X_train'].attrs['min']
        max_values = f['X_train'].attrs['max']
        normalize_min = f['X_train'].attrs['normalize_min']
        normalize_max = f['X_train'].attrs['normalize_max']

    data = {
        'min_values': min_values,
        'max_values': max_values,
        'normalize_min': normalize_min,
        'normalize_max': normalize_max
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, default=default_serializer)