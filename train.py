import argparse
import importlib
import os
import json


import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score, ExplainedVariance
from rich import print
from pathlib import Path

from core.dataset import HDF5Dataset
from core.trainer import Trainer

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser(description='Addestramento di un classificatore con PyTorch')
    parser.add_argument('--config', type=str, help='Configurazioni dei modelli', required=True)

    parser.add_argument('--model_path', type=str, default='models/', help='Percorso per il checkpoint del modello')
    parser.add_argument('--log_dir', type=str, default='runs/', help='Directory per TensorBoard logs')
    
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='Device to use')    
    parser.add_argument('--continue_training', help='Continue training', action='store_true')
    parser.add_argument('--reset_optimizer', help='Reset Optimizer', action='store_true')
    parser.add_argument('--reset_early_stopping', help='Reset Early Stopping', action='store_true')

    parser.add_argument('--verbose', help='Verbose', action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if not os.path.exists(args.config):
        print(f"[bold red]Configuration '{args.config}' not found[/bold red]")
        return
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"[bold green]Using device:[/bold green] {device}")
    print("")

    with open(args.config, 'r') as config_file:
        datasets = json.load(config_file)

    for dataset in datasets:
        dataset_path = dataset["dataset"]
        n_classes = dataset.get("n_classes", 3)

        if not os.path.exists(dataset_path):
            print(f"[bold red]Dataset '{dataset_path}' not found[/bold red]")
            continue

        file_name = Path(dataset_path).stem

        print(f"[bold blue]Sets:[/bold blue] {dataset_path}")

        train_dataset = HDF5Dataset(dataset_path, 'train')
        val_dataset = HDF5Dataset(dataset_path, 'val')

        print(f"[bold blue]Features:[/bold blue] {train_dataset.x_features}")
        print(f"[bold blue]Train samples:[/bold blue] {len(train_dataset)}")
        print(f"[bold blue]Val samples:[/bold blue] {len(val_dataset)}")

        configurations = dataset["models"]

        for configuration in configurations:
            module_name = configuration["module"]
            class_name = configuration["class"]
            kwargs = configuration.get("params", {})

            # Importa dinamicamente il modulo e ottiene la classe specificata
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)

            if not issubclass(model_class, nn.Module):
                print(f"[bold red]Invalid model:[/bold red] {module_name}.{class_name}")
                continue

            model = model_class(train_dataset.x_features, **kwargs)
            trainer_name = f'{model.name}_{file_name}'

            print(f"[bold green]Model name:[/bold green] {trainer_name}")

            checkpoint_path = os.path.join(args.model_path, trainer_name, 'best_model.pth')
            final_path = os.path.join(args.model_path, trainer_name, 'final_model.pth')
            model_path = os.path.join(args.model_path, trainer_name, 'model.pth')


            train_settings = configuration.get('train', {})

            epochs = train_settings.get('epochs', 1000) 
            min_epochs = train_settings.get('min_epochs', 1000)
            lr = train_settings.get('lr', 1e-4)
            max_lr = train_settings.get('max_lr', 1e-3)
            min_lr = train_settings.get('min_lr', 1e-6)
            reduce_lr = train_settings.get('reduce_lr', 10)
            warmup_lr = train_settings.get('warmup_lr', 10)
            lr_factor_down = train_settings.get('lr_factor_down', 0.8)
            lr_factor_up = train_settings.get('lr_factor_up', 1.2)
            batch_size = train_settings.get('batch_size', 2048)
            patience = train_settings.get('patience', 20)
            delta = train_settings.get('delta', 0.0)

            print(f"\n[bold blue]Training settings:[/bold blue]")
            print(f"[bold blue]Epochs:[/bold blue] {epochs}")
            print(f"[bold blue]Min epochs:[/bold blue] {min_epochs}")
            print(f"[bold blue]Batch size:[/bold blue] {batch_size}")
            print(f"[bold blue]Learning rate:[/bold blue] {lr}")
            print(f"[bold blue]Max learning rate:[/bold blue] {max_lr}")
            print(f"[bold blue]Min learning rate:[/bold blue] {min_lr}")
            print(f"[bold blue]Reduce LR:[/bold blue] {reduce_lr}")
            print(f"[bold blue]Warmup LR:[/bold blue] {warmup_lr}")
            print(f"[bold blue]LR factor down:[/bold blue] {lr_factor_down}")
            print(f"[bold blue]LR factor up:[/bold blue] {lr_factor_up}")
            print(f"[bold blue]Patience:[/bold blue] {patience}")
            print(f"[bold blue]Delta:[/bold blue] {delta}")
            print(f"[bold blue]Trainable parameters:[/bold blue] {count_trainable_parameters(model)}")
            print("")

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Initialize metrics
            train_metrics = [
                MeanSquaredError().to(device),
                MeanAbsoluteError().to(device),
                R2Score().to(device),
                ExplainedVariance().to(device)
            ]
            val_metrics = [
                MeanSquaredError().to(device),
                MeanAbsoluteError().to(device),
                R2Score().to(device),
                ExplainedVariance().to(device)
            ]

            # Initialize Trainer
            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=None,
                criterion=criterion,
                optimizer=optimizer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                device=device,
                batch_size=batch_size,
                validation_split=0.2,
                es_patience=patience,
                es_delta=delta,
                min_epochs=min_epochs,
                lr_factor_down=lr_factor_down, 
                lr_factor_up=lr_factor_up, 
                patience_lr_down=reduce_lr, 
                patience_lr_up=warmup_lr, 
                min_lr=min_lr, 
                max_lr=max_lr,
                checkpoint_path=checkpoint_path,
                final_model_path=final_path,
                model_path=model_path,
                log_dir=args.log_dir,
                name=trainer_name,
                verbose=args.verbose
            )

            # Start training
            trainer.fit(
                epochs=epochs, 
                class_names=[str(i) for i in range(n_classes)], 
                early_stopping_metrics='MeanSquaredError', 
                continue_training=args.continue_training,
                reset_optimizer=args.reset_optimizer, 
                reset_early_stopping=args.reset_early_stopping
            )

if __name__ == '__main__':
    main()
