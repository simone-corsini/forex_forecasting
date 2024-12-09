import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassConfusionMatrix
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import os
import re
from collections import defaultdict

class EarlyStopping:
    def __init__(self, trainer, patience=5, min_epochs=200, delta=0.00001, checkpoint_path='best_model.pth', smoothing_factor=0.1, verbose=True):
        self.trainer = trainer
        self.patience = patience
        self.min_epochs = min_epochs
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.smoothing_factor = smoothing_factor
        self.smoothed_score = None
        self.verbose = verbose

    def smooth_metric(self, current_score):
        if self.smoothed_score is None:
            self.smoothed_score = current_score
        else:
            self.smoothed_score = (self.smoothing_factor * current_score +
                                   (1 - self.smoothing_factor) * self.smoothed_score)
        return self.smoothed_score

    def __call__(self, current_score, epoch, console):
        smoothed_score = self.smooth_metric(current_score)

        if self.best_score is None:
            self.best_score = smoothed_score
            if self.verbose:
                console.print(f"[green]New improvement found. New score (smoothed/real): {smoothed_score:.5f}/{current_score:.5f}. Model and training state saved to {self.checkpoint_path}[/green]")
            self.trainer.save_model(self.checkpoint_path, backup=True, epoch=epoch)
        elif smoothed_score > self.best_score - self.delta:
            if epoch >= self.min_epochs:
                self.counter += 1
                if self.verbose:
                    console.print(f"[yellow]Validation improvement not detected. Counter: {self.counter}/{self.patience}[/yellow]")
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        console.print("[bold red]Early stopping activated[/bold red]")
        else:
            self.best_score = smoothed_score
            if self.verbose:
                console.print(f"[green]New improvement found. New score (smoothed/real): {smoothed_score:.5f}/{current_score:.5f}. Model and training state saved to {self.checkpoint_path}[/green]")
            self.trainer.save_model(self.checkpoint_path, backup=True, epoch=epoch)
            self.counter = 0

class DynamicLR:
    def __init__(self, optimizer, factor_down=0.8, factor_up=1.2, patience_lr_down=5, patience_lr_up=5, 
                 min_lr=1e-6, max_lr=1e-2, smoothing_factor=0.5, min_epochs=200, verbose=True):
        self.optimizer = optimizer
        self.factor_down = factor_down
        self.factor_up = factor_up
        self.patience_lr_down = patience_lr_down
        self.patience_lr_up = patience_lr_up
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
        self.min_epochs = min_epochs
        
        self.best_score = None
        self.counter_lr_down = 0
        self.counter_lr_up = 0
        self.lr = optimizer.param_groups[0]['lr']
        self.smoothing_factor = smoothing_factor
        self.smoothed_score = None

    def smooth_metric(self, current_score):
        if self.smoothed_score is None:
            self.smoothed_score = current_score
        else:
            self.smoothed_score = (self.smoothing_factor * current_score +
                                   (1 - self.smoothing_factor) * self.smoothed_score)
        return self.smoothed_score


    def step(self, current_score, epoch, console):
        smoothed_score = self.smooth_metric(current_score)

        if self.best_score is None:
            self.best_score = smoothed_score
        elif smoothed_score > self.best_score:
            if epoch >= self.min_epochs:
                self.counter_lr_down += 1
                self.counter_lr_up = 0
                if self.counter_lr_down >= self.patience_lr_down:
                    self._reduce_lr(console)
                    self.counter_lr_down = 0
        else:
            self.best_score = smoothed_score
            if epoch >= self.min_epochs:
                self.counter_lr_up += 1
                self.counter_lr_down = 0
                if self.counter_lr_up >= self.patience_lr_up:
                    self._increase_lr(console)
                    self.counter_lr_up = 0

    def _reduce_lr(self, console):
        new_lr = max(self.lr * self.factor_down, self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.lr = new_lr
        if self.verbose:
            console.print(f"[yellow]Reducing learning rate to {new_lr}[/yellow]")

    def _increase_lr(self, console):
        if self.lr >= self.max_lr:
            return

        real_new_lr = self.lr * self.factor_up
        new_lr = min(real_new_lr, self.max_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.lr = new_lr
        if self.verbose:
            console.print(f"[yellow]Increasing learning rate to {new_lr}[/yellow]")

        if real_new_lr > self.max_lr:
            self.max_lr *= 0.8
            if self.verbose:
                console.print("[bold red]Maximum learning rate reached[/bold red]")

class Trainer:
    def __init__(self, model, 
                 train_dataset, val_dataset, test_dataset,
                 criterion, optimizer, 
                 train_metrics, val_metrics,
                 device='cpu', 
                 batch_size=64, 
                 validation_split=0.2,
                 es_patience=5, es_delta=0.00001, min_epochs=200,
                 lr_factor_down=0.8, lr_factor_up=1.2, patience_lr_down=5, patience_lr_up=10, min_lr=1e-6, max_lr=1e-2,
                 checkpoint_path='best_model.pth',
                 final_model_path='final_model.pth',
                 log_dir='runs',
                 name='trainer',
                 verbose=True):
        self._name = name
        self.model = model.to(device)
        self.device = device
        self.verbose = verbose

        # Initialize Rich Console
        self.console = Console()
        self._total_steps = 0

        # Dataloaders
        self.train_loader, self.val_loader, self.test_loader = self.get_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size, validation_split
        )

        self.criterion = criterion
        self.optimizer = optimizer

        self.train_metrics = [metric.to(device) for metric in train_metrics]
        self.val_metrics = [metric.to(device) for metric in val_metrics]

        log_dir_base = log_dir
        log_dir_name = None

        if not os.path.exists(log_dir_base):
            os.makedirs(log_dir_base)

        log_pattern = re.compile(f'^{re.escape(self._name)}_(\\d+)$')
        existing_runs = [d for d in os.listdir(log_dir_base) if os.path.isdir(os.path.join(log_dir_base, d)) and log_pattern.match(d)]
        run_numbers = [int(log_pattern.match(d).group(1)) for d in existing_runs]
        if run_numbers:
            next_run_number = max(run_numbers) + 1
        else:
            next_run_number = 1
        log_dir_name = f'{self._name}_{next_run_number:04d}'
        log_dir = os.path.join(log_dir_base, log_dir_name)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)

        # Early Stopping
        self.early_stopping = EarlyStopping(self, patience=es_patience, delta=es_delta, checkpoint_path=checkpoint_path, min_epochs=min_epochs, verbose=self.verbose)
        self.final_model_path = final_model_path

        # Dynamic LR
        self.dynamic_lr = DynamicLR(self.optimizer, 
                                    factor_down=lr_factor_down, 
                                    factor_up=lr_factor_up, 
                                    patience_lr_down=patience_lr_down, 
                                    patience_lr_up=patience_lr_up,
                                    min_lr=min_lr, max_lr=max_lr,
                                    min_epochs=min_epochs,
                                    verbose=self.verbose)

        self._epoch = 1
        self._total_steps = 0
    
        self._all_metrics = []

        for metric in self.train_metrics:
            self._all_metrics.append(metric.__class__.__name__)
        
        for metric in self.val_metrics:
            self._all_metrics.append(metric.__class__.__name__)

        self._all_metrics = list(set(self._all_metrics))

    def _load_training_state(self, checkpoint, reset_optimizer=False, reset_early_stopping=False):
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not reset_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if not reset_early_stopping:
            self.early_stopping.best_score = checkpoint['best_score']
        self._epoch = checkpoint['epoch'] + 1
        self._total_steps = checkpoint['total_steps'] + 1

    def load_training_state(self, reset_optimizer=False, reset_early_stopping=False):
        if os.path.exists(self.final_model_path):
            checkpoint = torch.load(self.final_model_path, weights_only=True)
            self._load_training_state(checkpoint, reset_optimizer=reset_optimizer, reset_early_stopping=reset_early_stopping)
            if self.early_stopping.best_score:
                self.console.print(f"[green]Resuming training from epoch {self._epoch}, step {self._total_steps}, best score: {self.early_stopping.best_score:.5f}[/green]")
            else:    
                self.console.print(f"[green]Resuming training from epoch {self._epoch}, step {self._total_steps}[/green]")
        elif self.early_stopping and os.path.exists(self.early_stopping.checkpoint_path):
            checkpoint = torch.load(self.early_stopping.checkpoint_path)
            self._load_training_state(checkpoint, reset_optimizer=reset_optimizer, reset_early_stopping=reset_early_stopping)
            if self.early_stopping.best_score:
                self.console.print(f"[green]Resuming training from epoch {self._epoch}, step {self._total_steps}, best score: {self.early_stopping.best_score:.5f}[/green]")
            else:
                self.console.print(f"[green]Resuming training from epoch {self._epoch}, step {self._total_steps}[/green]")
        else:
            self.console.print(f"[bold red]No checkpoint found at {self.early_stopping.checkpoint_path}[/bold red]")
            
    def log_metrics(self, step, phase, metrics):
        for name, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{name}', value, step)

        self.writer.flush()

    def save_model(self, path, backup=False, epoch=0):
        folder_path = os.path.dirname(path)
        os.makedirs(folder_path, exist_ok=True)

        if backup:
            backup_path = os.path.join(folder_path, "backup")
            os.makedirs(backup_path, exist_ok=True)
            backup_file_name = os.path.basename(path)
            backup_file_name = backup_file_name.replace('.pth', f'__e{epoch:04d}.pth')
            backup_full_path = os.path.join(backup_path, backup_file_name)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_score': self.early_stopping.best_score,
                'epoch': self._epoch,
                'total_steps': self._total_steps
            }, backup_full_path)

            if self.verbose:
                self.console.print(f"[green]Model saved to {backup_full_path}, epoch {self._epoch}, steps {self._total_steps}, best score: {self.early_stopping.best_score:.5f}[/green]")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.early_stopping.best_score,
            'epoch': self._epoch,
            'total_steps': self._total_steps
        }, path)

        if self.verbose:
            self.console.print(f"[green]Model saved to {path}, epoch {self._epoch}, steps {self._total_steps}, best score: {self.early_stopping.best_score:.5f}[/green]")

    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, weights_only=True)['model_state_dict'])
            self.model.to(self.device)
            self.console.print(f"[green]Model loaded from {path}[/green]")
        else:
            self.console.print(f"[bold red]Model not found at {path}[/bold red]")
            self.console.print("[bold yellow]Model not loaded[/bold yellow]")

    def get_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size, validation_split):
        # If validation dataset is not provided, split the training dataset
        if val_dataset is None:
            total_train = len(train_dataset)
            val_size = int(total_train * validation_split)
            train_size = total_train - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None

        return train_loader, val_loader, test_loader

    def train_one_epoch(self, progress, early_stopping_metrics=None):
        self.model.train()
        running_loss = 0.0
        steps = len(self.train_loader)
        training_task = progress.add_task("", total=steps)

        value_metric = None
        if early_stopping_metrics is not None:
            for metric in self.train_metrics:
                if metric.__class__.__name__ == early_stopping_metrics:
                    value_metric = metric
                    

        for idx, batch in enumerate(self.train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs =  outputs.view(-1)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            self._total_steps += 1
            self.writer.add_scalar('Step/Loss', loss.item(), self._total_steps)

            # Update metrics
            for metric in self.train_metrics:
                metric.update(outputs, targets)
                value = metric.compute()
                if torch.is_tensor(value) and value.ndim == 0:
                    self.writer.add_scalar(f'Step/{metric.__class__.__name__}', value, self._total_steps)
    
            value_metric_string = ""
            if value_metric is not None:
                value_metric_string = f" | {early_stopping_metrics}: {value_metric.compute().item():.10f}"

            progress.update(training_task, advance=1, description=f"Train Batch {idx+1}/{steps} | Loss: {loss.item():.10f}{value_metric_string}")  
        
        progress.remove_task(training_task)

        epoch_loss = running_loss / len(self.train_loader.dataset)

        epoch_metrics = {}

        for metric in self.train_metrics:
            metric_value = metric.compute()
            if torch.is_tensor(metric_value) and metric_value.ndim == 0:
                epoch_metrics[metric.__class__.__name__] = metric_value.item()

            metric.reset()

        return epoch_loss, epoch_metrics

    def validate(self, progress):
        self.model.eval()
        running_loss = 0.0

        steps = len(self.val_loader)
        val_task = progress.add_task("", total=steps)

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                outputs =  outputs.view(-1)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                # Update metrics
                for metric in self.val_metrics:
                    metric.update(outputs, targets)
                
                progress.update(val_task, advance=1, description=f"Eval Batch {idx+1}/{steps}")  
        
            progress.remove_task(val_task)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        
        epoch_metrics = {}

        for metric in self.val_metrics:
            metric_value = metric.compute()
            if torch.is_tensor(metric_value) and metric_value.ndim == 0:
                epoch_metrics[metric.__class__.__name__] = metric_value.item()

            metric.reset()

        return epoch_loss, epoch_metrics


    def fit(self, epochs=20, class_names=None, early_stopping_metrics=None, continue_training=False, reset_optimizer=False, reset_early_stopping=False):
        if continue_training:
            self.load_training_state(reset_optimizer, reset_early_stopping)

        progress = Progress(
            TextColumn("[progress.description]"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TextColumn("{task.description}"),
            console=self.console
        )

        if self.verbose:
            progress.console.print("[bold blue]Epochs[/bold blue]")

        with progress:
            train_loss = 0.0
            val_loss = 0.0
            start_epoch = self._epoch
            total_epochs = start_epoch + epochs
            epoch_task = progress.add_task(f"Epoch {self._epoch}/{total_epochs - 1}", total=epochs)

            for epoch_counter in range(start_epoch, total_epochs):
                self._epoch = epoch_counter
                progress.update(epoch_task, description=f"Epoch {self._epoch}/{total_epochs - 1} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}")

                #Training
                train_loss, train_metric_values = self.train_one_epoch(progress, early_stopping_metrics)
                self.log_metrics(self._epoch, 'Train', train_metric_values)
                self.writer.add_scalar(f'Train/Loss', train_loss, self._epoch)

                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name, param, self._epoch)

                # Validation
                val_loss, val_metric_values = self.validate(progress)
                self.log_metrics(self._epoch, 'Validation', val_metric_values)
                self.writer.add_scalar(f'Validation/Loss', val_loss, self._epoch)

                # Early Stopping
                if early_stopping_metrics is not None:
                    self.early_stopping(val_metric_values[early_stopping_metrics], self._epoch, self.console)
                    self.writer.add_scalar(f'Parameters/Early Stopping Patience', self.early_stopping.counter, self._epoch)
                    if self.early_stopping.early_stop:
                        break
                
                #self.dynamic_lr.step(val_metric_values[early_stopping_metrics], self._epoch, self.console)
                self.dynamic_lr.step(val_loss, self._epoch, self.console)
                self.writer.add_scalar(f'Parameters/Dynamic Learning Rate Up Patience', self.dynamic_lr.counter_lr_up, self._epoch)
                self.writer.add_scalar(f'Parameters/Dynamic Learning Rate Down Patience', self.dynamic_lr.counter_lr_down, self._epoch)
                self.writer.add_scalar(f'Parameters/Learning Rate', self.optimizer.param_groups[0]['lr'], self._epoch)

                if self.verbose:
                    table = Table(title=f"Epoch {self._epoch} Metrics")
                    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
                    table.add_column("Train", justify="right", style="green")
                    table.add_column("Val", justify="right", style="green")
                    table.add_row("Loss", f"{train_loss:.4f}", f"{val_loss:.4f}")

                    for metric in self._all_metrics:
                        if metric not in train_metric_values and metric not in val_metric_values:
                            continue
                        train_string = ""
                        val_string = ""
                        if metric in train_metric_values:
                            train_string = f"{train_metric_values[metric]:.4f}"
                        if metric in val_metric_values:
                            val_string = f"{val_metric_values[metric]:.4f}"

                        table.add_row(metric, train_string, val_string)
                    
                    self.console.print(table)

                progress.update(epoch_task, advance=1, description=f"Epoch {self._epoch}/{total_epochs - 1} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}")

        #Load the best model
        self.load_model(self.early_stopping.checkpoint_path)

        # Save the final model
        self.save_model(self.final_model_path)

        self.writer.close()
