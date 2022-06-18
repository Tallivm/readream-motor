import os
import numpy as np
from torch import Tensor, nn, no_grad, manual_seed, flatten, cuda, optim
from torch import save as torch_save
from torch import load as torch_load
from dataset import DatasetBuilder
from tqdm import trange

from helper_func import import_params, generate_name
from typing import List, Tuple


class NNModelFull(nn.Module):
    def __init__(self, sample_shape: Tuple[int, ...], time_layer_output: int, freq_layer_output: int):
        """
        Create 4-layered NN model with one layer for temporal and another for spectral dimensions.

        sample_shape : the shape of initial data sample
        time_layer_output : the output size for the temporal CNN layer
        freq_layer_output : the output size for the spectral CNN layer
        """
        manual_seed(42)
        super().__init__()
        self.conv_time = nn.Conv3d(sample_shape[0], time_layer_output, kernel_size=(5, 5, 1),
                                   padding=(2, 2, 0), padding_mode='zeros')
        nn.init.kaiming_normal_(self.conv_time.weight)
        self.norm_time = nn.BatchNorm3d(num_features=time_layer_output)
        self.conv_freq = nn.Conv3d(sample_shape[-1], freq_layer_output, kernel_size=(5, 5, 1),
                                   padding=(2, 2, 0), padding_mode='zeros', groups=2)
        nn.init.kaiming_normal_(self.conv_freq.weight)
        self.norm_freq = nn.BatchNorm3d(num_features=freq_layer_output)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool3d((3, 3, 1))    # TODO: CHECK THE SHAPES!!!
        self.dropout = nn.Dropout(0.3)
        self.lin1 = nn.Linear(80, 80)  # TODO: un-hardcode the 80
        self.lin2 = nn.Linear(80, 3)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        """Forward the data sample"""
        x = self.conv_time(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm_time(x)
        x = self.dropout(x)
        x = x.transpose(-1, 1)
        x = self.conv_freq(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm_freq(x)
        x = self.dropout(x)
        x = flatten(x, 1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x


class NNModelTime(nn.Module):
    def __init__(self, sample_shape: Tuple[int, ...], time_layer_output: int):
        """
        Create 4-layered NN model with two layers for temporal dimension.

        sample_shape : the shape of initial data sample
        time_layer_output : the output size for the temporal CNN layer
        """
        manual_seed(42)
        super().__init__()
        self.conv_time0 = nn.Conv2d(sample_shape[0], sample_shape[0], kernel_size=(5, 5),
                                    padding=(2, 2), padding_mode='zeros')
        nn.init.kaiming_normal_(self.conv_time0.weight)
        self.norm_time0 = nn.BatchNorm2d(num_features=sample_shape[0])
        self.conv_time1 = nn.Conv2d(sample_shape[0], time_layer_output, kernel_size=(5, 5),
                                    padding=(2, 2), padding_mode='zeros')
        nn.init.kaiming_normal_(self.conv_time1.weight)
        self.norm_time1 = nn.BatchNorm2d(num_features=time_layer_output)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.3)
        self.lin1 = nn.Linear(320, 320)  # TODO: un-hardcode the 320
        self.lin2 = nn.Linear(320, 3)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        """Forward the data sample"""
        x = self.conv_time0(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm_time0(x)
        x = self.dropout(x)
        x = self.conv_time1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm_time1(x)
        x = self.dropout(x)
        x = flatten(x, 1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x


class NNModelFreq(nn.Module):
    def __init__(self, sample_shape: Tuple[int, ...], freq_layer_output: int):
        """
        Create 4-layered NN model with two layers for temporal dimension.

        sample_shape : the shape of initial data sample
        freq_layer_output : the output size for the spectral CNN layer
        """
        manual_seed(42)
        super().__init__()
        self.conv_freq0 = nn.Conv2d(sample_shape[0], sample_shape[0], kernel_size=(5, 5),
                                    padding=(2, 2), padding_mode='zeros', groups=2)
        nn.init.kaiming_normal_(self.conv_freq0.weight)
        self.norm_freq0 = nn.BatchNorm2d(num_features=sample_shape[0])
        self.conv_freq1 = nn.Conv2d(sample_shape[0], freq_layer_output, kernel_size=(5, 5),
                                    padding=(2, 2), padding_mode='zeros', groups=2)
        nn.init.kaiming_normal_(self.conv_freq1.weight)
        self.norm_freq1 = nn.BatchNorm2d(num_features=freq_layer_output)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.3)
        linear_size = freq_layer_output * (sample_shape[1] // 4) * (sample_shape[2] // 4)
        self.lin1 = nn.Linear(linear_size, linear_size)
        self.lin2 = nn.Linear(linear_size, 3)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        """Forward the data sample"""
        x = self.conv_freq0(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm_freq0(x)
        x = self.dropout(x)
        x = self.conv_freq1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm_freq1(x)
        x = self.dropout(x)
        x = flatten(x, 1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x


class NNModelTrainer:
    def __init__(self, dataset: DatasetBuilder):
        """
        Build, train and test the model.

        dataset : the source of data and some parameters.
        """
        manual_seed(42)
        self.params = import_params()
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model, self.criterion, self.optimizer = None, None, None
        self.dataset = dataset
        self.loss = {'train': [], 'validate': [], 'test': []}

    def get_model(self) -> None:
        """Get the model either by loading or building anew."""
        build_model_params = self.params["MODEL BUILD"]
        if build_model_params["LOAD MODEL"]:
            self._load_model(build_model_params["MODEL NAME"])
        else:
            self._build_model()
        self.model = self.model.to(self.device)
        self._load_optimizer(build_model_params["OPTIMIZER"], lr=build_model_params["LEARNING RATE"],
                             weight_decay=build_model_params["WEIGHT DECAY"])
        self._load_criterion()

    def _load_model(self, model_path: str) -> None:
        """Load a pre-trained PyTorch model."""
        self.model = torch_load(model_path)

    def _build_model(self) -> None:
        """Choose and build the model fit for the chosen data type."""
        time_layer_output = 5
        freq_layer_output = 4  # TODO: add to customizable parameters after un-hardcoding linear layer shapes
        # TODO: rewrite into config for custom data features
        if self.dataset.model_type == 'power-only':
            self.model = NNModelFreq(self.dataset.sample_shape, freq_layer_output)
        elif self.dataset.model_type == 'time-only':
            self.model = NNModelTime(self.dataset.sample_shape, time_layer_output)
        elif self.dataset.model_type == 'time-power':
            self.model = NNModelFull(self.dataset.sample_shape, time_layer_output, freq_layer_output)

    def _load_optimizer(self, optimizer_name: str, lr: float, weight_decay: float) -> None:
        """Load optimizer for the training."""
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer: {optimizer_name}')  # TODO: get optimizers from PyTorch by names

    def _load_criterion(self) -> None:
        """Load loss function (criterion) for the training."""
        self.criterion = nn.CrossEntropyLoss(weight=Tensor(self.dataset.class_weights).to(self.device))

    def train_and_test(self) -> None:
        """Train and validate the model, save the loss values."""
        train_params = self.params["TRAIN PARAMETERS"]
        n_epochs = train_params["N EPOCHS"]
        test_each_epoch = train_params["TEST AFTER N EPOCH"]
        to_save_model = train_params["SAVE MODEL"]
        n_batches = len(self.dataset.batches['train'])
        for epoch in range(n_epochs):
            epoch_loss = 0
            self.model.train()
            for _ in trange(n_batches, desc=f'Epoch {epoch+1}'):
                x, y = self.dataset.load_next_batch_(data_type='train')
                if x is None:
                    continue
                x = Tensor(np.array(x)).to(self.device)
                y = Tensor(np.array(y)).to(self.device)
                if not self.params["MODEL BUILD"]["MULTI-LABEL"]:
                    y = y.argmax(1).long()
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                epoch_loss += loss.item()
                self.optimizer.step()
            self.loss['train'].append(epoch_loss / n_batches)
            if (test_each_epoch != 0) and (epoch % test_each_epoch == test_each_epoch - 1):
                self.validate()
        if to_save_model:
            model_name = generate_name(self.dataset.runtime, '.pth')
            self.save_model(os.path.join(self.params["FILE STRUCTURE"]["MODEL DIRECTORY"], model_name))

    def _test_batch(self, run_type: str) -> (float, np.ndarray, np.ndarray):
        x, y = self.dataset.load_next_batch_(data_type=run_type)
        x = Tensor(np.array(x)).to(self.device)
        y = Tensor(np.array(y)).to(self.device)
        if not self.params["MODEL BUILD"]["MULTI-LABEL"]:
            y = y.argmax(1).long()
        with no_grad():
            outputs = self.model(x)
        loss = self.criterion(outputs, y).item()
        return loss, y.cpu(), outputs.cpu()

    def validate(self) -> None:
        run_type = 'validate'
        n_batches = len(self.dataset.batches[run_type])
        validation_loss = 0
        self.model.eval()
        for _ in trange(n_batches, desc='Validating...'):
            loss, _, _ = self._test_batch(run_type)
            validation_loss += loss
        self.loss[run_type].append(validation_loss / n_batches)
        print(f'Train loss: {self.loss["train"][-1]:.4f} | Validation loss: {self.loss[run_type][-1]:.4f}')

    def test(self) -> (np.ndarray, np.ndarray):
        run_type = 'test'
        n_batches = len(self.dataset.batches[run_type])
        y_true, y_pred = [], []
        test_loss = 0
        self.model.eval()
        for _ in trange(n_batches, desc='Testing...'):
            loss, y, yhat = self._test_batch(run_type)
            test_loss += loss
            y_true.append(y)
            y_pred.append(yhat)
        self.loss[run_type].append(test_loss / n_batches)
        print(f'Test loss: {self.loss[run_type][-1]:.4f}')
        return np.concatenate(y_true), np.concatenate(y_pred)

    def predict(self) -> List[np.ndarray]:
        """Predict labels from unlabeled data."""
        predictions = []
        self.model.eval()
        n_batches = len(self.dataset.batches['predict'])
        for _ in trange(n_batches, desc='Predicting...'):
            x = self.dataset.load_next_batch_(data_type="predict")
            x = Tensor(np.array(x)).to(self.device)
            with no_grad():
                outputs = self.model(x).cpu().numpy()
                predictions.append(outputs)
        return predictions

    def save_model(self, model_path):
        """Save the model in PyTorch format."""
        torch_save(self.model, model_path)
