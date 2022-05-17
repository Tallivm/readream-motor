import numpy as np
from torch import save as torch_save
from torch import Tensor, nn, no_grad, cuda, flatten, manual_seed
from tqdm import trange

from typing import Callable, Tuple, Union
from dataset_final import Dataset


class NNModelFull(nn.Module):
    def __init__(self, sample_shape: Tuple[int, ...], smack_time_to: int, smack_freq_to: int):
        manual_seed(42)
        super().__init__()
        self.conv_time = nn.Conv3d(sample_shape[0], smack_time_to, kernel_size=(3, 3, 1),
                                   padding=(1, 1, 0), padding_mode='zeros')
        nn.init.kaiming_normal_(self.conv_time.weight)
        self.conv_freq = nn.Conv3d(sample_shape[-1], smack_freq_to, kernel_size=(3, 3, 1),
                                   padding=(1, 1, 0), padding_mode='zeros', groups=2)
        nn.init.kaiming_normal_(self.conv_freq.weight)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d((3, 3, 1))
        self.norm_time = nn.BatchNorm3d(num_features=smack_time_to)
        self.norm_freq = nn.BatchNorm3d(num_features=smack_freq_to)
        self.dropout = nn.Dropout(0.5)
        self.lin1 = nn.Linear(80, 80)  # TODO: un-hardcode?
        self.lin2 = nn.Linear(80, 3)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
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
    def __init__(self, sample_shape: Tuple[int, ...], smack_time_to: int, smack_freq_to: int):
        manual_seed(42)
        super().__init__()
        self.conv_time = nn.Conv3d(sample_shape[0], smack_time_to, kernel_size=(3, 3, 1),
                                   padding=(1, 1, 0), padding_mode='zeros')
        nn.init.kaiming_normal_(self.conv_time.weight)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d((3, 3, 1))
        self.norm_time = nn.BatchNorm3d(num_features=smack_time_to)
        self.dropout = nn.Dropout(0.5)
        self.lin1 = nn.Linear(280, 280)  # TODO: un-hardcode?
        self.lin2 = nn.Linear(280, 3)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.conv_time(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.norm_time(x)
        x = self.dropout(x)
        x = flatten(x, 1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.softmax(x)
        return x


class NNModelFreq(nn.Module):
    def __init__(self, sample_shape: Tuple[int, ...], smack_time_to: int, smack_freq_to: int):
        manual_seed(42)
        super().__init__()
        self.conv_freq = nn.Conv3d(sample_shape[-1], smack_freq_to, kernel_size=(3, 3, 1),
                                   padding=(1, 1, 0), padding_mode='zeros', groups=2)
        nn.init.kaiming_normal_(self.conv_freq.weight)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d((3, 3, 1))
        self.norm_freq = nn.BatchNorm3d(num_features=smack_freq_to)
        self.dropout = nn.Dropout(0.5)
        self.lin1 = nn.Linear(224, 224)  # TODO: un-hardcode?
        self.lin2 = nn.Linear(224, 3)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
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


class NNModelTrainer:
    def __init__(self, dataset: Dataset):
        manual_seed(42)
        self.model = None
        self.dataset = dataset
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.train_loss, self.test_loss = [], []

    def build_model(self, model_class: Union[NNModelFull, NNModelFreq, NNModelTime], time_out, freq_out) -> None:
        self.model = model_class(self.dataset.sample_shape, time_out, freq_out)

    def train_and_record(self, loss_func: Callable, class_weights: Tuple[float, ...], optim_func: Callable,
                         lr: float, decay: float, n_epochs: int, test_each_epoch: int,
                         convert_to_single_classes: bool):
        optimizer = optim_func(self.model.parameters(), lr=lr, weight_decay=decay)
        criterion = loss_func(weight=Tensor(np.array(class_weights)).to(self.device))
        self.model = self.model.to(self.device)
        for epoch in range(n_epochs):
            running_loss = 0
            self.model.train()
            for _ in trange(len(self.dataset.train_ix), desc=f'Epoch {epoch+1}'):
                x, y = self.dataset.load_next_batch_(is_test=False)
                x = Tensor(np.array(x)).to(self.device)
                y = Tensor(np.array(y)).to(self.device)
                if convert_to_single_classes:
                    y = y.argmax(1).long()
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                running_loss += loss.item()
                optimizer.step()
            self.train_loss.append(running_loss / len(self.dataset.train_ix))
            if epoch % test_each_epoch == test_each_epoch - 1:
                self.test(loss_func, return_clases=False, convert_to_single_classes=convert_to_single_classes)

    def test(self, loss_func, return_clases: bool, convert_to_single_classes: bool):
        self.model = self.model.to(self.device)
        self.model.eval()
        running_loss = 0
        y_true, preds = [], []
        for _ in trange(len(self.dataset.test_ix), desc='Testing...'):
            x, y = self.dataset.load_next_batch_(is_test=True)
            x = Tensor(np.array(x)).to(self.device)
            y = np.array(y)
            if convert_to_single_classes:
                y = y.argmax(1)
            with no_grad():
                outputs = self.model(x).cpu()
                y_true.append(y)
                preds.append(outputs.numpy())
                if not return_clases:
                    metric = loss_func()
                    if convert_to_single_classes:
                        y = Tensor(y).long()
                    else:
                        y = Tensor(y)
                    running_loss += metric(outputs, y)
        if not return_clases:
            self.test_loss.append(running_loss / len(self.dataset.test_ix))
            print(f'Train loss: {self.train_loss[-1]:.4f} | Test loss: {self.test_loss[-1]:.4f}')
        else:
            return y_true, preds

    def save_model(self, model_path):
        torch_save(self.model, model_path)
