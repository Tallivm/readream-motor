import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from typing import List
from helper_func import import_params, generate_name


def plot_loss_curves(runtime: str, train_loss: List, test_loss: List) -> None:
    # TODO: add an option to split test losses by datasets into separate curves
    params = import_params()
    n_epochs = params["TRAIN PARAMETERS"]["N EPOCHS"]
    epochs_train = np.linspace(1, n_epochs, len(train_loss))
    epochs_test = np.linspace(1, n_epochs, len(test_loss))
    plt.plot(epochs_train, train_loss, c='#3388FF', label='train')
    plt.scatter(epochs_train, train_loss, c='#3388FF')
    plt.plot(epochs_test, test_loss, c='#FF8833', label='test')
    plt.scatter(epochs_test, test_loss, c='#FF8833')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plot_name = generate_name(runtime, '.svg')  # TODO: add option to save in other formats
    plt.savefig(os.path.join(params["FILE STRUCTURE"]["LOSS PLOTS DIRECTORY"], plot_name))
    plt.close()


def plot_confusion_matrix(runtime: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    # TODO: support multi-label classification, maybe in a separate function
    assert len(y_true.shape) == 1, 'Multi-label confusion matrices are not yet supported.'
    params = import_params()
    n_classes = params["DATA BUILDER PARAMETERS"]["N CLASSES"]
    class_names = params["DATA BUILDER PARAMETERS"]["CLASS NAMES"]
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred.argmax(1), labels=range(n_classes),
                                            display_labels=class_names, cmap='pink_r', colorbar=False)
    # plt.locator_params(axis="x", integer=True)  # FIXME: set_params() got an unexpected keyword argument 'integer'
    plt.tight_layout()
    plot_name = generate_name(runtime, '.svg')  # TODO: add option to save in other formats
    plt.savefig(os.path.join(params["FILE STRUCTURE"]["CONFUSION MATRICES DIRECTORY"], plot_name))
    plt.close()
