import numpy as np
import os
import jsonpickle
from torch import nn, optim, manual_seed
from torch import load as torch_load
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import data_tranforms as dtf
from dataset_final import Dataset
from models_final import NNModelTrainer, NNModelFull, NNModelTime, NNModelFreq


if __name__ == "__main__":
    manual_seed(42)

    model_classes = {'fft': NNModelFreq,
                     'cwt': NNModelFull,
                     'bandpass': NNModelTime}

    model_type = 'fft'
    model_name = None
    permute_locations = False
    convert_to_single_classes = True
    test_only = True
    frag_by_stim = True

    for current_patient in ['bp']:  # 'fp', 'hh', 'jc', 'jm', 'rh', 'rr'

        params = {
            "dataset_params": {
                "data_filename": "Data/imagery_basic_data/{patient}_{data_type}_t_h.mat",
                "data_types": ['im'],
                "electrode_filename": "Data/imagery_basic_data/locs/{patient}_electrodes.mat",

                "patients_for_grid": ['bp', 'fp', 'hh', 'jc', 'jm', 'rh', 'rr'],
                "drop_electrodes": {
                    "fp": [40, 48, 49, 56],
                    "jm": [32, 33, 40, 48, 49, 56, 57, 58]
                },
                "patients_for_train": [current_patient],
                "permute_locations": permute_locations,

                "classes": {0: 0, 11: 1, 12: 2},
                "test_size": 0.1,
                "batch_size": 16,
                "fs": 1000,

                "frag_len": 200,
                "frag_overlap": 100,
                "frag_by_stim": frag_by_stim,
                "freqs": list(range(8, 32)) + list(range(76, 100)),
                "gsp": 1.5,

                "denoise_functions": {
                    dtf.car_filter: {"car_filter_type": "mean"},
                    dtf.zero_mean_norm: {},
                },
                "main_tf": model_type,
                "main_tf_kwargs": {'fft': {'nperseg': 25, 'noverlap': 10, 'nfft': 1000},
                                   'cwt': {'w': 5},
                                   'bandpass': {'freq_range_to_drop': (33, 75)},
                                   'simple': {}},
                "squeeze_time": 2,
                "squeeze_freq": 2,
                "frag_start_crop": 500
            },
            "dataset_spatial": {
                "min_step": 5.12,
                "brain_filename": "Data/halfbrains.mat",
                "hemisphere": "right"
             },
            "load_model": None if model_name is None else os.path.join('results_final', model_name, 'model.pth'),
            "to_save_model": True,
            "test_only": test_only,
            "model_build": {
                "model_class": model_classes[model_type],
                "time_out": 5,
                "freq_out": 4
            },
            "train_params": {
                "loss_func": nn.CrossEntropyLoss,
                "class_weights": (1, 2, 2),
                "optim_func": optim.Adam,
                "lr": 0.0005,
                "decay": 1e-3 / 200,
                "n_epochs": 50,
                "test_each_epoch": 5,
                "convert_to_single_classes": convert_to_single_classes
            }
        }

        runtime = datetime.now()

        print("Initializing the dataset...")
        dataset = Dataset(**params['dataset_params'])
        dataset.prepare_spatial_remapping(**params['dataset_spatial'])
        print(f"Sample size will be {dataset.sample_shape}")
        dataset.prepare_batches()

        print("Initializing the model...")
        trainer = NNModelTrainer(dataset=dataset)
        if params['load_model'] is None:
            trainer.build_model(**params['model_build'])
        else:
            trainer.model = torch_load(params['load_model'])

        if params['test_only']:
            if model_name is None:
                raise ValueError('If to test only, please choose a model to load.')
            y_true, y_pred = trainer.test(params['train_params']['loss_func'], return_clases=True,
                                          convert_to_single_classes=convert_to_single_classes)
            y_true_classes = [y.argmax(1) for y in y_true]
            y_pred_classes = [y.argmax(1) for y in y_pred]
            y_true_classes = [x for y in y_true_classes for x in y]
            y_pred_classes = [x for y in y_pred_classes for x in y]
            confm = confusion_matrix(y_true_classes, y_pred_classes, labels=(0, 1, 2))
            disp = ConfusionMatrixDisplay(confm, display_labels=('rest', 'hand', 'tongue'))
            disp.plot(cmap='pink_r', colorbar=False)
            plt.tight_layout()
            plt.savefig(os.path.join('results_final', model_name,
                                     f'conf_matrix_{"-".join(params["dataset_params"]["data_types"])}.png'))
            print(classification_report(y_true_classes, y_pred_classes,
                                        target_names=('rest', 'hand', 'tongue')))
        else:
            print("Logging parameters...")
            directory = os.path.join("results_final",
                                     '_'.join([
                                         runtime.strftime("%m.%d.%H.%M.%S"),
                                         "-".join(params["dataset_params"]["patients_for_train"]),
                                         "-".join(params["dataset_params"]["data_types"]),
                                         f'fl{params["dataset_params"]["frag_len"]}',
                                         ["stim" if params["dataset_params"]["frag_by_stim"] else "window"][0],
                                         params["dataset_params"]["main_tf"]
                                     ]))
            os.mkdir(directory)
            logfile = os.path.join(directory, 'log.json')
            with open(logfile, 'w') as f:
                f.write(jsonpickle.encode(params, unpicklable=False, indent=4))

            print("Training...")
            trainer.train_and_record(**params["train_params"])

            if params['to_save_model']:
                print("Saving the model...")
                trainer.save_model(os.path.join(directory, 'model.pth'))

            print("Plotting losses...")
            epochs_train = np.linspace(1, params['train_params']['n_epochs'], len(trainer.train_loss))
            epochs_test = np.linspace(1, params['train_params']['n_epochs'], len(trainer.test_loss))
            plt.plot(epochs_train, trainer.train_loss, c='#3388FF', label='train')
            plt.scatter(epochs_train, trainer.train_loss, c='#3388FF')
            plt.plot(epochs_test, trainer.test_loss, c='#FF8833', label='test')
            plt.scatter(epochs_test, trainer.test_loss, c='#FF8833')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join('results_final', f'{os.path.basename(directory)}.png'))
            plt.close()
