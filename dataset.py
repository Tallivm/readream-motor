import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from sklearn.decomposition import PCA
from numpy.lib.stride_tricks import sliding_window_view as slide_view

import data_tranforms as dtf
from typing import List, Dict, Tuple, Callable, Any


def transform_with_pca(x: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2)
    pca.fit(x)
    return pca.transform(x)


class Dataset:
    def __init__(self, data_filename: str, data_types: List[str], electrode_filename: str,
                 patients_for_grid: List[str], drop_electrodes: Dict[str, List[int]], patients_for_train: List[str],
                 classes: Dict[int, int], test_size: float, batch_size: int, fs: int,
                 frag_len: int, frag_overlap: int, frag_by_stim: bool, freqs: List[int], gsp: float,
                 denoise_functions: Dict[Callable, Dict[str, Any]], main_tf: str, main_tf_kwargs: Dict[str, Any],
                 permute_locations: bool, squeeze_time: int = None, squeeze_freq: int = None,
                 frag_start_crop: int = 500
                 ):
        self.rng = np.random.default_rng(42)
        self.data_filename = data_filename
        self.data_types = data_types
        self.electrode_filename = electrode_filename
        self.patients_for_grid = patients_for_grid
        self.drop_electrodes = drop_electrodes
        self.patients_for_train = patients_for_train
        self.patient_raw_data = {}
        for patient in self.patients_for_train:
            self.patient_raw_data[patient] = {}
            for data_type in self.data_types:
                data_file = self.data_filename.format(patient=patient, data_type=data_type)
                self.patient_raw_data[patient][data_type] = self._load_file(data_file, 'data', patient)
        self.classes = classes
        self.n_cls = len(set(classes.values()))
        self.test_size = test_size
        self.batch_size = batch_size
        self.current_batch_i = 0
        self.fs = fs

        self.permute_locations = permute_locations

        self.frag_len = frag_len
        self.frag_overlap = frag_overlap
        self.frag_by_stim = frag_by_stim
        self.frag_start_crop = frag_start_crop
        self.freqs = freqs
        self.gsp = gsp
        self.denoise_functions = denoise_functions
        self.main_tf = main_tf
        self.main_tf_kwargs = main_tf_kwargs
        self.squeeze_time = squeeze_time
        self.squeeze_freq = squeeze_freq

        self.brain_file, self.hemisphere, self.min_step = None, None, None
        self.electrode_labels, self.grid_shape, self.sample_shape = None, None, None
        self.locations, self.labels, self.borders = {}, {}, {}
        self.train_ix, self.train_pat, self.test_ix, self.test_pat = None, None, None, None
        pass

    def prepare_spatial_remapping(self, min_step: float,
                                  brain_filename: str = "Data/halfbrains.mat", hemisphere: str = 'right'):
        """Get A and B sizes, calculate sample shape"""
        self.hemisphere = hemisphere
        self.brain_file = brain_filename
        all_electrodes, electrode_labels = [], []
        for p_nr, patient in enumerate(self.patients_for_grid):
            el_file = self.electrode_filename.format(patient=patient)
            electrodes = self._load_file(el_file, 'electrodes', patient)
            all_electrodes.append(electrodes)
            electrode_labels.extend([p_nr] * len(electrodes))
        self.electrode_labels = np.array(electrode_labels)
        self.min_step = min_step
        self._get_simplified_grid(np.concatenate(all_electrodes))
        self._get_sample_shape()

    def _load_file(self, filename: str, file_type: str, patient: str):
        data = loadmat(filename)[file_type]
        if file_type == 'data':
            if patient in self.drop_electrodes.keys():
                data = np.delete(data, self.drop_electrodes[patient], axis=1)
        elif file_type == 'stim':
            data = data.reshape(-1)
            data = self._convert_classes(data)
        elif file_type == 'electrodes':
            if self.hemisphere == 'right':
                data[:, 0] = np.abs(data)[:, 0]
            elif self.hemisphere == 'left':
                data[:, 0] = -np.abs(data)[:, 0]
            if patient in self.drop_electrodes.keys():
                data = np.delete(data, self.drop_electrodes[patient], axis=0)
        else:
            raise ValueError(f'Unknown file type: {file_type}')
        return data

    def prepare_batches(self):
        for patient in self.patients_for_train:
            self.labels[patient], self.borders[patient] = {}, {}
            for data_type in self.data_types:
                data_file = self.data_filename.format(patient=patient, data_type=data_type)
                stim = self._load_file(data_file, 'stim', patient)
                if self.frag_by_stim:
                    stim, frag_borders = self._fragment_data_by_stim(stim)
                else:
                    stim, frag_borders = self._fragment_data_with_window(stim, return_borders=True)
                self.labels[patient][data_type] = self._stim_to_percentages(stim)
                self.borders[patient][data_type] = frag_borders
        train_ix, train_pat, test_ix, test_pat = self._stratified_train_test_split()
        self.train_ix, self.train_pat = self._split_to_batches(train_ix, train_pat)
        self.test_ix, self.test_pat = self._split_to_batches(test_ix, test_pat)

    def load_next_batch_(self, is_test: bool) -> (List[np.ndarray], List[np.ndarray]):
        if is_test:
            ix_all, pat_all = self.test_ix, self.test_pat
        else:
            ix_all, pat_all = self.train_ix, self.train_pat
        ix, pat = ix_all[self.current_batch_i], pat_all[self.current_batch_i]
        x, y = [], []
        for i in range(len(ix)):
            patient, data_type = pat[i]
            fragment_ix = ix[i]
            y.append(self.labels[patient][data_type][fragment_ix])
            x_start, x_end = self.borders[patient][data_type][fragment_ix]
            x_i = self.patient_raw_data[patient][data_type][x_start: x_end]
            x_i = self._transform_data(x_i)  # [T, E, F]
            x_i = self._spatial_transform(x_i, patient)  # [T, A, B, F]
            x_i = self._normalize_data(x_i)
            x.append(x_i)
        self.current_batch_i += 1
        if self.current_batch_i >= len(ix_all):
            self.current_batch_i = 0
        return x, y

    def _get_simplified_grid(self, x: np.ndarray) -> None:
        init_vert = loadmat(self.brain_file)[f'{self.hemisphere}brain'][0][0][0]
        projected = dtf.project_points_to_other(init_vert, x)
        projected = transform_with_pca(projected)
        projected = (projected / self.min_step).round()
        projected = (projected - projected.min(0)).astype(int)
        self.grid_shape = projected.max(0) + 1
        projected = projected[:, 0] * self.grid_shape[1] + projected[:, 1]
        for p_nr, patient in enumerate(self.patients_for_grid):
            if patient in self.patients_for_train:
                el_ix = np.argwhere(self.electrode_labels == p_nr).ravel()
                self.locations[patient] = projected[el_ix]

    def _get_sample_shape(self):
        """Sample shape is always [Time, A, B, Freq]"""
        t = 1 if self.main_tf == 'fft' else self.frag_len // self.squeeze_time
        f = len(self.freqs) // self.squeeze_freq if self.main_tf in ['fft', 'cwt'] else 1
        self.sample_shape = (t, self.grid_shape[0], self.grid_shape[1], f)

    def _convert_classes(self, stim_x: np.ndarray) -> np.ndarray:
        given_classes = np.unique(list(self.classes.keys()))
        sort_idx = np.argsort(given_classes)
        idx = np.searchsorted(given_classes, stim_x, sorter=sort_idx)
        return np.asarray(list(self.classes.values()))[sort_idx][idx].astype('uint8')

    def _fragment_data_by_stim(self, stim: np.ndarray) -> (List[np.ndarray], List[np.ndarray]):
        frag_ix = np.nonzero(np.diff(stim))[0] + 1
        res = []
        for s in [stim, np.arange(len(stim))]:
            fragments_raw = np.split(s, frag_ix)[1:]
            fragments_raw = [frag[self.frag_start_crop:] for frag in fragments_raw]
            fragments = []
            for frag in fragments_raw:
                fragments.extend(self._fragment_data_with_window(frag, return_borders=False))
            res.append(fragments)
        fragments, frag_borders = res
        frag_borders = [[x[0], x[-1]] for x in frag_borders]
        return fragments, frag_borders

    def _fragment_data_with_window(self, stim: np.ndarray, return_borders: bool) -> (List[np.ndarray], ...):
        frag_ix = np.arange(0, len(stim) - self.frag_len, self.frag_len - self.frag_overlap)
        fragments = []
        for frag_i in frag_ix:
            fragments.append(stim[frag_i: frag_i + self.frag_len + 1])
        if return_borders:
            pseudo = np.arange(len(stim))
            pseudo_fragments = []
            for frag_i in frag_ix:
                pseudo_fragments.append(pseudo[frag_i: frag_i + self.frag_len + 1])
            frag_borders = [[x[0], x[-1]] for x in pseudo_fragments]
            return fragments, frag_borders
        return fragments

    def _stim_to_percentages(self, fragments: List[np.ndarray]) -> List[np.ndarray]:
        percentages = []
        for frag in fragments:
            counts = np.zeros(self.n_cls)
            v, c = np.unique(frag, return_counts=True)
            counts[v] = c
            percentages.append(counts / counts.sum())
        return percentages

    def _stratified_train_test_split(self) -> (List[List[int]], List[List[str]], List[List[int]], List[List[str]]):
        train_ix, train_pat, test_ix, test_pat = [], [], [], []
        for patient in self.patients_for_train:
            for data_type in self.data_types:
                n_samples = len(self.labels[patient][data_type])
                n_test = int(n_samples * self.test_size)
                test_ix.extend(self.rng.choice(range(n_samples), size=n_test, replace=False))
                test_pat.extend([(patient, data_type)] * n_test)
                train_ix.extend([i for i in range(n_samples) if i not in test_ix[-n_test:]])
                train_pat.extend([(patient, data_type)] * (n_samples - n_test))
        return train_ix, train_pat, test_ix, test_pat

    def _split_to_batches(self, ix, pat) -> (List[List[int]], List[List[str]]):
        batches_ix, batches_pat = [], []
        random_order = list(zip(ix, pat))
        self.rng.shuffle(random_order)
        ix, pat = zip(*random_order)
        for i in range(0, len(ix), self.batch_size):
            batches_ix.append(ix[i: i + self.batch_size])
            batches_pat.append(pat[i: i + self.batch_size])
        return batches_ix, batches_pat

    def _transform_data(self, x: np.ndarray) -> np.ndarray:
        for denoise_func, denoise_kwargs in self.denoise_functions.items():
            x = denoise_func(x, **denoise_kwargs)  # [T, E]
        if self.main_tf == 'fft':
            x = self._get_psd(x, **self.main_tf_kwargs[self.main_tf])  # [E, F]
            x = np.expand_dims(x, 0)   # [1, E, F]
        elif self.main_tf == 'cwt':
            x = self._morlet_wavelet_transform(x, **self.main_tf_kwargs[self.main_tf])  # [E, F, T]
            x = np.moveaxis(x, -1, 0)  # [T, E, F]
        elif self.main_tf == 'bandpass':
            x = self._bandpass_n_bandstop(x, **self.main_tf_kwargs[self.main_tf])  # [E, T]
            x = np.expand_dims(np.moveaxis(x, -1, 0), -1)  # [T, E, 1]
        else:
            raise ValueError(f'Unsupported transform type: {self.main_tf}')
        if (self.squeeze_time > 1) and (self.main_tf != 'fft'):
            x = np.mean(slide_view(x, self.squeeze_time, axis=0)[::self.squeeze_time], axis=-1)
        if (self.squeeze_freq > 1) and (self.main_tf != 'bandpass'):
            x = np.mean(slide_view(x, self.squeeze_freq, axis=2)[:, :, ::self.squeeze_freq], axis=-1)
        return x

    def _spatial_transform(self, x: np.ndarray, patient: str) -> np.ndarray:
        ravel_grid_locations = self.locations[patient]
        if self.permute_locations:
            ravel_grid_locations = self.rng.permutation(ravel_grid_locations)
        grid = np.zeros([x.shape[0], self.grid_shape[0] * self.grid_shape[1], x.shape[2]])
        for el in range(x.shape[1]):
            grid[:, ravel_grid_locations[el], :] = x[:, el, :]
        grid = np.reshape(grid, (x.shape[0], self.grid_shape[0], self.grid_shape[1], x.shape[2]))
        padding = np.ceil(self.gsp).astype(int)
        grid = np.pad(grid, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
        grid = gaussian_filter1d(gaussian_filter1d(grid, self.gsp, axis=1), self.gsp, axis=2)
        return grid[:, padding:-padding, padding:-padding, ]

    def _get_psd(self, x: np.ndarray, nperseg: int, noverlap: int, nfft: int) -> np.ndarray:
        psds = []
        for el in range(x.shape[1]):
            psd = signal.welch(x[:, el], nfft=nfft, fs=self.fs, window='hann',
                               nperseg=nperseg, noverlap=noverlap, detrend=False)[1]
            psds.append(psd[self.freqs])
        return np.array(psds)  # [E, F]

    def _morlet_wavelet_transform(self, x: np.ndarray, w: int) -> np.ndarray:
        widths = w * self.fs / (2 * np.array(self.freqs).ravel() * np.pi)
        transformed = []
        for el in range(x.shape[1]):
            res = np.real(signal.cwt(x[:, el], signal.morlet2, widths, w=w))
            transformed.append(res.astype('float32'))
        return np.array(transformed)  # [E, F, T]

    def _bandpass_n_bandstop(self, x: np.ndarray, freq_range_to_drop: Tuple[int, int]) -> np.ndarray:
        sos = signal.butter(5, (min(self.freqs), max(self.freqs)), 'bandpass', fs=self.fs, output='sos')
        filtered = []
        for el in range(x.shape[1]):
            filtered.append(signal.sosfilt(sos, x[:, el]))
        sos = signal.butter(5, freq_range_to_drop, 'bandstop', fs=self.fs, output='sos')
        for el in range(len(filtered)):
            filtered[el] = signal.sosfilt(sos, filtered[el])
        return np.array(filtered)

    def _normalize_data(self, x: np.ndarray) -> np.ndarray:
        if self.main_tf == 'fft':
            return x * 1e-5
        elif self.main_tf == 'cwt':
            return x * 1e-3 - (x * 1e-3).min()
        else:
            return x / 100 - (x / 100).min()
