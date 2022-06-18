import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy import signal

from data_transforms import TRANSFORM_FUNCTIONS
from helper_func import import_params, log_params, transform_with_pca, project_points_onto_other, split_to_batches
from typing import List, Dict, Tuple, Any, Optional


def _load_file(filename: str, keyword: Optional[str] = None):
    """Load MAT or NPY/NPZ file, with a keyword if provided."""
    if filename.endswith('.mat'):
        f = loadmat(filename)
    elif filename.endswith('.npy') or filename.endswith('.npz'):
        f = np.load(filename)
    else:
        raise ValueError(f'Unknown file extension for file: {filename}')
    if keyword is not None:
        f = f[keyword]
    return f


class Dataset:
    def __init__(self, params: Dict[str, Any], n_cls: int, final_sample_rate: int):
        """Build a dataset."""
        self.dataset_name = params["DATASET NAME"]
        self.n_cls = n_cls
        self.final_sample_rate = final_sample_rate

        self.files = params["FILES"]
        assert len(self.files['DATA FILES']) == len(self.files['ELECTRODE FILES']), \
            f"Dataset {self.dataset_name}: not the same number of data and location files!"

        self.data_type = params["DATASET TYPE"]
        if self.data_type in ['train', 'test']:
            assert len(self.files['DATA FILES']) == len(self.files['CUE FILES']), \
                f"Dataset {self.dataset_name}: not the same number of data and cue files!"
            self.classes = params["CONVERT CLASSES"]
            self.to_fragment_by_cues = params["FRAGMENT BY CUES"]
            self.frag_start_trim = params["FRAGMENTATION PARAMETERS"]["CUED FRAGMENT START TRIM"]
        elif self.data_type == 'predict':
            self.to_fragment_by_cues = False
            self.frag_start_trim = 0
        else:
            raise ValueError(f'{self.dataset_name}: unknown data type "{self.data_type}"')

        self.file_keywords = params["FILE KEYWORDS"]
        self.drop_electrodes = params["DROP ELECTRODES"]

        self.frag_len = params["FRAGMENTATION PARAMETERS"]["FRAGMENT LENGTH"]
        self.frag_overlap = params["FRAGMENTATION PARAMETERS"]["FRAGMENT OVERLAP"]
        self.sample_rate = params["SAMPLE RATE"]
        self.amp_coef = params["AMPLITUDE COEFFICIENT"]
        self.filt_before_frag = params["FILTERING BEFORE FRAGMENTING"]
        self.filt_after_frag = params["FILTERING AFTER FRAGMENTING"]
        self.fragment_data = {}

    def prepare_fragments(self) -> None:
        """Split labeled or unlabeled dataset into batches."""
        self.fragment_data = {'file_ix': [], 'labels': [], 'borders': []}
        for i, filename in enumerate(self.files['DATA FILES']):
            if self.data_type in ['train', 'test']:
                data = self.load_cue_file(self.files['CUE FILES'][i])
            else:
                data = self.load_data_file(filename, i)
            if self.to_fragment_by_cues:
                labels, frag_borders = self._fragment_by_cues(data)
            else:
                labels, frag_borders = self._fragment_with_window(data, return_borders=True)
            if self.data_type in ['train', 'test']:
                labels = self._stim_to_percentages(labels)
                self.fragment_data['labels'].extend(labels)
            self.fragment_data['borders'].extend(frag_borders)
            self.fragment_data['file_ix'].extend([i] * len(frag_borders))
        self.fragment_data['labels'] = np.array(self.fragment_data['labels'])
        self.fragment_data['borders'] = np.array(self.fragment_data['borders'])
        self.fragment_data['file_ix'] = np.array(self.fragment_data['file_ix'])

    def _fragment_by_cues(self, x: np.ndarray) -> (List[np.ndarray], List[Tuple[int, int]]):
        """
        Get fragment labels, start and end positions from label diff in cues array.

        x : cues array.
        """
        diff_ix = np.nonzero(np.diff(x))[0] + 1
        res = []
        for label in [x, np.arange(len(x))]:
            fragments_raw = np.split(label, diff_ix)
            fragments_raw = [frag[self.frag_start_trim:] for frag in fragments_raw]
            all_fragments = []
            for fragment in fragments_raw:
                fragments, _ = self._fragment_with_window(fragment, return_borders=False)
                all_fragments.extend(fragments)
            res.append(all_fragments)
        labels, fragment_borders = res
        fragment_borders = [(x[0], x[-1]) for x in fragment_borders]
        return labels, fragment_borders

    def _fragment_with_window(self, x: np.ndarray, return_borders: bool = True) -> (List[np.ndarray],
                                                                                    Optional[List[Tuple[int, int]]]):
        """
        Get fragment labels, start and end positions by applying a rolling window with strides.

        x : data or cues array.
        return_borders : if True, return fragment start and end positions.
        """
        frag_ix = np.arange(0, len(x) - self.frag_len, self.frag_len - self.frag_overlap)
        fragments = []
        for frag_i in frag_ix:
            fragments.append(x[frag_i: frag_i + self.frag_len + 1])
        frag_borders = None
        if return_borders:
            positions = np.arange(len(x))
            fragment_positions = []
            for frag_i in frag_ix:
                fragment_positions.append(positions[frag_i: frag_i + self.frag_len + 1])
            frag_borders = [(x[0], x[-1]) for x in fragment_positions]
        return fragments, frag_borders

    def get_samples(self, file_i: int, borders: List[Tuple[int, int]]) -> (List[np.ndarray], np.ndarray):
        """Load, filter and fragment one data file by indices asked by data builder."""
        data_filename = self.files['DATA FILES'][file_i]
        data = self.load_data_file(data_filename, file_i)
        data = self._apply_transforms_before_fragmentation(data)
        fragments = []
        for border in borders:
            fragments.append(data[border[0]: border[1] + 1])
        if self.final_sample_rate != self.sample_rate:
            fragments = self._resample_fragments(fragments)
        fragments = [f[:-1] for f in fragments]
        fragments = self._apply_transforms_after_fragmentation(fragments)
        locs_filename = self.files['ELECTRODE FILES'][file_i]
        locs = self.load_electrodes_file(locs_filename, file_i)
        return fragments, locs

    def _resample_fragments(self, fragments: List[np.ndarray]) -> List[np.ndarray]:
        resampled = []
        for fragment in fragments:
            resampled.append(signal.resample(fragment, len(fragment) * self.final_sample_rate / self.sample_rate))
        return resampled

    def _apply_transforms_before_fragmentation(self, x: np.ndarray) -> np.ndarray:
        """Apply one or more transforms on the whole data."""
        for func_name, func_params in self.filt_before_frag:
            x = TRANSFORM_FUNCTIONS[func_name](x, **func_params)
        return x

    def _apply_transforms_after_fragmentation(self, fragments: List[np.ndarray]) -> List[np.ndarray]:
        """Apply one or more transforms on each data fragment."""
        filtered = []
        for fragment in fragments:
            for func_name, func_params in self.filt_after_frag:
                fragment = TRANSFORM_FUNCTIONS[func_name](fragment, **func_params)
                filtered.append(fragment)
        return filtered

    def load_data_file(self, filename: str, sub_i: int) -> np.ndarray:
        """
        Load the data file.

        filename : path to data file.
        sub_i : subject index.
        """
        data = _load_file(filename, keyword=self.file_keywords['data'])
        if str(sub_i) in self.drop_electrodes.keys():
            data = np.delete(data, self.drop_electrodes[str(sub_i)], axis=1)
        return data

    def load_cue_file(self, filename: str) -> np.ndarray:
        """
        Load cue file to get the array of labels.

        filename : path to cut file.
        """
        cues = _load_file(filename, keyword=self.file_keywords['cues'])
        cues = cues.ravel()
        cues = self._remap_classes(cues)
        return cues

    def load_electrodes_file(self, filename: str, sub_i: int):
        """
        Load the electrodes file and mirror all electrodes to the right hemisphere.

        filename : path to file with electrode coordinates.
        sub_i : subject index.
        """
        locs = _load_file(filename, keyword=self.file_keywords['locs'])
        if str(sub_i) in self.drop_electrodes.keys():
            locs = np.delete(locs, self.drop_electrodes[str(sub_i)], axis=0)
        locs[:, 0] = np.abs(locs)[:, 0]
        return locs

    def _remap_classes(self, x: np.ndarray) -> np.ndarray:
        """
        Remap labels in data to labels needed for training.

        x : raw labels loaded from the cue file.
        """
        init_labels = np.unique(list(self.classes.keys()))
        sort_idx = np.argsort(init_labels)
        idx = np.searchsorted(init_labels, x, sorter=sort_idx)
        return np.asarray(list(self.classes.values()))[sort_idx][idx].astype('uint8')

    def _stim_to_percentages(self, fragments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Convert cue arrays to multi-class labels (percentages of each class in a label array).

        fragments : fragmented cue arrays.
        """
        percentages = []
        for fragment in fragments:
            counts = np.zeros(self.n_cls)
            v, c = np.unique(fragment, return_counts=True)
            counts[v] = c
            percentages.append(counts / counts.sum())
        return percentages


class DatasetBuilder:
    def __init__(self):
        self.rng = np.random.default_rng(42)
        self.params = import_params()
        builder_params = self.params["DATA BUILDER PARAMETERS"]
        self.model_type = builder_params['DATA FEATURES']
        assert self.model_type in ['time-only', 'power-only', 'time-power'], f"Unsupported model type {self.model_type}"
        self.freqs = builder_params['BANDS OF INTEREST']
        self.permute_locations = builder_params['PERMUTE LOCATIONS']
        self.n_cls = builder_params['N CLASSES']
        self.sample_rate = builder_params['SAMPLE RATE']
        self.brain_file = builder_params['BRAIN FILE']
        self.datasets = {}
        self.pca, self.grid_shape = None, None
        self.time_size, self.freq_size = None, None
        self.delta, self.alpha, self.gsp = None, None, None
        self.batches, self.current_batch_n = {}, {}
        self.class_weights = None
        self.runtime = log_params()

    def collect_datasets(self):
        """Collect all datasets together to get simplified grid shape and feed batches into the model."""
        dataset_params = self.params["DATASETS"]
        for i, raw_params in enumerate(dataset_params):
            dataset = Dataset(raw_params, self.n_cls, self.sample_rate)
            dataset.prepare_fragments()
            self.datasets[i] = dataset

    def prepare_spatial_remapping(self) -> None:
        """Calculate remapped coordinates for each electrode used in spatial transformation."""
        spatial_transform_params = self.params["SPATIAL TRANSFORM PARAMETERS"]
        self.delta = spatial_transform_params["SET DELTA"]
        assert self.delta > 0, f'Delta parameter should be bigger than 0, currently {self.delta}'
        self.time_size = spatial_transform_params["TEMPORAL RESAMPLE TO"]
        self.freq_size = spatial_transform_params["SPECTRAL RESAMPLE TO"]
        self.gsp = spatial_transform_params["GAUSSIAN BLUR POWER"]
        all_electrodes = []
        for _, dataset in self.datasets.items():
            if dataset.data_type == "train":
                for file_i, el_file in enumerate(dataset.files['ELECTRODE FILES']):
                    electrodes = dataset.load_electrodes_file(el_file, file_i)
                    all_electrodes.append(electrodes)
        all_electrodes = np.concatenate(all_electrodes)
        self._get_simplified_grid(all_electrodes)
        self._calculate_sample_shape()

    def _get_simplified_grid(self, x: np.ndarray):
        """Calculate the shape of the simplified spatial grid."""
        init_vert = np.load(self.brain_file)
        projected = project_points_onto_other(init_vert, x)
        self.pca, projected = transform_with_pca(projected)
        projected = (projected / self.delta).round()
        self.alpha = projected.min(0)
        projected = (projected - self.alpha).astype(int)
        self.grid_shape = projected.max(0) + 1

    def _calculate_sample_shape(self) -> None:
        # TODO: prepare an actual sample rather than estimating it
        if self.model_type == 'time-only':
            self.sample_shape = (self.time_size, self.grid_shape[0], self.grid_shape[1])
        elif self.model_type == 'power-only':
            self.sample_shape = (self.freq_size, self.grid_shape[0], self.grid_shape[1])
        elif self.model_type == 'time-power':
            self.sample_shape = (self.time_size, self.grid_shape[0], self.grid_shape[1], self.freq_size)

    def prepare_batches(self) -> None:
        """Split all datasets into batches."""
        batch_parameters = self.params["BATCH PARAMETERS"]
        test_size = batch_parameters["TEST RATIO"]
        batch_size = batch_parameters["BATCH SIZE"]
        self._prepare_train_batches(test_size, batch_size)
        self._prepare_test_or_predict_batches(batch_size, 'test')
        self._prepare_test_or_predict_batches(batch_size, 'predict')
        self._get_class_weights()

    def _prepare_train_batches(self, test_size: float, batch_size: int) -> None:
        """
        Split train datasets into batches. Train set is stratified by files.

        test_size : ratio of validation set from all train data, in range from 0 to 1.
        batch_size : batch size.
        """
        train_ix, validate_ix = [], []
        for dataset_i, dataset in self.datasets.items():
            if dataset.data_type == 'train':
                for file_i in range(len(dataset.files['DATA FILES'])):
                    ix = [i for i, x in enumerate(dataset.fragment_data['file_ix']) if x == file_i]
                    train_i = self.rng.choice(ix, int(len(ix) * (1 - test_size)), replace=False)
                    valid_i = [i for i in ix if i not in train_i]
                    train_ix.extend([(dataset_i, i) for i in train_i])
                    validate_ix.extend([(dataset_i, i) for i in valid_i])
        train_ix = self.rng.permutation(train_ix)
        self.batches['train'] = split_to_batches(train_ix, batch_size)
        self.current_batch_n['train'] = 0
        self.batches['validate'] = split_to_batches(validate_ix, batch_size)
        self.current_batch_n['validate'] = 0

    def _prepare_test_or_predict_batches(self, batch_size: int, data_type: str) -> None:
        """
        Split test/predict datasets into batches.

        batch_size : batch size.
        data_type : "test" or "predict"
        """
        ix = []
        for dataset_i, dataset in self.datasets.items():
            if dataset.data_type == data_type:
                for file_i in range(len(dataset.files['DATA FILES'])):
                    i = [i for i, x in enumerate(dataset.fragment_data['file_ix']) if x == file_i]
                    ix.extend([(dataset_i, i) for i in i])
        self.batches[data_type] = split_to_batches(ix, batch_size)
        self.current_batch_n[data_type] = 0

    def load_next_batch_(self, data_type: str) -> (List[np.ndarray], List[np.ndarray]):
        """
        Load the data into memory by the batch indices and apply pre-training transformations.

        data_type : "train", "validate", "test" or "predict".
        """
        # TODO: simplify the function (split into several functions)
        batch_ix = self.batches[data_type][self.current_batch_n[data_type]]
        if self.current_batch_n[data_type] < len(self.batches[data_type]) - 1:
            self.current_batch_n[data_type] += 1
        else:
            self.current_batch_n[data_type] = 0
        unique_dataset_ix, ix_by_dataset = np.unique(batch_ix[:, 0], return_inverse=True)
        all_fragments = []
        all_labels = []
        for i in range(ix_by_dataset.max() + 1):
            dataset_i = unique_dataset_ix[i]
            current_dataset_fragments = batch_ix[ix_by_dataset == i, 1]
            files_ix = self.datasets[dataset_i].fragment_data['file_ix'][current_dataset_fragments]
            borders = self.datasets[dataset_i].fragment_data['borders'][current_dataset_fragments]
            if data_type in ['train', 'validate', 'test']:
                labels = self.datasets[dataset_i].fragment_data['labels'][current_dataset_fragments]
                all_labels.extend(labels)
            unique_files_ix, ix_by_files = np.unique(files_ix, return_inverse=True)
            for j in range(ix_by_files.max() + 1):
                file_i = unique_files_ix[j]
                fragments, locs = self.datasets[dataset_i].get_samples(file_i, borders[ix_by_files == j])
                fragments = self.get_features(fragments)
                fragments = self._resample_fragments(fragments)  # TODO: vectorize by converting to array of frags?
                locs = self.electrodes_spatial_transform(locs)
                fragments = [self.spatial_transform(x, locs) for x in fragments]
                fragments = [self._normalize_data(x) for x in fragments]
                all_fragments.extend(fragments)
        return all_fragments, all_labels

    def get_features(self, fragments: List[np.ndarray]) -> List[np.ndarray]:
        """Choose which transformation to apply on each data fragment to extract features."""
        # TODO: have a config file to be able to add new model types
        if self.model_type == 'time-only':
            fragments = self._get_temporal_features(fragments)  # [E, T]
            fragments = [np.moveaxis(x, -1, 0) for x in fragments]  # [T, E]
        elif self.model_type == 'power-only':
            fragments = self._get_spectral_features(fragments)  # [E, F]
            fragments = [np.moveaxis(x, -1, 0) for x in fragments]  # [E, F]
        elif self.model_type == 'time-power':
            fragments = self._get_temporal_spectral_features(fragments)  # [E, F, T]
            fragments = [np.moveaxis(x, -1, 0) for x in fragments]  # [T, E, F]
        return fragments

    def _get_temporal_features(self, fragments: List[np.ndarray]) -> List[np.ndarray]:
        """Get samples after filtering with Butterworth filters. Resulting sample shape: [E, T]"""
        # TODO: support any number of breaks between frequencies of interest (currently fixed to 2 ranges)
        sos_bandpass = signal.butter(5, (self.freqs[0][0], self.freqs[1][1]), 'bandpass',
                                     fs=self.sample_rate, output='sos')
        sos_bandstop = signal.butter(5, (self.freqs[0][1], self.freqs[1][0]), 'bandstop',
                                     fs=self.sample_rate, output='sos')
        result = []
        for fragment in fragments:
            filtered = []
            for electrode in range(fragment.shape[1]):
                filt = signal.sosfilt(sos_bandpass, fragment[:, electrode])
                filt = signal.sosfilt(sos_bandstop, filt)
                filtered.append(filt)
            result.append(np.array(filtered))
        return result

    def _get_spectral_features(self, fragments: List[np.ndarray]) -> List[np.ndarray]:
        """Get samples after calculating power spectral density. Resulting sample shape: [E, F]"""
        result = []
        for fragment in fragments:
            psds = []
            for electrode in range(fragment.shape[1]):
                psd = signal.welch(fragment[:, electrode], nfft=self.sample_rate, fs=self.sample_rate, window='hann',
                                   nperseg=len(fragment)//4, noverlap=len(fragment)//10, detrend=False)[1]
                frequencies = np.hstack([np.arange(freq[0], freq[1]) for freq in self.freqs])
                psds.append(psd[frequencies])
            result.append(np.array(psds))
        return result

    def _get_temporal_spectral_features(self, fragments: List[np.ndarray]) -> List[np.ndarray]:
        """Get samples after continuous wavelet transformation. Resulting sample shape: [E, F, T]"""
        frequencies = np.array([list(range(freq[0], freq[1])) for freq in self.freqs]).ravel()
        widths = 5 * self.sample_rate / (2 * frequencies * np.pi)
        result = []
        for fragment in fragments:
            transformed = []
            for electrode in range(fragment.shape[1]):
                # TODO: add the w parameter into configurable
                res = np.real(signal.cwt(fragment[:, electrode], wavelet=signal.morlet2, widths=widths, w=5))
                transformed.append(res.astype('float32'))
            result.append(np.array(transformed))
        return result

    def _resample_fragments(self, fragments: List[np.ndarray]) -> List[np.ndarray]:
        """Resample arrays to get required temporal and spectral sizes."""
        # TODO: rewrite to be configurable
        if (self.model_type == 'time-only') and (self.time_size != fragments[0].shape[0]):
            fragments = [signal.resample(x, self.time_size, axis=0) for x in fragments]
        elif (self.model_type == 'power-only') and (self.freq_size != fragments[0].shape[0]):
            fragments = [signal.resample(x, self.freq_size, axis=0) for x in fragments]
        elif self.model_type == 'time-power':
            if self.time_size != fragments[0].shape[0]:
                fragments = [signal.resample(x, self.time_size, axis=0) for x in fragments]
            if self.freq_size != fragments[0].shape[2]:
                fragments = [signal.resample(x, self.freq_size, axis=2) for x in fragments]
        return fragments

    def electrodes_spatial_transform(self, electrodes: np.ndarray) -> np.ndarray:
        """Project and transform electrodes to get their simplified locations."""
        init_vert = np.load(self.brain_file)
        electrodes = project_points_onto_other(init_vert, electrodes)
        electrodes = self.pca.transform(electrodes)
        electrodes = (electrodes / self.delta).round()
        electrodes = (electrodes - self.alpha).astype(int)
        if self.permute_locations:
            electrodes = self.rng.permutation(electrodes)
        return electrodes

    def spatial_transform(self, x: np.ndarray, electrodes: np.ndarray) -> np.ndarray:
        """Place fragment data on simplified grid locations."""
        grid = np.zeros(self.sample_shape)
        for electrode in range(x.shape[1]):
            grid[:, electrodes[electrode][0], electrodes[electrode][1]] = x[:, electrode]
        padding = np.ceil(self.gsp).astype(int)
        if self.model_type == 'time-power':  # TODO: rewrite
            grid = np.pad(grid, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
        else:
            grid = np.pad(grid, ((0, 0), (padding, padding), (padding, padding)))
        grid = gaussian_filter1d(gaussian_filter1d(grid, self.gsp, axis=1), self.gsp, axis=2)
        grid = grid[:, padding:-padding, padding:-padding]
        return grid

    def _normalize_data(self, x: np.ndarray) -> np.ndarray:
        """Normalize data by a constant based on extracted feature type."""
        # TODO: un-hardcode constants
        if self.model_type == 'time-only':
            return x / 100 - (x / 100).min()
        elif self.model_type == 'power-only':
            return x * 1e-5
        elif self.model_type == 'time-power':
            return x * 1e-3 - (x * 1e-3).min()

    def _get_class_weights(self) -> None:
        """Get labels from all train batches to get weights for model criterion."""
        labels = []
        for _, dataset in self.datasets.items():
            if dataset.data_type == 'train':
                labels.extend(dataset.fragment_data['labels'])
        counts = np.zeros(self.n_cls)
        v, c = np.unique([label.argmax() for label in labels], return_counts=True)
        counts[v] = c
        self.class_weights = 1 / counts
