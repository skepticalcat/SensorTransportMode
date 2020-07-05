import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureBuilder:

    def __init__(self, folder, sample_rate, window_size_cnn_sec, window_size_lstm_sec):
        self.folder = folder
        self.motions = None
        self.sample_rate = sample_rate
        self.window_size_cnn = window_size_cnn_sec
        self.window_size_lstm = window_size_lstm_sec

    def clean_nans(self, motion, labels):
        ts_nans = motion[motion[1].isna()][0].values.tolist() # clean nans
        for elem in range(2,11):
            ts_nans.extend(motion[motion[elem].isna()][0].values.tolist()) # clean nans
        motion = motion[~motion[0].isin(ts_nans)]
        labels = labels[~labels[0].isin(ts_nans)]
        return motion, labels

    def remove_label_null(self, motion, labels):
        ts_null = labels[labels[1] == 0][0].values.tolist() # clean label 0
        motion = motion[~motion[0].isin(ts_null)]
        labels = labels[~labels[0].isin(ts_null)]
        return motion, labels

    def get_clean_motion_and_label_df(self, motion_path, label_path):
        print("cleaning", motion_path)

        motion = pd.read_csv(motion_path, delimiter=" ", header=None, float_precision="high")
        labels = pd.read_csv(label_path, delimiter=" ", header=None, dtype={0: np.int64})

        motion[0] = motion[0].astype(np.int64)

        motion, labels = self.clean_nans(motion, labels)
        motion, labels = self.remove_label_null(motion, labels)

        motion.reset_index(inplace=True, drop=True)
        labels.reset_index(inplace=True, drop=True)

        assert len(motion) == len(labels)

        columns = list(range(0, 10))
        columns.append(20)
        motion = motion.iloc[:, columns]  # get acc, gyro, magnet, pressure
        labels = labels.iloc[:, 0:2]
        motion["label"] = labels[1]

        return motion

    def recordings(self):
        data = {}
        for rec_folder in os.listdir(self.folder):
            if os.path.isdir(os.path.join(self.folder, rec_folder)):
                data[rec_folder] = {"motion": os.path.join(self.folder, rec_folder, "Hips_Motion.txt"),
                                    "labels": os.path.join(self.folder, rec_folder, "Label.txt")
                                    }
        for key, elem in data.items():
            try:
                motion = self.get_clean_motion_and_label_df(elem["motion"], elem["labels"])
            except Exception as e:
                print("Failed to process",key,"because of",e)
                continue
            yield motion

    def raw_data_to_cleaned_df(self, abort_after=25, force_repickle=False):
        if not force_repickle:
            self.motions = self._load_data("cleaned_raw_data")
            return
        cnt = 0
        data = []
        for motion in self.recordings():
            data.append(motion)
            if cnt >= abort_after:
                self._write_pickle(data, "cleaned_raw_data")
                self.motions = data
            cnt += 1
        self.motions = data

    def extract_at_hz(self, motion, hz):
        # dataset is sampled with 100Hz
        # with e.g., hz = 25 we need 100/25 = every fourth entry
        rate = 100//hz
        motion = motion.iloc[::rate, :]
        motion = motion.reset_index(drop=True)
        return motion

    def segment(self, motion, cnn_window_in_seconds, lstm_window_in_secs, sample_rate):
        cnn_size = cnn_window_in_seconds*sample_rate
        lstm_size = lstm_window_in_secs*sample_rate
        motion = motion.groupby(motion.index // lstm_size)
        lstm_segments = []
        for name, group in motion:
            _g = group.reset_index(drop=True)
            _g = _g.groupby(_g.index // cnn_size)
            lstm_segments.append(_g)
        return lstm_segments

    def normalize(self, motion):
        cols_to_norm = [1,2,3,4,5,6,7,8,9,20]
        motion[cols_to_norm] = StandardScaler().fit_transform(motion[cols_to_norm])
        return motion

    def clean_data(self):
        _motion = []
        for motion in self.motions:
            motion = self.extract_at_hz(motion, self.sample_rate)
            motion = self.normalize(motion)
            motion = self.segment(motion, self.window_size_cnn, self.window_size_lstm, self.sample_rate)
            _motion.extend(motion)
        return _motion

    def _write_pickle(self, obj, name):
        with open(name+'.pickle', 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_data(self, name):
        with open(name+'.pickle', 'rb') as handle:
            return pickle.load(handle)


fb = FeatureBuilder("H:\SHLDataset_User1Hips_v1\\release\\User1", 25, 3, 10*60)
fb.raw_data_to_cleaned_df()
fb.clean_data()