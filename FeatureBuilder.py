import os
import pickle
import pandas as pd
import numpy as np
import torch
import sys

from collections import Counter
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
            if os.path.isfile("cleaned_raw_data.pickle"):
                self.motions = self._load_data("cleaned_raw_data")
                return
            else:
                print("No pickle found, creating")
        cnt = 0
        data = []
        for motion in self.recordings():
            data.append(motion)
            if cnt >= abort_after:
                self._write_pickle(data, "cleaned_raw_data")
                self.motions = data
                break
            cnt += 1
        self.motions = data

    def extract_at_hz(self, motion, hz):
        # dataset is sampled with 100Hz
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
            if len(motion) < 600:
                continue
            motion = self.normalize(motion)
            motion = self.segment(motion, self.window_size_cnn, self.window_size_lstm, self.sample_rate)
            _motion.extend(motion)
        return _motion

    def df_to_lstm_tensors(self, motions):
        examples = []
        cols = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for lstm_example in motions:
            if len(lstm_example) < 600/3:
                continue
            cnn_tensors, cnn_labels = [], []
            for name, group in lstm_example:
                if len(group) < self.sample_rate * self.window_size_cnn:
                    continue
                label = Counter(group.iloc[:, -1].values).most_common(1)[0][0]
                cnn_tensors.append(torch.tensor(group.iloc[:, cols].values, dtype=torch.float))
                cnn_labels.append(torch.tensor(label, dtype=torch.long))
            examples.append((torch.stack(cnn_tensors), torch.stack(cnn_labels)))
        self._write_pickle(examples, "lstm_examples_{}_{}_{}".format(self.sample_rate,
                                                                     self.window_size_cnn,
                                                                     self.window_size_lstm))

    def _write_pickle(self, obj, name):
        with open(name+'.pickle', 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_data(self, name):
        with open(name+'.pickle', 'rb') as handle:
            return pickle.load(handle)

if __name__ == '__main__':

    if len(sys.argv) < 7:
        print("""Usage: python FeatureBuilder.py path/to/SHLDataset_User1Hips_v1/release/User1 
        sample_rate cnn_window_size_sec lstm_window_size_sec num_of_files_to_load boolean_force_reload
        
        Example:
        python FeatureBuilder.py /home/xyz/SHLData/User1 25 3 600 50 False
        
        """)

    path = sys.argv[1]
    sample_rate = int(sys.argv[2])
    cnn_window_size_sec = int(sys.argv[3])
    lstm_window_size_sec = int(sys.argv[4])
    abort_loading_after_n_files = int(sys.argv[5])
    force_reload_of_files = bool(sys.argv[6])

    fb = FeatureBuilder(path, sample_rate, cnn_window_size_sec, lstm_window_size_sec)
    fb.raw_data_to_cleaned_df(abort_loading_after_n_files,force_reload_of_files)
    fb.df_to_lstm_tensors(fb.clean_data())