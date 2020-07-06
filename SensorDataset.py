import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


class SensorDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'frame': self.examples[idx][0], 'labels': self.examples[idx][1]}
        return sample

    def train_test_dataset(self, test_split=0.25, val_split=0.1):
        train_idx, test_idx = train_test_split(list(range(len(self))), test_size=test_split)
        train_idx, val_idx = train_test_split(list(range(len(train_idx))), test_size=val_split)
        datasets = {'train': Subset(self, train_idx),
                    'test': Subset(self, test_idx),
                    'val': Subset(self, val_idx)}
        return datasets