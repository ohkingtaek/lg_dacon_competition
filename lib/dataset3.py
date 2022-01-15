from glob import glob
import json
import pandas as pd
import cv2
import numpy as np
from typing import Dict
import csv

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from lib.utils3 import csv_to_dict, csv_label


class GrowthDataset(Dataset):
    def __init__(
        self,
        files,
        transform=None,
        stage='train',
    ) -> None:
        self.files = files
        self.label_encoder = csv_label()
        self.max_len = 24*6
        self.csv_feature_dict = csv_to_dict()
        if files is not None:
            self.csv_feature_check = [0]*len(self.files)
            self.csv_features = [None]*len(self.files)
        self.transforms = transform
        self.stage = stage
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> Dict:
        file = self.files[index]
        file_name = file.split('/')[-1]

        if self.csv_feature_check[index] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)
            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]
            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[index] = csv_feature
            self.csv_feature_check[index] = 1
        else:
            csv_feature = self.csv_features[index]
        
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        if self.stage == 'train':
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'
            
            return {
                'img': img,
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32),
                'label': torch.tensor(self.label_encoder[label], dtype=torch.long)
            }
        else:
            return {
                'img': img,
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32)
            }


class GrowthDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train=None,
        val=None,
        test=None,
        train_transform=None,
        test_transform=None,
        mode='train'
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train = train
        self.val = val
        self.test = test
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.stage = mode
        self.num_workers = 2*torch.cuda.device_count()
    
    def setup(self, stage='train') -> None:
        self.train_dataset = GrowthDataset(
            self.train,
            self.train_transform
        )
        self.valid_dataset = GrowthDataset(
            self.val, 
            self.test_transform
        )
        self.test_dataset = GrowthDataset(
            self.test, 
            self.test_transform,
            stage='test'
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            num_workers = self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, 
            batch_size=self.batch_size,
            num_workers = self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers = self.num_workers,
        )

if __name__ == '__main__':
    pass