import argparse
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lib.model3 import DiseaseClassifier
from lib.dataset3 import GrowthDataModule
from lib.utils3 import csv_to_dict, split_data, csv_label, submission

def get_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def eval(
    ckpt_path: str,
    model_name: str,
    args
):
    test_data = split_data(mode='test')

    test_transform = A.Compose([
        A.Resize(height=args.height, width=args.width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    data_module = GrowthDataModule(
        args.batch_size,
        test=test_data,
        test_transform=test_transform
    )

    model = DiseaseClassifier(
        model_name,
        args.lr,
        num_features=len(csv_to_dict()),
        num_classes=len(csv_label()),
        drop_rate=args.drop_rate
    )
    
    trainer = pl.Trainer(
        gpus=0,
        precision=16
    )

    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['state_dict'])

    outputs = trainer.predict(model, data_module)

    submission(outputs, model_name)


if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(args.seed)
    ckpt_dir = ''
    ckpt_name = ''
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    eval(ckpt_path, 'efficientnet_b0', args)