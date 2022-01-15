import argparse

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lib.model3 import DiseaseClassifier
from lib.dataset3 import GrowthDataModule
from lib.utils3 import csv_to_dict, split_data, csv_label

def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def train(
    model_name: str,
    kfold: bool,
    fold_num: int,
    args
):
    if kfold == True:
        train_data, val_data = split_data(args.seed, mode='train', kfold=True, id=fold_num)
    elif kfold == False:
        train_data, val_data = split_data(args.seed, mode='train', kfold=False)

    train_transform = A.Compose([
        A.Resize(height=args.height, width=args.width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(height=args.height, width=args.width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    data_module = GrowthDataModule(
        args.batch_size,
        train=train_data,
        val=val_data,
        train_transform=train_transform,
        test_transform=test_transform
    )

    label_encoder, _ = csv_label()
    model = DiseaseClassifier(
        model_name,
        args.lr,
        num_features=len(csv_to_dict()),
        num_classes=len(label_encoder),
        drop_rate=args.drop_rate
    )

    ckpoint_path = f'weight/{model_name}'
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=ckpoint_path,
        filename='{epoch}-{lr}-{val_loss:.2f}',
        save_top_k=-1,
        mode='min',
        save_weights_only=True
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],#early_stopping
        gpus=1,
        max_epochs=args.max_epochs,
        precision=16,
    )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    args = get_args()
    pl.seed_everything(args.seed)
    train('efficientnet_b0', False, 0, args)
    # for i in range(5):s
    #     train('efficientnet_b0', True, i, args)