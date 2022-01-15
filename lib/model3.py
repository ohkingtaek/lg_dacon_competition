from turtle import forward
import torch
from torch import nn, Tensor, optim
from torchvision import models
import pytorch_lightning as pl
from sklearn.metrics import f1_score


def accuracy_function(real, pred):    
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

class ImageClassifier(nn.Module):
    def __init__(self, model_name: str = 'resnet101') -> None:
        super().__init__()
        self.feature_extractor = getattr(models, model_name)(pretrained=True)

    def forward(self, images):
        outputs = self.feature_extractor(images)
        return outputs


class SeriesClassifier(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(24*6, 1024) #, bidirectional=True(change model feature)
        self.fc = nn.Linear(num_features*1024, 1000)

    def forward(self, series):
        hidden, _ = self.lstm(series) #gru
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.fc(hidden)
        return hidden


class FeatureFusion(nn.Module):
    def __init__(
        self, 
        model_name: str, 
        num_features: int, 
        num_classes: int, 
        drop_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.image_classifier = ImageClassifier(model_name)
        self.series_classifier = SeriesClassifier(num_features)
        self.fc = nn.Linear(1000 + 1000, num_classes) # imgclassify out_dim + seriesclassify out_dim
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, images, series):
        out1 = self.image_classifier(images)
        out2 = self.series_classifier(series)
        outputs = torch.cat([out1, out2], dim=1)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs


class DiseaseClassifier(pl.LightningModule):
    def __init__(
        self, 
        model_name: str, 
        lr: float, 
        num_features: int, 
        num_classes: int, 
        drop_rate: float
    ) -> None:
        super().__init__()
        self.model = FeatureFusion(model_name, num_features, num_classes, drop_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=1, 
            eta_min=1e-4,
            last_epoch=-1
        )
        return [optimizer], [scheduler]

    def forward(self, images, series):
        outputs = self.model(images, series)
        return outputs

    def training_step(self, batch, batch_idx):
        img = batch['img']
        csv_feature = batch['csv_feature']
        label = batch['label']
        
        output = self(img, csv_feature)
        loss = self.criterion(output, label)
        score = accuracy_function(label, output)
        
        self.log(
            'train_loss', loss, prog_bar=True, logger=True
        )
        self.log(
            'train_score', score, prog_bar=True, logger=True
        )
        return {'loss': loss, 'train_score': score}

    def validation_step(self, batch, batch_idx):
        img = batch['img']
        csv_feature = batch['csv_feature']
        label = batch['label']
        
        output = self(img, csv_feature)
        loss = self.criterion(output, label)
        score = accuracy_function(label, output)
        
        self.log(
            'val_loss', loss, prog_bar=True, logger=True
        )
        self.log(
            'val_score', score, prog_bar=True, logger=True
        )
        
        return {'val_loss': loss, 'val_score': score}
    
    def predict_step(self, batch, batch_idx):
        img = batch['img']
        seq = batch['csv_feature']
        
        output = self(img, seq)
        output = torch.argmax(output, dim=1)
        
        return output
