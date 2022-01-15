from lib.dataset3 import GrowthDataModule
from lib.model3 import FeatureFusion
from lib.utils3 import csv_make, csv_to_dict, csv_label, split_data
import pandas as pd

if __name__ == '__main__':
    csv_make()
    df = pd.read_csv('csv_feature.csv', index_col=0)
    if df is None:
        print('CSV_make break')
    elif csv_to_dict() is None:
        print('csv_feature_dict break')
    elif csv_label() is None:
        print('csv_label break')
    elif split_data() is None:
        print('split data break')
    else:
        print('utils right')


    print('Dataset :', GrowthDataModule)
    print('Model :', FeatureFusion)
