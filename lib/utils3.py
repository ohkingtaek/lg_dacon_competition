from typing import Dict
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold


def csv_make():
    csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', 
                    '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
    
    csv_files = sorted(glob('data/train/*/*.csv'))
    temp_csv = pd.read_csv(csv_files[0])[csv_features]
    max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    for csv in tqdm(csv_files[1:]):                                  
        temp_csv = pd.read_csv(csv)[csv_features]
        temp_csv = temp_csv.replace('-', np.nan).dropna()
        if len(temp_csv) == 0:
            continue
        temp_csv = temp_csv.astype(float)
        temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
        max_arr = np.max([max_arr,temp_max], axis=0)
        min_arr = np.min([min_arr,temp_min], axis=0)
    
    csv_feature_dict = {csv_features[i]: [min_arr[i], max_arr[i]] for i in range(len(csv_features))}
    df = pd.DataFrame(csv_feature_dict)
    df.to_csv('csv_feature.csv')

def csv_label():
    crop = {'1': '딸기','2': '토마토','3': '파프리카','4': '오이','5': '고추','6': '시설포도'}
    disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
        '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
    risk = {'1': '초기','2': '중기','3': '말기'}

    label_description = {}
    for key, value in disease.items():
        label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
        for disease_code in value:
            for risk_code in risk:
                label = f'{key}_{disease_code}_{risk_code}'
                label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'

    label_encoder = {key:idx for idx, key in enumerate(label_description)}
    label_decoder = {val:key for key, val in label_encoder.items()}
    return label_encoder, label_decoder

def split_data(
    split_rate: float = 0.2, 
    seed: int = 42, 
    mode: str = 'train', 
    kfold: bool = False, 
    id: int = 0
):
    if mode == 'test':
            test = sorted(glob('data/test/*'))
            return test
    else:
        if kfold == True:
            df = pd.read_csv('data/train.csv')
            df['fold'] = 0
            kf = KFold(n_splits=5, random_state=seed, shuffle=True)

            for fold, (_, valid) in enumerate(kf.split(X=df.index)):
                df.loc[df.iloc[valid].index, 'fold'] = int(fold)
            train = df[df['fold'] != id]
            val = df[df['fold'] == id]
            tmp, tmp_2 = [], []
            for a in train['image'][0:]:
                tmp.append(f'data/train/{a}')
            for b in val['image'][0:]:
                tmp_2.append(f'data/train/{b}')
            train, val = tmp, tmp_2
            return train, val
        else:
            train_files = sorted(glob('data/train/*'))
            labels = pd.read_csv('data/train.csv')['label']
            train, val = train_test_split(
                train_files,
                test_size=split_rate,
                random_state=seed,
                shuffle=True,
                stratify=labels
            )
            print(type(train))
            return train, val

def csv_to_dict() -> Dict:
    dic = {}
    csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', 
                    '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
    df = pd.read_csv('csv_feature.csv', index_col=0)
    for i in csv_features:
        dic[i] = [df[i][0], df[i][1]]
    return dic

def submission(preds, file_name):
    _, label_decoder = csv_label()
    preds = [outputs.detach().cpu().numpy() for batch in preds for outputs in batch]
    preds = np.array([label_decoder[int(val)] for val in preds])
    submission = pd.read_csv('data/sample_submission.csv')
    submission['label'] = preds
    file_name = 'submission/' + file_name
    submission.to_csv(file_name, index=False)


if __name__ == '__main__':
    csv_make()