import glob
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .MFCC import extract_mfcc, pad_or_clip_features


def load_file(_train):
    folders = []
    if _train:
        file_path = './DataSet/Train'
    else:
        file_path = './DataSet/Test'

    for file in os.listdir(file_path):
        ab_path = os.path.join(file_path, file)
        if os.path.isdir(ab_path):
            folders.append(ab_path)

    return folders

def data_classify(_folders, _need_visible_data, target_length=200):
    features = []
    labels = []

    label_encoder = LabelEncoder()
    for folder in _folders:
        label = os.path.basename(folder)

        for file in glob.glob(os.path.join(folder, '*.wav')):
            mfcc, _ = extract_mfcc(file, _need_visible_data)
            if mfcc is not None:
                mfcc = pad_or_clip_features(mfcc, target_length)
                features.append(mfcc)
                labels.append(label)

    labels = label_encoder.fit_transform(labels)
    return np.array(features), np.array(labels), label_encoder
