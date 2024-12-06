import glob
import os
import librosa
import numpy as np
import soundfile
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from .MFCC import extract_mfcc, pad_or_clip_features


def load_file(_train):
    folders = []
    if _train:
        file_path = './DataSet/Train'
    else:
        file_path = './DataSet/Predict'

    for file in os.listdir(file_path):
        ab_path = os.path.join(file_path, file)
        if os.path.isdir(ab_path):
            folders.append(ab_path)

    return folders

def remove_silence(_file, _threshold, _save_path, _need_graphic=False):
    y, sr = librosa.load(_file)
    # 获取音频的能量值
    intervals = librosa.effects.split(y, top_db=_threshold)
    if len(intervals) == 0:
        print(f"Warning: Audio file {_file} is all below the threshold!")
        return y
    # 通过间隔剪裁音频
    non_silent_audio = np.concatenate([y[start:end] for start, end in intervals])
    if _need_graphic:
        # 绘制裁剪前后的时域图
        plt.figure(figsize=(12, 8))

        # 绘制裁剪前的时域图
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, len(y) / sr, len(y)), y)
        plt.title('原始音频')
        plt.xlabel('时间(s)')
        plt.ylabel('幅度')

        # 绘制裁剪后的时域图
        plt.subplot(2, 1, 2)
        plt.plot(np.linspace(0, len(non_silent_audio) / sr, len(non_silent_audio)), non_silent_audio)
        plt.title('裁剪低音段后的音频')
        plt.xlabel('时间(s)')
        plt.ylabel('幅度')

        # 显示图像
        plt.tight_layout()
        plt.show()

    if _save_path is not None:
        soundfile.write(_save_path, non_silent_audio, sr)

    return non_silent_audio, sr

def data_classify(_folders, _need_visible_data):
    temp_features = []
    features = []
    labels = []
    frame_len = []

    label_encoder = LabelEncoder()
    for folder in _folders:
        label = os.path.basename(folder)

        for file in tqdm(glob.glob(os.path.join(folder, '*.wav'))):
            data = remove_silence(file,30, None)
            mfcc, _ = extract_mfcc(file, data, _need_visible_data)
            # print(mfcc.shape)
            if mfcc is not None:
                frame_len.append(mfcc.shape[0])
                temp_features.append(mfcc)
                labels.append(label)

    median = int(np.median(frame_len))
    print("所有文件帧长中位数" + str(median))
    minimun = int(np.min(frame_len))
    print("所有文件帧长最小值" + str(minimun))

    for data in tqdm(temp_features):
        mfcc = pad_or_clip_features(data, median)
        features.append(mfcc)

    labels = label_encoder.fit_transform(labels)
    return np.array(features), np.array(labels), label_encoder, median
