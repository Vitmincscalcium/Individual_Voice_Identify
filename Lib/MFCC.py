import os.path

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features as psf
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 黑体字体
rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def extract_mfcc(_file, _data, _need_visible_data, _mel=13, _nfft=2048):
    if _data:
        signal, fs = _data
    else:
        signal, fs = librosa.load(_file, sr=None)
    # 提取 MFCC 特征
    try:
        mfcc = psf.mfcc(signal, fs, numcep=_mel, nfft=_nfft)
        if _need_visible_data:
            mfcc_heat_map(mfcc, fs, _file)
        return mfcc, fs
    except Exception as e:
        print(f"Error processing {_file}: {e}")
        return None

def pad_or_clip_features(features, target_length):
    # 如果特征数小于目标帧数，填充零；如果大于目标帧数，裁剪
    if features.shape[0] < target_length:
        padding = np.zeros((target_length - features.shape[0], features.shape[1]))
        features = np.vstack([features, padding])
    elif features.shape[0] > target_length:
        features = features[:target_length, :]
    return features

def mfcc_heat_map(_mfcc, _fs, _file_name):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(_mfcc.T, x_axis='frames', sr=_fs, cmap='coolwarm')
    # 添加标题和标签
    plt.colorbar(format='%+2.0f dB')
    plt.title(_file_name + ' Mel倒谱系数热度图')
    plt.xlabel('帧数')
    plt.ylabel('MFCC系数')
    plt.tight_layout()
    # 显示热图
    plt.show()