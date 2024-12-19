# Individual_Voice_Identify
 利用MFCC提取特征，再使用CNN训练的语音身份识别脚本

---
# 功能
- **音频预处理**：去除静音部分进行音频提纯。
- **特征提取**：从音频中提取 MFCC 特征。
- **模型训练**：使用提取的特征训练神经网络模型。
- **预测**：使用训练好的模型对新音频进行预测。
---
# 函数和执行流程

## 1. **数据加载与预处理**

### `load_file(_train)`
该函数用于加载训练数据的文件夹路径。

- **参数**：`_train` (布尔值，决定加载训练集还是验证集)
- **返回**：返回包含文件夹路径的列表。

### `remove_silence(_file, _threshold, _save_path, _need_graphic=False)`
移除音频文件中的静音部分。

- **参数**：
  - `_file`：音频文件路径。
  - `_threshold`：静音的能量阈值。
  - `_save_path`：裁剪后的音频保存路径。
  - `_need_graphic`：是否需要绘制时域图。
- **返回**：去除静音后的音频数据。

### `data_classify(_folders, _need_graphics)`
将音频文件数据分类，提取 MFCC 特征并标注标签。

- **参数**：
  - `_folders`：包含音频文件夹路径的列表。
  - `_need_visible_data`：是否需要可视化数据。
- **返回**：返回提取的特征、标签和标签编码器。

---

## 2. **特征提取**

### `extract_mfcc(_file, _data, _need_graphics, _mel=13, _nfft=2048)`
从音频中提取 MFCC 特征。

- **参数**：
  - `_file`：音频文件路径。
  - `_data`：音频数据（如果已经加载）。
  - `_need_visible_data`：是否需要绘制 MFCC 热图。
  - `_mel`：MFCC 系数的数量（默认 13）。
  - `_nfft`：FFT 点数（默认 2048）。
- **返回**：返回提取的 MFCC 特征和采样率。

### `pad_or_clip_features(_features, _target_length)`
填充或裁剪特征，使其长度符合模型要求。

- **参数**：
  - `features`：MFCC 特征。
  - `target_length`：目标特征长度。
- **返回**：返回处理后的特征。

### `mfcc_heat_map(_mfcc, _fs, _file_name)`
绘制 MFCC 特征的热图。

- **参数**：
  - `_mfcc`：MFCC 特征。
  - `_fs`：音频采样率。
  - `_file_name`：音频文件名。
- **返回**：无（绘制图表）。

---

## 3. **模型构建与训练**

### `build_model(_input_shape, _num_classes)`
构建一个 1D 卷积神经网络（CNN）模型，用于语音识别。

- **参数**：
  - `input_shape`：输入数据的形状。
  - `num_classes`：分类数目（即说话人数量）。
- **返回**：返回构建好的模型。

### `train_model(_dataset, _epochs=10, _batch_size=32)`
训练模型并保存。

- **参数**：
  - `_dataset`：包含训练集和测试集的元组。
  - `_epochs`：训练周期数（默认 10）。
  - `_batch_size`：批次大小（默认 32）。
- **返回**：返回训练好的模型。

---

## 4. **模型评估与预测**

### `evaluation(_model, _dataset)`
评估训练好的模型在测试集上的表现。

- **参数**：
  - `_model`：训练好的模型。
  - `_dataset`：包含测试集的元组。
- **返回**：无（输出评估结果）。

### `prediction(_file_path, _confidence, _label_encoder, _model, _target_len)`
对指定的音频文件进行预测。

- **参数**：
  - `_file_path`：音频文件路径。
  - `_confidence`：置信度阈值，低于该值返回“陌生人”。
  - `_label_encoder`：标签编码器。
  - `_model`：训练好的模型。
  - `_target_len`：特征的目标长度。
- **返回**：返回预测的标签和该标签的预测概率。

---

## 5. **主程序执行**

### `main()`
训练阶段：
1. 加载训练数据。
2. 提取特征并进行标签编码。
3. 划分数据集为训练集和测试集。
4. 构建并训练模型。
5. 评估模型性能，并保存模型、标签编码器和音频帧数的中位数。

### `main_prediction(_confidence)`
预测阶段：
1. 加载训练好的模型、标签编码器和音频帧数的中位数。
2. 对给定音频文件进行预测并输出结果。
---
# LICENSE
- 该项目使用 MIT License，详情请查看LICENSE。
## 所用库
- **Keras**: MIT License
- **librosa**: MIT License
- **numpy**: BSD License
- **soundfile**: BSD License
- **scikit-learn**: BSD License
- **matplotlib**: PSF License
- **python_speech_features**: MIT License
