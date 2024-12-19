import os.path
import numpy as np
from keras import layers, models
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from matplotlib import pyplot as plt
from .MFCC import extract_mfcc, pad_or_clip_features
from .Preprocess import remove_silence

def build_model(_input_shape, _num_classes):
    model = models.Sequential()
    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=_input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(256, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.4))  # 40% 的神经元会被随机丢弃
    model.add(layers.Dense(_num_classes, activation='softmax'))  # 输出每个类别的概率

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(_dataset, _epochs=20, _batch_size=32):
    _x_train, _x_test, _y_train, _y_test, _labels = _dataset
    input_shape = (_x_train.shape[1], _x_train.shape[2])
    num_classes = len(np.unique(_labels))  # 类别数（即说话人数量）

    model = build_model(input_shape, num_classes)

    # 学习率调节器：根据验证集的损失动态调整学习率
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    # 提前停止：当验证集的损失在连续 5 个 epoch 不再降低时停止训练
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.summary()
    history = model.fit(_x_train, _y_train, epochs=_epochs, batch_size=_batch_size, validation_data=(_x_test, _y_test), callbacks=[lr_scheduler, early_stopping])
    plot_training_history(history)

    save_path = './Model/model.keras'
    model.save(save_path)
    return model

def evaluation(_model, _dataset):
    _, _x_test, _, _y_test, _ = _dataset
    # 评估模型
    test_loss, test_acc = _model.evaluate(_x_test, _y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")

def prediction(_file_path, _confidence, _label_encoder, _model, _target_len):
    data = remove_silence(_file_path, 30, None)
    mfcc, _ = extract_mfcc(_file_path, data, True)
    print(mfcc.shape)
    mfcc = pad_or_clip_features(mfcc, _target_len)
    mfcc = np.array(mfcc).reshape(1, mfcc.shape[0], mfcc.shape[1])  # 重塑为模型输入的形状
    predict = _model.predict(mfcc)

    predicted_label = _label_encoder.inverse_transform([np.argmax(predict)])
    predicted_prob = predict[0][np.argmax(predict)]  # 获取该类别的预测概率

    if predicted_prob < _confidence:
        return '陌生人', predicted_prob
    return predicted_label[0], predicted_prob


def plot_training_history(_history, _save_path='./Sample/training_curve.png'):
    """
    显示训练过程的曲线并保存图像。

    :param _history: Keras 模型训练的历史对象
    :param _save_path: 存储图像的路径
    """
    # 确保保存路径的文件夹存在
    os.makedirs(os.path.dirname(_save_path), exist_ok=True)

    # 绘制训练与验证的损失值曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(_history.history['loss'], label='训练集损失')
    plt.plot(_history.history['val_loss'], label='测试集损失')
    plt.title('损失曲线')
    plt.xlabel('训练次数')
    plt.ylabel('损失')
    plt.legend()

    # 绘制训练与验证的准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(_history.history['accuracy'], label='训练集准确率')
    plt.plot(_history.history['val_accuracy'], label='测试集准确率')
    plt.title('准确率曲线')
    plt.xlabel('训练次数')
    plt.ylabel('准确率')
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(_save_path)
    plt.show()

    print(f"训练曲线已存储到 {_save_path}")
