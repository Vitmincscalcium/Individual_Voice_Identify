import os.path
import numpy as np
from keras import layers, models
from .MFCC import extract_mfcc, pad_or_clip_features
from .Preprocess import remove_silence

def build_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(256, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # 50% 的神经元会被随机丢弃
    model.add(layers.Dense(num_classes, activation='softmax'))  # 输出每个类别的概率

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(_dataset, _epochs=10, _batch_size=32):
    _x_train, _x_test, _y_train, _y_test, _labels = _dataset
    input_shape = (_x_train.shape[1], _x_train.shape[2])
    num_classes = len(np.unique(_labels))  # 类别数（即说话人数量）

    model = build_model(input_shape, num_classes)
    model.summary()
    model.fit(_x_train, _y_train, epochs=_epochs, batch_size=_batch_size, validation_data=(_x_test, _y_test))

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
    mfcc, _ = extract_mfcc(_file_path, data, False)
    print(mfcc.shape)
    mfcc = pad_or_clip_features(mfcc, _target_len)
    mfcc = np.array(mfcc).reshape(1, mfcc.shape[0], mfcc.shape[1])  # 重塑为模型输入的形状
    predict = _model.predict(mfcc)

    predicted_label = _label_encoder.inverse_transform([np.argmax(predict)])
    predicted_prob = predict[0][np.argmax(predict)]  # 获取该类别的预测概率

    if predicted_prob < _confidence:
        return '陌生人', predicted_prob
    return predicted_label[0], predicted_prob
