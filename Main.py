from sklearn.model_selection import train_test_split
from Lib import load_file, data_classify, mfcc_heat_map, train_model, evaluation, prediction


def main():
    folders = load_file(True)
    print(folders)

    features, labels, label_encoder = data_classify(folders, False)

    # 数据划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 为CNN模型调整输入形状
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])  # 将特征维度转换为通道
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
    print(x_train.shape)
    print(x_test.shape)
    dataset = (x_train, x_test, y_train, y_test, labels)

    model = train_model(dataset)
    evaluation(model, dataset)
    predictions = prediction('./DataSet/Test/', label_encoder, model)
    print("识别结果为：" + predictions)
    # 测试
if __name__ == '__main__':
    main()
