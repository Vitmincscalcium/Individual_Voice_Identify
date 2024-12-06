import pickle
from keras.src.saving import load_model
from sklearn.model_selection import train_test_split
from Lib import load_file, data_classify, train_model, evaluation, prediction
from Lib.Preprocess import remove_silence

# 该项目为2024学年西北工业大学机器学习的课程作业，如果学弟学妹想使用，请随意使用并完善该代码
# 当前仍有识别数据集外音频会硬套一个已有标签并高置信的问题，我们组的成员均非人工智能相关课题组的硕士生，希望优秀的大家指正
# This project is for the NWPU machine learning course in 2024. Anyone is welcome to use it and contribute by forking the repository.
# Currently, there is an issue with identifying voices outside of the dataset. None of our team members specialize in AI, so we hope others will help improve and upgrade this project.

def main():
    folders = load_file(True)
    print(folders)

    features, labels, label_encoder, median = data_classify(folders, False)

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
    with open('./Model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('./Model/median.pkl', 'wb') as f:
        pickle.dump(median, f)

def main_prediction(_confidence):
    # 测试
    model = load_model('./Model/model.keras')
    with open('./Model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('./Model/median.pkl', 'rb') as f:
        median = pickle.load(f)
    predictions = prediction('./DataSet/Predict/wph.wav', _confidence, label_encoder, model, median)
    print("识别结果为：" + str(predictions))

    predictions = prediction('./DataSet/Predict/Azuma_2.wav', _confidence, label_encoder, model, median)
    print("识别结果为：" + str(predictions))

    predictions = prediction('./DataSet/Predict/dingzhen_9.wav', _confidence, label_encoder, model, median)
    print("识别结果为：" + str(predictions))

    predictions = prediction('./DataSet/Predict/Diana_201.wav', _confidence, label_encoder, model, median)
    print("识别结果为：" + str(predictions))

if __name__ == '__main__':
    remove_silence('./DataSet/Predict/wph.wav', 30, None, True)
    main()
    main_prediction(.95)
