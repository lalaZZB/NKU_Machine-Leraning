import numpy as np
from collections import Counter


# 读取 Semeion 数据集
def load_semeion_data(file_path):
    data = np.loadtxt(file_path)
    features = data[:, :256]  # 前256列为特征
    labels = np.argmax(data[:, 256:], axis=1)  # 后10列为one-hot编码的标签
    return features, labels


# 计算欧氏距离
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# kNN算法实现
def knn(train_data, train_labels, test_data, k):
    distances = []
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_data)
        distances.append((dist, train_labels[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = [label for _, label in distances[:k]]

    most_common_label = Counter(k_nearest_neighbors).most_common(1)[0][0]
    return most_common_label


# 留一法实现
def loocv_knn(data, labels, k):
    correct_predictions = 0

    for i in range(len(data)):
        test_data = data[i]
        test_label = labels[i]

        train_data = np.delete(data, i, axis=0)
        train_labels = np.delete(labels, i)

        predicted_label = knn(train_data, train_labels, test_data, k)

        if predicted_label == test_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(data)
    return accuracy


# 加载数据
file_path = 'semeion.data'
features, labels = load_semeion_data(file_path)

# 计算不同k值下的识别精度
k_values = [5, 9, 13]
for k in k_values:
    accuracy = loocv_knn(features, labels, k)
    print(f'k={k}, Accuracy={accuracy:.4f}')
