import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, confusion_matrix
import scipy.stats


# 加载 Semeion 数据集
def load_semeion_data(file_path):
    data = np.loadtxt(file_path)
    features = data[:, :256]  # 前256列为特征
    labels = np.argmax(data[:, 256:], axis=1)  # 后10列为one-hot编码的标签
    return features, labels


# 自实现 kNN 算法
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn(train_data, train_labels, test_data, k):
    distances = []
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_data)
        distances.append((dist, train_labels[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = [label for _, label in distances[:k]]

    most_common_label = Counter(k_nearest_neighbors).most_common(1)[0][0]
    return most_common_label


def loocv_knn(data, labels, k):
    correct_predictions = 0
    predicted_labels = []

    for i in range(len(data)):
        test_data = data[i]
        test_label = labels[i]

        train_data = np.delete(data, i, axis=0)
        train_labels = np.delete(labels, i)

        predicted_label = knn(train_data, train_labels, test_data, k)
        predicted_labels.append(predicted_label)

        if predicted_label == test_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(data)
    return accuracy, predicted_labels


# 加载数据
file_path = 'semeion.data'
features, labels = load_semeion_data(file_path)

# 使用自实现 kNN 算法，计算识别精度和预测标签
k = 5  # 示例中选一个k值，如5
accuracy, self_predicted_labels = loocv_knn(features, labels, k)
print(f'Self-implemented kNN (k={k}), Accuracy={accuracy:.4f}')

# 使用 scikit-learn 的 kNN 分类器进行对比
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(features, labels)
sklearn_predicted_labels = clf.predict(features)

# 计算 scikit-learn 的性能指标
sklearn_accuracy = accuracy_score(labels, sklearn_predicted_labels)
print(f'Scikit-learn kNN (k={k}), Accuracy={sklearn_accuracy:.4f}')

# 计算归一化互信息（NMI）
self_nmi = normalized_mutual_info_score(labels, self_predicted_labels)
sklearn_nmi = normalized_mutual_info_score(labels, sklearn_predicted_labels)
print(f'Self-implemented kNN (k={k}), NMI={self_nmi:.4f}')
print(f'Scikit-learn kNN (k={k}), NMI={sklearn_nmi:.4f}')

# 计算混淆熵（CEN），假设用归一化的互信息作为近似
self_conf_matrix = confusion_matrix(labels, self_predicted_labels)
self_cen = scipy.stats.entropy(self_conf_matrix.flatten())
sklearn_conf_matrix = confusion_matrix(labels, sklearn_predicted_labels)
sklearn_cen = scipy.stats.entropy(sklearn_conf_matrix.flatten())
print(f'Self-implemented kNN (k={k}), CEN={self_cen:.4f}')
print(f'Scikit-learn kNN (k={k}), CEN={sklearn_cen:.4f}')
