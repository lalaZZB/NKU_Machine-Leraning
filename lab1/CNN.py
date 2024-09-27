import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import rotate


# 加载 Semeion 数据集
def load_semeion_data(file_path):
    data = np.loadtxt(file_path)
    features = data[:, :256].reshape(-1, 16, 16, 1)  # 前256列为特征，转换为16x16图像
    labels = np.argmax(data[:, 256:], axis=1)  # 后10列为one-hot编码的标签
    return features, labels


# 数据增强 - 旋转图像
def augment_data(features, labels):
    augmented_features = []
    augmented_labels = []

    for feature, label in zip(features, labels):
        augmented_features.append(feature)  # 原始图像
        augmented_labels.append(label)

        # 旋转图像（左上和左下）
        augmented_features.append(rotate(feature, angle=15, mode='wrap'))
        augmented_labels.append(label)

        augmented_features.append(rotate(feature, angle=-15, mode='wrap'))
        augmented_labels.append(label)

    return np.array(augmented_features), np.array(augmented_labels)


# 构建 CNN 模型
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # 10 类输出
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 加载和增强数据
file_path = 'semeion.data'  
features, labels = load_semeion_data(file_path)
augmented_features, augmented_labels = augment_data(features, labels)

# 将标签转换为 one-hot 编码
augmented_labels = to_categorical(augmented_labels, num_classes=10)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(augmented_features, augmented_labels, test_size=0.2,
                                                    random_state=42)

# 构建和训练 CNN 模型
input_shape = (16, 16, 1)
cnn_model = create_cnn_model(input_shape)
cnn_model.summary()
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.01)

# 在测试集上评估模型
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
print(f'CNN Model Test Accuracy: {test_accuracy:.4f}')
