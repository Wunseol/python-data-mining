import os
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# 本项目通过卷积神经网络（CNN）对二分类问题进行建模、训练、评估并可视化结果。

# 1. 导入所需库
# h5py: 用于读取HDF5文件格式的数据。
# tensorflow 及其子模块 keras: 构建和训练深度学习模型，特别是卷积神经网络。
# sklearn.metrics: 应用于模型性能评估的多种指标。
# matplotlib.pyplot: 数据可视化，用于绘制ROC曲线。
# 2. 加载数据
# 使用h5py.File以只读模式打开指定路径的HDF5文件，分别加载训练集(X_train, y_train)和测试集(X_test, y_test)的数据。HDF5是一种高效存储大量数组数据的文件格式，常用于机器学习和数据科学项目中存储数据集。
# 3. 构建卷积神经网络模型
# Sequential模型: Keras中的一个线性堆叠模型，按顺序添加层。
# Conv2D: 添加两个卷积层，分别有32和64个卷积核，尺寸为(3, 3)，激活函数为ReLU，用于特征提取。
# MaxPooling2D: 在每个卷积层后添加最大池化层，尺寸为(2, 2)，用于降低数据维度，提取更鲁棒的特征。
# 第三个Conv2D层进一步提取特征。
# Flatten: 将前面的三维特征图展平为一维，以便输入到全连接层。
# Dense: 添加两个全连接层，第一个有64个节点，第二个有1个节点（因为是二分类问题，所以输出层通常只有一个节点，使用Sigmoid激活函数进行概率输出）。
# 4. 编译模型
# optimizer='adam': 选用Adam优化器进行权重更新。
# loss='binary_crossentropy': 因为是二分类任务，所以损失函数选择二元交叉熵。
# metrics=['accuracy']: 评估模型时关注的指标是准确率。
# 5. 数据预处理
# 将X_train和X_test的数据形状调整为适合CNN输入的四维张量，即(-1, 200, 768, 1)，其中-1表示样本数量可以动态调整，1表示单通道图像（灰度图）。
# 6. 训练模型
# 使用fit方法在训练数据上训练模型，指定训练轮数（epochs）为5，批次大小（batch_size）为64。
# 7. 模型评估
# 在测试集上进行预测，得到预测概率y_pred_prob，并通过阈值0.5将概率转换为类别标签y_pred。
# 计算并打印出模型的准确率、精确率、召回率和曲线下面积(AUC)。
# 8. 绘制ROC曲线
# 使用roc_curve函数计算ROC曲线所需的假正率(fpr)和真正率(tpr)，然后使用matplotlib绘制ROC曲线，并标注标题、坐标轴标签等。


# # 加载数据
with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_onehot.hdf5'), 'r') as train_file:
    X_train = train_file['X_train'][:]
    y_train = train_file['y_train'][:]

with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_onehot.hdf5'), 'r') as test_file:
    X_test = test_file['X_test'][:]
    y_test = test_file['y_test'][:]


# 构建卷积神经网络模型
model = models.Sequential([# Sequential模型
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 768, 1)),# 卷积层，用于检测图像中的局部特征。
    layers.MaxPooling2D((2, 2)),# 在每个卷积层后添加最大池化层
    layers.Conv2D(64, (3, 3), activation='relu'),# 进一步提取特征
    layers.MaxPooling2D((2, 2)),# 降低数据维度，提取更鲁棒特征
    layers.Conv2D(64, (3, 3), activation='relu'),# 进一步提取特征
    layers.Flatten(),# 将前面的三维特征图展平为一维
    layers.Dense(64, activation='relu'),# 添加两个全连接层
    layers.Dense(1, activation='sigmoid')# 输出层，使用Sigmoid激活函数进行概率输出
])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 将数据reshape为适合CNN的形状
X_train = X_train.reshape(-1, 200, 768, 1)
X_test = X_test.reshape(-1, 200, 768, 1)

# 训练模型
model.fit(X_train, y_train, epochs=5, batch_size=64)

# 在测试集上进行评估
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# 计算评价指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"AUC: {auc:.3f}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='CNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

print("Done!")
print("20211113492 陈文聪 \n")

