"""CNN文本分类GPU模块

使用卷积神经网络（CNN）对文本特征进行二分类，
支持GPU加速训练，包含Conv2D特征提取与全连接分类层。
"""
import os
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'train_onehot.hdf5'), 'r') as train_file:
        X_train = train_file['X_train'][:]
        y_train = train_file['y_train'][:]

    with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_onehot.hdf5'), 'r') as test_file:
        X_test = test_file['X_test'][:]
        y_test = test_file['y_test'][:]

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 768, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    X_train = X_train.reshape(-1, 200, 768, 1)
    X_test = X_test.reshape(-1, 200, 768, 1)

    history = model.fit(X_train, y_train, epochs=5, batch_size=64)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"AUC: {auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label='CNN')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()

    print("Done!")
