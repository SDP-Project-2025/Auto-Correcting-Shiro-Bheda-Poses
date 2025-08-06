# prepare_dataset.py

import os
import numpy as np

DATA_DIR = 'keypoints_data'
OUTPUTS = ['X_train.npy', 'y_train.npy', 'X_test.npy', 'y_test.npy']

def load_keypoints(data_type):
    X, y = [], []
    class_names = sorted(os.listdir(os.path.join(DATA_DIR, data_type)))
    label_map = {label: idx for idx, label in enumerate(class_names)}

    for label in class_names:
        folder = os.path.join(DATA_DIR, data_type, label)
        for file in os.listdir(folder):
            if file.endswith('.npy'):
                filepath = os.path.join(folder, file)
                keypoints = np.load(filepath)
                X.append(keypoints)
                y.append(label_map[label])

    return np.array(X), np.array(y), label_map

# Load and save
X_train, y_train, label_map = load_keypoints('train')[:3]
X_test, y_test, _ = load_keypoints('test')

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
np.save('label_classes.npy', label_map)

print("✅ Dataset saved: X_train.npy, y_train.npy, X_test.npy, y_test.npy, label_classes.npy")
