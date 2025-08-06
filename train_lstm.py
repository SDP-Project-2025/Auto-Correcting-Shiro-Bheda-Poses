import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

KEYPOINT_DIR = 'keypoints_data'

def load_sequences(split):
    X, y = [], []
    split_dir = os.path.join(KEYPOINT_DIR, split)
    for label in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, label)
        for file in os.listdir(class_dir):
            if file.endswith('.npy'):
                sequence = np.load(os.path.join(class_dir, file))
                if sequence.shape == (30, 99):  # ensure correct format
                    X.append(sequence)
                    y.append(label)
    return np.array(X), np.array(y)

# Load keypoint sequences
X_train, y_train = load_sequences('train')
X_test, y_test = load_sequences('test')

# Label encoding
le = LabelEncoder()
y_train_enc = to_categorical(le.fit_transform(y_train))
y_test_enc = to_categorical(le.transform(y_test))

# Model parameters
num_classes = y_train_enc.shape[1]
sequence_length = X_train.shape[1]
num_features = X_train.shape[2]

# LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, num_features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training the model
history = model.fit(
    X_train, y_train_enc,
    validation_data=(X_test, y_test_enc),
    epochs=30,
    batch_size=32
)

# Save model and labels
model.save('shirobheda_lstm_model.h5')
np.save('label_classes.npy', le.classes_)
print("Model and label classes saved.")

# Plot training vs validation accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
