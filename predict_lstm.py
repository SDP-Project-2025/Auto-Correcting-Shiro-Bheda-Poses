import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from mediapipe.python.solutions.pose import Pose
import cv2

# Load model and labels
model = load_model('shirobheda_lstm_model.h5')
labels = np.load('label_classes.npy')

def extract_keypoints_from_frames(folder_path):
    mp_pose = Pose(static_image_mode=True)
    keypoints = []
    
    frames = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    for frame_name in frames[:30]:  # Use first 30 frames
        img_path = os.path.join(folder_path, frame_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = mp_pose.process(img_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_keypoints = []
            for lm in landmarks:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
            keypoints.append(frame_keypoints)
    
    mp_pose.close()

    # Ensure we have 30 frames
    keypoints = pad_sequences([keypoints], maxlen=30, dtype='float32', padding='post', truncating='post')[0]
    return keypoints

# Predict on a folder of frames
folder = "predict_frames"
sequence = extract_keypoints_from_frames(folder)

if sequence.shape == (30, 99):
    sequence = np.expand_dims(sequence, axis=0)  # Shape: (1, 30, 99)
    predictions = model.predict(sequence)
    predicted_label = labels[np.argmax(predictions)]
    print(f"🔍 Predicted class: {predicted_label}")
else:
    print("❌ Invalid sequence shape. Make sure 30 valid frames with pose keypoints are present.")
