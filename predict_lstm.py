import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from mediapipe.python.solutions.pose import Pose

# Load model and labels
model = load_model('shirobheda_lstm_model.h5')
labels = np.load('label_classes.npy')

def extract_all_keypoints(video_path):
    mp_pose = Pose(static_image_mode=True)
    keypoints = []
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_keypoints = []
            for lm in landmarks:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
            keypoints.append(frame_keypoints)

    cap.release()
    mp_pose.close()

    return keypoints  # List of [99] keypoints per frame

def predict_on_video(video_path, window_size=30, stride=30):
    all_keypoints = extract_all_keypoints(video_path)
    total_frames = len(all_keypoints)

    print(f"🔍 Total valid frames with keypoints: {total_frames}")

    segment_count = 0

    # Process full windows
    for start in range(0, total_frames - window_size + 1, stride):
        window = all_keypoints[start:start + window_size]
        window = np.array(window)

        if window.shape == (30, 99):
            sequence = np.expand_dims(window, axis=0)
            predictions = model.predict(sequence, verbose=0)
            predicted_label = labels[np.argmax(predictions)]
            print(f"Segment {segment_count + 1} [{start}-{start+window_size}]: {predicted_label}")
            segment_count += 1

    # Process remaining frames (if any)
    remainder = total_frames % stride
    if total_frames >= window_size and (total_frames - window_size) % stride != 0:
        start = total_frames - remainder
        last_window = all_keypoints[start:]
        last_window = pad_sequences([last_window], maxlen=window_size, dtype='float32', padding='post', truncating='post')[0]
        
        if last_window.shape == (30, 99):
            sequence = np.expand_dims(last_window, axis=0)
            predictions = model.predict(sequence, verbose=0)
            predicted_label = labels[np.argmax(predictions)]
            print(f"Segment {segment_count + 1} [final {remainder} frames]: {predicted_label}")
            segment_count += 1
    elif 0 < total_frames < window_size:
        # Handle case where entire video is shorter than 30 frames
        last_window = pad_sequences([all_keypoints], maxlen=window_size, dtype='float32', padding='post')[0]
        sequence = np.expand_dims(last_window, axis=0)
        predictions = model.predict(sequence, verbose=0)
        predicted_label = labels[np.argmax(predictions)]
        print(f"Segment {segment_count + 1} [entire video {total_frames} frames]: {predicted_label}")

# Path to video
video_path = r"C:\\shirobheda_videos\\testing.mp4"  # Replace with your actual video file path

# Run prediction
predict_on_video(video_path)
