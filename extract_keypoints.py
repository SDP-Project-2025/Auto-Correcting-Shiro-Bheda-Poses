import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

DATASET_DIR = 'dataset'
SAVE_DIR = 'keypoints_data'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_keypoints_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return keypoints
    else:
        return None

SEQUENCE_LENGTH = 30

for split in ['train', 'test']:
    for label in os.listdir(os.path.join(DATASET_DIR, split)):
        class_dir = os.path.join(DATASET_DIR, split, label)
        save_class_dir = os.path.join(SAVE_DIR, split, label)
        os.makedirs(save_class_dir, exist_ok=True)

        frames = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Chunk frames into sequences of 30
        for i in range(0, len(frames) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            keypoints_sequence = []
            for j in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(class_dir, frames[i + j])
                keypoints = extract_keypoints_from_image(frame_path)
                if keypoints:
                    keypoints_sequence.append(keypoints)

            if len(keypoints_sequence) == SEQUENCE_LENGTH:
                filename = f"{label}_{i//SEQUENCE_LENGTH}.npy"
                save_path = os.path.join(save_class_dir, filename)
                np.save(save_path, keypoints_sequence)
