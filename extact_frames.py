# save as extract_frames_from_video.py
import cv2
import os

video_path = 'C:\\shirobheda_videos\\test.mp4'
output_dir = 'predict_frames/some_new_sequence'

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
success = True

while success and frame_count < 30:
    success, frame = cap.read()
    if success:
        frame_path = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1

cap.release()
print(f"✅ Extracted {frame_count} frames to {output_dir}")
