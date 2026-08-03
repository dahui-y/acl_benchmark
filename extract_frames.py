import cv2
from pathlib import Path
import numpy as np

def extract_frames(video_path: Path, output_folder: Path, num_frames=20):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    indices_set = set(indices)

    i = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in indices_set:
            frame_filename = f"frame_{i+1:03d}.jpg"
            cv2.imwrite(str(output_folder / frame_filename), frame)
            i += 1

        frame_id += 1

    cap.release()

def main():
    video_path = Path('/Users/hanxianjing/proj/Video benchmark/video_gen')
    output_path = Path('/Users/hanxianjing/proj/Video benchmark/selected_frames')

    output_path.mkdir(parents=True, exist_ok=True)

    for video_file in video_path.glob('*.mp4'):
        video_name = video_file.stem
        video_output_folder = output_path / video_name

        video_output_folder.mkdir(parents=True, exist_ok=True)

        print(f"Extracting frames from: {video_file.name}")

        extract_frames(video_file, video_output_folder, num_frames=20)

    print("All frames extracted successfully!")

if __name__ == "__main__":
    main()
