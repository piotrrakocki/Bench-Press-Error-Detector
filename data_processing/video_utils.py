import os
import glob
import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, num_frames, img_size, error_mapping):
        self.num_frames = num_frames
        self.img_size = img_size
        self.error_mapping = error_mapping

    def extract_labels_from_filename(filename, error_mapping):
        basename = os.path.splitext(os.path.basename(filename))[0]
        parts = basename.split('_')
        labels = np.zeros(len(error_mapping), dtype=np.int32)

        if len(parts) > 1:
            for err in parts[1:]:
                if err in error_mapping:
                    labels[error_mapping[err]] = 1
                else:
                    print(f"Uwaga: Nieznany błąd '{err}' w nazwie pliku {filename}")
        return labels

    def save_video_frames_to_disk(self, video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Film {video_path} nie zawiera żadnych klatek.")
        if total_frames < self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int).tolist()
            while len(frame_indices) < self.num_frames:
                frame_indices.append(frame_indices[len(frame_indices) % total_frames])
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        os.makedirs(output_dir, exist_ok=True)
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.img_size)
            output_file = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_file, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cap.release()
        print(f"Klatki zapisane z filmu: {video_path} do folderu: {output_dir}")


    def load_frames_from_folder(self, folder_path):
        frame_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        frames = []
        for frame_file in frame_files[:self.num_frames]:
            img = cv2.imread(frame_file)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            frames.append(img)
        if len(frames) != self.num_frames:
            raise ValueError(f"Folder {folder_path} zawiera {len(frames)} klatek, oczekiwano {self.num_frames}.")
        return np.array(frames, dtype=np.float32) / 255.0
