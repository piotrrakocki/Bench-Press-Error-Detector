# data_augmentation/video_augmentation.py
import cv2
import numpy as np
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

class VideoAugmenter:
    def __init__(self, output_folder="augmented_videos"):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def change_brightness_contrast(frame, alpha=1.2, beta=30):
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    @staticmethod
    def add_noise(frame):
        noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
        return cv2.add(frame, noise)

    @staticmethod
    def rotate_frame(frame, angle=5):
        h, w = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        return cv2.warpAffine(frame, matrix, (w, h))

    @staticmethod
    def flip_frame(frame):
        return cv2.flip(frame, 1)

    @staticmethod
    def crop_zoom(frame, zoom_factor=1.2):
        h, w = frame.shape[:2]
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        start_x, start_y = (w - new_w) // 2, (h - new_h) // 2
        cropped = frame[start_y:start_y + new_h, start_x:start_x + new_w]
        return cv2.resize(cropped, (w, h))

    def augment_video(self, input_video):
        try:
            clip = VideoFileClip(input_video)
        except Exception as e:
            print(f"Nie udało się otworzyć pliku {input_video}: {e}")
            return

        filename = os.path.splitext(os.path.basename(input_video))[0]

        augmentations = {
            "bright": lambda f: self.change_brightness_contrast(f, 1.4, 40),
            "dark": lambda f: self.change_brightness_contrast(f, 0.8, -30),
            "rotate": lambda f: self.rotate_frame(f, angle=5),
            "flip": self.flip_frame,
            "noise": self.add_noise,
            "zoom": self.crop_zoom
        }

        for aug_name, aug_func in augmentations.items():
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                print(f"Nie udało się otworzyć wideo {input_video} dla augmentacji {aug_name}.")
                continue

            output_path = os.path.join(self.output_folder, f"{filename}_{aug_name}.mp4")
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = clip.fps

            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                augmented_frame = aug_func(frame)
                out.write(augmented_frame)

            cap.release()
            out.release()
            print(f"Zapisano: {output_path}")

    def process_input_path(self, input_path):
        if os.path.isdir(input_path):
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
            for file in os.listdir(input_path):
                if file.lower().endswith(video_extensions):
                    video_file = os.path.join(input_path, file)
                    print(f"Przetwarzanie wideo: {video_file}")
                    self.augment_video(video_file)
        elif os.path.isfile(input_path):
            print(f"Przetwarzanie wideo: {input_path}")
            self.augment_video(input_path)
        else:
            print(f"Ścieżka {input_path} nie istnieje.")
