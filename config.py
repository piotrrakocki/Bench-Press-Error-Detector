# config.py
NUM_FRAMES = 32
IMG_SIZE = (112, 112)
BATCH_SIZE = 6
EPOCHS = 50

ERROR_MAPPING = {
    "lokcie": 0,
    "biodra": 1,
    "nieDotykaKlatki": 2,
    "polRuchy": 3,
    "malpiChwyt": 4,
    "wyciskanieGilotynowe": 5,
}

VIDEO_DIR = "data/video"
FRAMES_BASE_DIR = "data/frames"
