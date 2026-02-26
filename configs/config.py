# configs/config.py

DATASET_PATH = r"C:\Users\jange\Downloads\archive\tiny-imagenet-200"
PROCESSED_PATH = "data/processed"

IMAGE_SIZE = 64
NUM_CLASSES = 200
IMAGES_PER_CLASS = 10  # 200 x 10 = 2000

IMAGE_SHAPE = (64, 64, 3)
BETA = 1.0
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50