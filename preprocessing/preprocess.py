import os
import cv2
import numpy as np
import random
from configs.config import *

def preprocess_data():

    train_path = os.path.join(DATASET_PATH, "train")
    
    x_train = np.empty((2000, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    
    a = 0
    class_folders = os.listdir(train_path)

    for i in range(NUM_CLASSES):
        class_name = class_folders[i]
        class_image_path = os.path.join(train_path, class_name, "images")
        
        image_files = os.listdir(class_image_path)
        
        # randomly select 10 images per class
        selected_ids = np.random.choice(len(image_files), IMAGES_PER_CLASS, replace=False)
        
        for idx in selected_ids:
            img_path = os.path.join(class_image_path, image_files[idx])
            image = cv2.imread(img_path)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            
            x_train[a] = image
            a += 1

    # Normalize
    x_train = x_train / 255.0

    # Split
    input_S = x_train[:1000]
    input_C = x_train[1000:]

    # Save
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    np.save(os.path.join(PROCESSED_PATH, "input_S.npy"), input_S)
    np.save(os.path.join(PROCESSED_PATH, "input_C.npy"), input_C)

    print("Preprocessing completed successfully.")
    print("Secret shape:", input_S.shape)
    print("Cover shape:", input_C.shape)

if __name__ == "__main__":
    preprocess_data()