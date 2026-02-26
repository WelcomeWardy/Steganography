import numpy as np
from models.stego_model import build_model
from configs.config import BATCH_SIZE, EPOCHS

def train():

    input_S = np.load("data/processed/input_S.npy")
    input_C = np.load("data/processed/input_C.npy")

    model = build_model()

    y_train = np.concatenate([input_S, input_C], axis=-1)

    model.fit([input_S, input_C],
              y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS)

    model.save("stego_model.h5")

if __name__ == "__main__":
    train()