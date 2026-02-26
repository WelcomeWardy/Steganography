import numpy as np
import tensorflow as tf
from models.stego_model import build_model
from models.decoder import reveal_network
from models.loss import rev_loss
from training.lr_schedule import lr_schedule
from configs.config import IMAGE_SHAPE

# -----------------------------
# GPU CONFIGURATION
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Running on GPU")
else:
    print("Running on CPU")

# -----------------------------
# TRAINING FUNCTION
# -----------------------------

def train():

    input_S = np.load("data/processed/input_S.npy")
    input_C = np.load("data/processed/input_C.npy")

    m = input_S.shape[0]
    batch_size = 32
    epochs = 1000  # official paper

    deep_stegan = build_model()
    decoder = reveal_network(IMAGE_SHAPE)

    decoder.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=rev_loss
    )

    loss_history = []

    for epoch in range(epochs):

        idx = np.random.permutation(m)
        input_S = input_S[idx]
        input_C = input_C[idx]

        lr = lr_schedule(epoch)
        tf.keras.backend.set_value(deep_stegan.optimizer.learning_rate, lr)

        itera = m // batch_size

        f_loss_mean = 0
        r_loss_mean = 0

        for i in range(itera):

            batch_S = input_S[i*batch_size:(i+1)*batch_size]
            batch_C = input_C[i*batch_size:(i+1)*batch_size]

            y_true = np.concatenate([batch_S, batch_C], axis=-1)

            # Train full model
            f_loss = deep_stegan.train_on_batch(
                [batch_S, batch_C],
                y_true
            )

            # Predict container
            pred = deep_stegan.predict(
                [batch_S, batch_C],
                verbose=0
            )

            container_pred = pred[..., 3:6]

            # Train decoder
            r_loss = decoder.train_on_batch(
                container_pred,
                batch_S
            )

            f_loss_mean += f_loss
            r_loss_mean += r_loss

        f_loss_mean /= itera
        r_loss_mean /= itera

        loss_history.append((f_loss_mean, r_loss_mean))

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Full Loss: {f_loss_mean:.4f} | "
            f"Reveal Loss: {r_loss_mean:.4f} | "
            f"LR: {lr}"
        )

    np.save("loss_history.npy", loss_history)
    deep_stegan.save("deep_stegan.h5")
    decoder.save("decoder.h5")

    print("Training Complete")


if __name__ == "__main__":
    train()