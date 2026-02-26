"""
models/encoder.py
Hide-Network (prep_and_hide_network):
  Accepts a secret message image and a cover image, fuses them using
  multi-scale convolutional layers and outputs a steganographic container.
"""

from tensorflow.keras.layers import Input, Conv2D, concatenate
from tensorflow.keras.models import Model

from configs.config import IMAGE_SIZE


def build_encoder(input_shape: tuple = None) -> Model:
    """
    Build and return the encoder (hiding) network.

    Parameters
    ----------
    input_shape : tuple, optional
        (H, W, C).  Defaults to IMAGE_SIZE from config.

    Returns
    -------
    keras.Model  with inputs=[input_message, input_cover]
                      output= image_container
    """
    if input_shape is None:
        input_shape = IMAGE_SIZE

    # ── Inputs ────────────────────────────────────────────────────────────────
    input_message = Input(shape=input_shape, name="input_message")
    input_cover   = Input(shape=input_shape, name="input_cover")

    # ── Prepare message  (multi-scale feature extraction) ────────────────────
    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding="same",
                activation="relu", name="msg_conv3x3")(input_message)
    x2 = Conv2D(10, (4, 4), strides=(1, 1), padding="same",
                activation="relu", name="msg_conv4x4")(input_message)
    x3 = Conv2D( 5, (5, 5), strides=(1, 1), padding="same",
                activation="relu", name="msg_conv5x5")(input_message)

    # Concatenate multi-scale message features
    x  = concatenate([x1, x2, x3], name="msg_concat")

    # ── Intermediate processing on merged features ────────────────────────────
    z1 = Conv2D(50, (3, 3), strides=(1, 1), padding="same",
                activation="relu", name="merged_conv3x3")(x)
    z2 = Conv2D(10, (4, 4), strides=(1, 1), padding="same",
                activation="relu", name="merged_conv4x4")(x)
    z3 = Conv2D( 5, (5, 5), strides=(1, 1), padding="same",
                activation="relu", name="merged_conv5x5")(x)

    x  = concatenate([z1, z2, z3], name="merged_concat")

    # ── Fuse processed message with cover image ───────────────────────────────
    x  = concatenate([x, input_cover], name="fuse_with_cover")

    # ── Output layer: steganographic container ────────────────────────────────
    image_container = Conv2D(3, (3, 3), strides=(1, 1), padding="same",
                             activation="relu", name="image_container")(x)

    # ── Build Keras model ─────────────────────────────────────────────────────
    encoder = Model(
        inputs =[input_message, input_cover],
        outputs= image_container,
        name   = "encoder_hide_network",
    )
    return encoder