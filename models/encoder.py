"""
models/encoder.py  —  Hide Network (unchanged from v1)
Multi-scale convolutional encoder that hides a secret image inside a cover image.
"""

from tensorflow.keras.layers import Input, Conv2D, concatenate
from tensorflow.keras.models import Model
from configs.config import IMAGE_SIZE


def build_encoder(input_shape: tuple = None) -> Model:
    if input_shape is None:
        input_shape = IMAGE_SIZE

    input_message = Input(shape=input_shape, name="input_message")
    input_cover   = Input(shape=input_shape, name="input_cover")

    # ── Block 1: multi-scale message processing ───────────────────────────────
    x1 = Conv2D(50, (3,3), strides=(1,1), padding="same", activation="relu",
                name="msg_conv3x3")(input_message)
    x2 = Conv2D(10, (4,4), strides=(1,1), padding="same", activation="relu",
                name="msg_conv4x4")(input_message)
    x3 = Conv2D( 5, (5,5), strides=(1,1), padding="same", activation="relu",
                name="msg_conv5x5")(input_message)
    x  = concatenate([x1, x2, x3], name="msg_concat")

    # ── Block 2: deeper processing ────────────────────────────────────────────
    z1 = Conv2D(50, (3,3), strides=(1,1), padding="same", activation="relu",
                name="merged_conv3x3")(x)
    z2 = Conv2D(10, (4,4), strides=(1,1), padding="same", activation="relu",
                name="merged_conv4x4")(x)
    z3 = Conv2D( 5, (5,5), strides=(1,1), padding="same", activation="relu",
                name="merged_conv5x5")(x)
    x  = concatenate([z1, z2, z3], name="merged_concat")

    # ── Fuse with cover ────────────────────────────────────────────────────────
    x  = concatenate([x, input_cover], name="fuse_with_cover")

    # ── Output ────────────────────────────────────────────────────────────────
    image_container = Conv2D(3, (3,3), strides=(1,1), padding="same",
                             activation="relu", name="image_container")(x)

    encoder = Model(inputs=[input_message, input_cover],
                    outputs=image_container,
                    name="encoder_hide_network")
    return encoder