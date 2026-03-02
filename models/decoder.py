"""
models/decoder.py  —  Reveal Network
Updated to accept an optional jpeg_flag input so the decoder knows
which images underwent JPEG compression. This lets the decoder learn
different extraction strategies for compressed vs uncompressed containers.

Architecture:
  reveal_input  (64,64,3)  — the container image (possibly JPEG compressed)
  jpeg_flag     (1,)       — scalar: 1.0 if JPEG was applied, 0.0 if not

The jpeg_flag is tiled to (64,64,1) and concatenated with the input,
giving the decoder a per-pixel signal about compression status.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, concatenate, GaussianNoise, Lambda, Reshape
)
from tensorflow.keras.models import Model
from configs.config import IMAGE_SIZE


def build_decoder(input_shape: tuple = None, fixed: bool = False) -> Model:
    """
    Build the decoder (reveal) network with JPEG-awareness.

    Inputs
    ------
    reveal_input : (H, W, 3)  container image
    jpeg_flag    : (1,)       1.0 = JPEG compressed, 0.0 = clean

    Output
    ------
    recovered message : (H, W, 3)
    """
    if input_shape is None:
        input_shape = IMAGE_SIZE

    H, W, C = input_shape

    # ── Inputs ────────────────────────────────────────────────────────────────
    reveal_input = Input(shape=input_shape,  name="reveal_input")
    jpeg_flag    = Input(shape=(1,),          name="jpeg_flag")

    # ── Tile the flag to (H, W, 1) so it can be concatenated per-pixel ───────
    # This broadcasts the single scalar to every pixel location
    flag_tiled = Lambda(
        lambda f: tf.tile(
            tf.reshape(f, (-1, 1, 1, 1)),
            [1, H, W, 1]
        ),
        name="tile_jpeg_flag"
    )(jpeg_flag)

    # ── Concatenate container + flag → (H, W, 4) ─────────────────────────────
    combined_input = concatenate([reveal_input, flag_tiled],
                                  axis=-1, name="input_with_flag")

    # ── Noise injection ───────────────────────────────────────────────────────
    input_with_noise = GaussianNoise(0.01, name="gaussian_noise")(combined_input)

    # ── Block 1: first multi-scale extraction ─────────────────────────────────
    x1 = Conv2D(50, (3,3), strides=(1,1), padding="same", activation="relu",
                name="rev_conv3x3_a")(input_with_noise)
    x2 = Conv2D(10, (3,3), strides=(1,1), padding="same", activation="relu",
                name="rev_conv3x3_b")(input_with_noise)
    x3 = Conv2D( 5, (3,3), strides=(1,1), padding="same", activation="relu",
                name="rev_conv3x3_c")(input_with_noise)
    x  = concatenate([x1, x2, x3], name="rev_concat_a")

    # ── Block 2: deeper extraction ────────────────────────────────────────────
    y1 = Conv2D(50, (3,3), strides=(1,1), padding="same", activation="relu",
                name="rev_conv3x3_d")(x)
    y2 = Conv2D(10, (3,3), strides=(1,1), padding="same", activation="relu",
                name="rev_conv3x3_e")(x)
    y3 = Conv2D( 5, (3,3), strides=(1,1), padding="same", activation="relu",
                name="rev_conv3x3_f")(x)
    x  = concatenate([y1, y2, y3], name="rev_concat_b")

    # ── Output ────────────────────────────────────────────────────────────────
    message = Conv2D(3, (5,5), strides=(1,1), padding="same",
                     activation="relu", name="recovered_message")(x)

    reveal = Model(inputs=[reveal_input, jpeg_flag],
                   outputs=message,
                   name="decoder_reveal_network")
    return reveal