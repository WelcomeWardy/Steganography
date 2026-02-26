"""
models/decoder.py
Reveal-Network:
  Accepts a steganographic container image (with optional Gaussian noise
  injection) and extracts / recovers the hidden secret message.
"""

from tensorflow.keras.layers import Input, Conv2D, concatenate, GaussianNoise
from tensorflow.keras.models import Model

from configs.config import IMAGE_SIZE


def build_decoder(input_shape: tuple = None, fixed: bool = False) -> Model:
    """
    Build and return the decoder (reveal) network.

    Parameters
    ----------
    input_shape : tuple, optional
        (H, W, C).  Defaults to IMAGE_SIZE from config.
    fixed : bool
        If True the injected Gaussian noise is kept constant during training
        (acts as a regulariser without adding randomness each step).

    Returns
    -------
    keras.Model  with input = reveal_input (steganographic image)
                      output= message      (recovered secret image)
    """
    if input_shape is None:
        input_shape = IMAGE_SIZE

    # ── Input ─────────────────────────────────────────────────────────────────
    reveal_input = Input(shape=input_shape, name="reveal_input")

    # ── Noise injection for robustness ────────────────────────────────────────
    # GaussianNoise is only active during training (Keras default behaviour).
    # Setting trainable=False on the whole model is what makes it "fixed".
    input_with_noise = GaussianNoise(0.01, name="gaussian_noise")(reveal_input)

    # ── First multi-scale feature extraction ─────────────────────────────────
    x1 = Conv2D(50, (3, 3), strides=(1, 1), padding="same",
                activation="relu", name="rev_conv3x3_a")(input_with_noise)
    x2 = Conv2D(10, (3, 3), strides=(1, 1), padding="same",
                activation="relu", name="rev_conv3x3_b")(input_with_noise)
    x3 = Conv2D( 5, (3, 3), strides=(1, 1), padding="same",
                activation="relu", name="rev_conv3x3_c")(input_with_noise)

    x  = concatenate([x1, x2, x3], name="rev_concat_a")

    # ── Second multi-scale extraction ─────────────────────────────────────────
    y1 = Conv2D(50, (3, 3), strides=(1, 1), padding="same",
                activation="relu", name="rev_conv3x3_d")(x)
    y2 = Conv2D(10, (3, 3), strides=(1, 1), padding="same",
                activation="relu", name="rev_conv3x3_e")(x)
    y3 = Conv2D( 5, (3, 3), strides=(1, 1), padding="same",
                activation="relu", name="rev_conv3x3_f")(x)

    x  = concatenate([y1, y2, y3], name="rev_concat_b")

    # ── Output layer: recovered message ───────────────────────────────────────
    message = Conv2D(3, (5, 5), strides=(1, 1), padding="same",
                     activation="relu", name="recovered_message")(x)

    # ── Build Keras model ─────────────────────────────────────────────────────
    reveal = Model(
        inputs = reveal_input,
        outputs= message,
        name   = "decoder_reveal_network",
    )
    return reveal