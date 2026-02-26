"""
models/model.py
Loss functions and the combined deep-steganography model.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model

from configs.config import IMAGE_SIZE, BETA
from models.encoder import build_encoder
from models.decoder import build_decoder


# ─── Loss functions ───────────────────────────────────────────────────────────

def rev_loss(true: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """
    Reveal loss: scaled sum of squared pixel differences.
    Higher BETA = network prioritises recovering the secret more.
    """
    return BETA * tf.reduce_sum(tf.square(true - pred))


def full_loss(true: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """
    Full loss = message_loss + container_loss
    Channels 0:3 = message (secret), channels 3:6 = container (cover)
    """
    message_true,   container_true  = true[..., :3],  true[..., 3:6]
    message_pred,   container_pred  = pred[..., :3],  pred[..., 3:6]

    # Weight message loss higher so decoder is forced to learn
    message_loss   = 2.0 * rev_loss(message_true, message_pred)
    container_loss = tf.reduce_sum(tf.square(container_true - container_pred))

    return message_loss + container_loss


# ─── Combined model ───────────────────────────────────────────────────────────

def build_deep_steganography_model(input_shape: tuple = None):
    """
    Build and compile the full deep steganography model.

    KEY CHANGE vs original: decoder is NOT frozen inside the combined model.
    Both encoder and decoder train together end-to-end, which forces the
    decoder to actually learn to extract the hidden secret.
    """
    if input_shape is None:
        input_shape = IMAGE_SIZE

    encoder = build_encoder(input_shape)
    decoder = build_decoder(input_shape)

    # Compile decoder standalone for the separate decoder training step
    decoder.compile(optimizer="adam", loss=rev_loss)

    # ── Wire encoder → decoder ────────────────────────────────────────────────
    input_message   = Input(shape=input_shape, name="msg_input")
    input_container = Input(shape=input_shape, name="cov_input")

    output_container = encoder([input_message, input_container])
    output_message   = decoder(output_container)

    # Concatenate for full_loss (message first, container second)
    combined_output = concatenate(
        [output_message, output_container],
        axis=-1,
        name="combined_output",
    )

    # NOTE: decoder.trainable is TRUE here (not frozen)
    # This means the combined model trains BOTH encoder and decoder together
    deep_stegan = Model(
        inputs =[input_message, input_container],
        outputs= combined_output,
        name   = "deep_steganography",
    )
    deep_stegan.compile(optimizer="adam", loss=full_loss)

    return deep_stegan, encoder, decoder