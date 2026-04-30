"""
models/model.py
Loss functions and combined deep steganography model with JPEG awareness.

Combined model inputs:
  input_message   (H,W,3)  — secret image
  input_container (H,W,3)  — cover image
  jpeg_flag       (1,)     — 1.0 if JPEG applied, 0.0 if not

Combined model output:
  (H,W,6) — concat(recovered_message, output_container)
  used by full_loss during training
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model

from configs.config import IMAGE_SIZE, BETA
from models.encoder import build_encoder
from models.decoder import build_decoder


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def rev_loss(true: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """
    Reveal loss: scaled sum of squared pixel differences.
    BETA weights how much we penalise failed secret recovery.
    """
    return BETA * tf.reduce_sum(tf.square(true - pred))


def full_loss(true: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """
    Combined loss for the full steganography model.

    Tensor layout (6 channels):
      channels 0:3 = message  (secret image)
      channels 3:6 = container (encoded cover)

    message_loss   = 2 × rev_loss on secret channels
    container_loss = sum of squared differences on cover channels
    total          = message_loss + container_loss
    """
    message_true,   container_true  = true[..., :3],  true[..., 3:6]
    message_pred,   container_pred  = pred[..., :3],  pred[..., 3:6]

    message_loss   = 2.0 * rev_loss(message_true, message_pred)
    container_loss = tf.reduce_sum(tf.square(container_true - container_pred))

    return message_loss + container_loss


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_deep_steganography_model(input_shape: tuple = None):
    """
    Build and compile the full deep steganography model.

    Architecture
    ────────────
    input_message ──┐
                    ├──► encoder ──► container ──► [optional JPEG] ──► decoder ──► recovered_message
    input_cover   ──┘                                                       ↑
                                                               jpeg_flag ───┘

    Note: JPEG compression is NOT applied inside this Keras model graph.
    It is applied OUTSIDE in the training loop (numpy operation) because
    JPEG compression is not differentiable. The jpeg_flag tells the decoder
    whether compression was applied so it can adapt its extraction.

    Returns
    -------
    deep_stegan : compiled full model
    encoder     : hide network
    decoder     : reveal network (JPEG-aware)
    """
    if input_shape is None:
        input_shape = IMAGE_SIZE

    encoder = build_encoder(input_shape)
    decoder = build_decoder(input_shape)

    # Compile decoder standalone for separate training step
    decoder.compile(optimizer="adam", loss=rev_loss)

    # ── Wire inputs ───────────────────────────────────────────────────────────
    input_message   = Input(shape=input_shape, name="msg_input")
    input_container = Input(shape=input_shape, name="cov_input")
    jpeg_flag       = Input(shape=(1,),         name="jpeg_flag_input")

    # ── Forward pass ──────────────────────────────────────────────────────────
    output_container = encoder([input_message, input_container])
    output_message   = decoder([output_container, jpeg_flag])

    # ── Concatenate for full_loss ──────────────────────────────────────────────
    combined_output = concatenate(
        [output_message, output_container],
        axis=-1,
        name="combined_output"
    )

    deep_stegan = Model(
        inputs =[input_message, input_container, jpeg_flag],
        outputs= combined_output,
        name   = "deep_steganography_v2"
    )
    deep_stegan.compile(optimizer="adam", loss=full_loss)

    return deep_stegan, encoder, decoder