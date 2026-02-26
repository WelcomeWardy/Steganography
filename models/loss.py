import tensorflow.keras.backend as K
from configs.config import BETA

def rev_loss(true, pred):
    loss = BETA * K.sum(K.square(true - pred))
    return loss

def full_loss(true, pred):

    message_true = true[..., 0:3]
    container_true = true[..., 3:6]

    message_pred = pred[..., 0:3]
    container_pred = pred[..., 3:6]

    message_loss = rev_loss(message_true, message_pred)
    container_loss = K.sum(K.square(container_true - container_pred))

    loss = message_loss + container_loss
    return loss