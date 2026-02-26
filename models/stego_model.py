from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from models.encoder import prep_and_hide_network
from models.decoder import reveal_network
from models.loss import full_loss, rev_loss
from configs.config import IMAGE_SHAPE, LEARNING_RATE

def build_model():

    prep_and_hide = prep_and_hide_network(IMAGE_SHAPE)
    reveal = reveal_network(IMAGE_SHAPE)

    reveal.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                   loss=rev_loss)

    reveal.trainable = False

    input_message = prep_and_hide.input[0]
    input_container = prep_and_hide.input[1]

    output_container = prep_and_hide([input_message, input_container])
    output_message = reveal(output_container)

    deep_stegan = Model(inputs=[input_message, input_container],
                        outputs=concatenate([output_message, output_container]))

    deep_stegan.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                        loss=full_loss)

    return deep_stegan