from tensorflow.keras.layers import Input, Conv2D, concatenate
from tensorflow.keras.models import Model

def prep_and_hide_network(input_shape):

    input_message = Input(shape=input_shape)
    input_cover = Input(shape=input_shape)

    # Message branch
    x1 = Conv2D(50, (3,3), padding='same', activation='relu')(input_message)
    x2 = Conv2D(50, (4,4), padding='same', activation='relu')(input_message)
    x3 = Conv2D(50, (5,5), padding='same', activation='relu')(input_message)

    x = concatenate([x1, x2, x3])

    x = Conv2D(50, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(50, (4,4), padding='same', activation='relu')(x)
    x = Conv2D(50, (5,5), padding='same', activation='relu')(x)

    x = concatenate([input_cover, x])

    x = Conv2D(50, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(50, (4,4), padding='same', activation='relu')(x)
    x = Conv2D(50, (5,5), padding='same', activation='relu')(x)

    image_container = Conv2D(3, (3,3), padding='same', activation='relu')(x)

    model = Model(inputs=[input_message, input_cover], outputs=image_container)

    return model