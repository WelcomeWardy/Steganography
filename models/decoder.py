from tensorflow.keras.layers import Input, Conv2D, concatenate, GaussianNoise
from tensorflow.keras.models import Model

def reveal_network(input_shape):

    reveal_input = Input(shape=input_shape)

    input_with_noise = GaussianNoise(0.01)(reveal_input)

    x1 = Conv2D(50, (3,3), padding='same', activation='relu')(input_with_noise)
    x2 = Conv2D(50, (4,4), padding='same', activation='relu')(input_with_noise)
    x3 = Conv2D(50, (5,5), padding='same', activation='relu')(input_with_noise)

    x = concatenate([x1, x2, x3])

    x = Conv2D(50, (3,3), padding='same', activation='relu')(x)
    x = Conv2D(50, (4,4), padding='same', activation='relu')(x)
    x = Conv2D(50, (5,5), padding='same', activation='relu')(x)

    message = Conv2D(3, (3,3), padding='same', activation='relu')(x)

    model = Model(inputs=reveal_input, outputs=message)

    return model