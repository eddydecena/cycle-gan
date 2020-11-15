from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.initializers import Initializer

#
from cycle.model_helper import downample

# Hyperparameters
from cycle.config import INPUT_IMG_SIZE

def get_discriminator(
    filters: int = 64,
    num_downsampling: int = 3,
    kernel_initializer: Initializer = None,
    use_bias: bool = False,
    name: str = None) -> Model:
    
    inputs: layers.Input = layers.Input(shape=INPUT_IMG_SIZE, name = f'{name}_img_input')
    
    # Early layers
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer)(inputs)
    x = layers.LeakyReLU(0.2)(x)
    
    # Middle layers
    for num in range(num_downsampling):
        filters *= 2
        if num < 2:
            x = downample(
                x, 
                filters=filters, 
                activation=layers.LeakyReLU(0.2), 
                kernel_size=(4, 4),
                strides=(2, 2),
                kernel_initializer=kernel_initializer, 
                use_bias=use_bias)
        else:
            x = downample(
                x, 
                filters=filters, 
                activation=layers.LeakyReLU(0.2), 
                kernel_size=(4, 4),
                strides=(1, 1), 
                kernel_initializer=kernel_initializer, 
                use_bias=use_bias)
    
    x = layers.Conv2D(
        1, 
        (4, 4),
        strides=(1, 1),
        padding='same', 
        kernel_initializer=kernel_initializer)(x)
    
    return Model(inputs, x, name=name)