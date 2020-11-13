from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow_addons.layers import InstanceNormalization

#
from cycle.model_helper import downample
from cycle.model_helper import residual_block
from cycle.model_helper import upsample
from cycle.padding import ReflectionPadding2D

# Hyperparameters
from cycle.config import INPUT_IMG_SIZE

def get_resnet_generator(
    gamma_initializer,
    kernel_initializer,
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsampling_blocks=2,
    use_bias=False,
    name=None
):
    inputs = layers.Input(shape=INPUT_IMG_SIZE, name=f'{name}_img_input') # shape: (None, 256, 256, 3)
    
    # Early layers
    x = ReflectionPadding2D(padding=(3, 3))(inputs) # shape: (None, 258, 258, 3)
    x = layers.Conv2D(
        filters, 
        (7, 7), 
        kernel_initializer=kernel_initializer,
        use_bias=use_bias)(x) # shape: (None, 258, 258, filters)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x) # shape: (None, 258, 258, filters)
    x = layers.Activation('relu')(x) # shape: (None, 258, 258, filters)
    
    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downample(
            x,
            filters=filters, 
            activation=layers.Activation('relu'),
            use_bias=use_bias,
            gamma_initializer=gamma_initializer,
            kernel_initializer=kernel_initializer) # shape: (None, W/2, H/2, filters * 2)
        
    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(
            x, 
            layers.Activation('relu'), 
            gamma_initializer=gamma_initializer, 
            kernel_initializer=kernel_initializer,
            use_bias=use_bias) # shape: (None, W, H, filters)
        
    for _ in range(num_upsampling_blocks):
        filters //= 2
        x = upsample(
            x, 
            filters, 
            layers.Activation('relu'),
            kernel_initializer=kernel_initializer,
            gamma_initializer=gamma_initializer,
            use_bias=use_bias) # shape: (None, W * 2, H * 2, filters // 2)
        
    # Latest layers
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding='valid')(x)
    x = layers.Activation('tanh')(x)
    
    return Model(inputs, x, name=name)  