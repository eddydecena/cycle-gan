from typing import Tuple

from tensorflow.keras import layers
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.initializers import Initializer

#
from cycle.padding import ReflectionPadding2D

def residual_block(
    x: layers.Layer,
    activation: layers.Activation,
    kernel_size: Tuple[int] = (3, 3),
    strides: Tuple[int] = (1, 1),
    padding: str = 'valid',
    kernel_initializer: Initializer = None,
    gamma_initializer: Initializer = None,
    use_bias: bool = False) -> layers.Layer:
    
    dim: int = x.shape[-1]
    input_tensor: layers.Layer = x
    
    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
            dim, 
            kernel_size, 
            strides=strides, 
            kernel_initializer=kernel_initializer, 
            padding=padding, 
            use_bias=use_bias)(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)
    
    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
            dim, 
            kernel_size, 
            strides=strides, 
            kernel_initializer=kernel_initializer, 
            padding=padding, 
            use_bias=use_bias)(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    
    return layers.add([input_tensor, x])

def downample(
    x: layers.Layer,
    filters: int,
    activation: layers.Activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding='same',
    kernel_initializer: Initializer = None,
    gamma_initializer: Initializer = None,
    use_bias: bool = False) -> layers.Layer:
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias)(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    
    if activation:
        x = activation(x)
        
    return x

def upsample(
    x: layers.Layer,
    filters: int,
    activation: layers.Activation,
    kernel_size: Tuple[int] = (3, 3),
    strides: Tuple[int] = (2, 2),
    padding: str = 'same',
    kernel_initializer: Initializer = None,
    gamma_initializer: Initializer = None,
    use_bias: bool = False) -> layers.Layer:
    x = layers.Conv2DTranspose(
        filters,
        kernel_size, 
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias)(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    
    if activation:
        x = activation(x)
        
    return x