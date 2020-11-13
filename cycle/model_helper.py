from tensorflow.keras import layers
from tensorflow_addons.layers import InstanceNormalization

#
from cycle.padding import ReflectionPadding2D

def residual_block(
    x,
    activation,
    kernel_initializer,
    gamma_initializer,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='valid',
    use_bias=False
):
    dim =- x.shape[-1]
    input_tensor = x
    
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
    x,
    filters,
    activation,
    kernel_initializer,
    gamma_initializer,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding='same',
    use_bias=False
):
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
    x,
    filters,
    activation,
    kernel_initializer,
    gamma_initializer,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding='same',
    use_bias=False
):
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