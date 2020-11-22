import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.config.experimental import set_memory_growth
from tensorflow.config import list_physical_devices

# Failed to get convolution algorithm
physical_devices = list_physical_devices('GPU')
set_memory_growth(physical_devices[0], True)

# Hyperparameters
from cycle.config import ORIG_IMG_SIZE
from cycle.config import INPUT_IMG_SIZE
from cycle.config import BUFFER_SIZE
from cycle.config import BATCH_SIZE

from cycle import CycleGAN
from cycle.callbacks import GANMonitor
from cycle.loss import generator_loss_fn
from cycle.loss import discriminator_loss_fn
from cycle.generator import get_resnet_generator
from cycle.discriminator  import get_discriminator

from cycle.preprocessing import preprocess_train_image
from cycle.preprocessing import preprocess_test_image

datasets = tfds.load('cycle_gan/horse2zebra', as_supervised=True)

train_horses, train_zebras = datasets['trainA'], datasets['trainB']
test_horses, test_zebras = datasets['testA'], datasets['testB']

# Apply preprocessing

train_horses = (
    train_horses.map(preprocess_train_image, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

train_zebras = (
    train_zebras.map(preprocess_train_image, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)


test_horses = (
    test_horses.map(preprocess_test_image, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

test_zebras = (
    test_zebras.map(preprocess_test_image, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

_, ax_train = plt.subplots(4, 2, figsize=(10, 15))
for i, samples in enumerate(zip(train_horses.take(4), train_zebras.take(4))):
    horse = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    zebra = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    ax_train[i, 0].imshow(horse)
    ax_train[i, 1].imshow(zebra)
plt.show()

_, ax_test = plt.subplots(4, 2, figsize=(10, 15))
for i, samples in enumerate(zip(test_horses.take(4), test_zebras.take(4))):
    horse = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    zebra = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    ax_test[i, 0].imshow(horse)
    ax_test[i, 1].imshow(zebra)
plt.show()

# Putting all together
generator_G = get_resnet_generator(name='generator_G')
generator_F = get_resnet_generator(name='generator_F')

discriminator_X = get_discriminator(name='discriminator_X')
discriminator_Y = get_discriminator(name='discriminator_Y')

cycle_model = CycleGAN(
                generator_G=generator_G,
                generator_F=generator_F,
                discriminator_X=discriminator_X,
                discriminator_Y=discriminator_Y)

cycle_model.compile(
    generator_G_opt=Adam(learning_rate=2e-4, beta_1=0.5),
    generator_F_opt=Adam(learning_rate=2e-4, beta_1=0.5),
    discriminator_X_opt=Adam(learning_rate=2e-4, beta_1=0.5),
    discriminator_Y_opt=Adam(learning_rate=2e-4, beta_1=0.5),
    generator_loss_fn=generator_loss_fn,
    discriminator_loss_fn=discriminator_loss_fn)

plotter = GANMonitor(data=test_horses)
checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath
)

cycle_model.fit(
    Dataset.zip((train_horses, train_zebras)),
    epochs=90,
    callbacks=[plotter, model_checkpoint_callback],
)