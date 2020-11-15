import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.data.experimental import AUTOTUNE

# Hyperparameters
from cycle.config import ORIG_IMG_SIZE
from cycle.config import INPUT_IMG_SIZE
from cycle.config import BUFFER_SIZE
from cycle.config import BATCH_SIZE

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