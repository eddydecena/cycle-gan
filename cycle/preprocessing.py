import tensorflow as tf

# Hyperparameters
from cycle.config import ORIG_IMG_SIZE
from cycle.config import INPUT_IMG_SIZE

def normalize_img(img):
    img = tf.cast(img, tf.float32)
    return (img / 127.5) - 1.0

def preprocess_train_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [*ORIG_IMG_SIZE])
    img = tf.image.random_crop(img, [*INPUT_IMG_SIZE])
    return normalize_img(img)

def preprocess_test_image(img, label):
    img = tf.image.resize(img, [INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]])
    return normalize_img(img)