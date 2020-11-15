from typing import Any

from tensorflow import Tensor
from tensorflow import cast
from tensorflow import float32
from tensorflow.image import random_flip_left_right
from tensorflow.image import resize
from tensorflow.image import random_crop

# Hyperparameters
from cycle.config import ORIG_IMG_SIZE
from cycle.config import INPUT_IMG_SIZE

def normalize_img(img: Tensor) -> Tensor:
    img = cast(img, float32)
    return (img / 127.5) - 1.0

def preprocess_train_image(img: Tensor, label: Any) -> Tensor:
    img = random_flip_left_right(img)
    img = resize(img, [*ORIG_IMG_SIZE])
    img = random_crop(img, [*INPUT_IMG_SIZE])
    return normalize_img(img)

def preprocess_test_image(img: Tensor, label: Any) -> Tensor:
    img = resize(img, [INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]])
    return normalize_img(img)