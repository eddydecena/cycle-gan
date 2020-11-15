from typing import Tuple

# original image size for cycle_gan/horse2zebra dataset
ORIG_IMG_SIZE: Tuple[int] =(286, 286)

# Random Crop
INPUT_IMG_SIZE: Tuple[int] = (256, 256, 3)

BATCH_SIZE: int = 1
BUFFER_SIZE: int  = 256