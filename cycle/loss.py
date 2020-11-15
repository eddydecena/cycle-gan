from tensorflow import Tensor
from tensorflow import ones_like
from tensorflow import zeros_like
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import MeanSquaredError

adv_loss_fn: Loss = MeanSquaredError()

def generator_loss_fn(fake: Tensor) -> Tensor:
    return adv_loss_fn(ones_like(fake), fake)

def discriminator_loss_fn(real: Tensor, fake: Tensor) -> Tensor:
    real_loss = adv_loss_fn(ones_like(real), real)
    fake_loss = adv_loss_fn(zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5