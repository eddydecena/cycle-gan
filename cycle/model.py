from tensorflow.keras import Model
from tensorflow.keras.losses import Loss
from tensorflow import GradientTape
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

class CycleGAN(Model):
    def __init__(
        self, 
        generator_G: Model, 
        generator_F: Model, 
        discriminator_X: Model, 
        discriminator_Y: Model, 
        lambda_cycle: int = 10.0,
        lambda_identity: int = 0.5) -> None:
        
        super(CycleGAN, self).__init__()
        self.generator_G = generator_G
        self.generator_F = generator_F
        self.discriminator_X = discriminator_X
        self.discriminator_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
    
    def compile(
        self,
        generator_G_opt: optimizer_v2.OptimizerV2,
        generator_F_opt: optimizer_v2.OptimizerV2,
        discriminator_X_opt: optimizer_v2.OptimizerV2,
        discriminator_Y_opt: optimizer_v2.OptimizerV2,
        generator_loss_fn,
        discriminator_loss_fn) -> None:
        
        super(CycleGAN, self).compile()
        self.generator_G_opt = generator_G_opt
        self.generator_F_opt = generator_F_opt
        self.discriminator_X_opt = discriminator_X_opt
        self.discriminator_Y_opt = discriminator_Y_opt
        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn
        self.cycle_loss_fn: Loss = MeanAbsoluteError()
        self.identity_loss_fn: Loss = MeanAbsoluteError()
    
    def train_step(self, batch_data) -> dict:
        real_x, real_y = batch_data
        
        with GradientTape(persistent=True) as tape:
            fake_y = self.generator_G(real_x, training=True) # x -> y
            fake_x = self.generator_F(real_y, training=True) # y -> x
            
            cycled_x = self.generator_F(fake_y, training=True) # x -> y -> x
            cycled_y = self.generator_G(fake_x, training=True) # y -> x -> y
            
            # Identity mapping
            same_x = self.generator_F(real_x, training=True) # x -> x
            same_y = self.generator_G(real_y, training=True) # y -> y
            
            # Discriminator output
            discriminator_real_x = self.discriminator_X(real_x, training=True)
            discriminator_fake_x = self.discriminator_X(fake_x, training=True)
            
            discriminator_real_y = self.discriminator_Y(real_y, training=True)
            discriminator_fake_y = self.discriminator_Y(fake_y, training=True)
            
            # Generator Adversarial Loss
            generator_F_loss = self.generator_loss_fn(discriminator_fake_x)
            generator_G_loss = self.generator_loss_fn(discriminator_fake_y)
            
            # Generator Cycle Loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle
            
            # Generator Identity Loss
            identity_loss_G = self.identity_loss_fn(real_y, same_y) * self.lambda_cycle * self.lambda_identity
            identity_loss_F = self.identity_loss_fn(real_x, same_x) * self.lambda_cycle * self.lambda_identity
            
            # Total Generator Loss
            total_loss_G = generator_G_loss + cycle_loss_G + identity_loss_G
            total_loss_F = generator_F_loss + cycle_loss_F + identity_loss_F
            
            # Discriminator loss
            discriminator_x_loss = self.discriminator_loss_fn(discriminator_real_x, discriminator_fake_x)
            discriminator_y_loss = self.discriminator_loss_fn(discriminator_real_y, discriminator_fake_y)
            
        # Get gradients
        gradients_G = tape.gradient(total_loss_G, self.generator_G.trainable_variables)
        gradients_F = tape.gradient(total_loss_F, self.generator_F.trainable_variables)
        
        gradients_disc_X = tape.gradient(discriminator_x_loss, self.discriminator_X.trainable_variables)
        gradients_disc_Y = tape.gradient(discriminator_y_loss, self.discriminator_Y.trainable_variables)
        
        # Update weights
        self.generator_G_opt.apply_gradients(
            zip(gradients_G, self.generator_G.trainable_variables)
        )
        
        self.generator_F_opt.apply_gradients(
            zip(gradients_F, self.generator_F.trainable_variables)
        )
        
        self.discriminator_X_opt.apply_gradients(
            zip(gradients_disc_X, self.discriminator_X.trainable_variables)
        )
        
        self.discriminator_Y_opt.apply_gradients(
            zip(gradients_disc_Y,  self.discriminator_Y.trainable_variables)
        )
        
        return {
            'G_loss': total_loss_G,
            'F_loss': total_loss_F,
            'D_X_loss': discriminator_x_loss,
            'D_Y_loss': discriminator_y_loss
        }