import numpy as np
import matplotlib.pyplot as plt
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import array_to_img

class GANMonitor(Callback):
    def __init__(self, data: Dataset, img_num: int = 4):
        super(GANMonitor, self).__init__()
        self.img_num: int = img_num
        self.data = data
    
    def on_epoch_end(self, epoch: int, logs=None):
        _, ax = plt.subplots(self.img_num, 2, figsize=(12, 12))
        for i, img in enumerate(self.data.take(self.img_num)):
            output = self.model.generator_G(img)[0]
            output = (output * 127.5 + 127.5).numpy().astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
            
            output = array_to_img(output)
            img = array_to_img(img)
            
            output.save(f'./examples/generated_img_{i}_{epoch}.png')
            img.save('./examples/original_img_{i}_{epoch}.png')
