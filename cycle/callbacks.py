import numpy as np
import matplotlib.pyplot as plt
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import CallBack
from tensorflow.keras.preprocessing import array_to_img

class GANMonitor(CallBack):
    def __init__(self, img_num: int):
        super(GANMonitor, self).__init__()
        self.img_num: int = img_num
    
    def on_epoch_end(self, epoch: int, test_horses: Dataset, logs=None):
        _, ax = plt.subplots(self.img_num, 2, figsize=(12, 12))
        for i, img in enumerate(test_horses.take(self.img_num)):
            output = self.model.gen_G(img)[0].numpy()
            output = (output * 127.5 + 127.5).numpy().astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)
            
            output = array_to_img(output)
            img = array_to_img(img)
            
            output.save(f'generated_img_{i}_{epoch}.png')
            img.save('original_img_{i}_{epoch}.png')
