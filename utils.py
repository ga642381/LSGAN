import torch.nn as nn
import cv2
import os
import numpy as np

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
            
def load_training_data(root_dir):
    data_train = []
    for root, dirs, files in os.walk(root_dir, True):
        for file in files:
            if file.find('jpg'):
                image = cv2.imread(root_dir + "/" + file,1)
                image = cv2.resize(image, (65, 65), interpolation=cv2.INTER_CUBIC)
                RGB_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                data_train.append(RGB_im/255 * 2 - 1 )
    return data_train
