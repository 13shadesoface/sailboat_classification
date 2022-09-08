from PIL import Image, ImageOps
import numpy as np
import os
from tqdm import tqdm
from typing import *


def load_data(path: str, img_size: int, class_index: int):
    x_list, y_list = [], []
    for filename in tqdm(os.listdir(path)):
        img = ImageOps.grayscale(Image.open(path + "/" + filename).resize(img_size))
        array = np.asarray(img).flatten().tolist()

        x_list.append(array)
        class_array = [0, 0, 0]
        class_array[class_index] = 1
        y_list.append(class_array)

    X, Y = np.array(x_list), np.array(y_list)
    return X / 255, Y

def accuracy(X_train, Y_train):
    total = len(Y_train)

def experiment(x_train, y_train, x_validation, y_validation, \
               nb_layers, nb_per_layer, rate, nb_iter) -> loss_per_iteration: np.array

