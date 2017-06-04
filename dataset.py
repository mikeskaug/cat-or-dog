import os
import numpy as np
import cv2
import math
import random
from collections import namedtuple

# Prepare the data for training and evaluating the CNN model.

ROOT = os.path.dirname(os.path.realpath(__file__))
SET_FRACTIONS = {'train': 0.6, 'validation': 0.2, 'test': 0.2}
CLASSES = ['cat', 'dog']
IMG_SIZE = 64


def get_data_sets(data_dir='data/train'):
    # The callable that will return sets of images to be used in training, validation and testing
    #
    # OUTPUT: a dictionary containing a Dataset object for each set.
    DataAttrs = namedtuple('DataAttrs', ['name', 'images', 'label'])
    classes = []
    label = 0
    full_path = os.path.join(ROOT, data_dir)
    for category in CLASSES:
        jpgs = [os.path.join(full_path, jpg) for jpg in os.listdir(full_path) if category in jpg]
        classes.append(DataAttrs(full_path, jpgs, label))
        label += 1

    # which class contains the fewest images and use this as the number of images sampled from each class
    # TODO: augment the data (flip, rotate, dilate, etc.)
    min_images = min([len(data_cls.images) for data_cls in classes])

    data_sets = {}
    start_idx = 0
    for set_name in SET_FRACTIONS.keys():
        image_set = []
        size = math.floor(min_images * SET_FRACTIONS[set_name])
        idxs = range(start_idx, start_idx + size)
        start_idx += size

        for data_cls in classes:
            images = [data_cls.images[i] for i in idxs]
            image_set += [(image, data_cls.label) for image in images]

        random.shuffle(image_set)
        data_sets[set_name] = Dataset(image_set)

    return data_sets


class Dataset:

    def __init__(self, data):
        self.batch_idx = 0
        self.data = data
        self.num_examples = len(data)

    def next_batch(self, batch_size):
        # return a new batch of images and labels
        if self.batch_idx + batch_size > self.num_examples:
            self.batch_idx = 0
            random.shuffle(self.data)

        image_array = np.empty(shape=(batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        label_array = np.empty(shape=(batch_size), dtype=np.int32)

        for i, j in zip(range(self.batch_idx, self.batch_idx + batch_size), range(batch_size)):
            # read the image file
            img = cv2.imread(self.data[i][0])
            # resize images
            resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            image_array[j, :, :, :] = np.array(resized)
            label_array[j] = self.data[i][1]
            self.batch_idx += 1

        return (image_array, label_array)
