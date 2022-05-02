import os
import cv2
import numpy as np
from PIL import Image

SIZE = 16


def load_set(dir, num_px):
    dataset_x = []
    dataset_y = []

    cat_images = os.listdir(f"{dir}/cats")

    for _, image_name in enumerate(cat_images):
        if image_name.endswith("jpg"):
            image = cv2.imread(f"{dir}/cats/" + image_name)
            image = Image.fromarray(image, "RGB")
            image = image.resize((num_px, num_px))

            dataset_x.append(np.array(image))
            dataset_y.append(0)

    dog_images = os.listdir(f"{dir}/dogs")

    for _, image_name in enumerate(dog_images):
        if image_name.endswith("jpg"):
            image = cv2.imread(f"{dir}/dogs/" + image_name)
            image = Image.fromarray(image, "RGB")
            image = image.resize((num_px, num_px))

            dataset_x.append(np.array(image))
            dataset_y.append(1)

    # (m_train, SIZE, SIZE, n_channels=3)
    set_x = np.array(dataset_x)

    # (m_train,)
    set_y = np.array(dataset_y)

    return set_x, set_y


dataset_dir = "datasets"

train_dir = f"{dataset_dir}/train_set"
test_dir = f"{dataset_dir}/test_set"


def load_dataset(num_px = SIZE):
    train_set_x, train_set_y = load_set(train_dir, num_px)
    test_set_x, test_set_y = load_set(test_dir, num_px)

    return (train_set_x, train_set_y), (test_set_x, test_set_y)
