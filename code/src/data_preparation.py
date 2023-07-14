import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_data(cover_images_path, secret_images_path, image_shape=(64, 64)):
    cover_train_paths = glob.glob(cover_images_path+'*')
    secret_train_paths = glob.glob(secret_images_path+'*')

    cover_images = []
    secret_images = []

    for cover, sc in zip(sorted(cover_train_paths), sorted(secret_train_paths)):
        cov = Image.open(cover).convert('RGB').resize(image_shape)
        secret = Image.open(sc).convert('RGB').resize(image_shape)
        cover_images.append(cov)
        secret_images.append(secret)

    return cover_images, secret_images


def normalize_images(images):
    return np.array([np.array(img) / 255.0 for img in images])

def split_data(cover_images, secret_images):
    cover_images = normalize_images(cover_images)
    secret_images = normalize_images(secret_images)

    X_cover_train, X_cover_val, X_secret_train, X_secret_val = train_test_split(cover_images, secret_images, test_size=0.2, random_state=42)

    return X_cover_train, X_cover_val, X_secret_train, X_secret_val

def create_dataset(X_cover, X_secret, batch_size, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(((X_cover, X_secret), (X_cover, X_secret)))
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
