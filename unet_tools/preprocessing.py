# U-net tools - Utilities for training U-nets
# Copyright (C) 2023  Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR a PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import skimage.transform as skt
import skimage.io as io
import numpy as np
import os


def resize(photos, labels, resolution):
    photos_resized = np.stack(
        [skt.resize(photo, resolution, anti_aliasing=True) for photo in photos]
    )
    labels_resized = np.stack(
        [skt.resize(label, resolution, anti_aliasing=True) for label in labels]
    )
    return photos_resized, labels_resized


def balance_weights(labels):
    labels_int = np.argmax(labels, axis=-1)
    class_prop = np.stack(
        [np.bincount(im.ravel()) for im in labels_int]
    )
    total_pixels = np.sum(class_prop, axis=0)
    class_weights = 1 / total_pixels
    class_weights = class_weights / np.sum(class_weights)
    return class_weights


def augment_data(photos, labels, n_rolls=5):
    """
    Data augmentation by horizontal flipping and sliding window.

    :param photos:
    :param labels:
    :param n_rolls:
    :return:
    """
    resolution = photos.shape[1:3]

    pixels_rolled = int(resolution[1] / n_rolls)
    aug_x, aug_y = [], []
    for tx, ty, in zip(photos, labels):
        flipped_x = np.flip(tx, axis=1)
        flipped_y = np.flip(ty, axis=1)

        aug_x.append(flipped_x)
        aug_y.append(flipped_y)

        for r in range(n_rolls):
            tx = np.roll(tx, pixels_rolled, axis=1)
            ty = np.roll(ty, pixels_rolled, axis=1)
            flipped_x = np.roll(flipped_x, pixels_rolled, axis=1)
            flipped_y = np.roll(flipped_y, pixels_rolled, axis=1)

            aug_x.append(tx)
            aug_y.append(ty)
            aug_x.append(flipped_x)
            aug_y.append(flipped_y)
    aug_x = np.stack(aug_x)
    aug_y = np.stack(aug_y)

    return aug_x, aug_y


def shuffle(photos, labels, seed=None):
    if seed is not None:
        np.random.seed(seed)

    order = np.random.choice(photos.shape[0], photos.shape[0], replace=False)
    photos = photos[order, :, :, :]
    labels = labels[order, :, :, :]

    return photos, labels


def compile_dataset(path, class_labels,
                    downscaling_factor=(10, 10)):
    downscaling_factor = tuple(list(downscaling_factor) + [1])

    photo_names = os.listdir(path)
    photo_names.sort()

    # main loop
    photos = []
    labels = []
    photo_names_complete = []
    for n, name in enumerate(photo_names):
        print("\rProcessing image %d of %d" % (n + 1, len(photo_names)),
              end="")
        
        name_length = len(name)

        photo_path = os.path.join(os.path.join(path, name), name + ".jpg")
        if os.path.exists(photo_path):
            photo = io.imread(photo_path) / 255
            photo_down = skt.downscale_local_mean(
                photo, downscaling_factor)
            photos.append(photo_down)

            cube = np.zeros(
                [photo_down.shape[0], photo_down.shape[1], len(class_labels)])

            label_files = os.listdir(os.path.join(path, name))
            for i, label in enumerate(class_labels):
                for filename in label_files:

                    # for PNG files the mask is extracted from the alpha channel
                    is_png = (".png" in filename) or (".PNG" in filename)
                    if (label in filename[name_length:]) and is_png:
                        label_path = os.path.join(path, name, filename)
                        photo = io.imread(label_path) / 255
                        photo_down = skt.downscale_local_mean(
                            photo, downscaling_factor)[:, :, 3]
                        photo_down = np.where(photo_down > 0,
                                              np.ones_like(photo_down),
                                              np.zeros_like(photo_down))
                        cube[:, :, i] = photo_down
                        break

                    # for JPG files the mask is extracted from the non-zero pixels
                    is_jpg = (".jpg" in filename) or (".JPG" in filename)
                    if (label in filename) and is_jpg:
                        label_path = os.path.join(path, name, filename)
                        photo = io.imread(label_path)
                        photo_down = skt.downscale_local_mean(
                            photo, downscaling_factor)
                        photo_down = np.mean(photo_down, axis=2)
                        photo_down = photo_down / (np.max(photo_down) + 1e-6)
                        cube[:, :, i] = photo_down
                        break

            # possibility of pixels with multiple labels
            cube = cube / (np.sum(cube, axis=2, keepdims=True) + 1e-6)

            # 'other' class
            cube = np.concatenate(
                [cube, 1 - np.sum(cube, axis=2, keepdims=True)],
                axis=2)

            labels.append(cube)
            photo_names_complete.append(name)

    photos = np.stack(photos)
    labels = np.stack(labels)

    return photos, labels


def load_photos(path, downscaling_factor=(10, 10)):
    downscaling_factor = tuple(list(downscaling_factor) + [1])

    photo_names = os.listdir(path)
    photo_names.sort()

    # main loop
    photos = []
    for n, name in enumerate(photo_names):
        print("\rProcessing image %d of %d" % (n + 1, len(photo_names)),
              end="")
        
        photo = io.imread(os.path.join(path, name))
        photo_down = skt.downscale_local_mean(
            photo, downscaling_factor) / 255
        photos.append(photo_down)

    photos = np.stack(photos)

    return photos
