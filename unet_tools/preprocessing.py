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

# import skimage.transform as skt
import skimage.exposure as ske
import skimage.io as io
from skimage.util import view_as_windows
import numpy as np
import os
import pyexiv2
import pyproj
import cv2
# import warnings


# def resize(photos, labels, resolution):
#     photos_resized = np.stack(
#         [skt.resize(photo, resolution, anti_aliasing=True) for photo in photos]
#     )
#     labels_resized = np.stack(
#         [skt.resize(label, resolution, anti_aliasing=True) for label in labels]
#     )
#     return photos_resized, labels_resized


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
    width = resolution[1]

    pixels_rolled = int(resolution[1] / n_rolls)
    aug_x, aug_y = [], []
    for tx, ty, in zip(photos, labels):
        flipped_x = np.flip(tx, axis=1)
        flipped_y = np.flip(ty, axis=1)

        tx = np.concatenate([tx, flipped_x], axis=1)
        ty = np.concatenate([ty, flipped_y], axis=1)

        for r in range(n_rolls * 2):
            tx = np.roll(tx, pixels_rolled, axis=1)
            ty = np.roll(ty, pixels_rolled, axis=1)

            tx_gamma = ske.adjust_gamma(tx[:, :width, :], np.exp(np.random.uniform(-0.4, 0.4)))

            aug_x.append(tx_gamma)
            aug_y.append(ty[:, :width, :])
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


def compile_dataset(path, class_labels, resolution):
    # downscaling_factor = tuple(list(downscaling_factor) + [1])

    photo_names = os.listdir(path)
    photo_names.sort()

    # main loop
    photos = []
    labels = []
    coordinates = []
    photo_names_complete = []
    for n, name in enumerate(photo_names):
        print(f"\rProcessing image {n + 1} of {len(photo_names)}", end="")
        
        name_length = len(name)

        photo_path = os.path.join(os.path.join(path, name), name + ".jpg")
        if os.path.exists(photo_path):
            photo = io.imread(photo_path) #/ 255
            # photo_down = skt.downscale_local_mean(
            #     photo, downscaling_factor)
            photo_down = cv2.resize(photo, resolution, interpolation=cv2.INTER_AREA)
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

                        # with warnings.catch_warnings():
                        #     warnings.simplefilter("ignore")
                        #     photo = cv2.imread(label_path, flags=cv2.IMREAD_UNCHANGED) #/ 255
                        photo = io.imread(label_path)
                        # photo_down = skt.downscale_local_mean(
                        #     photo, downscaling_factor)[:, :, 3]
                        photo_down = cv2.resize(photo, resolution, interpolation=cv2.INTER_AREA)[:, :, 3]
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
                        # photo_down = skt.downscale_local_mean(
                        #     photo, downscaling_factor)
                        photo_down = cv2.resize(photo, resolution, interpolation=cv2.INTER_AREA)
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
            photo_names_complete.append(photo_path)

            # coordinates
            try:
                coordinates.append(get_coordinates(photo_path))
            except Exception:
                coordinates.append(None)

    photos = np.stack(photos)
    labels = np.stack(labels)
    coordinates = np.stack(coordinates)

    return photos, labels, coordinates, photo_names_complete


def load_photos(path, resolution):
    # downscaling_factor = tuple(list(downscaling_factor) + [1])

    photo_names = os.listdir(path)
    photo_names.sort()

    # main loop
    photos = []
    coordinates = []
    for n, name in enumerate(photo_names):
        print("\rProcessing image %d of %d" % (n + 1, len(photo_names)),
              end="")

        photo_path = os.path.join(path, name)
        photo = io.imread(photo_path)
        # photo_down = skt.downscale_local_mean(
        #     photo, downscaling_factor) / 255
        photo_down = cv2.resize(photo, resolution, interpolation=cv2.INTER_AREA)
        photos.append(photo_down)

        coordinates.append(get_coordinates(photo_path))

    photos = np.stack(photos)
    coordinates = np.stack(coordinates)

    return photos, coordinates


def get_coordinates(path):
    with pyexiv2.Image(path) as image:
        metadata = image.read_exif()

    lat = metadata['Exif.GPSInfo.GPSLatitude'].split(' ')
    lon = metadata['Exif.GPSInfo.GPSLongitude'].split(' ')
    lat_sign = 1 if metadata['Exif.GPSInfo.GPSLatitudeRef'] == 'N' else -1
    lon_sign = 1 if metadata['Exif.GPSInfo.GPSLongitudeRef'] == 'E' else -1
    alt = metadata['Exif.GPSInfo.GPSAltitude']
    alt_ref = metadata['Exif.GPSInfo.GPSAltitudeRef']

    lat_num = lat_sign * (eval(lat[0]) + eval(lat[1]) / 60 + eval(lat[2]) / 3600)
    lon_num = lon_sign * (eval(lon[0]) + eval(lon[1]) / 60 + eval(lon[2]) / 3600)
    alt_num = eval(alt) - eval(alt_ref)

    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon_num,
            south_lat_degree=lat_num,
            east_lon_degree=lon_num,
            north_lat_degree=lat_num,
        ),
    )
    utm_crs = pyproj.CRS.from_epsg(utm_crs_list[0].code)

    transformer = pyproj.Transformer.from_crs(4326, utm_crs, always_xy=True)

    utm_coords = transformer.transform(lon_num, lat_num)

    xyz = np.array(list(utm_coords) + [alt_num])

    return xyz


def feature_matching(image_1, image_2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image_1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image_2, None)

    index_params = dict(algorithm=1, trees=10)
    search_params = dict(checks=20)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

    if len(good_matches) > 4:
        good_points_1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        good_points_2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        _, inliers = cv2.findFundamentalMat(good_points_1, good_points_2, cv2.FM_RANSAC, ransacReprojThreshold=1)
        good_matches = [good_matches[i] for i in range(len(inliers)) if inliers[i]]

    table_1, table_2 = [], []
    for match in good_matches:
        point_1 = tuple(map(int, keypoints1[match.queryIdx].pt))
        point_2 = tuple(map(int, keypoints2[match.trainIdx].pt))

        table_1.append(np.array(point_1))
        table_2.append(np.array(point_2))
    table_1 = np.stack(table_1)
    table_2 = np.stack(table_2)

    return table_1.astype(int), table_2.astype(int)


def slice_image(image, window_width, window_height, step_w, step_h):
    if len(image.shape) == 3:
        window_shape = (window_height, window_width, image.shape[-1])
        step = (step_h, step_w, 1)
        sliced = view_as_windows(image, window_shape, step)[:, :, 0, :, :, :]
    else:
        window_shape = (window_height, window_width)
        step = (step_h, step_w)
        sliced = view_as_windows(image, window_shape, step)

    return sliced


def flatten_sliced_images(images):
    batch_size = np.prod(images.shape[:3]) # batch, grid_y, grid_x
    new_shape = [batch_size] + list(images.shape[3:])  # height, width, channels
    return np.reshape(images, new_shape)

def compile_sliced_dataset(path, class_labels, resolution,
                           window_width, window_height, step_w, step_h):
    photo_names = os.listdir(path)
    photo_names.sort()

    # main loop
    photos = []
    labels = []
    coordinates = []
    photo_names_complete = []
    for n, name in enumerate(photo_names):
        print(f"\rProcessing image {n + 1} of {len(photo_names)}", end="")

        name_length = len(name)

        photo_path = os.path.join(os.path.join(path, name), name + ".jpg")
        if os.path.exists(photo_path):
            photo = io.imread(photo_path)
            photo_sliced = slice_image(photo, window_width, window_height, step_w, step_h)
            photo_down = np.stack([
                np.stack([cv2.resize(im, resolution, interpolation=cv2.INTER_AREA) for im in row])
                for row in photo_sliced]
            )
            photos.append(photo_down)

            cube = np.zeros(list(photo_down.shape[:-1]) + [len(class_labels)])

            label_files = os.listdir(os.path.join(path, name))
            for i, label in enumerate(class_labels):
                for filename in label_files:

                    # for PNG files the mask is extracted from the alpha channel
                    is_png = (".png" in filename) or (".PNG" in filename)
                    if (label in filename[name_length:]) and is_png:
                        label_path = os.path.join(path, name, filename)

                        photo = io.imread(label_path)
                        photo_sliced = slice_image(photo, window_width, window_height, step_w, step_h)
                        photo_down = np.stack([
                            np.stack([cv2.resize(im, resolution, interpolation=cv2.INTER_AREA)[:, :, 3] for im in row])
                            for row in photo_sliced]
                        )
                        photo_down = np.where(photo_down > 0,
                                              np.ones_like(photo_down),
                                              np.zeros_like(photo_down))
                        cube[:, :, :, :, i] = photo_down
                        break

                    # for JPG files the mask is extracted from the non-zero pixels
                    is_jpg = (".jpg" in filename) or (".JPG" in filename)
                    if (label in filename) and is_jpg:
                        label_path = os.path.join(path, name, filename)
                        photo = io.imread(label_path)
                        photo_sliced = slice_image(photo, window_width, window_height, step_w, step_h)
                        photo_down = np.stack([
                            np.stack([cv2.resize(im, resolution, interpolation=cv2.INTER_AREA) for im in row])
                            for row in photo_sliced]
                        )
                        photo_down = np.mean(photo_down, axis=-1)
                        photo_down = photo_down / (np.max(photo_down) + 1e-6)
                        cube[:, :, :, :, i] = photo_down
                        break

            # possibility of pixels with multiple labels
            cube = cube / (np.sum(cube, axis=-1, keepdims=True) + 1e-6)

            # 'other' class
            cube = np.concatenate(
                [cube, 1 - np.sum(cube, axis=-1, keepdims=True)],
                axis=-1)

            labels.append(cube)
            photo_names_complete.append(photo_path)

            # coordinates
            try:
                coordinates.append(get_coordinates(photo_path))
            except Exception:
                coordinates.append(None)

    photos = np.stack(photos)
    labels = np.stack(labels)
    coordinates = np.stack(coordinates)

    return photos, labels, coordinates, photo_names_complete