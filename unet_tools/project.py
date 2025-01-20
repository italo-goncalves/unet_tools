# U-net tools - Utilities for training U-nets
# Copyright (C) 2024  Ítalo Gomes Gonçalves
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

import os

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from cmcrameri import cm
import tensorflow as tf
from sklearn.metrics import classification_report
import skimage.transform as skt
import skimage.io as io
import warnings

import unet_tools.preprocessing as pr
import unet_tools.utils as ut
import unet_tools.keras as k

from scipy.spatial.distance import cdist


class SegmentationProject:
    def __init__(self, image_width, image_height, labels, colors, seed=0):
        self.image_width = image_width
        self.image_height = image_height
        self.labels = tuple(labels)
        self.colors = tuple(colors)
        self.n_classes = len(self.labels)
        self.seed = seed
        # self.downscaling_factor = downscaling_factor
        self._resolution = None

        gcd = math.gcd(image_width, image_height)
        self.aspect_ratio = (image_width / gcd, image_height / gcd)

        self.photos = None
        self.masks = None
        self.coordinates = None
        self.photo_paths = None
        # self.photos_resized = None
        # self.masks_resized = None
        self.u_net = None
        self.class_weights = None
        self.split = None
        self.train_x = None
        self.train_y = None
        self.train_x_full = None
        self.train_y_full = None
        self.report = None
        self.photos_transfer = None
        self.sparse_labels_transfer = None

        self.train_history = []
        self.validation_history = []

        self.color_patches = [mpatches.Patch(color=c, label=l)
                              for c, l in zip(self.colors, self.labels)]
        self.color_patches.append(mpatches.Patch(color='#888888', label='Other'))
        self.numeric_cmap = ListedColormap(list(self.colors) + ['#888888'])
        self.cmap_entropy = cm.lajolla

    @property
    def resolution(self):
        return self._resolution

    def set_resolution(self, p):
        self._resolution = (self.aspect_ratio[0] * 2 ** p, self.aspect_ratio[1] * 2 ** p)

    def compile_dataset(self, path):
        photos, masks, coordinates, photo_paths = pr.compile_dataset(path, self.labels, self.resolution)
        self.photos = photos
        self.masks = masks
        self.coordinates = coordinates
        self.photo_paths = photo_paths

    def transfer_labels(self, path, distance_buffer=10.0):
        photo_names = os.listdir(path)
        photo_names.sort()

        # unlabeled_photos, unlabeled_coordinates = pr.load_photos(path, (self.image_width, self.image_height))

        w_factor = self.resolution[0] / self.image_width
        h_factor = self.resolution[1] / self.image_height

        # labels = np.argmax(self.masks, axis=3)

        # dist = cdist(unlabeled_coordinates, self.coordinates)

        new_labels = []
        new_photos = []
        for i, name in enumerate(photo_names):
            print(f"\nTransferring labels: image {i + 1} of {len(photo_names)}", end="")

            photo_i = cv2.imread(os.path.join(path, name))
            coords_i = pr.get_coordinates(os.path.join(path, name))

            dist = np.squeeze(cdist(coords_i[None, :], self.coordinates))

            idx = dist <= distance_buffer
            if np.any(idx):
                labels = np.argmax(self.masks[idx], axis=3)
                idx = np.where(idx)[0]
                close_photos = [self.photo_paths[j] for j in idx]
                label_i = np.full(self.resolution[::-1], -1)
                for j, file in enumerate(close_photos):
                    labeled_photo = cv2.imread(file)
                    # try:
                    tab_source, tab_dest = pr.feature_matching(labeled_photo, photo_i)
                    # print(f'\nPixels: {tab_source.shape[0]}', end='')
                    print('.', end='')

                    # downsizing
                    tab_source = np.floor(
                        np.stack([tab_source[:, 0] * h_factor, tab_source[:, 1] * w_factor], axis=1)
                    ).astype(int)
                    tab_dest = np.floor(
                        np.stack([tab_dest[:, 0] * h_factor, tab_dest[:, 1] * w_factor], axis=1)
                    ).astype(int)

                    # removing out of bounds pixels (last row or column)
                    keep = (tab_source[:, 0] < self.resolution[0]) & (tab_source[:, 1] < self.resolution[1])
                    tab_source = tab_source[keep]
                    tab_dest = tab_dest[keep]

                    keep = (tab_dest[:, 0] < self.resolution[0]) & (tab_dest[:, 1] < self.resolution[1])
                    tab_source = tab_source[keep]
                    tab_dest = tab_dest[keep]

                    label_i[tab_dest[:, 1], tab_dest[:, 0]] = labels[j][tab_source[:, 1], tab_source[:, 0]]
                    # except Exception:
                    #     pass

                if np.max(label_i) > -1:
                    new_labels.append(label_i)
                    new_photos.append(cv2.resize(photo_i, self.resolution, interpolation=cv2.INTER_AREA))

        self.photos_transfer = np.stack(new_photos)
        self.sparse_labels_transfer = np.stack(new_labels)
        print('')

    # def resize_dataset(self):
    #     self.photos_resized, self.masks_resized = pr.resize(self.photos, self.masks, self.resolution)
    #
    #     self.class_weights = pr.balance_weights(self.masks_resized)#.tolist()

    def train_test_split(self, train_perc=0.5):
        self.split = ut.train_test_label(
            self.photos.shape[0], train_perc=train_perc, seed=self.seed)

        self.train_x = self.photos[self.split == "train", :, :, :]
        self.train_y = self.masks[self.split == "train", :, :, :]

        # data augmentation
        self.train_x, self.train_y = pr.augment_data(
            self.train_x, self.train_y, n_rolls=5)

        self.train_x, self.train_y = pr.shuffle(self.train_x, self.train_y, seed=self.seed)

        self.train_x_full, self.train_y_full = pr.augment_data(
            self.photos, self.masks, n_rolls=5)

        self.train_x_full, self.train_y_full = pr.shuffle(self.train_x_full, self.train_y_full, seed=self.seed)

    def build_u_net(self, channels=32, blocks=4, block_depth=2, residual=False,
                    dropout_prob=0.1, filter_size=5):
        net_input = tf.keras.layers.Input(self.train_x.shape[1:])
        net_output = k.u_net(
            input_layer=net_input,
            output_size=self.n_classes + 1,
            channels=channels,
            blocks=blocks,
            residual=residual,
            block_depth=block_depth,
            dropout_prob=dropout_prob,
            filter_size=filter_size,
            end_activation="softmax")

        self.u_net = tf.keras.Model(inputs=net_input, outputs=net_output)
        self.u_net.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            # loss="categorical_crossentropy",
            # loss=tf.keras.losses.CategoricalFocalCrossentropy(), #alpha=self.class_weights),
            # loss=tf.keras.losses.CategoricalFocalCrossentropy(
            #     alpha=0.2 + 0.1 * self.class_weights / np.max(self.class_weights)
            # ),
            # loss_weights={net_output.name: [self.class_weights]},
            loss=tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=-1),
            metrics=['accuracy'])

    def train_u_net(self, batch_size=10, epochs=50, validation_split=0.1, full_data=False):
        x = self.train_x_full if full_data else self.train_x
        y = self.train_y_full if full_data else self.train_y
        y = np.argmax(y, axis=3)

        if self.photos_transfer is not None:
            x = np.concatenate([x, self.photos_transfer])
            y = np.concatenate([y, self.sparse_labels_transfer])

        x = x / 255.0

        self.u_net.fit(x, y,
                       batch_size=batch_size,
                       epochs=epochs,
                       # class_weight={i: val for i, val in enumerate(self.class_weights)},
                       validation_split=validation_split)
        self.train_history.extend(self.u_net.history.history["accuracy"])

        try:
            self.validation_history.extend(self.u_net.history.history["val_accuracy"])
        except KeyError:
            pass

    def plot_history(self):
        if self.u_net is None:
            return Exception('U-net not trained')
        else:
            fig = plt.figure()
            plt.plot(self.train_history)
            plt.plot(self.validation_history)
            return fig

    def validate(self, path, dpi=150):
        if self.u_net is None:
            return Exception('U-net not trained')
        else:
            pred_y = self.u_net.predict(self.photos / 255, batch_size=5)
            entropy = - np.sum(pred_y * np.log(pred_y + 1e-6), axis=-1)
            true_num = np.argmax(self.masks, axis=-1)
            pred_num = np.argmax(pred_y, axis=-1)

            # metrics
            self.report = classification_report(
                true_num[self.split == "test"].ravel(),
                pred_num[self.split == "test"].ravel() ,
                labels=np.arange(self.n_classes + 1),
                target_names=list(self.labels) + ["Other"],
                zero_division=0.0
            )
            print(self.report)

            # figures
            for line in range(self.masks.shape[0]):
                fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[12, 12])
                axes[0, 0].imshow(self.photos[line, :, :, :])
                axes[0, 0].imshow(np.argmax(self.masks, axis=-1)[line, :, :],
                                  alpha=0.5, cmap=self.numeric_cmap,
                                  vmin=0, vmax=self.n_classes)
                axes[0, 0].set_title("True")
                axes[0, 0].set_axis_off()

                axes[0, 1].imshow(self.photos[line, :, :, :])
                axes[0, 1].imshow(np.argmax(pred_y, axis=-1)[line, :, :],
                                  alpha=0.5, cmap=self.numeric_cmap,
                                  vmin=0, vmax=self.n_classes)
                axes[0, 1].set_title("Predicted (" + self.split[line] + ")")
                axes[0, 1].set_axis_off()

                axes[1, 0].set_aspect("equal")
                axes[1, 0].set_axis_off()
                axes[1, 0].legend(handles=self.color_patches, mode="expand",
                                  loc="upper left", frameon=False, fontsize=14)
                axes[1, 0].set_axis_off()

                axes[1, 1].imshow(self.photos[line, :, :, :])
                axes[1, 1].imshow(entropy[line, :, :],
                                  alpha=0.25, cmap=self.cmap_entropy,
                                  vmin=0, vmax=np.log(self.n_classes))
                axes[1, 1].set_title("Entropy")
                axes[1, 1].set_axis_off()

                plt.savefig(os.path.join(path, f'Prediction_{line}.jpg'),
                            bbox_inches='tight', dpi=dpi)

                plt.close(fig)

    def test(self, images_path, prediction_path, dpi=150):
        if self.u_net is None:
            return Exception('U-net not trained')
        else:
            # photos, masks = pr.compile_dataset(images_path, self.labels, [self.downscaling_factor] * 2)
            # photos, masks = pr.resize(photos, masks, self.resolution)

            photos, masks, _, _ = pr.compile_dataset(images_path, self.labels, self.resolution)

            pred_y = self.u_net.predict(photos / 255, batch_size=5)
            entropy = - np.sum(pred_y * np.log(pred_y + 1e-6), axis=-1)
            true_num = np.argmax(masks, axis=-1)
            pred_num = np.argmax(pred_y, axis=-1)

            # metrics
            report = classification_report(
                true_num.ravel(),
                pred_num.ravel() ,
                labels=np.arange(self.n_classes + 1),
                target_names=list(self.labels) + ["Other"],
                zero_division=0.0
            )
            with open(os.path.join(prediction_path, "report.txt"), "w") as file:
                file.write(report)

            # figures
            for line in range(masks.shape[0]):
                fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[12, 12])
                axes[0, 0].imshow(photos[line, :, :, :])
                axes[0, 0].imshow(np.argmax(masks, axis=-1)[line, :, :],
                                  alpha=0.5, cmap=self.numeric_cmap,
                                  vmin=0, vmax=self.n_classes)
                axes[0, 0].set_title("True")
                axes[0, 0].set_axis_off()

                axes[0, 1].imshow(photos[line, :, :, :])
                axes[0, 1].imshow(np.argmax(pred_y, axis=-1)[line, :, :],
                                  alpha=0.5, cmap=self.numeric_cmap,
                                  vmin=0, vmax=self.n_classes)
                axes[0, 1].set_title("Predicted")
                axes[0, 1].set_axis_off()

                axes[1, 0].set_aspect("equal")
                axes[1, 0].set_axis_off()
                axes[1, 0].legend(handles=self.color_patches, mode="expand",
                                  loc="upper left", frameon=False, fontsize=14)
                axes[1, 0].set_axis_off()

                axes[1, 1].imshow(photos[line, :, :, :])
                axes[1, 1].imshow(entropy[line, :, :],
                                  alpha=0.25, cmap=self.cmap_entropy,
                                  vmin=0, vmax=np.log(self.n_classes))
                axes[1, 1].set_title("Entropy")
                axes[1, 1].set_axis_off()

                plt.savefig(os.path.join(prediction_path, f'Prediction_{line}.jpg'),
                            bbox_inches='tight', dpi=dpi)

                plt.close(fig)

    def predict_masks(self, images_path, prediction_path, dpi=150):
        for label in (self.labels + ('Predictions',)):
            try:
                os.mkdir(os.path.join(prediction_path, label))
            except FileExistsError:
                continue

        # photos_prediction = pr.load_photos(
        #     path=images_path,
        #     downscaling_factor=[self.downscaling_factor]*2)


        # photos_prediction = np.stack(
        #     [skt.resize(photo, self.resolution, anti_aliasing=True) for photo in photos_prediction]
        # )

        photos_prediction, _ = pr.load_photos(images_path, self.resolution)

        pred_masks = self.u_net.predict(photos_prediction / 255, batch_size=5)
        entropy = - np.sum(pred_masks * np.log(pred_masks + 1e-6), axis=-1)

        photo_names = os.listdir(images_path)

        for line in range(photos_prediction.shape[0]):
            print(f'\rSaving predictions for image {line + 1} of '
                  f'{photos_prediction.shape[0]}: {photo_names[line]}          ',
                  end='')

            # mask_big = skt.resize(
            #     pred_masks[line],
            #     (self.image_height, self.image_width), #self.n_classes),
            #     anti_aliasing=True)
            mask_big = cv2.resize(pred_masks[line],
                                  (self.image_width, self.image_height),
                                  interpolation=cv2.INTER_AREA)
            mask_cat = np.argmax(mask_big, axis=-1)

            name, ext = photo_names[line].split('.')

            for i, label in enumerate(self.labels):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # io.imsave(os.path.join(prediction_path, label, f'{name}_mask.{ext}'),
                    #           np.round(mask_big[:, :, i], 0).astype(np.uint8) * 255)
                    io.imsave(os.path.join(prediction_path, label, f'{name}_mask.{ext}'),
                              (mask_cat == i).astype(np.uint8) * 255)

            # visualization of predictions
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,
                                     figsize=[12, 12])
            axes[0, 0].imshow(photos_prediction[line, :, :, :])
            axes[0, 0].set_title("Image")
            axes[0, 0].set_axis_off()

            axes[0, 1].imshow(photos_prediction[line, :, :, :])
            axes[0, 1].imshow(np.argmax(pred_masks, axis=-1)[line, :, :],
                              alpha=0.5, cmap=self.numeric_cmap,
                              vmin=0, vmax=self.n_classes)
            axes[0, 1].set_title("Predicted")
            axes[0, 1].set_axis_off()

            axes[1, 0].set_aspect("equal")
            axes[1, 0].set_axis_off()
            axes[1, 0].legend(handles=self.color_patches, mode="expand",
                              loc="upper left", frameon=False, fontsize=14)
            axes[1, 0].set_axis_off()

            axes[1, 1].imshow(photos_prediction[line, :, :, :])
            axes[1, 1].imshow(entropy[line, :, :],
                              alpha=0.25, cmap=self.cmap_entropy,
                              vmin=0, vmax=np.log(self.n_classes))
            axes[1, 1].set_title("Entropy")
            axes[1, 1].set_axis_off()

            plt.savefig(os.path.join(prediction_path, 'Predictions', name + '_prediction.' + ext),
                        bbox_inches='tight', dpi=dpi)

            plt.close(fig)

        print('')