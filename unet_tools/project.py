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

import unet_tools.preprocessing as pr
import unet_tools.utils as ut
import unet_tools.keras as k


class SegmentationProject:
    def __init__(self, image_width, image_height, labels, colors, seed=0,
                 downscaling_factor=10):
        self.image_width = image_width
        self.image_height = image_height
        self.labels = tuple(labels)
        self.colors = tuple(colors)
        self.n_classes = len(self.labels)
        self.seed = seed
        self.downscaling_factor = downscaling_factor
        self._resolution = None

        gcd = math.gcd(image_width, image_height)
        self.aspect_ratio = (image_width / gcd, image_height / gcd)

        self.photos = None
        self.masks = None
        self.photos_resized = None
        self.masks_resized = None
        self.u_net = None
        self.class_weights = None
        self.split = None
        self.train_x = None
        self.train_y = None
        self.report = None

        self.color_patches = [mpatches.Patch(color=c, label=l)
                              for c, l in zip(self.colors, self.labels)]
        self.color_patches.append(mpatches.Patch(color='#888888', label='Other'))
        self.numeric_cmap = ListedColormap(list(self.colors) + ['#888888'])
        self.cmap_entropy = cm.lajolla

    @property
    def resolution(self):
        return self._resolution

    def set_resolution(self, p):
        self._resolution = (self.aspect_ratio[1] * 2 ** p, self.aspect_ratio[0] * 2 ** p)

    def compile_dataset(self, path):
        photos, masks = pr.compile_dataset(path, self.labels, [self.downscaling_factor]*2)
        self.photos = photos
        self.masks = masks

    def resize_dataset(self):
        self.photos_resized, self.masks_resized = pr.resize(self.photos, self.masks, self.resolution)

        self.class_weights = pr.balance_weights(self.masks_resized)

    def train_test_split(self, train_perc=0.5):
        self.split = ut.train_test_label(
            self.photos_resized.shape[0], train_perc=train_perc, seed=self.seed)

        self.train_x = self.photos_resized[self.split == "train", :, :, :]
        self.train_y = self.masks_resized[self.split == "train", :, :, :]

        # data augmentation
        self.train_x, self.train_y = pr.augment_data(
            self.train_x, self.train_y, n_rolls=5)

        self.train_x, self.train_y = pr.shuffle(self.train_x, self.train_y, seed=self.seed)

        self.train_x_full, self.train_y_full = pr.augment_data(
            self.photos_resized, self.masks_resized, n_rolls=5)

        self.train_x_full, self.train_y_full = pr.shuffle(self.train_x_full, self.train_y_full, seed=self.seed)

    def build_u_net(self, channels=32, blocks=4, residual=False):
        net_input = tf.keras.layers.Input(self.train_x.shape[1:])
        net_output = k.u_net(
            input_layer=net_input,
            output_size=self.n_classes + 1,
            channels=channels,
            blocks=blocks,
            residual=residual,
            end_activation="softmax")

        self.u_net = tf.keras.Model(inputs=net_input, outputs=net_output)
        self.u_net.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="categorical_crossentropy",
            loss_weights=self.class_weights,
            metrics=['accuracy'])

    def train_u_net(self, batch_size=10, epochs=50, validation_split=0.1, full_data=False):
        self.u_net.fit(self.train_x_full if full_data else self.train_x,
                       self.train_y_full if full_data else self.train_y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=validation_split)

    def plot_history(self):
        if self.u_net is None:
            return Exception('U-net not trained')
        else:
            fig = plt.figure()
            plt.plot(self.u_net.history.history["accuracy"])
            plt.plot(self.u_net.history.history["val_accuracy"])
            return fig

    def validate(self, path, dpi=150):
        if self.u_net is None:
            return Exception('U-net not trained')
        else:
            pred_y = self.u_net.predict(self.photos_resized, batch_size=5)
            entropy = - np.sum(pred_y * np.log(pred_y + 1e-6), axis=-1)
            true_num = np.argmax(self.masks_resized, axis=-1)
            pred_num = np.argmax(pred_y, axis=-1)

            # metrics
            self.report = classification_report(
                true_num[self.split == "test"].ravel() + 1,
                pred_num[self.split == "test"].ravel() + 1,
                labels=np.arange(self.n_classes) + 1,
                target_names=list(self.labels) + ["Other"])
            print(self.report)

            # figures
            for line in range(self.masks.shape[0]):
                fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[12, 12])
                axes[0, 0].imshow(self.photos_resized[line, :, :, :])
                axes[0, 0].imshow(np.argmax(self.masks_resized, axis=-1)[line, :, :],
                                  alpha=0.5, cmap=self.numeric_cmap,
                                  vmin=0, vmax=self.n_classes - 1)
                axes[0, 0].set_title("True")
                axes[0, 0].set_axis_off()

                axes[0, 1].imshow(self.photos_resized[line, :, :, :])
                axes[0, 1].imshow(np.argmax(pred_y, axis=-1)[line, :, :],
                                  alpha=0.5, cmap=self.numeric_cmap,
                                  vmin=0, vmax=self.n_classes - 1)
                axes[0, 1].set_title("Predicted (" + self.split[line] + ")")
                axes[0, 1].set_axis_off()

                axes[1, 0].set_aspect("equal")
                axes[1, 0].set_axis_off()
                axes[1, 0].legend(handles=self.color_patches, mode="expand",
                                  loc="upper left", frameon=False, fontsize=14)
                axes[1, 0].set_axis_off()

                axes[1, 1].imshow(self.photos_resized[line, :, :, :])
                axes[1, 1].imshow(entropy[line, :, :],
                                  alpha=0.25, cmap=self.cmap_entropy,
                                  vmin=0, vmax=np.log(self.n_classes))
                axes[1, 1].set_title("Entropy")
                axes[1, 1].set_axis_off()

                plt.savefig(os.path.join(path, r'.\Prediction_%d.jpg' % line),
                            bbox_inches='tight', dpi=dpi)

                plt.close(fig)

    def predict_masks(self, images_path, prediction_path):
        photos_prediction = pr.load_photos(
            path=images_path,
            downscaling_factor=[self.downscaling_factor]*2)

        photos_prediction = np.stack(
            [skt.resize(photo, self.resolution, anti_aliasing=True) for photo in photos_prediction]
        )

        pred_masks = self.u_net.predict(photos_prediction, batch_size=5)

        photo_names = os.listdir(images_path)

        for line in range(photos_prediction.shape[0]):
            mask_big = skt.resize(
                pred_masks[line],
                (self.image_height, self.image_width, self.n_classes),
                anti_aliasing=True)

            for i, label in enumerate(self.labels):
                io.imsave(os.path.join(prediction_path, label + '_' + photo_names[line]),
                          np.round(mask_big[:, :, i], 0).astype(np.uint8) * 255)
