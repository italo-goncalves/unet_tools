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

import tensorflow as tf


def encoder_block(input_layer, out_channels, filter_size=3,
                  depth=2, residual=False,
                  dropout_prob=0.3, max_pooling=True):
    conv = input_layer

    conv_out = conv
    for _ in range(depth):
        conv_out = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=filter_size,
            activation='relu',
            padding="same",
            kernel_initializer='HeNormal')(conv_out)
    if residual:
        conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            activation='relu',
            padding="same",
            kernel_initializer='HeNormal')(conv)
        conv_out = tf.keras.layers.Add()([conv_out, conv])
    conv = conv_out

    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = tf.keras.layers.MaxPool2D()(conv)
        return next_layer, conv
    else:
        return conv


def decoder_block(down_layer, out_channels, skip_layer=None, filter_size=3,
                  depth=2, residual=False,
                  dropout_prob=0.3):
    up = tf.keras.layers.UpSampling2D()(down_layer)
    up = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=2,
        activation='linear',
        padding="same",
        kernel_initializer='HeNormal')(up)

    conv = up
    if skip_layer is not None:
        # conv = tf.keras.layers.Concatenate()([skip_layer, up])
        conv = tf.keras.layers.Add()([skip_layer, up])

    conv_out = conv
    for _ in range(depth):
        conv_out = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=filter_size,
            activation='relu',
            padding="same",
            kernel_initializer='HeNormal')(conv_out)

    if residual:
        conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            activation='relu',
            padding="same",
            kernel_initializer='HeNormal')(conv)
        conv_out = tf.keras.layers.Add()([conv_out, conv])

    if dropout_prob > 0:
        conv_out = tf.keras.layers.Dropout(dropout_prob)(conv_out)

    return conv_out


def dice_loss(y_true, y_pred):
    return 1 - 2 * tf.reduce_sum(y_true * y_pred) / (
            tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


def focal_loss(y_true, y_pred):
    cross_entropy = - (1 - y_pred) ** 2 * y_true * tf.math.log(
        y_pred * 0.9999 + 0.0001)
    return tf.reduce_mean(cross_entropy)


def u_net(input_layer, output_size, channels=16, blocks=4,
          block_depth=2, residual=False,
          dropout_prob=0.1, filter_size=5,
          end_activation=None):
    # input_layer = tf.keras.layers.Input(shape=input_shape)
    out_layer = input_layer
    skips = []
    for i in range(blocks):
        out_layer, skip = encoder_block(out_layer, channels * 2 ** i,
                                        dropout_prob=dropout_prob,
                                        filter_size=filter_size,
                                        depth=block_depth,
                                        residual=residual)
        skips.append(skip)

    out_layer = encoder_block(out_layer, channels * 2 ** blocks,
                              max_pooling=False)

    for i in range(blocks):
        out_layer = decoder_block(out_layer,
                                  channels * 2 ** (blocks - i - 1),
                                  skip_layer=skips[-i - 1],
                                  dropout_prob=dropout_prob,
                                  filter_size=filter_size,
                                  depth=block_depth,
                                  residual=residual)

    out_layer = tf.keras.layers.Conv2D(
        filters=output_size,
        kernel_size=1,
        activation=end_activation,
        padding="same",
        kernel_initializer='HeNormal')(out_layer)

    return out_layer

    # if return_layers:
    #     return input_layer, out_layer
    # else:
    #     model = tf.keras.Model(inputs=input_layer, outputs=out_layer)
    #     return model


def multi_res_u_net(input_shape, output_size, dropout_prob=0.1,
                    end_activation=None):
    input_layer = tf.keras.layers.Input(shape=input_shape)

    def base_conv(in_layer, out_channels, dp):
        conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=3,
            activation='relu',
            padding="same",
            kernel_initializer='HeNormal')(in_layer)
        if dp > 0:
            conv = tf.keras.layers.Dropout(dp)(conv)
        return conv

    def multi_res_block(in_layer, out_channels, dp):
        block_a = base_conv(in_layer, out_channels[0], dropout_prob)
        block_b = base_conv(block_a, out_channels[1], dropout_prob)
        block_c = base_conv(block_b, out_channels[2], dropout_prob)
        block_d = tf.keras.layers.Conv2D(
                filters=sum(out_channels),
                kernel_size=1,
                activation='relu',
                padding="same",
                kernel_initializer='HeNormal')(in_layer)
        concat = tf.keras.layers.Concatenate()([block_a, block_b, block_c])
        final_conv = tf.keras.layers.Add()([concat, block_d])
        return final_conv

    def residual_path(in_layer, out_channels, dp, steps):
        conv = in_layer

        for _ in range(steps):
            path_a = base_conv(conv, out_channels, dp)
            path_b = tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=1,
                activation='relu',
                padding="same",
                kernel_initializer='HeNormal')(conv)
            conv = tf.keras.layers.Add()([path_a, path_b])

        return conv

    def upsample(in_layer, ch):
        up = tf.keras.layers.UpSampling2D()(in_layer)
        up = tf.keras.layers.Conv2D(
            filters=ch,
            kernel_size=2,
            activation='linear',
            padding="same",
            kernel_initializer='HeNormal')(up)
        return up

    block_1 = multi_res_block(input_layer, [8, 17, 26], dropout_prob)
    path_1 = residual_path(block_1, 32, dropout_prob, 4)
    block_1 = tf.keras.layers.MaxPool2D()(block_1)

    block_2 = multi_res_block(block_1, [17, 35, 53], dropout_prob)
    path_2 = residual_path(block_2, 64, dropout_prob, 3)
    block_2 = tf.keras.layers.MaxPool2D()(block_2)

    block_3 = multi_res_block(block_2, [35, 71, 106], dropout_prob)
    path_3 = residual_path(block_3, 128, dropout_prob, 2)
    block_3 = tf.keras.layers.MaxPool2D()(block_3)

    block_4 = multi_res_block(block_3, [71, 142, 213], dropout_prob)
    path_4 = residual_path(block_4, 256, dropout_prob, 1)
    block_4 = tf.keras.layers.MaxPool2D()(block_4)

    block_5 = multi_res_block(block_4, [142, 284, 427], dropout_prob)

    block_6 = upsample(block_5, 426)
    block_6 = tf.keras.layers.Concatenate()([path_4, block_6])
    block_6 = multi_res_block(block_6, [71, 142, 213], dropout_prob)

    block_7 = upsample(block_6, 212)
    block_7 = tf.keras.layers.Concatenate()([path_3, block_7])
    block_7 = multi_res_block(block_7, [35, 71, 106], dropout_prob)

    block_8 = upsample(block_7, 105)
    block_8 = tf.keras.layers.Concatenate()([path_2, block_8])
    block_8 = multi_res_block(block_8, [17, 35, 53], dropout_prob)

    block_9 = upsample(block_8, 51)
    block_9 = tf.keras.layers.Concatenate()([path_1, block_9])
    block_9 = multi_res_block(block_9, [8, 17, 26], dropout_prob)

    output_layer = tf.keras.layers.Conv2D(
            filters=output_size,
            kernel_size=3,
            activation=end_activation,
            padding="same",
            kernel_initializer='HeNormal')(block_9)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def autoencoder(input_layer, channels=16, blocks=4,
                block_depth=2, residual=False,
                dropout_prob=0.1, filter_size=5,
                end_activation=None):
    # input_layer = tf.keras.layers.Input(shape=input_shape)
    out_layer = input_layer
    # skips = []
    for i in range(blocks):
        out_layer, _ = encoder_block(out_layer, channels * 2 ** i,
                                     dropout_prob=dropout_prob,
                                     filter_size=filter_size,
                                     depth=block_depth,
                                     residual=residual)
        # skips.append(skip)

    out_layer = encoder_block(out_layer, channels * 2 ** blocks,
                              max_pooling=False)

    for i in range(blocks):
        out_layer = decoder_block(out_layer,
                                  channels * 2 ** (blocks - i - 1),
                                  dropout_prob=dropout_prob,
                                  filter_size=filter_size,
                                  depth=block_depth,
                                  residual=residual)

    out_layer = tf.keras.layers.Conv2D(
        filters=input_layer.shape[-1],
        kernel_size=1,
        activation=end_activation,
        padding="same",
        kernel_initializer='HeNormal')(out_layer)

    return out_layer
