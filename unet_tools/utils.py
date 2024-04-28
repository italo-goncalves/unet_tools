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

import numpy as np


def train_test_label(n_data, train_perc=0.25, seed=None):
    # if train_perc < 0.5:
    #     n_test = int(np.ceil((1 - train_perc) / train_perc))
    #     n_train = int(np.ceil(1 / n_test))
    # else:
    #     n_train = int(np.ceil(train_perc / (1 - train_perc)))
    #     n_test = int(np.ceil(1 / n_train))
    #
    # split = np.array(
    #     (["train"] * n_train + ["test"] * n_test)
    #     * int(np.ceil(n_data / (n_test + n_train)))
    # )[:n_data]

    n_train = int(np.ceil(train_perc * n_data))
    n_test = n_data - n_train
    split = ["train"] * n_train + ["test"] * n_test

    if seed is not None:
        np.random.seed(seed)

    split = np.random.choice(split, size=n_data, replace=False)

    return split
