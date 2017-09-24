from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def zero_center(data):
    # calculates the mean of each paramater across all examples
    return data - np.mean(data, axis=0)
