import numpy as np
import numba
from numba import cuda

@cuda.jit
def row_wise_histogram(feature, output,  feature_length):
    xmin = np.float32(0)
    xmax = np.float32(1)
    idx = cuda.grid(1)
    nbins = 10
    bin_width = (xmax - xmin) / nbins
    if  idx < output.shape[0]:
        for i in range(feature_length):
            # Each thread will take all the row features to generate historgram
            input = feature[idx][i]
            bin_number = np.int32(nbins * (np.float32(input) - np.float32(xmin)) / (np.float32(xmax) - np.float32(xmin)))
            # mybin[idx][i] = bin_number
            if bin_number >= 0 and bin_number < output.shape[1]:
                cuda.atomic.add(output[idx], bin_number, 1)