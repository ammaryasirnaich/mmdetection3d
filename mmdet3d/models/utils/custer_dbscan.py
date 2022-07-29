import numpy as np
import numba
from numba import cuda


@cuda.jit
def custer_dbscan(feature, output,  feature_length):
    idx = cuda.grid(1)
    if  idx < output.shape[0]:
        shr = cuda.shared.array(feature_length, dtype=float16)
        
        # Declare an array in shared memory
        # transfer the elements to shared memory for clustering later
        for i in range(feature_length):
            # Each thread will take all the row features
            shr[i] = feature[idx][i]
        
        ## perform clustering on shared memory


        ## transfer custered data to <output>
           



