import numpy as np
import matplotlib.pyplot as plt


def compute_segment_location(y, indice_start, indice_end):
        duration = []
        ind = 0
        for label_ts_prev, label_ts in zip(y[indice_start:indice_end - 1], y[indice_start + 1:indice_end]):
            ind += 1
            if label_ts_prev != label_ts:
                duration.append([int(y[ind - 1]), ind])
        if duration[-1][-1] < indice_end:
            duration.append([int(y[-1]), indice_end-indice_start])
        return duration

def get_mean_segment_length(y, indice_start=0, indice_end=None):
    if indice_end == None:
        indice_end=len(y)
    dur = compute_segment_location(y, indice_start, indice_end)
    dur = np.array(compute_segment_location(y, 0, len(y)))
    dur = dur[1:,1] - dur[:-1,1]
    return np.average(dur)