import numpy as np


def segment_recursive(signal, start, end, threshold, segments):
    part = signal[start:end]
    if len(part) <= 2:
        segments.append((start, end))
        return

    var = np.var(part)
    if var > threshold:
        mid = (start + end) // 2
        segment_recursive(signal, start, mid, threshold, segments)
        segment_recursive(signal, mid, end, threshold, segments)
    else:
        segments.append((start, end))


def run_segmentation(signal, threshold=1.0):
    segments = []
    segment_recursive(signal, 0, len(signal), threshold, segments)
    return segments
