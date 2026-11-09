import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 128
BAND_LIMITS = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40))


def make_layers(raw, fs=FS):
    sos = [butter(4, limits, btype="bandpass", fs=fs, output="sos")
           for limits in BAND_LIMITS]
    filtered = [sosfiltfilt(section, raw, axis=-1) for section in sos]
    return np.stack((raw, *filtered), axis=-1)
