import math
import numpy as np


def snr(original, converted):
    a = math.sqrt(np.mean(original ** 2))
    b = math.sqrt(np.mean(converted ** 2))
    return 10 * np.log10(a / b)
