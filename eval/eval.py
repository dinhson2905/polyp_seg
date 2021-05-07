import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as distance
from scipy.misc import imfilter
import math

def wfm(FG, GT):
    dGT = np.asarray(GT, np.float32)
    dGT /= (GT.max() + 1e-8)

    E = np.abs(FG - dGT)
    Dst, Idxt = distance(dGT, return_indices=True)
    Et = E
    Et[~GT] = Et[Idxt[~GT]]
    EA = imfilter(Et, 'blur')
    MIN_E_EA = E
    MIN_E_EA[GT & EA<E] = EA[GT & EA<E]
    B = np.ones(GT.shape)
    B[~GT] = 2.0 - 1*math.exp(math.log(1-0.5)/5. * Dst[~GT])
    Ew = np.multiply(MIN_E_EA, B)

    TPw = np.sum(dGT[:] - np.sum(np.sum(Ew[GT])))
    FPw = np.sum(np.sum(Ew[~GT]))
    R = np.mean(Ew[GT])
    P = TPw


def mean2(input, target):
    input_flat = np.reshape(input, -1)
    target_flat = np.reshape(target, -1)
    n = input_flat.shape[0]
    