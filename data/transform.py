import numpy as np
import random
import torchvision.transforms as T
import cv2
import math
from data import transform as base_transform
from utils import is_list, is_dict, get_valid_args


def Compose(trf_cfg):
    assert is_list(trf_cfg)
    transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    return transform


def get_transform(trf_cfg=None):
    if is_dict(trf_cfg):
        transform = getattr(base_transform, trf_cfg['type'])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
        return transform(**valid_trf_arg)
    if trf_cfg is None:
        return lambda x: x
    if is_list(trf_cfg):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    raise "Error type for -Transform-Cfg-"


class RandomPerspective(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            h, w = seq.shape[-2:]
            cutting = int(w // 44) * 10
            x_left = list(range(0, cutting))
            x_right = list(range(w - cutting, w))
            TL = (random.choice(x_left), 0)
            TR = (random.choice(x_right), 0)
            BL = (random.choice(x_left), h)
            BR = (random.choice(x_right), h)
            srcPoints = np.float32([TL, TR, BR, BL])
            canvasPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            perspectiveMatrix = cv2.getPerspectiveTransform(
                np.array(srcPoints), np.array(canvasPoints))
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpPerspective(_[0, ...], perspectiveMatrix, (w, h))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq


class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            dh, dw = seq.shape[-2:]
            # rotation
            degree = random.uniform(-self.degree, self.degree)
            M1 = cv2.getRotationMatrix2D((dh // 2, dw // 2), degree, 1)
            # affine
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq
        
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            return seq[..., ::-1]
        
        
class BaseSilCuttingTransform():
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        if cutting != 0: 
            x = x[..., cutting:-cutting]
        return x / self.divsor