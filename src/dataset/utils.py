import PIL
import torch
import numpy as np
from typing import List
from torch.utils.data import default_collate


def normalize_image(image: PIL.Image):
    size = image.size[0]
    image = np.array(image.convert("RGB"), dtype=np.float32)
    image = image.transpose(2, 0, 1)
    image = image / 127.5 - 1.0
    return image


def prepare_npy_image_mask(image: PIL.Image, mask):
    size = image.size[0]
    image = np.array(image.convert("RGB"), dtype=np.float32)
    image = image.transpose(2, 0, 1)
    image = image / 127.5 - 1.0
    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    masked_image = image * (mask < 0.5)

    nonzeros = mask[0].nonzero()  # (2, N)
    minx, maxx = min(nonzeros[0], default=0), max(nonzeros[0], default=size)
    miny, maxy = min(nonzeros[1], default=0), max(nonzeros[1], default=size)
    mask_coordinate = np.array((minx, maxx, miny, maxy), dtype=np.int16)
    return image, mask, masked_image, mask_coordinate


def char_inpaint_collate_fn(features):
    """this collate function concate list/set into a list instead of merging into a tensor"""
    feature_keys = features[0].keys()
    collated = {k: [] for k in feature_keys}
    for feature in features:
        for k, v in feature.items():
            collated[k].append(v)
    for k, v in collated.items():
        if not isinstance(v[0], list) and not isinstance(v[0], set):
            collated[k] = default_collate(v)
    return collated


class LenCounter:
    def __init__(self, min_len=1, max_len=15, eachnum=10, inf=False) -> None:
        self.bucket = {k: eachnum for k in range(min_len, max_len + 1)}
        self.inf = inf

    def ended(self):
        if self.inf:
            return False
        else:
            return sum(list(self.bucket.values())) == 0

    def __call__(self, label_str):
        if self.inf:
            return True
        else:
            propose_len = len(label_str)
            if propose_len not in self.bucket or self.bucket[propose_len] == 0:
                return False  # not adding anything
            self.bucket[propose_len] -= 1  # adding one to this bucket
            return True


def sample_random_angle(
    cat_prob: List,
    angle_list: List,
    rotate_range: int,
    generator: torch.Generator = None,
):
    """Return a random angle according to the probability distribution of each category.

    Args:
        cat_prob (List): 3-element list, the probability of each category for stay/rotate in angle_list/rotate in random angle
        angle_list (List): possible angles for category 1
        rotate_range (int): maximum possible angle for category 2
        generator (torch.Generator, optional): let the function be deterministic. Defaults to None.
    """
    assert len(cat_prob) == 3
    # sample category
    cat_sample = torch.rand(size=(), generator=generator).item()
    if cat_sample < cat_prob[0]:
        # no rotate
        angle = 0
    elif cat_sample < (cat_prob[0] + cat_prob[1]):
        # rotate in angle_list
        angle_list = list(angle_list)
        angle = angle_list[
            torch.randint(0, len(angle_list), size=(), generator=generator)
        ]
    else:
        # rotate in random angle
        angle = torch.randint(
            -rotate_range, rotate_range, size=(), generator=generator
        ).item()
    return angle
