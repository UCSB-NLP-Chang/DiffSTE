import os
import csv
import itertools
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageChops, ImageDraw, ImageOps
from scipy.spatial import ConvexHull

from .utils import LenCounter, prepare_npy_image_mask, sample_random_angle


def filter_out_synthtext(line, size=256):
    img_size = eval(line[2])
    if img_size[0] < 10 or img_size[1] < 10:
        return None
    if img_size[0] > 3 * size or img_size[1] > 3 * size:  # ignore too long image
        return None
    path = line[0]
    label = path.split("/")[-1].split("_")[1]
    if not label.isalnum() or not label.isascii():
        return None
    return (path, label, img_size)


def load_SynthText(source, config, lencounter: LenCounter = None):
    labels = []
    max_num = config.get("max_num", 99999999999)
    with open(config["label_path"]) as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            if lencounter.ended():
                break
            if len(labels) >= max_num:
                break
            if line[0] == "path":
                continue
            sample = filter_out_synthtext(line)
            if sample and lencounter(sample[2]):
                labels.append((source,) + sample)
    if not lencounter.inf:
        labels.sort(key=lambda x: len(x[2]))
    labels = labels[:max_num]
    return labels


def filter_out_synthtiger(row):
    return row


def load_Synthtiger(source, config, lencounter=None):
    labels = []
    max_num = config.get("max_num", 99999999999)
    if config.get("use_textbbox", False):
        bbox_type = "text_bbox"
    else:
        bbox_type = "char_bbox"
    with open(config["label_path"]) as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            if lencounter.ended():
                break
            if len(labels) >= max_num:
                break
            row = {k: eval(v) if not k in ["path"]
                   else v for k, v in row.items()}
            sample = filter_out_synthtiger(row)
            if sample and lencounter(sample["label"][0]):
                if len(row) == 6:  # ! has style information
                    style = row['style']
                    labels.append((source,) + (os.path.join("images",
                                  row['path']), row['label'], row[bbox_type], style))
                else:
                    labels.append(
                        (source,) + (os.path.join("images", row["path"]), row["label"], row[bbox_type]))
    if not lencounter.inf:
        labels.sort(key=lambda x: len(x[2][0]))
    labels = labels[:max_num]
    return labels


def loadSynthOCRData(source, config):
    """Return a list of datas"""
    print(f"Collecting data from {config['label_path']}")
    labels = []
    if "len_counter" in config:
        lencounter = LenCounter(**config["len_counter"])
    else:
        lencounter = LenCounter(inf=True)

    if "synthtext" in source.lower():
        labels = load_SynthText(source, config, lencounter)
    elif "synthtiger" in source.lower():
        labels = load_Synthtiger(source, config, lencounter)
    else:
        raise NotImplementedError
    print(f"Collected {len(labels)} samples from {source}")
    return labels


def get_most_distant_points(points):
    max_dist = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = (points[i][0] - points[j][0]) ** 2 + \
                (points[i][1] - points[j][1]) ** 2
            if dist > max_dist:
                max_dist = dist
                p1, p2 = i, j
    return points[p1], points[p2]


def get_k(point0, point1):
    x0, y0 = point0
    x1, y1 = point1
    if (x0 == x1):
        k = 0
    else:
        k = (y1 - y0) / (x1 - x0)
    return k


def match_kb(k, point):
    b = point[1] - k * point[0]
    return b


def get_upline_downline(k, points):
    maxb, minb = 0, 1e8
    for i in range(len(points)):
        b = match_kb(k, points[i])
        if b <= minb:
            minb = b
        if b >= maxb:
            maxb = b
    return minb, maxb


def rand_mask_points(char_bboxes, direction="none"):
    # * Randomly create a polygon mask inside the bounding box,
    # * but big enough to cover than all characters
    if not any(isinstance(i, list) for i in char_bboxes):
        char_bboxes = [char_bboxes]
    points = []
    for ch in char_bboxes:
        x1, y1, x2, y2 = ch
        tmp_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        points.extend(tmp_points)

    hull = ConvexHull(points, incremental=True)
    if direction == "longer":
        # we are safe to make the mask longer
        # left right
        def x_max(bbox): return int(max(bbox[0::2]))
        def y_max(bbox): return int(max(bbox[1::2]))
        if any(x_max(bbox) == x_max(char_bboxes[0]) for bbox in char_bboxes[1:]):
            # horizontal text
            k = 0
            min_ratio = 0.1
            max_ratio = 0.3
        else:
            # left, right = get_most_distant_points(points)
            left = points[3]
            right = points[-1]
            k = get_k(left, right)
            min_ratio = 0.2
            max_ratio = 0.4
        minb, maxb = get_upline_downline(k, points)
        left, right = hull.min_bound[0], hull.max_bound[0]
        expand_left0 = left - (min_ratio + np.random.rand()
                               * max_ratio) * (right - left)
        expand_right0 = right + \
            (min_ratio + np.random.rand() * max_ratio) * (right - left)
        expand_left1 = left - (min_ratio + np.random.rand()
                               * max_ratio) * (right - left)
        expand_right1 = right + \
            (min_ratio + np.random.rand() * max_ratio) * (right - left)
        hull.add_points(
            [
                (expand_left0, k * expand_left0 + minb),
                (expand_right0, k * expand_right0 + minb),
                (expand_left1, k * expand_left1 + maxb),
                (expand_right1, k * expand_right1 + maxb),
            ]
        )
        expanded_points = hull.points[hull.vertices]
        if (k == 0) or (k == 1e9):
            new_points = np.random.rand(10, 2)
            new_points[:, 1] *= 1.2 * (hull.max_bound[1] - hull.min_bound[1])
            new_points[:, 0] += hull.min_bound[0]
            new_points[:, 1] += hull.min_bound[1] + 1
            expanded_points = np.concatenate((expanded_points, new_points))
    elif direction == "higher":
        pass
    elif direction == "expand":
        new_points = np.random.rand(30, 2)
        new_points[:, 0] *= 1.2 * (hull.max_bound[0] - hull.min_bound[0])
        new_points[:, 1] *= 1.2 * (hull.max_bound[1] - hull.min_bound[1])
        new_points[:, 0] += hull.min_bound[0]
        new_points[:, 1] += hull.min_bound[1]
        expanded_points = np.concatenate((points, new_points))
        for p in expanded_points:
            if p[0] <= 0:
                p[0] = 0
            if p[0] >= 256:
                p[0] = 255
            if p[1] <= 0:
                p[1] = 0
            if p[1] >= 256:
                p[1] = 255
    else:
        expanded_points = points
    points = list(
        itertools.chain(
            *[expanded_points[x] for x in ConvexHull(expanded_points).vertices]
        )
    )
    return points


class AugForSynthOCR:
    def __init__(self, size, config, return_pil=False):
        self.size = size
        self.config = config
        self.return_pil = return_pil
        if "expand" in self.config:
            self.min_longside = int(
                self.config["expand"]["min_longside_ratio"] * self.size
            )
            self.max_longside = int(
                self.config["expand"]["max_longside_ratio"] * self.size
            )

    def pad_image(self, image, top=5, left=5, right=5, down=5, fillcolor="white"):
        width, height = image.size
        new_width = width + right + left
        new_height = height + top + down
        new_image = Image.new(image.mode, (new_width, new_height), fillcolor)
        new_image.paste(image, (left, top))
        return new_image

    # 1. expand original image to random ratio
    # 2. paste the expanded image to full black background
    # 3. create mask corresponding to the expanded image
    # 4. rotate image/mask to some possible angle
    def __call__(self, raw_image, generator, points=None):
        attribute = set()
        black = Image.new("RGB", (self.size, self.size), "black")
        if points is not None:
            # only mask out part of the image
            mask = Image.new("RGB", raw_image.size, "black")
            draw = ImageDraw.Draw(mask)
            # for point in points:
            draw.polygon(points, fill="white", outline="black")
            del draw
        else:
            # mask out full image
            mask = Image.new("RGB", raw_image.size, "white")

        if "pad" in self.config and self.config["pad"]:
            raw_image = self.pad_image(raw_image)
            # mask = self.pad_image(mask, fillcolor="white")
            mask = self.pad_image(mask, fillcolor="black")

        longside = max(raw_image.size)
        if "expand" in self.config:
            min_longside = min(longside, self.min_longside)
            max_longside = self.max_longside
            new_longside = torch.randint(
                min_longside, max_longside + 1, (1,), generator=generator
            ).item()
            # resize while keeping the short side to keep the aspect ratio
            ratio = new_longside / longside
            # raw_image.thumbnail((new_longside, new_longside))
            raw_image = raw_image.resize(
                (int(raw_image.size[0] * ratio), int(raw_image.size[1] * ratio))
            )

            mask = mask.resize(
                (int(mask.size[0] * ratio), int(mask.size[1] * ratio)))
            attribute.add("expandsize")
        else:
            # make sure we can put the image to fit black background
            if longside > self.size:
                newlongside = int(self.size * 0.9)
                raw_image.thumbnail((newlongside, newlongside))
                mask.thumbnail((newlongside, newlongside))
            attribute.add("fixsize")

        raw_x, raw_y = raw_image.size[0], raw_image.size[1]
        if "center" in self.config:
            center_prob = torch.rand((), generator=generator).item()
            if center_prob < self.config["center"]:  # at center
                attribute.add("center")
                left_upper_x = self.size // 2 - raw_x // 2
                left_upper_y = self.size // 2 - raw_y // 2
            else:
                attribute.add("rand_pos")
                max_left_upper_x = self.size - raw_x + 1
                max_left_upper_y = self.size - raw_y + 1
                left_upper_x = torch.randint(
                    0, max_left_upper_x, (), generator=generator
                ).item()
                left_upper_y = torch.randint(
                    0, max_left_upper_y, (), generator=generator
                ).item()
        else:
            attribute.add("center")
            left_upper_x = self.size // 2 - raw_x // 2
            left_upper_y = self.size // 2 - raw_y // 2

        black.paste(raw_image.convert("RGB"), (left_upper_x, left_upper_y))
        raw_image = black
        # create mask
        new_black = Image.new("RGB", black.size, "black")
        new_black.paste(mask, (left_upper_x, left_upper_y))
        mask = new_black

        if "rotate" in self.config:
            rand_angle = sample_random_angle(
                cat_prob=self.config["rotate"]["cat_prob"],
                angle_list=self.config["rotate"]["angle_list"],
                rotate_range=self.config["rotate"]["rotate_range"],
                generator=generator,
            )
            if rand_angle != 0:
                # most of the picture is black, so don't expand
                raw_image = TF.rotate(raw_image, rand_angle, expand=True)
                mask = TF.rotate(mask, rand_angle, expand=True)
                raw_image = TF.resize(raw_image, (self.size, self.size))
                mask = TF.resize(mask, (self.size, self.size))
            if rand_angle == 0:
                attribute.add("no_rotate")
            elif rand_angle in self.config["rotate"]["angle_list"]:
                attribute.add("normal_rotate")
            else:
                attribute.add("random_rotate")

        raw_image, mask, masked_image, mask_coordinate = prepare_npy_image_mask(
            raw_image, mask
        )

        return {
            "image": raw_image,
            "mask": mask,
            "masked_image": masked_image,
            "coordinate": mask_coordinate,
            "attribute": attribute,
        }
