import torch
import json
import math
from PIL import Image, ImageDraw, ImageChops, ImageOps
from .utils import prepare_npy_image_mask, sample_random_angle, LenCounter
import torchvision.transforms.functional as TF


def loadSceneOCRData(source, config):
    """Return a list of datas"""
    print(f"Collecting data from {config['label_path']}")
    if "len_counter" in config:
        lencounter = LenCounter(**config["len_counter"])
    else:
        lencounter = LenCounter(inf=True)

    max_num = config.get("max_num", 99999999999)
    rawlabels = json.load(open(config["label_path"]))
    filter_func = globals()[f"filter_{source}"]
    rawlabels = filter_func(rawlabels)
    rawlabels = [(source,) + tuple(x) for x in rawlabels]

    labels = []
    for label in rawlabels:
        if lencounter.ended():
            break
        if len(labels) >= max_num:
            break
        if lencounter(label[2]):
            labels.append(label)
    if not lencounter.inf:
        labels.sort(key=lambda x: len(x[2]))
    print(f"Collected {len(labels)} samples from {source}")
    return labels


ACCEPT_CHARS = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def filter_ICDAR13(labels):
    def filter_item(item):
        label = item[1]
        if len(label) > 16:
            return True
        for ch in label:  # filter out control characters
            if not (ch.isalnum() or ch in ACCEPT_CHARS):
                return True
        return False
    legal_labels = []
    for item in labels:
        if not filter_item(item):
            legal_labels.append(item)
    return legal_labels  # all legal


def filter_out(item):
    # * return true for those we want to filter out
    label = item[1]
    if len(label) > 16:  # char is too long
        return True
    for ch in label:  # filter out control characters
        if not (ch.isalnum() or ch in ACCEPT_CHARS):
            return True
    if len(item) == 4:
        # we have image size info
        points = item[2]
        all_x, all_y = points[0::2], points[1::2]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        if (max_x - min_x) * (max_y - min_y) * 500 > item[3][0] * item[3][1]:
            return True
    return False


def filter_ArT(labels):
    # ArT has no image size info
    # return labels with readable english
    legal_labels = []
    for img_name, label in labels.items():
        for item in label:
            if (
                not item["illegibility"]
                and item["language"] == "Latin"
                and item["transcription"].strip() != "###"
            ):
                points = []
                for x in item["points"]:
                    points.append(x[0])
                    points.append(x[1])
                chars = item["transcription"]
                tmp_item = (img_name + ".jpg", chars, points)
                if not filter_out(tmp_item):
                    legal_labels.append(tmp_item)
    return legal_labels


def filter_COCO(labels):
    legal_labels = []
    for _, item in labels["anns"].items():
        if item["language"].lower() == "english" and item["legibility"] == "legible":
            img_name = f"COCO_train2014_{item['image_id']:012d}.jpg"
            chars = item["utf8_string"]
            mask = item["mask"]
            points = []
            for i in range(len(mask) // 2):
                points.extend([mask[2 * i], mask[2 * i + 1]])
            img_size = (
                labels["imgs"][str(item["image_id"])]["width"],
                labels["imgs"][str(item["image_id"])]["height"],
            )
            tmp_item = (img_name, chars, points, img_size)
            if not filter_out(tmp_item):
                legal_labels.append(tmp_item)
    return legal_labels


def filter_TextOCR(labels):
    legal_labels = []
    for key, item in labels["anns"].items():
        if item["utf8_string"].strip(".") != "":
            img_name = key.split("_")[0] + ".jpg"
            chars = item["utf8_string"]
            points = item["points"]
            img_size = (
                labels["imgs"][key.split("_")[0]]["width"],
                labels["imgs"][key.split("_")[0]]["height"],
            )
            tmp_item = (img_name, chars, points, img_size)
            if not filter_out(tmp_item):
                legal_labels.append(tmp_item)
    return legal_labels


class AugForSceneOCR:
    # randomly choose some operations and apply to all input images

    def rand_crop_coordinate(
        h, w, min_x, max_x, min_y, max_y, max_size, generator=None
    ):
        left = int(max(max_x - max_size, 0))
        right = int(min(min_x, h - max_size))
        crop_x1 = torch.randint(
            left, max(left + 1, right + 1), size=(), generator=generator
        ).item()
        left = int(max(max_y - max_size, 0))
        right = int(min(min_y, w - max_size))
        crop_y1 = torch.randint(
            left, max(left + 1, right + 1), size=(), generator=generator
        ).item()
        crop_x2 = crop_x1 + max_size
        crop_y2 = crop_y1 + max_size
        return (crop_x1, crop_y1, crop_x2, crop_y2)

    def create_mask(image, polygons):
        mask = Image.new("1", size=image.size, color=0)
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
            draw.polygon(polygon, fill="white", outline="white")
        del draw
        return mask

    def __init__(self, size, config):
        self.size = size
        self.config = config  # with probability

    def __call__(self, image, points, generator=None, return_pil=False):
        attribute = set()  # denote what operations are applied to the image
        h, w = image.size
        all_x, all_y = points[0::2], points[1::2]
        min_x, max_x = int(min(all_x)), int(max(all_x))
        min_y, max_y = int(min(all_y)), int(max(all_y))
        # 1. add mask
        mask_points = [points]
        if "expand_mask" in self.config:
            mask_prob = torch.rand(size=(), generator=generator)
            if mask_prob < self.config["expand_mask"]["center_mask"]:
                attribute.add("center_mask")
                # center mask to left/right
                if max_x - min_x < max_y - min_y:
                    # expand from center to left/right
                    span = torch.randint(
                        1, max((max_x - min_x) // 4, 2), size=(), generator=generator
                    ).item()
                    left = min_x - span
                    span = torch.randint(
                        1, max((max_x - min_x) // 4, 2), size=(), generator=generator
                    ).item()
                    right = max_x + span
                    polygon = (left, min_y, right, min_y,
                               right, max_y, left, max_y)
                    mask_points.append(polygon)
                else:
                    # expand from center to up/down
                    span = torch.randint(
                        1, max((max_y - min_y) // 4, 2), size=(), generator=generator
                    ).item()
                    up = min_y - span
                    span = torch.randint(
                        1, max((max_y - min_y) // 4, 2), size=(), generator=generator
                    ).item()
                    down = max_y + span
                    polygon = (min_x, up, max_x, up, max_x, down, min_x, down)
                    mask_points.append(polygon)

            # additional mask
            mask_prob = torch.rand(size=(), generator=generator)
            if mask_prob < self.config["expand_mask"]["additional_mask"]:
                attribute.add("additional_mask")
                num = torch.randint(1, 3, size=(), generator=generator)
                left = max(0, min_x - (max_x - min_x) // 8)
                right = min(h, max_x + (max_x - min_x) // 8)
                up = max(0, min_y - (max_y - min_y) // 8)
                down = min(w, max_y + (max_y - min_y) // 8)
                for _ in range(num):
                    l = torch.randint(
                        left, max(left + 1, max_x), size=(), generator=generator
                    ).item()
                    r = torch.randint(
                        max_x, max(max_x + 1, right), size=(), generator=generator
                    ).item()
                    u = torch.randint(
                        up, max(up + 1, max_y), size=(), generator=generator
                    ).item()
                    d = torch.randint(
                        max_y, max(max_y + 1, down), size=(), generator=generator
                    ).item()
                    polygon = (l, u, r, u, r, d, l, d)
                    mask_points.append(polygon)

        # 2. crop
        if not "crop" in self.config:
            # basic crop, make sure the text is in cropped image
            mask_image_ratio = self.config["mask_image_ratio"]
            max_size = min(
                h,
                w,
                self.size,
                int(math.sqrt(mask_image_ratio * (max_x - min_x) * (max_y - min_y))),
            )
            # try center crop
            crop_x1 = max(0, int((max_x + min_x) / 2 - max_size / 2))
            crop_x2 = min(h, int(crop_x1 + max_size))
            crop_y1 = max(0, int((max_y + min_y) / 2 - max_size / 2))
            crop_y2 = min(w, int(crop_y1 + max_size))
            attribute.add("center_crop")
        else:
            # take additional masks into account
            # update min_x, max_x, min_y, max_y
            for polygon in mask_points:
                all_x, all_y = points[0::2], points[1::2]
                min_x = min(min_x, min(all_x))
                max_x = max(max_x, max(all_x))
                min_y = min(min_y, min(all_y))
                max_y = max(max_y, max(all_y))
            mask_image_ratio = self.config["crop"]["mask_image_ratio"]
            max_size = min(
                h,
                w,
                self.size,
                int(math.sqrt(mask_image_ratio * (max_x - min_x) * (max_y - min_y))),
            )
            max_size = max(
                max_size, (max_x - min_x + 5), (max_y - min_y + 5)
            )  # bigger than mask
            (crop_x1, crop_y1, crop_x2, crop_y2,) = AugForSceneOCR.rand_crop_coordinate(
                h, w, min_x, max_x, min_y, max_y, max_size, generator=generator
            )
            attribute.add("rand_crop")

        # create mask
        mask = AugForSceneOCR.create_mask(image, mask_points)
        image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        mask = mask.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # 3. rotate image&mask
        if "rotate" in self.config:
            rand_angle = sample_random_angle(
                cat_prob=self.config["rotate"]["cat_prob"],
                angle_list=self.config["rotate"]["angle_list"],
                rotate_range=self.config["rotate"]["rotate_range"],
                generator=generator,
            )
            if rand_angle != 0:
                image = TF.rotate(image, rand_angle, expand=True)
                mask = TF.rotate(mask, rand_angle, expand=True)
            if rand_angle == 0:
                attribute.add("0-no_rotate")
            elif rand_angle in self.config["rotate"]["angle_list"]:
                attribute.add(f"{rand_angle}-normal_rotate")
            else:
                attribute.add(f"{rand_angle}-random_rotate")

        # always resize back to (self.size, self.size)
        image = image.resize((self.size, self.size))
        mask = mask.resize((self.size, self.size))
        mask_coordinate = (-1, -1, -1, -1)
        if return_pil:
            masked_image = ImageChops.multiply(
                image, ImageOps.invert(mask.convert("RGB"))
            )
        else:
            image, mask, masked_image, mask_coordinate = prepare_npy_image_mask(
                image, mask
            )
        return {
            "image": image,
            "mask": mask,
            "masked_image": masked_image,
            "coordinate": mask_coordinate,  # to help OCR model retrieve only the generation part
            "attribute": attribute,
        }
