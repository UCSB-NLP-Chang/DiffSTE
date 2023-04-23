import os
import re
import glob
import random
from PIL import Image

import torch
import json
from torch import Generator
from torch.utils.data import Dataset
from .utils import char_inpaint_collate_fn
from .synthocr import loadSynthOCRData, AugForSynthOCR, rand_mask_points
from .sceneocr import loadSceneOCRData, AugForSceneOCR
import itertools
import numpy as np
from datasets import load_dataset
from datasets import Dataset as hfDataset
from scipy.spatial import ConvexHull

__all__ = ["CharInpaintDataset", "char_inpaint_collate_fn", "SimpleOCRData"]

AUG_CLASSES = {
    "synth": AugForSynthOCR,
    "scene": AugForSceneOCR,
}


def filter_center_label(label, char_bboxes):
    # expand long mask
    if " " in label:
        # randomly choose the center
        splits = label.split(" ")
        reduce_splits = list(filter(lambda x: len(x) > 0, splits))
        assert len(reduce_splits) == 3, "Should have 3 splits"
        # not always at center
        mid_idx = len(reduce_splits[0])
        last_idx = len(label) - len(reduce_splits[-1])
        label = reduce_splits[1]
        char_bboxes = char_bboxes[mid_idx: last_idx]
        return label, char_bboxes
    else:
        # don't do anything
        return label, char_bboxes


def rand_expand_mask(points, label=None, expand_longmask=True):
    points = list(zip(points[::2], points[1::2]))
    points = [list(x) for x in points]
    # depending on how we
    hull = ConvexHull(points)
    new_points = np.random.rand(20, 2)
    new_points[:, 0] *= 1.2 * (hull.max_bound[0] - hull.min_bound[0])
    new_points[:, 1] *= 1.2 * (hull.max_bound[1] - hull.min_bound[1])
    new_points[:, 0] += hull.min_bound[0]
    new_points[:, 1] += hull.min_bound[1]
    expanded_points = np.concatenate((points, new_points))
    points = list(
        itertools.chain(
            *[expanded_points[x] for x in ConvexHull(expanded_points).vertices]
        )
    )
    return points


def prepare_style_chars(chars, style):
    if len(style) == 0:
        return "Write \"" + chars + f"\""
    else:
        if style[0] != "":
            style[0] = re.sub(r'\[.*?\]', '', style[0])
        if "and" in style[0]: # emphasize font combination
            return f"in font: {style[0]}"
        if style[0] != "" and style[1] != "":  # font and color
            return "Write \"" + chars + f"\" in font: {style[0]} and color: {style[1]}"
        if style[0] == "" and style[1] != "":  # only color
            return "Write \"" + chars + f"\" in color: {style[1]}"
        if style[0] != "" and style[1] == "":  # only font
            return "Write \"" + chars + f"\" in font: {style[0]}"
        if style[0] == "" and style[1] == "":  # no information about font/color
            return "Write \"" + chars + f"\""


class CharInpaintDatasetTest:

    WORDS = json.load(open(os.path.dirname(
        os.path.abspath(__file__)) + "/vocab.json"))

    FONTNAMES = [
        "VesperLibre-Regular", "ChivoMono", "Mallanna-Regular", "ScopeOne-Regular", "MontserratAlternates-Regular",
        "SawarabiMincho-Regular", "Monoton-Regular", "VujahdayScript-Regular", "Cabin", "YujiHentaiganaAkebono-Regular",
        "LondrinaSolid-Regular", "Besley", "Sono", "GloriaHallelujah", "BubblegumSans-Regular", "Gorditas-Regular",
        "ExpletusSans", "MoonDance-Regular", "Bentham-Regular", "Podkova", "Pacifico-Regular", "DigitalNumbers-Regular",
        "RougeScript-Regular", "SometypeMono-Regular", "Spirax-Regular", "AzeretMono", "GrechenFuemen-Regular", "Sura-Regular",
        "GochiHand", "NovaMono", "Allura-Regular", "Wallpoet-Regular", "BadScrip", "PaytoneOne-Regular", "CourierPrime-Regular",
        "Oranienbaum-Regular", "MeowScript-Regular", "PressStart2P", "WorkSans", "OleoScriptSwashCaps-Regular", "Ligconsolata-Regular",
        "Kameron-Regular", "Questrial-Regular", "SyneMono-Regular", "Iceberg-Regular", "SedgwickAveDisplay-Regular", "Anaheim-Regular",
        "Rosarivo-Regular", "WendyOne-Regular", "LongCang-Regular", "Yesteryear-Regular", "RubikGemstones-Regular", "SansitaOne-Regular",
        "ArbutusSlab-Regular", "SourceCodePro", "AnekLatin", "AnonymousPro-Regular", "Glory", "NotoSans-Regular", "HerrVonMuellerhoff-Regular",
        "SecularOne-Regular", "Courgette", "FragmentMono-Regular", "Jomolhari-Regular", "HanaleiFill-Regular", "Gluten", "PTM55FT", "B612Mono-Regular",
        "Sansita-Regular", "NovaOval", "YujiSyuku-Regular", "Artifika-Regular", "CormorantUpright-Regular", "Lancelot-Regular",
        "CreteRound-Regular", "CroissantOne-Regular", "SedgwickAve-Regular", "SixCaps", "AlumniSansPinstripe-Regular", "Mohave",
        "EncodeSans", "MartianMono", "Geostar-Regular", "Inconsolata-Regular", "Suravaram-Regular", "PassionOne-Regular", "RedHatMono",
        "Ubuntu-Regular", "CutiveMono-Regular", "RammettoOne-Regular", "VT323-Regular", "TwinkleStar-Regular", "Inika-Regular",
        "AmiriQuran-Regular", "Andika-Regular", "OverpassMono", "Italianno-Regular", "Inconsolata", "BIZUDPMincho-Regular",
        "ShareTechMono-Regular", "Ramabhadra-Regular",
    ]

    COLORNAMES = [
        "black", "blue", "brown", "cyan", "green", "grey", "olive", "orange", "pink", "purple", "red"
    ]

    def __init__(self, base_dir, data_path=None, rand_label=False, rand_style=False):
        # self.data = data
        raw = json.load(open(data_path))
        self.base_dir = base_dir
        self.data = hfDataset.from_dict(raw)
        self.rand_label = rand_label
        self.rand_style = rand_style

    @staticmethod
    def load_dataset(base_dir, path):
        return CharInpaintDatasetTest(base_dir, path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(
            self.base_dir, self.data[idx]['path'] + ".png")
        mask_path = os.path.join(
            self.base_dir, self.data[idx]['path'] + "_mask.png")
        image = np.array(Image.open(image_path).convert(
            "RGB"), dtype=np.float32)
        image = image.transpose(2, 0, 1)
        image = image / 127.5 - 1.0

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.
        mask = mask[None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        d = self.data[idx]['coordinate']
        mask[:, d[0]:d[1]+1,
             d[2]:d[3]+1] = 1
        masked_image = image * (mask < 0.5)

        coordinate = self.data[idx]['coordinate']
        if self.rand_label:
            # change label
            charlen = len(self.data[idx]['chars'])
            chars = self.WORDS[charlen][idx % len(self.WORDS[charlen])]
        else:  # gt text
            chars = self.data[idx]['chars']
        style = self.data[idx]['styles']
        if self.rand_style:
            font = self.FONTNAMES[idx % len(self.FONTNAMES)]
            color = self.COLORNAMES[idx % len(self.COLORNAMES)]
            style = [font, color]

        item = {
            "image": image,
            "masked_image": masked_image,
            "mask": mask,
            "coordinate": coordinate,
            "rawpath": image_path,
            "chars": chars,
            "rawstyle": style,
            "rawlabel": self.data[idx]['chars'],
            "style": prepare_style_chars(chars, style),
            "idx": idx,
        }
        return item


class CharInpaintDataset:
    def __init__(self, config):
        self.config = config
        self.size = config.size
        self.data = []

        self.generator = Generator()
        # create base augmentation method
        self.augment_fn = {}
        for augname, augconf in config["augconf"].items():
            augcls = AUG_CLASSES[augname]
            self.augment_fn[augname] = augcls(self.size, augconf)

        # create load
        self.base_image_folder = {}
        max_num = config.get("max_num", None)
        for source, source_conf in config.dataconfs.items():
            self.base_image_folder[source] = source_conf["image_dir"]
            if max_num is not None:
                source_conf["max_num"] = min(
                    max_num, source_conf.get("max_num", max_num))
            if source_conf["type"] == "synth":
                labels = loadSynthOCRData(source, source_conf)
            elif source_conf["type"] == "scene":
                labels = loadSceneOCRData(source, source_conf)
            self.data.extend(labels)
            if "augconf" in source_conf:
                # * if custom augmentation
                augcls = AUG_CLASSES[source_conf["type"]]
                self.augment_fn[source] = augcls(
                    self.size, source_conf["augconf"])
            else:
                self.augment_fn[source] = self.augment_fn[source_conf["type"]]

        if max_num and len(self.data) > max_num:
            self.data = random.choices(self.data, k=max_num)
        print(f"In all collected {len(self.data)} sample")

    def __len__(self):
        return len(self.data)

    def style_dropout(self, style_mode, style, dropout_prob):
        def drop_out_fn(x, prob):
            return "" if np.random.rand() < prob else x
        res_style = []
        if style_mode == "same-same":  # we can drop out any of this
            res_style.append(drop_out_fn(style[0], dropout_prob[0]))
            res_style.append(drop_out_fn(style[1], dropout_prob[1]))
        elif style_mode == "same-diff":
            res_style.append(drop_out_fn(style[0], dropout_prob[0]))
            res_style.append(style[1])
        elif style_mode == "diff-same":
            res_style.append(style[0])
            res_style.append(drop_out_fn(style[1], dropout_prob[1]))
        elif style_mode == "diff-diff":
            res_style = style
        return res_style

    def __getitem__(self, idx):
        source = self.data[idx][0]
        path = self.data[idx][1]
        label = self.data[idx][2]
        if isinstance(label, list):
            label_idx = 0
            label = label[label_idx]
        img_path = os.path.join(self.base_image_folder[source], path)
        augfn = self.augment_fn[source]
        image = Image.open(img_path).convert("RGB")
        source_conf = self.config.dataconfs[source]

        if source_conf["type"] == "scene":
            points = self.data[idx][3]
            if source_conf.get("rand_mask_text", False):
                points = rand_expand_mask(points)
                print("hello")
            item = augfn(image, points=points)
        elif source_conf["type"] == "synth" and source_conf.get(
            "rand_mask_text", False
        ):  # multiple texts in synthetic image
            label_idx = torch.randint(0, len(self.data[idx][2]), ())
            label = self.data[idx][2][label_idx]
            points = self.data[idx][3][label_idx]
            # print(len(label), len(points))
            label, points = filter_center_label(label, points)
            direction = source_conf.get("direction", "none")
            if source_conf.get("direction_prob", None) is not None:
                if np.random.rand() < source_conf["direction_prob"]:
                    direction = "longer"
            points = rand_mask_points(sum([points], []), direction)
            item = augfn(image, points=points, generator=self.generator)
        else:  # no need todo center filter
            label = self.data[idx][2][label_idx]
            points = rand_mask_points(
                sum([self.data[idx][3][label_idx]], []), "none")
            item = augfn(image, points=points, generator=self.generator)

        item["chars"] = label
        item["idx"] = idx
        if len(self.data[idx]) == 5 and "synthtiger" in source.lower():  # has style information
            rawstyle = self.data[idx][-1]
            item['style'] = self.style_dropout(
                # same font-color by default
                source_conf.get('style_mode', 'same-same'),
                [rawstyle['fonts'][label_idx], rawstyle['colors'][label_idx]],
                source_conf.get('style_dropout', [0.0, 0.0])
            )
            item['style'] = prepare_style_chars(label, item['style'])
        else:
            item['style'] = prepare_style_chars(label, ["", ""])
        item['rawpath'] = img_path
        return item

    @staticmethod
    def export_to_Test(dataset, outdir):
        pass


class SimpleOCRData(Dataset):
    def __init__(self, base_dir, preprocess_fn) -> None:
        super().__init__()
        self.data = []
        # collect all images and labels
        all_paths = glob.glob(base_dir + "/**/char-*.png", recursive=True)
        all_paths = list(filter(lambda x: not "grid" in x, all_paths))
        for path in all_paths:
            label = path.split("/")[-2].split("-")[-1]
            self.data.append((path, label))

        if len(self.data) == 0:
            all_paths = glob.glob(base_dir + "/*-*.png", recursive=True)
            all_paths.sort()
            for path in all_paths:
                label = path.split("/")[-1].split("-")[-1].split(".")[0]
                self.data.append((path, label))
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index][0]).convert("RGB")
        img = self.preprocess_fn(img)
        return {"image": img[0], "label": self.data[index][1].lower()}
