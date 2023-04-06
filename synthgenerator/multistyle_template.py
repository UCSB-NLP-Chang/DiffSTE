import os
import cv2
import csv
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from synthtiger import components, layers, templates, utils
from new_components import (
    FixRotate, WordSampler, MultiColorSampler, MultiFontSampler
)
from utils import _blend_images


def clean_fontname(fontname):
    fontname = fontname.rstrip(".ttf")
    pass


class SynthForCharDiffusion(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        self.level = config.get("level", 1)
        self.coord_output = config.get("coord_output", True)
        self.mask_output = config.get("mask_output", True)
        self.glyph_coord_output = config.get("glyph_coord_output", True)
        self.glyph_mask_output = config.get("glyph_mask_output", True)
        self.vertical = config.get("vertical", False)
        self.multiline = config.get("multiline", False)
        self.max_word_count = config.get("max_word_count", 10)
        self.visibility_check = config.get("visibility_check", False)
        self.corpus = WordSampler(**config.get("word_sampler", {}))
        self.font = MultiFontSampler(**config.get("font", {}))
        if config.get("named_color", None) is not None:
            self.named_color = MultiColorSampler(**config.get('named_color'))
        else:
            self.named_color = None
        self.color = components.RGB(**config.get("color", {}))
        self.colormap2 = components.GrayMap(**config.get("colormap2", {}))
        self.colormap3 = components.GrayMap(**config.get("colormap3", {}))
        self.layout = components.FlowLayout(**config.get("layout", {}))
        self.text_layout = components.FlowLayout(
            **config.get("text_layout", {}))
        self.style_mode = self.font.mode + "-" + self.named_color.mode
        self.split_text = config.get("split_text", False)

        self.texture = components.Switch(
            components.BaseTexture(), **config.get("texture", {}))
        self.style = components.Switch(
            components.Selector(
                [
                    components.TextBorder(),
                    components.TextShadow(),
                    components.TextExtrusion(),
                ]
            ),
            **config.get("style", {}),
        )

        self.max_length = 256

        self.shape = components.Switch(
            components.Selector(
                [components.ElasticDistortion(), components.ElasticDistortion()]),
            **config.get("shape", {}),
        )

        self.transform = components.Switch(
            components.Selector(
                [
                    components.Perspective(),
                    components.Perspective(),
                    components.Trapezoidate(),
                    components.Trapezoidate(),
                    components.Skew(),
                    components.Skew(),
                    components.Rotate(),
                    FixRotate(),
                ]
            ),
            **config.get("transform", {}),
        )
        self.fit = components.Fit()
        self.pad = components.Switch(components.Pad(), **config.get("pad", {}))
        self.postprocess = components.Iterator(
            # self.postprocess = components.Selector(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.GaussianBlur()),
                # components.Switch(components.Resample()),
                components.Switch(components.MedianBlur()),
            ],
            **config.get("postprocess", {}),
        )
        # all same, same color, same font, diff
        self.style_type = config.get('style_type', "bothsame")
        self._adjust_to_level(self.level)

    def _adjust_to_level(self, level):
        self.corpus.level = level
        if level == 1:
            # adjust probibility of any transformation
            # self.texture.prob = 0.0
            self.style.prob = 0.0
            self.transform.prob = 0.0
            # self.corpus.max_num = 1
        elif level == 2:
            # self.texture.prob = 0.0
            pass
        elif level == 3:
            self.texture.prob = 1.0

    def _visualize_multi_words(self, texts, fonts):
        text_layers = []
        char_layers = []
        max_length = 0  # maximium width according to the rendered text
        layout_style = self.layout.sample(meta={"vertical": self.vertical})
        for i in range(len(texts)):
            text = texts[i]
            font = fonts[i]
            chars = utils.split_text(text, reorder=True)
            text = "".join(chars)
            chars = utils.split_text(text, reorder=True)
            text = "".join(chars)
            char_layer = [layers.TextLayer(char, **font) for char in chars]
            self.layout.apply(char_layer, meta=layout_style)
            char_layers.append(char_layer)  # as a group
            text_layer = layers.Group(char_layer).merge()
            max_length = max(max_length, text_layer.size[0])
            text_layers.append(text_layer)
        return text_layers, char_layers

    def _get_bg_layer(self, size, color, level=1):
        layer = layers.RectLayer(size)
        self.color.apply([layer], meta=color)
        self.texture.apply([layer], meta={"color": color})
        meta = self.texture.sample(
            meta={"meta": {"w": int(size[0]), "h": int(size[1])}}
        )
        self.texture.apply([layer], meta=meta)
        return layer

    def _generate_color(self, level, texts):
        if level == 1:
            if self.named_color is not None:
                fg_color = self.named_color.sample()
            else:
                fg_color = self.color.sample()
            fg_style = self.style.sample()
            mg_color = None
            mg_style = None
            _, bg_color = self.colormap2.sample()
        elif level == 2:
            if self.named_color is not None:
                fg_color = self.named_color.sample(meta={"num": len(texts)})
            else:
                fg_color = self.color.sample()
            fg_style = self.style.sample()
            _, bg_color = self.colormap2.sample()
            mg_color = None
            mg_style = None
        else:
            assert True, "Should not be here"
            mg_color = self.color.sample()
            fg_style = self.style.sample()
            mg_style = self.style.sample()

            if self.named_color is not None:
                fg_color = self.named_color.sample(meta={"num": len(texts)})
            else:
                fg_color = self.color.sample()
            if fg_style["state"]:
                bg_color, style_color = self.colormap3.sample()
                fg_style["meta"]["meta"]["rgb"] = style_color["rgb"]
            else:
                bg_color = self.colormap2.sample()
        return fg_color, fg_style, mg_color, mg_style, bg_color

    def _sample_texts(self, level=1):
        texts = self.corpus.data(self.corpus.sample())
        if self.split_text:
            new_texts = []
            ok = False
            left_num = np.random.randint(1, 5)
            right_num = np.random.randint(1, 5)
            for text in texts:
                if len(text) < 3:
                    continue
                ok = True
                split_idx1 = np.random.choice(range(1, len(text) - 1))
                split_idx2 = np.random.choice(range(split_idx1 + 1, len(text)))
                left, mid, right = text[:split_idx1], text[split_idx1:split_idx2], text[split_idx2:]
                text = left + " " * left_num + mid + " " * right_num + right
                new_texts.append(text)
            texts = new_texts
            if not ok:
                raise ValueError("Not a good split")
        return texts

    def generate(self):
        # sample words to visualize
        texts = self._sample_texts(self.level)
        max_text_len = max([len(x) for x in texts])
        # sample font
        size = np.random.randint(
            self.font.size[0],
            max(min(self.font.size[1] + 1, self.max_length //
                max_text_len + 1), self.font.size[0] + 1)
        )
        fonts = self.font.sample(
            meta={
                "num": len(texts),
                "vertical": self.vertical,
                "size": size,  # decrease the probability of too-wide image
            },
        )['fonts']
        font_names = [os.path.basename(x['path'].rstrip(".ttf")) for x in fonts]
        # visualize text from character-level to word-level
        text_layers, char_layers = self._visualize_multi_words(texts, fonts)

        # transform first then layout
        transform = self.transform.sample()
        fg_color, fg_style, mg_color, mg_style, bg_color = self._generate_color(
            self.level, texts=texts)
        if self.named_color is not None:
            color_names = [x['color_name'] for x in fg_color['colors']]
        else:
            color_names = 'unkown'

        text_glyph_layers = [text_layer.copy() for text_layer in text_layers]
        # apply color
        for i in range(len(char_layers)):
            char_layer = char_layers[i]
            text_layer = text_layers[i]
            color_meta = fg_color['colors'][i]
            self.named_color.apply([text_layer], color_meta)

        # apply other styles
        for char_layer, text_layer, text_glyph_layer in zip(char_layers, text_layers, text_glyph_layers):
            self.style.apply([text_layer], fg_style)
            self.style.apply([text_glyph_layer], fg_style)
            self.transform.apply([text_layer, *char_layer], transform)
            self.transform.apply([text_glyph_layer], transform)
            self.fit.apply([text_layer, *char_layer])
            self.fit.apply([text_glyph_layer])
            for char_layer in char_layer:
                char_layer.topleft -= text_layer.topleft

        # final layout for multiple words
        text_group = layers.Group(
            text_layers,
        )
        text_glyph_group = layers.Group(
            text_glyph_layers
        )
        text_sample = self.text_layout.sample()
        text_sample["length"] = self.max_length
        self.text_layout.apply(
            text_group,
            meta=text_sample,
        )
        self.text_layout.apply(
            text_glyph_group,
            meta=text_sample,
        )
        all_char_layers = sum(char_layers, [])
        self.fit.apply(
            [
                text_group,
                *text_layers,
                *all_char_layers,
            ]
        )
        self.fit.apply(
            [text_glyph_group, *text_glyph_layers]
        )
        for text_layer in text_layers:
            text_layer.topleft -= text_group.topleft
        for text_glyph_layer in text_glyph_layers:
            text_glyph_layer.topleft -= text_glyph_group.topleft
        for char_layer, text_layer in zip(char_layers, text_layers):
            for single_char in char_layer:
                single_char.topleft += text_layer.topleft

        text_group = text_group.merge()
        if text_group.height > self.max_length or text_group.width > self.max_length:
            raise RuntimeError("Not a valid generation")
        max_size = np.random.randint(max(text_group.height, text_group.width),
                                     self.max_length)
        up = np.random.randint(0, max_size - text_group.height)
        left = np.random.randint(0, max_size - text_group.width)
        pad_meta = self.pad.sample(
            meta={
                "meta": {
                    "pxs": [up, max_size - left - text_group.width,
                            max_size - up - text_group.height, left]
                }
            }
        )
        self.pad.apply([text_group], pad_meta)

        for text_layer in text_layers:
            text_layer.topleft -= text_group.topleft
        for text_glyph_layer in text_glyph_layers:
            text_glyph_layer.topleft -= text_glyph_group.topleft
        for char_layer, text_layer in zip(char_layers, text_layers):
            for single_char in char_layer:
                single_char.topleft -= text_group.topleft

        # background image layer
        bg_layer = self._get_bg_layer(
            text_group.size,
            color=bg_color,
            level=self.level,
        )
        bg_layer.topleft = text_group.topleft

        if self.level == 1:
            togrey = cv2.cvtColor(text_group.output().astype(
                np.uint8), cv2.COLOR_RGBA2GRAY)
            ret, mask = cv2.threshold(togrey, 0, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            text_image = text_group.output().astype(np.uint8)
            bg_image = bg_layer.output().astype(np.uint8)
            text_image = cv2.bitwise_and(text_image, text_image, mask=mask)
            bg_image = cv2.bitwise_or(bg_image, bg_image, mask=mask_inv)
            image = cv2.add(bg_image, text_image)
            image_layer = [layers.Layer(image)]
            image = image_layer[0].output()
        elif self.level == 2:
            togrey = cv2.cvtColor(text_group.output().astype(
                np.uint8), cv2.COLOR_RGBA2GRAY)
            ret, mask = cv2.threshold(togrey, 0, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            text_image = text_group.output().astype(np.uint8)
            bg_image = bg_layer.output().astype(np.uint8)
            text_image = cv2.bitwise_and(text_image, text_image, mask=mask)
            bg_image = cv2.bitwise_or(bg_image, bg_image, mask=mask_inv)
            image = cv2.add(bg_image, text_image)
            image_layer = [layers.Layer(image)]
            self.postprocess.apply(image_layer)
            image = image_layer[0].output()
        else:
            image = _blend_images(text_group.output(),
                                  bg_layer.output(),
                                  self.visibility_check)
            image_layer = [layers.Layer(image)]
            self.postprocess.apply(image_layer)
            image = image_layer[0].output()

        text_bboxes = [text_layer.bbox for text_layer in text_layers]
        bboxes = [[char.bbox for char in char_layer]
                  for char_layer in char_layers]
        label = texts
        glyph_image = text_glyph_group.output(bbox=text_group.bbox)

        if image.shape[0] > self.max_length or image.shape[1] > self.max_length:
            raise RuntimeError("Not a valid generation")
        if image.shape[0] < 20 or image.shape[1] < 20:
            raise RuntimeError("Not a valid generation")

        coords = [[[x, y, x + w, y + h] for x, y, w, h in bbox]
                  for bbox in bboxes]
        text_coords = [[x, y, x + w, y + h] for x, y, w, h in text_bboxes]
        data = {
            "image": image,
            "glyph_image": glyph_image,
            "label": label,  # N elem list
            "text_bbox": text_coords,  # N text bbox
            "char_bbox": coords,  # N list of char bbox
            "size": image.size,
            # N list of [font_name, color_name]
            "style": {
                "fonts": font_names,
                "colors": color_names,
            }
        }
        return data

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, "glyphs"), exist_ok=True)
        OmegaConf.save(config=self.config, f=open(
            os.path.join(root, "data_config.yaml"), "w"))
        data_path = os.path.join(root, "data.csv")
        self.data_file = open(data_path, "w", encoding="utf-8")
        self.csv_writer = csv.writer(
            self.data_file,
            delimiter=",",
        )
        if self.named_color is not None:
            self.csv_writer.writerow(
                ["path", "label", "size", "text_bbox", "char_bbox", "style"])
        else:
            self.csv_writer.writerow(
                ["path", "label", "size", "text_bbox", "char_bbox"])

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        text_bbox = data["text_bbox"]
        char_bbox = data["char_bbox"]
        glyph = data.get("glyph_image", None)
        style = data.get("style", None)

        shard = str(idx // 10000)
        image_key = os.path.join(shard, f"{idx}.png")
        image_path = os.path.join(root, "images", image_key)
        glyph_path = os.path.join(root, "glyphs", image_key)
        os.makedirs(os.path.dirname(glyph_path), exist_ok=True)
        img = Image.fromarray(glyph[..., 3].astype(np.uint8))
        img.save(glyph_path)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_path, quality=100)
        if style is not None:
            self.csv_writer.writerow(
                [image_key, label, image.size, text_bbox, char_bbox, style])
        else:
            self.csv_writer.writerow(
                [image_key, label, image.size, text_bbox, char_bbox])

    def end_save(self, root):
        self.data_file.close()
