import copy
import numpy as np
from synthtiger import components, utils
from synthtiger.components import Component, BaseFont, BaseTexture


class NoresizeTexture(BaseTexture):
    def __init__(self, paths=(), weights=(), alpha=(1, 1), grayscale=0, crop=0):
        super(NoresizeTexture, self).__init__(
            paths=paths,
            weights=weights,
            alpha=alpha,
            grayscale=grayscale,
            crop=crop,
        )

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        texture = self.data(meta)

        for layer in layers:
            height, width = layer.image.shape[:2]
            image = utils.resize_image(texture, (width, height))
            layer.image = utils.blend_image(layer.image, image, mode="normal")

        return meta


class FixRotate(components.Rotate):
    def __init__(self, angle=(10, 30, 45, 75, 90), ccw=0):
        super().__init__()
        self.angle = angle
        self.ccw = ccw

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        angle = meta.get("angle", np.random.choice(self.angle))
        ccw = meta.get("ccw", np.random.rand() < self.ccw)
        meta = {
            "angle": angle,
            "ccw": ccw,
        }
        return meta


class NamedColors(components.ColorMap):
    def __init__(self, pallate_name='TAB'):
        self.colors = self.get_matplotlib_colors(pallate_name)

    def get_matplotlib_colors(self, pallate_name):
        import matplotlib.colors as mcolors
        if pallate_name == 'XKCD':
            rawcolors = mcolors.XKCD_COLORS
        elif pallate_name == "TAB":
            # add black into the pallate
            rawcolors = mcolors.TABLEAU_COLORS
            rawcolors['tab:black'] = "#000000"
        elif pallate_name == "SUBSET":
            selectcolors = list(map(lambda x: x.split(
                ":")[1], list(mcolors.TABLEAU_COLORS.keys())))
            selectcolors.append("black")
            rawcolors = {}
            for x in selectcolors:
                if x == "gray":
                    x = "grey"
                rawcolors[x] = mcolors.XKCD_COLORS["xkcd:" + x]

        colors = {}
        for name in rawcolors.keys():
            color = rawcolors[name]
            color = mcolors.to_rgba(color)
            color = [int(x * 255) for x in color]
            name = name.split(":")[-1]
            colors[name] = color
        return colors

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        if meta.get('rgba', None) is None:
            rand_idx = np.random.randint(0, len(self.colors))
            color_name = list(self.colors.keys())[rand_idx]
            color = self.colors[color_name]
            meta['color_name'] = color_name
            meta['rgba'] = color
        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        for layer in layers:
            image = np.empty(layer.image.shape)
            image[..., :] = self.data(meta)
            layer.image = utils.blend_image(image, layer.image, mask=True)
        return meta

    def data(self, meta):
        rgba = meta['rgba']
        return rgba


class MultiFontSampler(Component):
    def __init__(
        self,
        paths=(),
        weights=(),
        size=(16, 48),
        bold=0,
        vertical=False,
        mode="same",
    ):
        self.font_sampler = BaseFont(
            paths=paths, weights=weights, size=size, bold=bold, vertical=vertical)
        self.size = size
        self.mode = mode

    def sample(self, meta=None):
        if meta is None:
            meta = {'num': 1}
        if meta.get('fonts', None) is None:
            num = meta.get('num')
            fonts = []
            if self.mode == "same":
                font = self.font_sampler.sample(meta)
                fonts = [font] * num  # all with same font
            elif self.mode == "random":
                for _ in range(num):
                    if len(fonts) == 0:
                        fonts.append(self.font_sampler.sample(meta))
                    else:
                        prev_font = fonts[-1]
                        tmp_meta = copy.deepcopy(prev_font)
                        tmp_meta.pop('path')  # sample another font name
                        tmp_meta = self.font_sampler.sample(tmp_meta)
                        fonts.append(tmp_meta)
            elif self.mode.startswith("group"):
                pass
            else:
                raise ValueError("Unknown mode: {}".format(self.mode))
            meta['fonts'] = fonts
        return meta


class MultiColorSampler(Component):
    def __init__(self, pallate_name="TAB", mode="same"):
        self.color_sampler = NamedColors(pallate_name)
        self.mode = mode

    def sample(self, meta=None):
        if meta is None:
            meta = {'num': 1}
        if meta.get('colors', None) is None:
            num = meta.get('num')
            colors = []
            if self.mode == "same":
                color = self.color_sampler.sample()
                colors = [color] * num  # all with same color
            elif self.mode == "random":
                colors = [self.color_sampler.sample() for _ in range(num)]
            elif self.mode.startswith("group"):
                pass
            else:
                raise ValueError("Unkown mode: {}".format(self.mode))
            meta['colors'] = colors
        return meta

    def data(self, meta=None):
        colors = meta['colors']
        return colors

    def apply(self, layers, meta=None):
        self.color_sampler.apply(layers, meta)


class WordSampler(Component):
    vocab = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
        "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    ]

    def __init__(
        self,
        prob=0.0,
        type="uniform",
        max_num=1,
        max_len=16,
        weight=None,
        level=1,
        repeat_text=False,
        same_len=False,
        corpus_conf=None,
        do_cache=False,
        source_weight=[1, 0],
    ):
        # if weight is None:
        # self.weight = weights # weight for length
        self.max_num = max_num
        self.max_len = max_len
        self.corpus_conf = corpus_conf
        self.repeat_text = repeat_text
        self.same_len = same_len
        if self.corpus_conf is not None:
            self.corpus = components.BaseCorpus(**corpus_conf)
        else:
            self.corpus = None
        self.do_cache = do_cache

    def cache_style(self, texts):
        # cache length
        if not hasattr(self, "len_cache"):
            setattr(self, "len_cache", [len(x) for x in texts])
            return self.len_cache
        else:
            return self.len_cache

    def clear_cache(self):
        delattr(self, "len_cache")

    def sample_rand_text(self, level):
        text_lens = getattr(self, "len_cache", None)

        texts = []
        if self.level == 1:
            if text_lens is None:
                text_len = np.random.randint(1, self.max_len + 1)
                texts.append("".join(np.random.choice(
                    self.vocab, text_len, replace=True)))
            else:
                text_len = text_lens[0]
                texts.append("".join(np.random.choice(
                    self.vocab, text_len, replace=True)))

        elif self.level == 2:
            if text_lens is None:
                if self.max_num == 1:
                    text_num = 1
                else:
                    # make sure they are close in length
                    text_num = np.random.randint(2, self.max_num + 1)
                mid_len = np.random.randint(2, self.max_len + 1)
                for _ in range(text_num):
                    text_len = max(2, mid_len + np.random.randint(-3, 4))
                    texts.append("".join(np.random.choice(
                        self.vocab, text_len, replace=True)))
            else:
                for cur_len in text_lens:
                    texts.append("".join(np.random.choice(
                        self.vocab, cur_len, replace=True)))

        if self.do_cache:
            self.cache_style(texts)
        return texts

    def sample_corpus(self, level):
        text_lens = getattr(self, "len_cache", None)

        texts = []

        def sample_text(text_len):
            label = self.corpus.data(self.corpus.sample())
            while len(label) != text_len:
                label = self.corpus.data(self.corpus.sample())
            return label
        if self.level == 1:
            if text_lens is None:
                text_len = np.random.randint(1, self.max_len + 1)
                texts.append(sample_text(text_len))
            else:
                text_len = text_lens[0]
                texts.append(sample_text(text_len))
        elif self.level == 2:
            if text_lens is None:
                if self.max_num == 1:
                    text_num = 1
                else:
                    # make sure they are close in length
                    text_num = np.random.randint(2, self.max_num + 1)
                mid_len = np.random.randint(2, self.max_len + 1)
                for _ in range(text_num):
                    text_len = max(2, mid_len + np.random.randint(-3, 4))
                    text = sample_text(text_len)
                    if self.repeat_text and len(texts) != 0:
                        texts.append(texts[-1])
                    else:
                        texts.append(text)
            else:
                for cur_len in text_lens:
                    text = sample_text(cur_len)
                    texts.append(text)
        if self.do_cache:
            self.cache_style(texts)
        return texts

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        if "text" not in meta:
            if self.corpus_conf is None:
                texts = self.sample_rand_text(self.level)
            else:
                if np.random.rand() < 0.2:
                    texts = self.sample_rand_text(self.level)
                else:
                    texts = self.sample_corpus(self.level)
            meta['text'] = texts
        return meta

    def data(self, meta):
        texts = meta["text"]
        return texts
