import os
import json
import collections
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn

from transformers.utils import logging
from transformers import PreTrainedTokenizer, PreTrainedModel, PretrainedConfig


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {
    "vocab_file": "char_vocab.json",
}


class CharTokenizer(PreTrainedTokenizer):
    vocab_files_names: Dict[str, str] = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        pad_token=" ",
        unk_token=" ",
        cls_token=None,
        model_max_length=16,
        **kwargs,
    ):
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            cls_token=cls_token,
            model_max_length=model_max_length,
            **kwargs,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find char vocab file at '{vocab_file}")
        with open(vocab_file, "r", encoding="utf-8") as reader:
            self.vocab = json.load(reader)  # a list of chars
            self.vocab = {x: int(i) for i, x in enumerate(self.vocab)}
            assert self.vocab[pad_token] == 0  # always first
        self.model_max_length = model_max_length
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        if kwargs.get("cliptokenizer", False):
            # map to CLIPencoder token id
            print("Map to clip tokens...")
            cliptokenizer = kwargs.get("cliptokenizer")
            self.cls_token = cliptokenizer.bos_token
            self.cls_token_id = cliptokenizer.bos_token_id
            for ch in self.vocab:
                vocab_id = cliptokenizer.convert_tokens_to_ids(ch)
                self.vocab[ch] = vocab_id
            self.ids_to_tokens = collections.OrderedDict(
                [(ids, tok) for tok, ids in self.vocab.items()]
            )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "")
                + VOCAB_FILES_NAMES["vocab_file"],
            )
        else:
            vocab_file = (
                filename_prefix + "-" if filename_prefix else ""
            ) + save_directory
        json.dump(self.vocab, open(vocab_file, "w", encoding="utf-8"))
        return (vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return [self.cls_token_id] + token_ids_0

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)


class CharEmbedderConfig(PretrainedConfig):
    model_type = "char_embedder"

    def __init__(
        self,
        vocab_size=95,
        embedding_dim=32,
        max_length=16,
        padding_idx=0,
        attention_head_dim=2,
        encoder_config=None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.padding_idx = padding_idx
        self.attention_head_dim = attention_head_dim
        self.encoder_config = encoder_config


class CharEmbedder(PreTrainedModel):
    config_class = CharEmbedderConfig
    base_model_prefix = "charembedder"

    def __init__(self, config: CharEmbedderConfig, **kwargs):
        super().__init__(config, **kwargs)
        print("Initilize char embedder...")
        self.max_length = config.max_length
        self.embed_dim = config.embedding_dim
        self.char_embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.padding_idx
        )
        self.position_embedding = nn.Embedding(
            config.max_length, config.embedding_dim)
        self.register_buffer(
            "position_ids", torch.arange(config.max_length).expand((1, -1))
        )

        if hasattr(config, "encoder") and config.encoder is not None:
            encoder_config = config.encoder
            self.projector = nn.Linear(
                self.embed_dim, encoder_config.contrast_hidden_dim
            )
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoarderLayer(
                    d_model=encoder_config.contrast_hidden_dim,
                    nhead=encoder_config.num_heads,
                    dim_feedforward=encoder_config.contrast_hidden_dim,
                ),
                num_layers=encoder_config.num_encoder_layers,
                norm=nn.LayerNorm(
                    encoder_config.contrast_hidden_dim
                ),  # layer norm same as CLIPTextModel
            )

    def forward(
        self, input_ids=None, position_ids=None, input_embeds=None, do_encode=False
    ):
        assert (input_ids is not None) ^ (
            input_embeds is not None
        ), "One and only one of char_input_ids and char_input_embeds should be provided."
        if input_embeds is None:
            embeds = self.char_embedding(input_ids)
            seq_length = input_ids.shape[-1]
            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]
            position_embedding = self.position_embedding(position_ids)
            embeds = embeds + position_embedding
        else:
            embeds = input_embeds

        if (
            hasattr(self, "encoder") and do_encode
        ):  # project to contrastive hidden space
            embeds = self.projector(embeds)
            embeds = self.encoder(embeds)

        return (embeds,)
