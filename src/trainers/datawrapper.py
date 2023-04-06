from .utils import get_obj_from_str
from torch.utils.data import DataLoader
from ..dataset import char_inpaint_collate_fn, CharInpaintDataset
import pytorch_lightning as pl
import torch
import torch.distributed as torchdist


class WrappedDataModule(pl.LightningDataModule):
    def __init__(self, data_config, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config = data_config
        self.batch_size = data_config.batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.train = CharInpaintDataset(self.config.train)
            self.val = CharInpaintDataset(self.config.validation)
        if stage == "test" or stage == "predict":
            self.val = CharInpaintDataset(self.config.test)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=char_inpaint_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=char_inpaint_collate_fn,
        )
