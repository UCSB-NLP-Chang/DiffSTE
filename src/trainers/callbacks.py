import os
import glob
import json
import torch
import pandas as pd
import editdistance
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import torch.distributed as torchdist
from tqdm import tqdm

from PIL import Image
from typing import List
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from ..dataset import char_inpaint_collate_fn, SimpleOCRData, CharInpaintDataset
from ..abinet import MultiLosses, prepare_label, create_ocr_model, postprocess
from ..abinet import preprocess as abinet_preprocess


class OCRAccLogger(Callback):
    """ Use pretrained ABI-Net to evaluate text accuracy in generated images """

    def __init__(self, train_eval_conf, val_eval_conf, base_log_dir, mode="synthtext"):
        super().__init__()
        print("Initializing OCR acc logger...")
        # each length, 1 to 16, sample 10 images and generate, in all
        self.ocr_model = None
        self.charset = None
        self.log_dir = os.path.join(base_log_dir, "images_gen")
        self.train_eval = CharInpaintDataset(train_eval_conf)
        self.val_eval = CharInpaintDataset(val_eval_conf)
        self.mode = mode

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.global_rank == 0:
            send = [self.log_dir] * torchdist.get_world_size()
        else:
            send = [None] * torchdist.get_world_size()
        torchdist.broadcast_object_list(send, src=0)
        if trainer.global_rank != 0:
            rec = send[trainer.local_rank]
            #! update log_dir
            self.log_dir = rec

    @torch.no_grad()
    def generate_images(self, model, eval_data, log_dir):
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_data, shuffle=False
        )
        loader = DataLoader(
            eval_data,
            batch_size=1,
            num_workers=0,  # try if this solves
            sampler=test_sampler,
            collate_fn=char_inpaint_collate_fn,
        )  # the batch size on each GPU
        test_sampler.set_epoch(0)
        print(
            f"Generating {len(loader)} images on rank: {torchdist.get_rank()} to {log_dir}")
        signal = torch.tensor(0, device=model.device)  # ! used for synchronize
        # generate
        for img_idx, batch in enumerate(tqdm(loader)):
            img_idx = batch["idx"][0]  # ! real index from dataset
            generation_kwargs = {
                "num_inference_steps": 30,
                "num_sample_per_image": 3,
                "guidance_scale": 7.5,
                "generator": torch.Generator(model.device).manual_seed(42),
            }
            char2coordinate = {
                f"{img_idx}-" + batch["chars"][i]: batch["coordinate"][i]
                for i in range(len(batch["chars"]))
            }
            with torch.no_grad():
                images = model.log_images(
                    batch, generation_kwargs, stage="validation")
            for sample_caption, all_char_results in images.items():
                sample_caption = str(sample_caption.split("-")[-1])
                sample_caption = f"{img_idx}-" + sample_caption
                os.makedirs(os.path.join(
                    log_dir, sample_caption), exist_ok=True)
                # save full image
                for i, x in enumerate(all_char_results):
                    img = ToPILImage()(x)
                    img.save(os.path.join(
                        log_dir, sample_caption, f"full-{i}.png"))
                grid = make_grid(
                    torch.cat(
                        [
                            (batch["image"] / 2.0 + 0.5).clamp(0.0, 1.0),
                            (batch["masked_image"] / 2.0 + 0.5).clamp(0.0, 1.0),
                            all_char_results,
                        ]
                    ),
                    nrow=generation_kwargs["num_sample_per_image"] + 2,
                    pad_value=1,
                    padding=2,
                )
                img = ToPILImage()(grid)
                img.save(os.path.join(log_dir, sample_caption, "full-grid.png"))

                if self.mode == "synthtext":
                    d = char2coordinate[sample_caption]
                    all_char_results = [
                        x[:, d[0]: d[1] + 1, d[2]: d[3] + 1] for x in all_char_results
                    ]
                    for i, x in enumerate(all_char_results):
                        img = ToPILImage()(x)
                        img.save(os.path.join(
                            log_dir, sample_caption, f"char-{i}.png"))
                    gt = (batch["image"][0] / 2.0 +
                          0.5).clamp(0.0, 1.0)  # only one
                    gt = gt[:, d[0]: d[1] + 1, d[2]: d[3] + 1]
                    all_char_results.insert(0, gt)
                    grid = make_grid(
                        all_char_results,
                        nrow=generation_kwargs["num_sample_per_image"] + 1,
                        pad_value=1,
                        padding=3,
                    )
                    img = ToPILImage()(grid)
                    img.save(os.path.join(log_dir, sample_caption, "grid.png"))
                elif self.mode == "textocr":
                    pass

        torchdist.all_reduce(
            signal
        )  # ! wait for all processes to finish image generation

    @torch.no_grad()
    def ocr_eval(self, log_dir, device):
        # evaluate
        ocr_data = SimpleOCRData(log_dir, abinet_preprocess)
        batch_results = []
        for batch in DataLoader(ocr_data, batch_size=32, shuffle=False, num_workers=2):
            batch["image"] = batch["image"].to(device)
            outputs = self.ocr_model(batch["image"], mode="validation")
            celoss_inputs = outputs[:3]
            gt_ids, gt_lengths = prepare_label(
                batch["label"], self.charset, device)
            loss = MultiLosses(True)(celoss_inputs, gt_ids, gt_lengths)
            text_preds = outputs[-1]
            pt_text, _, _ = postprocess(text_preds, self.charset, "alignment")
            batch_res = {"[loss]": loss.item() * len(batch["label"])}

            assert len(pt_text) == len(batch["label"])
            for pred, label in zip(pt_text, batch["label"]):
                if not label in batch_res:
                    batch_res[label] = {
                        "trial": 0,
                        "success": 0,
                        "edist": 0,
                        "pred": [],
                    }
                batch_res[label]["trial"] += 1
                batch_res[label]["edist"] += editdistance.eval(pred, label)
                batch_res[label]["success"] += pred.lower() == label.lower()
                batch_res[label]["pred"].append(pred)
            batch_results.append(batch_res)
        acc = self.reduce_results(batch_results)
        res = {"[raw]": batch_results, "[acc]": acc}
        return res

    def reduce_results(self, batch_results, return_raw=False):
        def _reduce(results):
            loss_all, trial_num = 0, 1e-12
            all_res = {}
            for batch_result in results:
                for label, res in batch_result.items():
                    if label == "[loss]":
                        loss_all += res
                        continue
                    if label not in all_res:
                        all_res[label] = {
                            "trial": 0,
                            "success": 0,
                            "edist": 0,
                            "pred": [],
                        }
                    for k, v in res.items():
                        all_res[label][k] += v
            trial_num += sum([x["trial"] for x in all_res.values()])
            sample_sum = len(all_res) + 1e-12
            trial_success = sum([x["success"] for x in all_res.values()])
            sample_success = sum(
                [1 for x in all_res.values() if x["success"] > 0])
            all_success_num = sum(
                [1 for x in all_res.values() if x["success"] == x["trial"]]
            )
            trial_edit_dist = sum([x["edist"] for x in all_res.values()])
            res = {
                "loss": loss_all / trial_num,
                "sample_num": int(sample_sum),
                "trial_num": int(trial_num),
                "trial_acc": trial_success / trial_num,
                "sample_acc": sample_success / sample_sum,
                "sample_oacc": all_success_num / sample_sum,
                "trial_edist": trial_edit_dist / trial_num,
            }
            return res

        res = {"[all]": _reduce(batch_results)}
        max_len = 0
        for batch_result in batch_results:
            for label in batch_result.keys():
                if label == "[loss]":
                    continue
                if len(label) > max_len:
                    max_len = len(label)

        for i in range(1, max_len + 1):
            len_samples = list()
            for batch_res in batch_results:
                len_samples.append(
                    dict(
                        filter(
                            lambda x: len(x[0]) == i and x[0] != "[loss]",
                            batch_res.items(),
                        )
                    )
                )
            res[f"[len-{i}]"] = _reduce(len_samples)
        return res

    def reduce_res_group(self, all_res):
        max_len = max([len(x) for x in all_res.keys()])
        length_groups = {}
        for i in range(1, 30):
            if i > max_len:
                break
            len_results = dict(
                filter(lambda x: len(x[0]) == i, all_res.items()))
            len_res = self.reduce_results(list(len_results.values()))
            length_groups[f"[len-{i}]"] = len_res
        return length_groups

    @rank_zero_only
    def run_test(self, eval_data, pl_module):
        self.charset, self.ocr_model = create_ocr_model(device=pl_module.device)
        log_dir_path = os.path.join(self.log_dir, "test", f"results-test")
        os.makedirs(log_dir_path, exist_ok=True)
        reduced_res = self.generate_images(
            pl_module, eval_data=eval_data, log_dir=log_dir_path
        )
        with open(os.path.join(self.log_dir, "test", f"results-test.json"), "w") as f:
            json.dump(reduced_res, f)
        acc = reduced_res["[acc]"]
        for k in acc.keys():
            acc[k]["name"] = k
        df = pd.DataFrame.from_records(list(acc.values()))
        return df

    def raw_ocr_eval(self, device, eval_dir, res_path):
        self.charset, self.ocr_model = create_ocr_model(device=device)
        reduced_res = self.ocr_eval(eval_dir, device)
        with open(res_path, "w") as f:
            json.dump(reduced_res, f, indent=2)
        del self.ocr_model
        torch.cuda.empty_cache()

    #########################################################################
    # pytorch_lightning related code
    #########################################################################

    @rank_zero_only
    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        os.makedirs(os.path.join(self.log_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "val"), exist_ok=True)

    def log_ocr_eval(self, trainer, pl_module, log_dir_path, split="train"):
        self.charset, self.ocr_model = create_ocr_model(device=pl_module.device)
        reduced_res = self.ocr_eval(log_dir_path, pl_module.device)
        style_metric = self.style_metric.eval_dir(log_dir_path)
        name = os.path.join(
            self.log_dir, split, f"results-step_{trainer.global_step}.json"
        )
        reduced_res['[style]'] = style_metric
        with open(
            name,
            "w",
        ) as f:
            json.dump(reduced_res, f, indent=2)
        acc = reduced_res["[acc]"]
        for k in acc.keys():
            acc[k]["name"] = k
        df = pd.DataFrame.from_records(list(acc.values()))
        if pl_module.logger is not None:
            logger = pl_module.logger
            if hasattr(logger, "log_table"):
                logger.log_table(
                    key=f"{split}/ocr-step-{trainer.global_step}",
                    dataframe=df,
                )
            logger.log_metrics(
                {f"{split}/trial_acc": acc["[all]"]["trial_acc"]})
            logger.log_metrics(
                {f"{split}/sample_acc": acc["[all]"]["sample_acc"]})
            # style acc
            for style_name in style_metric.keys():
                logger.log_metrics(
                    {f"{split}/{style_name}/mean": style_metric[style_name]['mean']})
                logger.log_metrics(
                    {f"{split}/{style_name}/best": style_metric[style_name]['best']})
        del self.ocr_model
        torch.cuda.empty_cache()

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if pl_module.config.get("sanity_check", False):
            if (
                trainer.current_epoch + 1
            ) % pl_module.config.train_eval_every_epoch != 0:
                return
        log_dir_path = os.path.join(
            self.log_dir, "train", f"step_{trainer.global_step}"
        )
        os.makedirs(log_dir_path, exist_ok=True)
        name = os.path.join(
            log_dir_path, "train", f"results-step_{trainer.global_step}.json"
        )
        if os.path.exists(name):
            return

        self.generate_images(
            pl_module, eval_data=self.train_eval, log_dir=log_dir_path)

        # ! used for synchronize
        signal = torch.tensor(0, device=pl_module.device)
        if pl_module.local_rank == 0:
            self.log_ocr_eval(trainer, pl_module, log_dir_path, split="train")
        torchdist.all_reduce(signal)
        torch.cuda.synchronize()
        print(f"rank: {trainer.global_rank} done")

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if (
            trainer.state.stage == "sanity_check"
        ):  # don't log images when running sanity check
            return
        log_dir_path = os.path.join(
            self.log_dir, "val", f"step_{trainer.global_step}")
        os.makedirs(log_dir_path, exist_ok=True)
        name = os.path.join(
            log_dir_path, "val", f"results-step_{trainer.global_step}.json"
        )
        if os.path.exists(name):
            return

        # distributed part
        self.generate_images(
            pl_module, eval_data=self.val_eval, log_dir=log_dir_path)

        # ! used for synchronize
        signal = torch.tensor(0, device=pl_module.device)
        if pl_module.local_rank == 0:
            self.log_ocr_eval(trainer, pl_module, log_dir_path, split="val")
        torchdist.all_reduce(signal)
        torch.cuda.synchronize()
        print(f"rank: {trainer.global_rank} done")


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            print(f"BASE LOG DIR: {self.logdir}")
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
            )

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )
        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass
