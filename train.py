import os
import glob
import argparse
import datetime
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf
from src.trainers.utils import *


def create_parser(**kwargs):
    parser = argparse.ArgumentParser(**kwargs)
    parser.add_argument("--project", type=str, default="charinpaint")
    parser.add_argument("--name", type=str, const=True, nargs="?")
    parser.add_argument("--resume", type=str, const=True, default="", nargs="?")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stage", type=str, default="fit")
    parser.add_argument(
        "--base", type=str, nargs="*", metavar="config.yaml", default=list()
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_logdir", type=str, default="logs")
    parser.add_argument("--postfix", type=str, default="")
    return parser


def create_model(resume, config, ckptdir=None):
    if not resume:
        print(f"Inititalize model from {config.pretrained_model_path}")
        trainer_cls = get_obj_from_str(config.target)
        model = trainer_cls(config=config)
    else:
        print(f"Resume model from checkpoint {resume}")
        trainer_cls = get_obj_from_str(config.target)
        model = trainer_cls.load_from_checkpoint(resume)
        model.config['ckpt_dir'] = ckptdir
    return model


def create_data(config):
    data_cls = get_obj_from_str(config.target)
    data = data_cls(data_config=config)
    return data


def main():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    parser = create_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unkown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths) - paths[::-1].index(opt.base_logdir) + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_name = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_name)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        if opt.debug:
            logdir = os.path.join(
                opt.base_logdir, opt.project, "debug", nowname)
        else:
            logdir = os.path.join(opt.base_logdir, opt.project, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # merge CLI config and config file
    configs = [OmegaConf.load(b) for b in opt.base]
    for conf in configs:
        conf["base_log_dir"] = logdir  # save base_log_dir to config object
        OmegaConf.resolve(conf)
    cli = OmegaConf.from_dotlist(unkown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.pop("trainer", OmegaConf.create())

    # trainer_config["distributed_backend"] = "ddp"
    # cli configs overwrite config.yaml
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if trainer_config["accelerator"] == "gpu":
        devices = trainer_config["devices"]
        print(f"Running on GPUS: {devices}")
    else:
        print(f"Running on cpu")
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # create logger
    trainer_kwargs = dict()
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "version": 0,  # always 0, for resume
            }
        }
    }
    logger_cfg = lightning_config.logger or OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfgs["tensorboard"], logger_cfg)
    os.makedirs(os.path.join(logdir, "wandb"),
                exist_ok=True)  # create wandb dir
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "checkpoint_callback": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:02}-{step:06}",
                "verbose": True,
                "save_last": False,  # by default, don't save las
                "save_top_k": -1,  # by default, save all checkpoints
                "every_n_epochs": 1,  # by defcault, save every checkpoint
                "monitor": None,  # by default, no monitor
            },
        },
        "setup_callback": {
            "target": "src.trainers.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            },
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {"logging_interval": "step", "log_momentum": True},
        },
    }

    callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

    # create model
    model_config = config.pop("model", OmegaConf.create())
    model_opt = model_config
    model_opt['ckpt_dir'] = ckptdir
    if opt.resume:
        model = create_model(opt.resume_from_checkpoint, model_opt, ckptdir)
    else:
        model = create_model(opt.resume, model_opt)
    # configure learning rate
    model.learning_rate = model_opt.base_learning_rate

    # create data
    data_config = config.pop("data", OmegaConf.create())
    data_opt = data_config
    data = create_data(data_opt)
    data.prepare_data()
    data.setup(stage=opt.stage)

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    import signal

    signal.signal(signal.SIGUSR1, melk)
    try:
        trainer.fit(model, datamodule=data)
    except Exception as e:
        print(f"Training failed due to {e}")
        if not opt.debug:
            melk()
        raise


if __name__ == "__main__":
    main()
