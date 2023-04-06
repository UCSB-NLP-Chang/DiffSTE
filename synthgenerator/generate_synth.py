from multiprocessing import Process, Queue
from omegaconf import OmegaConf
from synthtiger.gen import _task_generator, _run, _generate
from synthtiger.main import parse_args
import time
import pprint
import importlib
import itertools
import synthtiger


def read_template(path, name, config):
    template_cls = get_obj(path + "." + name, reload=True)
    template = template_cls(config=config)
    return template


synthtiger.read_template = read_template
synthtiger.gen.read_template = read_template


def get_obj(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def run(args):
    if args.config is not None:
        config = OmegaConf.load(args.config)
        OmegaConf.resolve(config)

    if args.seed is None:
        if 'seed' not in config:
            args.seed = 42
        else:
            args.seed = config['seed']
    if 'seed' not in config:
        config['seed'] = args.seed

    pprint.pprint(config)
    synthtiger.set_global_random_seed(args.seed)
    template = read_template(args.script, args.name, config)
    generator = synthtiger.generator(
        args.script,
        args.name,
        config=config,
        count=args.count,
        worker=args.worker,
        seed=args.seed,
        retry=True,
        verbose=args.verbose,
    )

    if args.output is not None:
        template.init_save(args.output)

    from tqdm import tqdm
    for idx, (task_idx, data) in tqdm(enumerate(generator)):
        if args.output is not None:
            template.save(args.output, data, task_idx)
        # print(f"Generated {idx + 1} data (task {task_idx})")

    if args.output is not None:
        template.end_save(args.output)


def main():
    start_time = time.time()
    args = parse_args()
    run(args)
    end_time = time.time()
    print(f"{end_time - start_time:.2f} seconds elapsed")


if __name__ == "__main__":
    main()
