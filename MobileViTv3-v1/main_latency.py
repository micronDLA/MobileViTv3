# For licensing see accompanying LICENSE file.

import multiprocessing
import torch
from utils import logger
from options.opts import get_training_arguments
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master, distributed_init
from cvnets import get_model, EMA
from loss_fn import build_loss_fn
from optim import build_optimizer
from optim.scheduler import build_scheduler
from data import create_train_val_loader
from utils.checkpoint_utils import load_checkpoint, load_model_state
from engine import Trainer
import math
from torch.cuda.amp import GradScaler
from common import DEFAULT_EPOCHS, DEFAULT_ITERATIONS, DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_EPOCHS

#import torch.utils.benchmark as benchmark
import numpy as np

@torch.no_grad()
def run_inference(model, input_tensor):
    return model(input_tensor)

def main(opts, **kwargs):
    num_gpus = getattr(opts, "dev.num_gpus", 0) # defaults are for CPU
    dev_id = getattr(opts, "dev.device_id", torch.device('cpu'))
    device = getattr(opts, "dev.device", torch.device('cpu'))

    is_master_node = is_master(opts)

    # set-up data loaders
    train_loader, val_loader, train_sampler = create_train_val_loader(opts)

    # set-up the model
    model = get_model(opts)

    model = model.to(device=device)
    model.eval()

    input_tensor = torch.randn(1,3,256,256, dtype=torch.float).to(device)

    batch_size = 100
    input_tensor_t = torch.randn(batch_size,3,256,256, dtype=torch.float).to(device)

    # reference: https://deci.ai/blog/measure-inference-time-deep-neural-networks

    # initialize
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    repetitions = 10000
    timings = np.zeros((repetitions,1))
    total_time = 0

    # GPU warm-up
    for _ in range(10):
        _ = model(input_tensor)

    # Latency
    # Measure performance
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(input_tensor)
            ender.record()
            # wait for gpu sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f"Mean Latency: {mean_syn}")

    # Throughput
    # Measure performance
    repetitions = 1000
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(input_tensor_t)
            ender.record()
            # wait for gpu sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            total_time += curr_time / 1000

    throughput = repetitions * batch_size / total_time
    print(f"Throughput: {throughput}")




def distributed_worker(i, main, opts, kwargs):
    setattr(opts, "dev.device_id", i)
    if torch.cuda.is_available():
        torch.cuda.set_device(i)

    ddp_rank = getattr(opts, "ddp.rank", None)
    if ddp_rank is None:  # torch.multiprocessing.spawn
        ddp_rank = kwargs.get('start_rank', 0) + i
        setattr(opts, "ddp.rank", ddp_rank)

    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts, **kwargs)


def main_worker(**kwargs):
    opts = get_training_arguments()
    print(opts)
    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error('--rank should be >=0. Got {}'.format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = '{}/{}'.format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    world_size = getattr(opts, "ddp.world_size", -1)
    use_distributed = getattr(opts, "ddp.enable", False)
    if num_gpus <= 1:
        use_distributed = False
    setattr(opts, "ddp.use_distributed", use_distributed)

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    norm_name = getattr(opts, "model.normalization.name", "batch_norm")
    if use_distributed:
        if world_size == -1:
            logger.log("Setting --ddp.world-size the same as the number of available gpus")
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)
        elif world_size != num_gpus:
            logger.log("--ddp.world-size does not match num. available GPUs. Got {} !={}".format(world_size, num_gpus))
            logger.log("Setting --ddp.world-size=num_gpus")
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)

        if dataset_workers == -1 or dataset_workers is None:
            setattr(opts, "dataset.workers", n_cpus // world_size)

        start_rank = getattr(opts, "ddp.rank", 0)
        setattr(opts, "ddp.rank", None)
        kwargs['start_rank'] = start_rank
        torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts, kwargs),
            nprocs=num_gpus,
        )
    else:
        if dataset_workers == -1:
            setattr(opts, "dataset.workers", n_cpus)

        if norm_name in ["sync_batch_norm", "sbn"]:
            setattr(opts, "model.normalization.name", "batch_norm")

        # adjust the batch size
        train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
        val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
        setattr(opts, "dataset.train_batch_size0", train_bsize)
        setattr(opts, "dataset.val_batch_size0", val_bsize)
        setattr(opts, "dev.device_id", None)
        main(opts=opts, **kwargs)


if __name__ == "__main__":
    #multiprocessing.set_start_method('spawn', force=True)

    main_worker()
