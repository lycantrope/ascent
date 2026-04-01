import functools
import glob
import logging
import os
import re
import time
from multiprocessing import resource_tracker

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2 import Compose

from ascent.models.loss import CombinedLoss
from ascent.utils.common import to_device


def setup_ddp(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup(fn):
    try:
        mp.set_sharing_strategy("file_system")
    except Exception as e:
        print(f"Warning: Could not set sharing strategy: {e}")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        """Manual cleanup of shared memory objects owned by the current user."""
        try:
            ret = fn(*args, **kwargs)
            return ret
        finally:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
                # This helps the resource tracker finalize its accounting
                resource_tracker._resource_tracker._stop()  # type: ignore
            except Exception:
                pass

    return wrapper


def gather_embeddings(embeddings):
    # Gather embeddings from all processes
    gathered = dist_nn.all_gather(embeddings)
    return torch.cat(gathered, dim=0)


def no_collate_fn(batch):
    return batch


def parse_args(cfg, **kwargs):
    """
    Update the configuration dictionary with provided command line arguments.
    """
    for k, v in kwargs.items():
        if v is not None:
            cfg[k] = v
            logging.info(f"Config parameter updated: {k} = {v}")


def setup_logging(cfg):
    """
    Set up logging based on the configuration. Logs both to console and file if 'logfile' is specified.

    Args:
        cfg (dict): Configuration dictionary containing logging settings.
    """
    log_format = "%(asctime)s | (%(levelname)s) %(message)s"
    log_level = logging.INFO

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter(log_format)

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # If logfile is specified, also log to the specified file
    logfile = cfg.get("logfile", None)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logging.info("Training Session Started")


def setup_tensorboard(cfg):
    """
    Set up TensorBoard for logging metrics and visualizations. Defaults to a directory next to the logfile or a default
    local directory if neither 'tensorboarddir' nor 'logfile' are specified.

    Args:
        cfg (dict): Configuration dictionary containing TensorBoard and logging settings.

    Returns:
        SummaryWriter: A SummaryWriter instance for TensorBoard logging.

    Notes:
    - For more detailed examples on how to use TensorBoard with PyTorch, refer to:
    - https://pytorch.org/docs/stable/tensorboard.html
    """
    tensorboard_dir = cfg.get("tensorboard_logdir", None)
    if not tensorboard_dir:
        if "logfile" in cfg and cfg["logfile"]:
            tensorboard_dir = os.path.join(
                os.path.dirname(cfg["logfile"]), "tensorboard_logs"
            )
        else:
            tensorboard_dir = "tensorboard_logs"

    os.makedirs(tensorboard_dir, exist_ok=True)
    return SummaryWriter(tensorboard_dir)

    # For more detailed examples on how to use TensorBoard with PyTorch, refer to:
    # https://pytorch.org/docs/stable/tensorboard.html


def select_device(cfg, rank):
    """
    Selects the device to be used for training based on the provided configuration. Supports automatic selection
    of the device based on availability, or manual selection of 'cuda', 'mps', or 'cpu'.

    Returns:
        str: Identifier of the selected device ('cuda', 'mps', or 'cpu').

    Notes:
    Please implement models that can dynamically adapt to different hardware capabilities
    ensures that your training process is robust and efficient.
    """
    device = cfg.get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
            logging.info(
                f"Automatically selected device: {device}, GPUs available: {torch.cuda.device_count()}"
            )
            torch.cuda.set_device(device)
        elif torch.backends.mps.is_available():
            device = "mps"  # Support for Apple silicon
            logging.info(f"Automatically selected device: {device}")
        else:
            device = "cpu"
            logging.info(f"Automatically selected device: {device}")
    elif device not in ["cuda", "mps", "cpu"]:
        raise Exception(f"Device {device} is not supported.")
    else:
        if device == "cuda":
            if torch.cuda.is_available():
                torch.cuda.set_device(rank)
                device = torch.device(f"cuda:{rank}")
                torch.cuda.set_device(device)
            else:
                raise Exception("CUDA is not available.")

        logging.info(f"Selected device: {device}")

    return device


def _init_dataloader(cfg, dataset, world_size, rank) -> DataLoader:
    batch_sampler = cfg.get("batch_sampler", None)
    if batch_sampler:
        # If a batch sampler is specified, do not use 'batch_size', 'shuffle', or 'drop_last'
        batch_sampler_instance = batch_sampler["class"](
            data_source=dataset,
            num_replicas=world_size,
            rank=rank,
            **batch_sampler.get("params", {}),
        )
        cfg_extra = cfg.copy()
        cfg_extra.pop("batch_sampler")
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler_instance,
            collate_fn=no_collate_fn,
            **cfg_extra,
        )
    else:
        # Normal DataLoader initialization
        dataloader = DataLoader(dataset, **cfg)

    return dataloader


def setup_dataloader(cfg, dataset, world_size, rank) -> DataLoader | list[DataLoader]:
    if isinstance(dataset, list):
        dataloaders = []
        for data in dataset:
            dataloaders.append(_init_dataloader(cfg, data, world_size, rank))
        return dataloaders
    else:
        return _init_dataloader(cfg, dataset, world_size, rank)


# Initialize datasets
def _init_dataset(dataset_cfg):
    dataset_class = dataset_cfg["class"]
    dataset_params = dataset_cfg.get("params", {})
    return dataset_class(**dataset_params)


def setup_dataset(cfg, world_size, rank):
    """
    Setup dataset based on the provided configuration.

    Args:
        cfg (dict): Configuration dictionary containing dataset settings and parameters.

    Returns:
        dict: A dictionary containing DataLoader instances for training set.
    """
    # Assertions to ensure required fields are present in the configuration
    assert "dataset" in cfg, "Configuration must include 'dataset'."
    assert "dataloader" in cfg, "Configuration must include 'dataloader'."

    # Initialize dataloaders dictionary to hold DataLoader instances for each dataset split
    dataset = {}
    dataloader = {}

    if isinstance(cfg["dataset"], list):
        dataset = [_init_dataset(item) for item in cfg["dataset"]]
    else:
        dataset = _init_dataset(cfg["dataset"])

    # Initialize dataloader
    dataloader_cfg = cfg["dataloader"]
    dataloader = setup_dataloader(dataloader_cfg, dataset, world_size, rank)

    logging.info("Dataset and DataLoader are set up.")
    return dataset, dataloader


def _setup_transforms(transforms_cfg):
    transforms = []
    for transform_cls, params in transforms_cfg:
        transforms.append(transform_cls(**params))
    if len(transforms) > 0:
        transforms = Compose(transforms)
    else:
        transforms = None
    return transforms


def setup_transforms(cfg):
    """
    Setup transforms based on the provided configuration.
    """
    assert "transforms" in cfg, "Configuration must include 'transforms'."
    transforms_cfg = cfg["transforms"]
    transforms = _setup_transforms(transforms_cfg)
    logging.info("Transforms are set up.")
    return transforms


def get_last_saved_model_path_epoch_time(cfg):
    model_save_path = cfg.get("model_save_path", "./model.pth")

    re_model_save_path = re.compile(r"(.*)_epoch_(\d+)_time_(\d+).pth")
    saved_models = glob.glob(model_save_path.replace(".pth", "_epoch_*_time_*.pth"))
    saved_models = [m for m in saved_models if re_model_save_path.match(m)]
    if len(saved_models) > 0:
        saved_models.sort(
            key=lambda x: int(re_model_save_path.match(x).group(2)), reverse=True
        )
        match = re_model_save_path.match(saved_models[0])
        epoch, savetime = int(match.group(2)), int(match.group(3))
        return saved_models[0], epoch, savetime
    else:
        return None, None, None


def setup_model(cfg, device, world_size, rank):
    """
    Instantiates and sets up the model for training or evaluation based on the provided configuration.
    Supports loading pre-trained models if specified, and moves the model to the appropriate device
    (CPU, GPU, etc.).

    Args:
        cfg (dict): Configuration dictionary containing model settings, which includes a model class
                    to instantiate, parameters for model initialization,
                    and optionally, a path to a pre-trained model state.
        device (str): The device to move the model to ('cpu', 'cuda', 'mps', etc.).

    Returns:
        torch.nn.Module: The model ready for training or evaluation.

    Example configuration:
    ```
    cfg = {
        'model': {
            'class': MyModelClass                        # Model class to instantiate
            'params': {'arg1': value1, 'arg2': value2},  # Parameters for model initialization
            'load_path': 'path/to/pretrained/model.pth'  # Optional path to a pre-trained model
        }
    }
    ```

    For more information on PyTorch models and device management:
    - PyTorch Models: https://pytorch.org/docs/stable/nn.html
    - PyTorch Device Management: https://pytorch.org/docs/stable/notes/cuda.html
    """
    assert "model" in cfg, "Model configuration is required."
    assert "class" in cfg["model"], "'model' configuration must include 'class'."

    # Initialize the model based on the provided configuration
    model_class = cfg["model"]["class"]
    model_params = cfg["model"].get("params", {})
    model = model_class(**model_params)

    # Load a pre-trained model if specified
    if "load_path" in cfg["model"]:
        strict = cfg["model"].get("load_state_dict_strict", True)
        state_dict = torch.load(cfg["model"]["load_path"], map_location=device)
        model.load_state_dict(state_dict, strict=strict)
        logging.info(f"Loaded pre-trained model state from {cfg['model']['load_path']}")
    elif "continue_training" in cfg and cfg["continue_training"]:
        model_path, epoch, savetime = get_last_saved_model_path_epoch_time(cfg)
        if model_path is not None:
            state_dict = torch.load(model_path, map_location=device)
            # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
            cfg["load_epoch"] = epoch
            cfg["load_time"] = savetime
            logging.info(f"Loaded pre-trained model state from {model_path}")
        else:
            logging.info("No pre-trained model found to continue training.")

    # Move the model to the specified device
    model.to(device)

    # Wrap the model using DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[rank])
        # Replace BatchNorm to SyncBatchNorm in distributed training setting
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        dist.barrier()
        logging.info(f"Model set up with DDP on rank {rank}.")

    logging.info(f"Model setup complete. Model moved to {device}.")
    return model


def setup_loss(cfg):
    """
    Initializes one or more loss functions based on the provided configuration, supporting the use of custom
    wrappers as specified through separate configurations. This function accommodates complex loss computation
    scenarios, including the dynamic provision of additional parameters.

    Args:
        cfg (dict): Configuration dictionary containing settings for one or more loss functions,
                    including their classes, initialization parameters, and optional wrappers.

    Returns:
        A function that, when called with predictions and targets, computes either a single loss or
        the weighted sum of multiple losses, based on the provided configuration.

    Example configuration:
    ```
    cfg = {
        'losses': [
            {
                'class': torch.nn.BCELoss,
                'params': {'reduction': 'mean'},
                'weight': 1.0
            },
            {
                'class': CustomLoss,
                'params': {},
                'weight': 0.5,
                'wrapper': CustomLossWrapper,  # wrapper class to apply
            }
        ]
    }
    ```

    """
    assert "losses" in cfg, "Configuration must include 'losses'."

    loss_fns = []
    weights = []

    for loss_cfg in cfg["losses"]:
        loss_class = loss_cfg["class"]
        loss_params = loss_cfg.get("params", {})
        loss = loss_class(**loss_params)

        # Check for and apply wrapper if specified
        if "wrapper" in loss_cfg:
            wrapper_class = loss_cfg["wrapper"]
            wrapper = wrapper_class(loss)
            loss = wrapper

        weight = loss_cfg.get("weight", 1.0)
        loss_fns.append(loss)
        weights.append(weight)

    combined_loss_fn = CombinedLoss(loss_fns, weights)
    logging.info("Loss functions are set up.")
    return combined_loss_fn


def setup_optimizer(cfg, model):
    """
    Initializes the optimizer with default settings and custom learning rates for specific layers, if specified.

    Args:
        model (torch.nn.Module): The model for which the optimizer is being set up.
        cfg (dict): Configuration dictionary possibly containing optimizer settings and custom learning rates.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    # Default optimizer configuration
    default_cfg = {
        "class": Adam,
        "params": {
            "lr": 1e-3,
            # Add other default parameters as needed
        },
        "layer_lrs": {},  # Example: {'layer1.*': 1e-4, 'layer2.*': 5e-4}
    }

    # Update default config with any user-provided settings
    optimizer_cfg = {**default_cfg, **cfg.get("optimizer", {})}
    lr = optimizer_cfg.get("params", {}).get("lr", 1e-3)

    # Construct parameter groups
    param_groups = []
    for name, param in model.named_parameters():
        # Check if this parameter matches any custom learning rate pattern
        custom_lr = next(
            (
                lr
                for pattern, lr in optimizer_cfg["layer_lrs"].items()
                if torch.fnmatch.fnmatch(name, pattern)
            ),
            None,
        )
        param_group = {"params": [param]}
        if custom_lr is not None:
            param_group["lr"] = custom_lr
        param_groups.append(param_group)

    # Initialize the optimizer with parameter groups
    optimizer_class = optimizer_cfg["class"]
    optimizer = optimizer_class(param_groups, **optimizer_cfg["params"])

    # load optimizer
    if "loadpath" in optimizer_cfg:
        optimizer.load_state_dict(torch.load(optimizer_cfg["loadpath"]))
        logging.info("Loaded PyTorch Optimizer State from %s".format())
    elif "continue_training" in cfg and cfg["continue_training"]:
        model_save_path = cfg.get("model_save_path", "./model.pth")
        if "load_epoch" in cfg:
            epoch = cfg["load_epoch"]
            savetime = cfg["load_time"]
        else:
            _, epoch, savetime = get_last_saved_model_path_epoch_time(cfg)

        if epoch is not None:
            optimizer_path = model_save_path.replace(
                ".pth", f"_epoch_{epoch}_time_{savetime}_optimizer.pth"
            )
            optimizer.load_state_dict(torch.load(optimizer_path))
            logging.info(f"Loaded PyTorch Optimizer State from {optimizer_path}")
        else:
            logging.info("No pre-trained optimizer found to continue training.")

    logging.info(f"Optimizer set up: {optimizer_class} with lr={lr}")
    return optimizer


class NoOpScheduler:
    def __init__(self, optimizer, *args, **kwargs):
        self.optimizer = optimizer

    def step(self, *args, **kwargs):
        pass  # Perform no action on step


def setup_scheduler(cfg, optimizer):
    """
    Sets up the learning rate scheduler based on the provided configuration, allowing for advanced
    tuning by experienced users. Defaults to a "no-op" scheduler that does nothing, simplifying usage
    for those not concerned with custom scheduling.

    Args:
        cfg (dict): Configuration dictionary that may contain scheduler settings.
        optimizer (torch.optim.Optimizer): The optimizer for which the scheduler will be applied.

    Returns:
        A learning rate scheduler. If no scheduler is configured, returns a NoOpScheduler.
    """

    if "scheduler" in cfg:
        scheduler_cfg = cfg["scheduler"]
        scheduler_class = scheduler_cfg["class"]
        scheduler_params = scheduler_cfg.get("params", {})

        scheduler = scheduler_class(optimizer, **scheduler_params)
        if "loadpath" in scheduler_cfg:
            scheduler.load_state_dict(torch.load(scheduler_cfg["loadpath"]))
            logging.info("Loaded PyTorch Scheduler State from %s".format())

        if "continue_training" in cfg and cfg["continue_training"]:
            model_save_path = cfg.get("model_save_path", "./model.pth")
            if "load_epoch" in cfg:
                epoch = cfg["load_epoch"]
                savetime = cfg["load_time"]
            else:
                _, epoch, savetime = get_last_saved_model_path_epoch_time(cfg)

            if epoch is not None:
                scheduler_path = model_save_path.replace(
                    ".pth", f"_epoch_{epoch}_time_{savetime}_scheduler.pth"
                )
                scheduler.load_state_dict(torch.load(scheduler_path))
                logging.info(f"Loaded PyTorch Scheduler State from {scheduler_path}")
            else:
                logging.info("No pre-trained scheduler found to continue training.")
    else:
        # Default to NoOpScheduler if no scheduler is specified
        scheduler_class = NoOpScheduler
        scheduler = scheduler_class(optimizer)

    logging.info(f"Scheduler set up: {scheduler_class}")
    return scheduler


def train_one_epoch(model, dataloader, transforms, loss_fn, optimizer, device, epoch):
    # torch.autograd.set_detect_anomaly(True)
    model.train()

    if not isinstance(dataloader, list):
        list_dataloader = [dataloader]
    else:
        list_dataloader = dataloader

    # 1. Set epoch for distributed samplers BEFORE creating iterators
    for i_dl, dl in enumerate(list_dataloader):
        if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
            logging.debug(f"Setting epoch {epoch} for dataloader {i_dl} sampler.")
            dl.sampler.set_epoch(epoch)
        elif hasattr(dl, "batch_sampler") and hasattr(dl.batch_sampler, "set_epoch"):
            # Batch samplers might also need set_epoch (less common for custom ones)
            logging.debug(f"Setting epoch {epoch} for dataloader {i_dl} batch sampler.")
            dl.batch_sampler.set_epoch(epoch)

    # 2. Create iterators and get lengths/weights
    data_iters = []
    dataloader_indices = []
    dl_total_loss = []
    for i_dl, dl in enumerate(list_dataloader):
        try:
            length = len(dl)  # Number of batches
            data_iters.append(iter(dl))
            dataloader_indices.extend(
                [i_dl] * length
            )  # Extend with the index of the dataloader
            dl_total_loss.append(0.0)
        except TypeError:
            # Iterable dataset without __len__ is not directly supported by this weighted approach
            # Fallback or error needed. Let's raise an error for now.
            logging.error(
                f"DataLoader {dl} does not support __len__, cannot use weighted random sampling."
            )
            raise TypeError(
                "All dataloaders must support __len__ for weighted random batch sampling."
            )

    # 3. Determine total steps and weights
    total_steps = len(dataloader_indices)
    # shuffle the indices to ensure randomness
    np.random.seed(epoch)  # Set seed for multi-GPU training
    np.random.shuffle(dataloader_indices)

    # 4. Main loop processing randomly sampled batches
    for step in range(total_steps):
        optimizer.zero_grad()

        i_dl = dataloader_indices[step]
        batch_data = next(data_iters[i_dl])

        # Forward pass
        batch_data = to_device(batch_data, device)
        if transforms is not None:
            v1 = transforms(batch_data)
            v2 = transforms(batch_data)
        else:
            v1 = batch_data
            v2 = batch_data
        z1 = model(v1)
        z2 = model(v2)

        # Backward pass and optimize
        if dist.is_initialized():
            z1_full = gather_embeddings(z1)
            z2_full = gather_embeddings(z2)
            loss = loss_fn(z1_full, z2_full)
        else:
            loss = loss_fn(z1, z2)

        loss.backward()

        optimizer.step()
        dl_total_loss[i_dl] += loss.item()

    for i_dl, dl in enumerate(list_dataloader):
        dl_total_loss[i_dl] /= len(dl)
    logging.info(f"Training ({device}) - Epoch {epoch} Loss: {sum(dl_total_loss)}")

    return dl_total_loss


def save_model(model, optimizer, scheduler, path):
    """
    Saves the model to the specified path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(model, DDP):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

    torch.save(optimizer.state_dict(), path.replace(".pth", "_optimizer.pth"))
    if not isinstance(scheduler, NoOpScheduler):
        torch.save(scheduler.state_dict(), path.replace(".pth", "_scheduler.pth"))
    logging.info(f"Model saved to {path}")


def periodic_save_model(
    model, optimizer, scheduler, epoch, start_time, last_save_time, cfg
):
    """
    Saves the model periodically based on epoch frequency or time frequency as specified in the configuration.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        scheduler (torch.optim.lr_scheduler): The scheduler to be saved.
        epoch (int): The current epoch number.
        start_time (float): The start time of the training session (timestamp).
        last_save_time (float): The last time the model was saved (timestamp).
        cfg (dict): Configuration dictionary containing model saving frequency settings.

    Returns:
        float: Updated last_save_time if the model was saved based on time frequency, else returns the input last_save_time.
    """
    current_time = time.time()
    model_save_path = cfg.get("model_save_path", "./model.pth")
    save_every_n_epochs = cfg.get("save_every_n_epochs", None)
    save_time_span = cfg.get("save_time_span", None)

    # Periodic saving based on epoch
    model_time = current_time - start_time
    if save_every_n_epochs is not None and (epoch + 1) % save_every_n_epochs == 0:
        model_save_path = model_save_path.replace(".pth", "")
        periodic_save_path = (
            f"{model_save_path}_epoch_{epoch + 1}_time_{int(model_time)}.pth"
        )
        save_model(model, optimizer, scheduler, periodic_save_path)

    # Periodic saving based on time
    if save_time_span is not None and (
        int((current_time - start_time) // save_time_span)
        > int((last_save_time - start_time) // save_time_span)
    ):
        model_save_path = model_save_path.replace(".pth", "")
        periodic_save_path = (
            f"{model_save_path}_epoch_{epoch + 1}_time_{int(model_time)}.pth"
        )
        save_model(model, optimizer, scheduler, periodic_save_path)
        last_save_time = current_time  # Update last save time

    return last_save_time
