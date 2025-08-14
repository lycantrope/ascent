import argparse
import logging
import time

import torch
import torch.multiprocessing as mp

from ascent.tools.train_helper import (
    cleanup_ddp,
    parse_args,
    periodic_save_model,
    save_model,
    select_device,
    setup_dataset,
    setup_ddp,
    setup_logging,
    setup_loss,
    setup_model,
    setup_optimizer,
    setup_scheduler,
    setup_tensorboard,
    setup_transforms,
    train_one_epoch,
)
from ascent.utils.common import load_config


def train_model(rank, world_size, config, **kwargs):
    """
    Train a pytorch model.
    """
    # Load the YAML config file and update with command line arguments
    cfg = load_config(config)
    parse_args(cfg, **kwargs)

    # Set up logging
    # Read https://realpython.com/python-logging/ for basic logging information
    setup_logging(cfg)

    # After parsing config and command line arguments
    logging.info("Starting training with the following configuration:")
    logging.info(f"{cfg}")
    # Set up DDP and device
    if world_size > 1:
        logging.info(f"Running DDP on rank {rank}.")
        setup_ddp(rank, world_size, cfg.get("port", 12345))
    device = select_device(cfg, rank)

    # set up tensorboard
    # read https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html for basic introduction to tensorboard in pytorch
    if rank == 0:
        writer = setup_tensorboard(cfg)

    # set up datasets and dataloaders
    # read https://pytorch.org/tutorials/beginner/data_loading_tutorial.html for basic introduction to dataloaders in pytorch
    dataset, dataloader = setup_dataset(cfg, world_size, rank)

    # set up transformations for augmentation
    transforms = setup_transforms(cfg)

    # set up model
    # read https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html for basic introduction to models in pytorch
    model = setup_model(cfg, device, world_size, rank)

    # set up loss funcion
    loss_fn = setup_loss(cfg)

    # set up optimizer and scheduler
    # read https://pytorch.org/docs/stable/optim.html for basic introduction to optimizers in pytorch
    # read https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate for basic introduction to schedulers in pytorch
    optimizer = setup_optimizer(cfg, model)
    scheduler = setup_scheduler(cfg, optimizer)

    # either time_limit or epochs should be specified
    assert "time_limit" in cfg or "epochs" in cfg, (
        "either time_limit or epochs should be specified in the config file"
    )

    # Retrieve time limit from configuration, defaulting to a large number if not specified
    time_limit = cfg.get("time_limit", float("inf"))  # Time limit in seconds

    if "load_epoch" in cfg:
        start_epoch = cfg["load_epoch"]
    else:
        start_epoch = 0

    if "load_time" in cfg:
        now = time.time()
        start_time = now - cfg["load_time"]
        last_save_time = now
        last_epoch_time = now
    else:
        start_time = time.time()
        last_save_time = start_time
        last_epoch_time = start_time

    if rank == 0:
        model_save_path = cfg.get("model_save_path", "./model.pth")
        model_save_path = model_save_path.replace(".pth", "_init.pth")
        save_model(model, optimizer, scheduler, model_save_path)

    # Training loop
    is_initial_epoch = True
    for epoch in range(start_epoch, cfg.get("epochs", 9999999999)):
        # Check if the current elapsed time for training has exceeded the time limit
        if time.time() - start_time > time_limit:
            logging.info("Training ended due to time limit.")
            break

        # Training phase
        train_loss = train_one_epoch(
            model,
            dataloader,
            transforms,
            loss_fn,
            optimizer,
            device,
            epoch,
            world_size,
        )

        # Tensorboard logging and model saving should be done only by rank 0
        if rank == 0:
            # Log the current epoch
            if len(train_loss) == 1:
                writer.add_scalar("Train/Loss", train_loss[0], epoch)
            else:
                for i, loss in enumerate(train_loss):
                    writer.add_scalar(f"Train/Loss_{i}", loss, epoch)
                writer.add_scalar("Train/Loss", sum(train_loss), epoch)

            # Call to periodic_save_model
            last_save_time = periodic_save_model(
                model,
                optimizer,
                scheduler,
                epoch,
                start_time,
                last_save_time,
                cfg,
            )

            # Log the current learing rate
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Train/Learning_rate", current_lr, epoch)

            # Log the time taken for the current epoch
            current_time = time.time()
            epoch_time = current_time - last_epoch_time
            last_epoch_time = current_time
            writer.add_scalar("Train/Epoch_time", epoch_time, epoch)

        # Step the scheduler
        scheduler.step()

        if is_initial_epoch and rank == 0:
            # Log memory usage
            # Print the amount of memory allocated
            logging.info("Memory allocated:")
            logging.info(torch.cuda.memory_allocated())

            # Print the amount of memory reserved
            logging.info("Memory reserved:")
            logging.info(torch.cuda.memory_reserved())

            # Print the maximum amount of memory reserved
            logging.info("Max memory reserved:")
            logging.info(torch.cuda.max_memory_reserved())

            # Print a summary of CUDA memory usage
            logging.info("CUDA memory summary:")
            logging.info(torch.cuda.memory_summary())
            is_initial_epoch = False

    # Save the last model
    if rank == 0:
        model_save_path = cfg.get("model_save_path", "./model.pth")
        save_model(model, optimizer, scheduler, model_save_path)
        # close tensorboard
        writer.close()

    logging.info("Training completed.")

    if world_size > 1:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    # config file path
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    world_size = num_gpus  # One process per GPU
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, args.config), nprocs=world_size, join=True)
    else:
        train_model(0, 1, args.config)


if __name__ == "__main__":
    main()
