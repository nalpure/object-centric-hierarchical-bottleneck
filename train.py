import argparse
import os
import numpy as np
from tqdm import tqdm

import math_utils
import io_utils
import train_classes as tc
import factory as fc


VALID_TYPES = ["slot_attention", "explicit_latents", "implicit_dynamics"]


def main():
    # ----- Parse arguments -----

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Model and training configuration")
    parser.add_argument("-n", "--name", help="Name of the training run.")
    parser.add_argument("-d", "--data", help="Dataset path.")
    parser.add_argument("-b", "--base", help="Base model name.")    
    parser.add_argument(
        "-e",
        "--base-epoch",
        help="Base model epoch. If not provided, best epoch is selected.",
    )
    parser.add_argument("-s", "--scheduler-adjust", action="store_true", help="Adjust LR scheduler for loaded checkpoint.")
    args = parser.parse_args()


    # ----- Load and add to configuration -----
    
    config = io_utils.load_config_by_name(args.config)

    if "seed" not in config:
        config["seed"] = np.random.randint(2**31)

    if args.name is None:
        if not "name" in config:
            raise "Provide a name for the run!"
    else:
        config["name"] = args.name

    if args.data is None:
        if not "path" in config["data"]:
            raise "Provide a dataset path!"
    else:
        config.setdefault("data", {})
        config["data"]["path"] = args.data

    if "num_workers" not in config:
        config["num_workers"] = 0

    if config["type"] == "slot_attention":
        out_dir = "out"
    else:
        out_dir = os.path.dirname(config["data"]["path"])

    config["base_ckpt"] = ""

    if args.base is None:
        print("No base model specified. Training from scratch.")
    else:        
        if args.base_epoch is None:
            config["base_ckpt"] = os.path.join(out_dir, args.base, "ckpt_best.pt")
        else:
            config["base_ckpt"] = os.path.join(out_dir, args.base, f"ckpt_epoch_{args.base_epoch}.pt")
        print(f"Using base model checkpoint: {config['base_ckpt']}")

    if config["type"] not in VALID_TYPES:
        raise ValueError(f"Unknown training type '{config['type']}'. Valid types are: {VALID_TYPES}")

    # For explicit latents, set sequence length to 1 (since disentanglement can only be applied to first frame)
    if config["type"] == "explicit_latents":
        config["data"]["seq_length"] = 1
    
    if "type" not in config["train"]["opt"]:
        config["train"]["opt"]["type"] = "adam"


    # ----- Load dataset, model, and train manager -----
    
    math_utils.set_seed(config['seed'])
    dataloader = fc.build_dataloader(config)
    model = fc.build_model(config, eval_mode=False)
    optimizer = fc.build_optimizer(config, model)
    scheduler = fc.build_scheduler(config, optimizer, adjust_for_checkpoint=args.scheduler_adjust)
    train_step = fc.build_train_step(config, model)
    anneal_epochs = config["train"]["opt"].get("weight_anneal_epochs", 0)
    train_manager = tc.TrainManager(train_step, dataloader, optimizer, scheduler, anneal_epochs)


    # ----- Create output folder -----
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    output_path = io_utils.make_unique_dir(out_dir, config["name"])
    io_utils.save_config(config, os.path.join(output_path, "config.toml"))
    print(f"Created new output directory at '{output_path}'.")


    # ----- Train model -----
    print("Starting training...")
    for epoch in tqdm(range(config["train"]["epochs"])):
        train_manager.train_epoch()
        if (epoch + 1) % config["train"]["ckpt_rate"] == 0:
            train_manager.save_checkpoint(f"{output_path}/ckpt_epoch_{epoch+1}.pt")
        train_manager.save_if_best(f"{output_path}/ckpt_best.pt")
        train_manager.save_losses_to_csv(f"{output_path}/losses.csv")

    print(f"Finished. Best epoch: {train_manager.best_epoch_idx} with loss {train_manager.best_loss:.4f}.")


if __name__ == "__main__":
    main()